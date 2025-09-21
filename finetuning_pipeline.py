import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, datasets
from timm.data import resolve_model_data_config
import timm
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler
import math


class PlanktonBottleneckModel(nn.Module):
    """Model with bottleneck as described in the research paper"""

    def __init__(self, backbone_name="beit_large_patch16_224", num_classes=8, bottleneck_dim=512):
        super().__init__()

        # Load pre-trained BEiT-Large backbone
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)

        # Get backbone output dimension
        backbone_dim = self.backbone.num_features

        # Bottleneck: Linear + LayerNorm + GELU + Classification layer
        self.bottleneck = nn.Sequential(
            nn.Linear(backbone_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU()
        )

        # Final classifier with Weight Normalization
        self.classifier = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, num_classes, bias=True)
        )

        # Initialize bottleneck and classifier weights
        self._init_weights()

    def _init_weights(self):
        """Initialize newly added layers"""
        for module in [self.bottleneck, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.backbone(x)
        bottleneck_features = self.bottleneck(features)
        logits = self.classifier(bottleneck_features)
        return logits

    def get_param_groups(self, lr_backbone=1e-3, lr_bottleneck=1e-2, lr_classifier=1e-2):
        """Get parameter groups with different learning rates"""
        return [
            {'params': self.backbone.parameters(), 'lr': lr_backbone},
            {'params': self.bottleneck.parameters(), 'lr': lr_bottleneck},
            {'params': self.classifier.parameters(), 'lr': lr_classifier}
        ]



def get_plankton_transforms(backbone_name: str = "beit_large_patch16_224"):
    """Data augmentation using BEiT's timm data config for mean/std and interpolation."""
    # Resolve model-specific data config (mean/std, input size, etc.)
    _tmp_model = timm.create_model(backbone_name, pretrained=True, num_classes=0)
    cfg = resolve_model_data_config(_tmp_model)

    input_size = cfg.get('input_size', (3, 224, 224))
    resize_hw = (int(input_size[1]), int(input_size[2]))
    mean = cfg.get('mean', [0.485, 0.456, 0.406])
    std = cfg.get('std', [0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize(resize_hw, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180),  # Plankton can be oriented in any direction
        transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),  # Scale variation
        transforms.Grayscale(num_output_channels=3),  # enforce grayscale while keeping 3 channels
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(resize_hw, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Grayscale(num_output_channels=3),  # enforce grayscale while keeping 3 channels
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, val_transform


def exponential_decay_lr(initial_lr, step, total_steps, gamma=10, beta=0.75):
    """Exponential learning rate decay as described in the paper"""
    decay_factor = (1 + gamma * step / total_steps) ** (-beta)
    return initial_lr * decay_factor


class ExponentialLRScheduler:
    """Custom exponential scheduler matching the paper"""

    def __init__(self, param_groups, total_steps, gamma=10, beta=0.75):
        self.param_groups = param_groups
        self.total_steps = total_steps
        self.gamma = gamma
        self.beta = beta
        self.step_count = 0
        self.initial_lrs = [group['lr'] for group in param_groups]

    def step(self):
        self.step_count += 1
        decay_factor = (1 + self.gamma * self.step_count / self.total_steps) ** (-self.beta)

        for i, param_group in enumerate(self.param_groups):
            param_group['lr'] = self.initial_lrs[i] * decay_factor


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, accumulation_steps=1, use_amp=True, scaler: GradScaler | None = None, is_main: bool = True):
    """Train for one epoch with gradient accumulation and per-step LR scheduling."""
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        with autocast(enabled=use_amp and device.type == 'cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels) / accumulation_steps

        if scaler is not None and device.type == 'cuda':
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            if scaler is not None and device.type == 'cuda':
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        # Statistics (scale back the loss for logging)
        total_loss += loss.item() * accumulation_steps
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    avg_loss = total_loss / max(1, len(train_loader))
    accuracy = correct_predictions / max(1, total_samples)
    return avg_loss, accuracy

def evaluate(model, test_loader, criterion, device, distributed: bool, is_main: bool):
    """Evaluate the model and gather predictions across processes if distributed."""
    model.eval()
    total_loss = 0.0
    local_preds = []
    local_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            local_preds.append(predicted.cpu())
            local_labels.append(labels.cpu())

    avg_loss = total_loss / max(1, len(test_loader))

    preds_t = torch.cat(local_preds) if len(local_preds) else torch.empty(0, dtype=torch.long)
    labs_t = torch.cat(local_labels) if len(local_labels) else torch.empty(0, dtype=torch.long)

    if distributed and dist.is_initialized():
        world_size = dist.get_world_size()
        # gather lengths
        len_t = torch.tensor([preds_t.numel()], device=device)
        lens = [torch.zeros_like(len_t) for _ in range(world_size)]
        dist.all_gather(lens, len_t)
        lens = [int(x.item()) for x in lens]
        max_len = max(lens)

        # pad to max_len
        pad_preds = torch.full((max_len,), -1, dtype=torch.long, device=device)
        pad_labs = torch.full((max_len,), -1, dtype=torch.long, device=device)
        if preds_t.numel() > 0:
            pad_preds[: preds_t.numel()] = preds_t.to(device)
            pad_labs[: labs_t.numel()] = labs_t.to(device)

        gather_preds = [torch.empty_like(pad_preds) for _ in range(world_size)]
        gather_labs = [torch.empty_like(pad_labs) for _ in range(world_size)]
        dist.all_gather(gather_preds, pad_preds)
        dist.all_gather(gather_labs, pad_labs)

        if is_main:
            all_preds = torch.cat([gp[:l] for gp, l in zip(gather_preds, lens)]).cpu().numpy()
            all_labels = torch.cat([gl[:l] for gl, l in zip(gather_labs, lens)]).cpu().numpy()
        else:
            all_preds = None
            all_labels = None
    else:
        all_preds = preds_t.cpu().numpy()
        all_labels = labs_t.cpu().numpy()

    if is_main:
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
    else:
        accuracy, f1 = 0.0, 0.0

    return avg_loss, accuracy, f1


def main():
    # Set device
    # Distributed detection
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    distributed = world_size > 1
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    is_main = True

    if torch.cuda.is_available():
        if distributed:
            torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank if distributed else 0)
    else:
        device = torch.device('cpu')

    if distributed and not dist.is_initialized():
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend)
        is_main = dist.get_rank() == 0
    else:
        is_main = True

    if is_main:
        print(f"Using device: {device}")

    # Data directory
    data_dir = r"/content/AI_backend/custom_data"

    # Get transforms (BEiT config for normalization + bicubic interpolation)
    backbone_name = "beit_large_patch16_224"
    train_transform, val_transform = get_plankton_transforms(backbone_name)

    # Create consistent split indices (avoid leakage) and restrict to first 5 classes
    base = datasets.ImageFolder(root=data_dir)  # no transforms for indexing
    selected_classes = base.classes[:5]
    class_names = selected_classes
    num_classes = len(class_names)
    selected_class_indices = [base.class_to_idx[c] for c in selected_classes]
    # Indices of samples belonging to the first 5 classes
    selected_indices = [i for i, (_, t) in enumerate(base.samples) if t in selected_class_indices]

    if is_main:
        print(f"Number of classes (restricted): {num_classes}")
        print(f"Classes: {class_names}")

    total_size = len(selected_indices)
    train_size = int(0.85 * total_size)
    val_size = total_size - train_size
    gen = torch.Generator().manual_seed(42)
    perm = torch.randperm(total_size, generator=gen).tolist()
    # Map permuted positions back to global indices in the ImageFolder
    train_indices = [selected_indices[i] for i in perm[:train_size]]
    val_indices = [selected_indices[i] for i in perm[train_size:]]

    # Remap original class indices -> [0..num_classes-1]
    id_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(selected_class_indices)}

    train_dataset = torch.utils.data.Subset(
        datasets.ImageFolder(
            root=data_dir,
            transform=train_transform,
            target_transform=(lambda t, m=id_map: m[t])
        ),
        train_indices,
    )
    val_dataset = torch.utils.data.Subset(
        datasets.ImageFolder(
            root=data_dir,
            transform=val_transform,
            target_transform=(lambda t, m=id_map: m[t])
        ),
        val_indices,
    )

    if is_main:
        print(f"Train size: {train_size}, Validation size: {val_size}")

    # Data loaders with batch size 64 as per paper
    batch_size = 64
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
        shuffle_train = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                            sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           sampler=val_sampler, num_workers=4, pin_memory=True)

    # Initialize model
    model = PlanktonBottleneckModel(backbone_name=backbone_name, num_classes=num_classes).to(device)
    # Optional SyncBN in multi-GPU
    if distributed and torch.cuda.is_available():
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # Wrap with DDP if distributed
    if distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if device.type == 'cuda' else None)

    # Loss function with label smoothing (0.1) as per paper
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # SGD optimizer with Nesterov momentum (0.9) and weight decay (10^-2) as per paper
    # Create param groups from the underlying module (DDP .module if needed)
    base_model = model.module if hasattr(model, 'module') else model
    param_groups = base_model.get_param_groups(lr_backbone=1e-3, lr_bottleneck=1e-2, lr_classifier=1e-2)
    optimizer = optim.SGD(param_groups, momentum=0.9, nesterov=True, weight_decay=1e-2)

    # Learning rate scheduler - exponential decay as per paper
    epochs = 300  # Increased as requested
    accumulation_steps = 1  # Set >1 to emulate larger batches
    updates_per_epoch = max(1, math.ceil(len(train_loader) / max(1, accumulation_steps)))
    total_steps = updates_per_epoch * epochs
    scheduler = ExponentialLRScheduler(optimizer.param_groups, total_steps, gamma=10, beta=0.75)

    # AMP scaler
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # Training history
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    val_f1_scores = []

    # Best model tracking for early stopping (by validation F1)
    best_val_accuracy = 0.0
    best_val_f1 = float('-inf')
    best_model_path = None
    epochs_without_improvement = 0
    early_stopping_patience = 50  # As per paper

    if is_main:
        print("Starting training...")
        print("=" * 80)

    for epoch in range(epochs):
        start_time = time.time()

        # Set epoch for samplers
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        if isinstance(val_loader.sampler, DistributedSampler):
            val_loader.sampler.set_epoch(epoch)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device,
            accumulation_steps=accumulation_steps, use_amp=True, scaler=scaler, is_main=is_main
        )

        # Validate
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device, distributed, is_main)

        # Record metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_f1_scores.append(val_f1)

        epoch_time = time.time() - start_time

        # Print epoch results (main only)
        if is_main:
            current_lrs = [group['lr'] for group in optimizer.param_groups]
            lr_str = ', '.join(f"{lr:.3e}" for lr in current_lrs)
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
                f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | val_f1: {val_f1:.4f} | "
                f"lrs: [{lr_str}] | time: {epoch_time:.2f}s"
            )

        # Early stopping and checkpointing (main process decides)
        stop_early = torch.tensor([0], device=device)

        if is_main:
            # Use validation F1 for model selection and early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_accuracy = val_acc
                epochs_without_improvement = 0

                # Save best checkpoint
                checkpoint_dir = Path('checkpoints')
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                best_model_path = checkpoint_dir / f"best_model_epoch_{epoch + 1}.pth"
                to_save = model.module if hasattr(model, 'module') else model
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'val_f1': val_f1,
                    'train_accuracy': train_acc,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'class_names': class_names,
                }, best_model_path)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping_patience:
                stop_early[0] = 1

            # Periodic checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_dir = Path('checkpoints')
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                periodic_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"
                to_save = model.module if hasattr(model, 'module') else model
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'val_f1': val_f1,
                    'train_accuracy': train_acc,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'class_names': class_names,
                }, periodic_path)

        # Broadcast stop signal to all ranks if distributed
        if distributed and dist.is_initialized():
            dist.broadcast(stop_early, src=0)

        if stop_early.item() == 1:
            if is_main:
                print(f"Early stopping at epoch {epoch + 1} after {epochs_without_improvement} epochs without improvement.")
            break

    # Final logging
    if is_main:
        print("Training finished.")
        if best_model_path is not None:
            print(f"Best model saved to: {best_model_path}")
        else:
            print("No checkpoint was saved.")

    # Plot training curves (main process only)
    if is_main:
        plt.figure(figsize=(15, 5))

        # Loss plot
        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label='Train Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Accuracy plot
        plt.subplot(1, 3, 2)
        plt.plot(train_accuracies, label='Train Accuracy', color='blue')
        plt.plot(val_accuracies, label='Validation Accuracy', color='red')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        # F1 Score plot
        plt.subplot(1, 3, 3)
        plt.plot(val_f1_scores, label='Validation F1 Score', color='green')
        plt.title('Validation F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
        print("Training plots saved as 'training_metrics.png'")
        plt.show()


if __name__ == "__main__":
    main()