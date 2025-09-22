import os
import io
import numpy as np
from typing import List

import torch
import torch.nn as nn
from PIL import Image, ImageOps, UnidentifiedImageError

import timm
from torchvision import transforms

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from uvicorn import Config, Server
from contextlib import asynccontextmanager

from pyngrok import ngrok
import nest_asyncio

# ---------------------------------------------------------------------
# ----------------------- USER CONFIGURATION ---------------------------
# ---------------------------------------------------------------------

# 1) Hardcode your ngrok token here


# 1) Put your ngrok token in an env var (recommended).
#    In Colab:  os.environ["NGROK_TOKEN"] = "YOUR_TOKEN"
NGROK_TOKEN = "332y8uSgsDk1RN02FaeqUPTInse_2BSfWYKUWQ3qQ6Kgwo357"
# 2) Path to your trained checkpoint (.pth)
MODEL_PATH = os.environ.get("PLANKTON_CKPT", "/content/drive/MyDrive/Classification_model.pth")

# 3) Backbone & bottleneck config
BACKBONE_NAME = os.environ.get("PLANKTON_BACKBONE", "beit_large_patch16_224")
BOTTLENECK_DIM = int(os.environ.get("PLANKTON_BOTTLENECK", "512"))

# 4) Batch endpoint limit
BATCH_LIMIT = int(os.environ.get("PLANKTON_BATCH_LIMIT", "10"))

# ---------------------------------------------------------------------
# --------------------------- MODEL CODE -------------------------------
# ---------------------------------------------------------------------

class PlanktonBottleneckModel(nn.Module):
    def __init__(self, backbone=BACKBONE_NAME, classes=8, bottleneck_dim=BOTTLENECK_DIM):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
        dim = getattr(self.backbone, "num_features", None)
        if dim is None:
            # Fallback for some timm models
            dim = self.backbone.get_classifier().in_features
        self.bottleneck = nn.Sequential(
            nn.Linear(dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
        )
        self.classifier = nn.utils.weight_norm(nn.Linear(bottleneck_dim, classes))
        self._init()

    def _init(self):
        # Initialize Linear layers in bottleneck and classifier
        for m in list(self.bottleneck) + [self.classifier]:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feats = self.backbone(x)
        z = self.bottleneck(feats)
        return self.classifier(z)

def build_val_tf(backbone=BACKBONE_NAME) -> transforms.Compose:
    # Use timm's data config to match model expectations
    cfg = timm.data.resolve_model_data_config(
        timm.create_model(backbone, pretrained=False, num_classes=0)
    )
    resize = (cfg["input_size"][1], cfg["input_size"][2])
    mean, std = cfg["mean"], cfg["std"]

    # Many plankton datasets are grayscale; ensure 3-ch input
    return transforms.Compose([
        transforms.Resize(resize, transforms.InterpolationMode.BICUBIC),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

# ---------------------------------------------------------------------
# --------------------------- FASTAPI APP ------------------------------
# ---------------------------------------------------------------------

app = FastAPI(title="Plankton Classification API", version="1.1.0")

# CORS: allow everything for testing; tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Globals set in lifespan
model = None
transform = None
class_names = None
device = None

def _load_checkpoint(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Mount Google Drive or set PLANKTON_CKPT to your checkpoint."
        )
    ckpt = torch.load(model_path, map_location="cpu")
    return ckpt

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, transform, class_names, device

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[startup] Using device: {device}")

    # Optional perf knobs (safe on both CPU/GPU)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    # Load checkpoint & build model/transform
    ckpt = _load_checkpoint(MODEL_PATH)
    if "class_names" in ckpt and isinstance(ckpt["class_names"], (list, tuple)):
        class_names = list(ckpt["class_names"])
        num_classes = len(class_names)
    else:
        # Fallback: infer classes from classifier weight shape if available
        num_classes = ckpt.get("num_classes", 8)
        class_names = [f"class_{i}" for i in range(num_classes)]
        print("[startup] 'class_names' missing in checkpoint; using fallback indices.")

    model = PlanktonBottleneckModel(classes=num_classes)
    missing, unexpected = model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    if missing:
        print(f"[startup] Missing keys in state_dict: {missing}")
    if unexpected:
        print(f"[startup] Unexpected keys in state_dict: {unexpected}")

    model.eval().to(device)
    transform = build_val_tf()

    print(f"[startup] Model ready with {len(class_names)} classes")
    print(f"[startup] Classes: {class_names}")

    yield

    # Optional: cleanup
    print("[shutdown] Cleaning up model from device.")
    try:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

# Recreate the app with lifespan (avoids deprecated on_event)
app.router.lifespan_context = lifespan

# ----------------------------- ROUTES --------------------------------

@app.get("/")
async def root():
    return {"message": "Plankton Classification API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if (model is not None and transform is not None) else "loading",
        "device": str(device) if device else None,
        "classes": len(class_names) if class_names else 0,
        "class_names": class_names,
        "backbone": BACKBONE_NAME,
        "model_path": MODEL_PATH,
    }

def _read_image_from_upload(file: UploadFile) -> Image.Image:
    """
    Robustly read an image:
    - accept non image/* content-types by trying anyway
    - apply EXIF orientation fix
    - convert to RGB (we'll grayscale->3ch in tf)
    """
    try:
        image_bytes = file.file.read()  # faster in FastAPI's UploadFile
        if not image_bytes:
            # fallback for .read() via await; shouldn't happen here
            image_bytes = file.file.read()
        img = Image.open(io.BytesIO(image_bytes))
        # Validate image by loading a minimal tile
        img.verify()  # verify doesn't decode full image
        # Re-open because verify() leaves it in an unusable state
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img).convert("RGB")
        return img
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {str(e)}")

def _predict_pytorch(img: Image.Image, topk: int = 3):
    if model is None or transform is None or device is None:
        raise HTTPException(status_code=503, detail="Model not initialized yet. Try again in a moment.")

    batch = transform(img).unsqueeze(0).to(device, non_blocking=True)

    # Inference
    with torch.inference_mode():
        use_amp = (device.type == "cuda")
        if use_amp:
            # AMP only on CUDA here; leave CPU in full precision for correctness
            with torch.amp.autocast(device_type="cuda"):
                logits = model(batch)
        else:
            logits = model(batch)

        probs = logits.softmax(dim=1).detach().cpu().numpy()[0]

    # Top-k
    top_indices = np.argsort(-probs)[:topk]
    predictions = [
        {"class": class_names[idx], "probability": float(probs[idx])}
        for idx in top_indices
    ]
    return predictions

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    # Be lenient on content-type: some clients send octet-stream
    if not (file.content_type or "").startswith("image/"):
        # We'll try anyway; PIL will throw if invalid
        pass

    img = _read_image_from_upload(file)
    predictions = _predict_pytorch(img, topk=3)
    top = predictions[0]
    return {"success": True, "predictions": predictions, "top_prediction": top}

@app.post("/classify_batch")
async def classify_batch(files: List[UploadFile] = File(...)):
    if len(files) > BATCH_LIMIT:
        raise HTTPException(status_code=400, detail=f"Maximum {BATCH_LIMIT} images per batch")

    results = []
    for f in files:
        try:
            img = _read_image_from_upload(f)
            preds = _predict_pytorch(img, topk=1)
            results.append({
                "filename": f.filename,
                "prediction": preds[0]
            })
        except HTTPException as he:
            results.append({"filename": f.filename, "error": he.detail})
        except Exception as e:
            results.append({"filename": f.filename, "error": str(e)})

    return {"results": results}

# ---------------------------------------------------------------------
# ---------------------------- LAUNCHING -------------------------------
# ---------------------------------------------------------------------

def setup_ngrok(port: int = 8000):
    ngrok.set_auth_token("332y8uSgsDk1RN02FaeqUPTInse_2BSfWYKUWQ3qQ6Kgwo357")  # <-- hardcoded right here

    # Explicit HTTP tunnel
    public_url = ngrok.connect(port, "http")
    public_url_str = str(public_url)
    print(f"\n🌍 Public URL: {public_url_str}")
    print(f"📡 API Docs:   {public_url_str}/docs")
    print(f"🔍 Health:     {public_url_str}/health")
    return public_url_str

async def run_server():
    config = Config(app=app, host="0.0.0.0", port=8000, log_level="info")
    server = Server(config)
    await server.serve()

# Apply nest_asyncio so Uvicorn can run inside Jupyter/Colab
nest_asyncio.apply()

# Start ngrok and the server
public_url = setup_ngrok(8000)
print(f"\n🚀 Starting server on http://localhost:8000")
print(f"📚 Local docs: http://localhost:8000/docs")

# IMPORTANT: In notebooks we can 'await' directly.
import asyncio
await run_server()
