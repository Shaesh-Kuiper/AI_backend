import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import io
import uvicorn
from pyngrok import ngrok
import os

class PlanktonBottleneckModel(nn.Module):
    def __init__(self, backbone="beit_large_patch16_224", classes=8, bottleneck_dim=512):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
        dim = self.backbone.num_features
        self.bottleneck = nn.Sequential(
            nn.Linear(dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
        )
        self.classifier = nn.utils.weight_norm(nn.Linear(bottleneck_dim, classes))
        self._init()

    def _init(self):
        for m in [*self.bottleneck, self.classifier]:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.classifier(self.bottleneck(self.backbone(x)))

def build_val_tf(backbone="beit_large_patch16_224"):
    cfg = timm.data.resolve_model_data_config(
        timm.create_model(backbone, pretrained=False, num_classes=0)
    )
    resize = (cfg["input_size"][1], cfg["input_size"][2])
    mean, std = cfg["mean"], cfg["std"]
    return transforms.Compose([
        transforms.Resize(resize, transforms.InterpolationMode.BICUBIC),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


app = FastAPI(title="Plankton Classification API", version="1.0.0")


model = None
transform = None
class_names = None
device = None

@app.on_event("startup")
async def startup_event():
    global model, transform, class_names, device

   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    
    model_path = "/content/checkpoints/best_model_epoch_8.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found!")

    ckpt = torch.load(model_path, map_location='cpu')
    class_names = ckpt['class_names']
    model = PlanktonBottleneckModel(classes=len(class_names))
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval().to(device)

  
    transform = build_val_tf()

    print(f"Model loaded successfully with {len(class_names)} classes")
    print(f"Classes: {class_names}")

@app.get("/")
async def root():
    return {"message": "Plankton Classification API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": str(device),
        "classes": len(class_names) if class_names else 0,
        "class_names": class_names
    }

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    try:
       
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

      
        batch = transform(image).unsqueeze(0).to(device)

    
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            logits = model(batch)
            probs = logits.softmax(dim=1).cpu().numpy()[0]

       
        top_indices = np.argsort(-probs)[:3]
        predictions = []

        for idx in top_indices:
            predictions.append({
                "class": class_names[idx],
                "probability": float(probs[idx])
            })

        return {
            "success": True,
            "predictions": predictions,
            "top_prediction": {
                "class": class_names[top_indices[0]],
                "probability": float(probs[top_indices[0]])
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

@app.post("/classify_batch")
async def classify_batch(files: list[UploadFile] = File(...)):
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")

    results = []
    for file in files:
        try:
            if not file.content_type.startswith('image/'):
                results.append({
                    "filename": file.filename,
                    "error": "File must be an image"
                })
                continue

            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            batch = transform(image).unsqueeze(0).to(device)

            with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                logits = model(batch)
                probs = logits.softmax(dim=1).cpu().numpy()[0]

            top_idx = np.argmax(probs)
            results.append({
                "filename": file.filename,
                "prediction": {
                    "class": class_names[top_idx],
                    "probability": float(probs[top_idx])
                }
            })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })

    return {"results": results}

def setup_ngrok():
   
    ngrok.set_auth_token("3318G79rGo3iRgnpTA6bjQnqHdC_6BwE39XTSThA1UNwJpmnM")

   
    public_url = ngrok.connect(8000)
    print(f"\n🌍 Public URL: {public_url}")
    print(f"📡 API Documentation: {public_url}/docs")
    print(f"🔍 Health Check: {public_url}/health")

    return public_url

if __name__ == "__main__":
  
    public_url = setup_ngrok()

 
    print(f"\n🚀 Starting server on http://localhost:8000")
    print(f"📚 Local docs: http://localhost:8000/docs")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
