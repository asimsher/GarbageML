import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter
import uvicorn
import io
from torchvision import transforms
import torch.nn as nn
from PIL import Image


labels_classes = torch.load('classes.pth')

transform_a = transforms.Compose([
    transforms.Resize((250, 250)),
    transforms.ToTensor()
])


class Garbage(nn.Module):
  def __init__(self):
     super().__init__()
     self.first = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d((8, 8))

      )
     self.second = nn.Sequential(
        nn.Flatten(),
        nn.Linear(128 * 8 * 8, 256), # Corrected input size
        nn.ReLU(),
        nn.Linear(256, 6)
     )

  def forward(self, image):
    image = self.first(image)
    image = self.second(image)
    return image


garbage_app = FastAPI()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Garbage()
model.load_state_dict(torch.load('model_garbage.pth', map_location=device))
model.to(device)
model.eval()


@garbage_app.post('/predict/')
async def check_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        if not image_data:
            raise HTTPException(status_code=400, detail='файл кошулган жок')

        img = Image.open(io.BytesIO(image_data))
        img_tensor = transform_a(img).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(img_tensor)
            pred = y_pred.argmax(dim=1).item()

        return {"Answer": labels_classes[pred]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(garbage_app, host='127.0.0.1', port=8001)