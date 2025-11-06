import os, torch, torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ["crack", "missing-head", "paint-off"]
IMG_SIZE = (128, 128)
MATRIX = 15

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
    nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
    nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
    nn.MaxPool2d(2),

    nn.AdaptiveAvgPool2d((MATRIX, MATRIX)),
    nn.Flatten(),
    nn.Linear(128 * MATRIX * MATRIX, 512), nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 128), nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, len(class_names))
).to(device)

state = torch.load("best_model.pt", map_location=device)

model.load_state_dict(state, strict=True)
model.eval()

@torch.no_grad()
def predict_image(img_path):
    image = Image.open(img_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    x = transform(image).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    idx = int(torch.argmax(probs))
    return class_names[idx], float(probs[idx].item()), image

test_images = {
    "test_crack":       "test/crack/test_crack.jpg",
    "test_missinghead": "test/missing-head/test_missinghead.jpg",
    "test_paintoff":    "test/paint-off/test_paintoff.jpg"
}

for name, path in test_images.items():
    lbl, p, img = predict_image(path)
    plt.figure(); plt.imshow(img); plt.axis("off")
    plt.title(f"Prediction: {lbl} ({p*100:.1f}%)")
    plt.show()