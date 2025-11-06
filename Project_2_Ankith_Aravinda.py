import os, time, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

IMG_SIZE = (500, 500)
BATCH_SIZE = 3
EPOCHS = 10
LR = 5e-3
PATIENCE = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_tfms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomAffine(0, shear=10, scale=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
valid_tfms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
train_ds = datasets.ImageFolder("train", transform=train_tfms)
valid_ds = datasets.ImageFolder("valid", transform=valid_tfms)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
print("Classes:", train_ds.classes)

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
    nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(),
    nn.Linear(128, 128), nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, len(train_ds.classes))
).to(device)

loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=LR)

best_loss, patience, wait = float("inf"), PATIENCE, 0
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

for epoch in range(1, EPOCHS + 1):
    start_time = time.time()
    model.train()
    total, correct, running_loss = 0, 0, 0
    train_bar = tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS} [Train]", leave=False, ncols=80)
    for x, y in train_bar:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()

        running_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

        avg_loss = running_loss / total
        avg_acc = correct / total
        elapsed = time.time() - start_time
        speed = elapsed / max(total, 1)
        est = speed * (len(train_ds) - total)
        train_bar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "acc": f"{avg_acc:.3f}",
            "elapsed": f"{elapsed/60:.1f}m",
        })

    train_loss, train_acc = running_loss / total, correct / total
    
    model.eval()
    total, correct, running_loss = 0, 0, 0
    val_bar = tqdm(valid_dl, desc=f"Epoch {epoch}/{EPOCHS} [Val]  ", leave=False, ncols=80)
    with torch.no_grad():
        for x, y in val_bar:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            running_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

            avg_loss = running_loss / total
            avg_acc = correct / total
            elapsed = time.time() - start_time
            est = elapsed * (len(valid_dl) * BATCH_SIZE / max(total, 1) - 1)
            val_bar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "acc": f"{avg_acc:.3f}",
                "elapsed": f"{elapsed/60:.1f}m",
            })

    val_loss, val_acc = running_loss / total, correct / total

    elapsed = time.time() - start_time
    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
          f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} | "
          )

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    if val_loss < best_loss:
        best_loss, wait = val_loss, 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

# Plots
plt.figure(figsize=(6,4))
plt.plot(history["train_acc"], label="Train Acc")
plt.plot(history["val_acc"], label="Val Acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.title("Accuracy vs Epoch"); plt.legend(); plt.grid()
plt.savefig("accuracy_plot.png", dpi=150)

plt.figure(figsize=(6,4))
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Loss vs Epoch"); plt.legend(); plt.grid()
plt.savefig("loss_plot.png", dpi=150)

print("Saved: best_model.pt, accuracy_plot.png, loss_plot.png")
