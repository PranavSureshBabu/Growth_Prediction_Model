"""
Grass classifier: tells trimmed lawns from overgrown ones.

Uses a fine-tuned MobileNetV2 with a color-based preprocessing step
to isolate the grass region before classification.
"""

import os
import glob
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix


# === Config ===
DATA_ROOT = "dataset"
SAVE_PATH = "grass_cnn.pth"
VAL_FRACTION = 0.2
BATCH_SIZE = 8
NUM_EPOCHS = 15
LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {DEVICE}")


def detect_grass_mask(img_bgr):
    """
    Finds grass pixels using a combo of Excess Green index and HSV filtering.
    Returns a binary mask where 255 = grass.
    
    The idea: real grass tends to be green in both RGB space (high G relative
    to R and B) and HSV space (hue in the green range). Combining both filters
    helps reject stuff like green cars or painted walls.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(img_bgr)

    # Excess Green index: 2G - R - B
    # High values = more green than you'd expect from a neutral gray
    exg = 2.0 * g.astype(np.float32) - r.astype(np.float32) - b.astype(np.float32)
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, exg_mask = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # HSV filter for green-ish hues
    lower = np.array([20, 40, 40], dtype=np.uint8)
    upper = np.array([90, 255, 255], dtype=np.uint8)
    hsv_mask = cv2.inRange(hsv, lower, upper)

    # Both conditions must be true
    mask = cv2.bitwise_and(exg_mask, hsv_mask)

    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Throw out tiny blobs (probably not actual lawn)
    h, w = mask.shape
    min_area = 0.05 * h * w
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    filtered = np.zeros_like(mask)
    for i in range(1, n_labels):  # skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 255

    return filtered


class GrassDataset(Dataset):
    """
    Loads lawn images and optionally crops to just the grass region.
    Labels: 0 = trimmed, 1 = grown
    """
    
    def __init__(self, items, transform=None, crop_to_grass=True):
        self.items = items  # list of (filepath, label)
        self.transform = transform
        self.crop_to_grass = crop_to_grass

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Couldn't load: {path}")

        if self.crop_to_grass:
            mask = detect_grass_mask(img)
            ys, xs = np.where(mask > 0)
            
            if len(xs) > 0:  # found some grass
                # Bounding box with a bit of padding
                pad = 5
                y0 = max(ys.min() - pad, 0)
                y1 = min(ys.max() + pad, img.shape[0] - 1)
                x0 = max(xs.min() - pad, 0)
                x1 = min(xs.max() + pad, img.shape[1] - 1)
                img = img[y0:y1+1, x0:x1+1]

        # OpenCV uses BGR, PIL expects RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        return img, label


def gather_dataset():
    """Scan the dataset folder for images."""
    items = []
    
    for folder_name, label in [("trimmed", 0), ("grown", 1)]:
        folder = os.path.join(DATA_ROOT, folder_name)
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            paths = glob.glob(os.path.join(folder, ext))
            items.extend((p, label) for p in paths)
    
    print(f"Found {len(items)} images total")
    return items


def build_model():
    """MobileNetV2 with frozen backbone, fresh classifier head."""
    model = models.mobilenet_v2(pretrained=True)
    
    # Freeze the feature extractor—we're just fine-tuning the classifier
    for p in model.features.parameters():
        p.requires_grad = False

    # Swap out the final layer for binary classification
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)
    
    return model


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct += (out.argmax(1) == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_true, all_pred = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out = model(imgs)
        loss = criterion(out, labels)

        total_loss += loss.item() * len(labels)
        preds = out.argmax(1)
        correct += (preds == labels).sum().item()
        total += len(labels)

        all_true.append(labels.cpu().numpy())
        all_pred.append(preds.cpu().numpy())

    avg_loss = total_loss / total if total else 0
    acc = correct / total if total else 0
    y_true = np.concatenate(all_true) if all_true else np.array([])
    y_pred = np.concatenate(all_pred) if all_pred else np.array([])

    return avg_loss, acc, y_true, y_pred


def main():
    items = gather_dataset()
    if not items:
        raise RuntimeError("No images found! Check dataset/trimmed and dataset/grown")

    # Standard ImageNet preprocessing, plus some augmentation for training
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Train/val split
    val_size = int(len(items) * VAL_FRACTION)
    train_size = len(items) - val_size
    train_items, val_items = random_split(
        items, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_ds = GrassDataset(list(train_items), transform=train_tf)
    val_ds = GrassDataset(list(val_items), transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    # Only train the classifier head
    optimizer = optim.Adam(model.classifier[-1].parameters(), lr=LR, weight_decay=1e-4)

    best_acc = 0.0
    best_weights = None

    print("\nTraining...")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)

        print(f"Epoch {epoch:2d} | "
              f"train: loss={train_loss:.3f} acc={train_acc:.3f} | "
              f"val: loss={val_loss:.3f} acc={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = model.state_dict().copy()

    
    if best_weights:
        model.load_state_dict(best_weights)

    _, final_acc, y_true, y_pred = evaluate(model, val_loader, criterion)
    
    print(f"\nBest validation accuracy: {final_acc:.3f}")
    if len(y_true):
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, target_names=["trimmed", "grown"]))
        print("Confusion matrix:")
        print(confusion_matrix(y_true, y_pred))

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\nModel saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()