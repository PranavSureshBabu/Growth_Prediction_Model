import sys
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

from train_model import detect_grass_mask, SAVE_PATH, DEVICE

# Same preprocessing as validation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def create_model():
    model = models.mobilenet_v2(pretrained=False)
    in_feats = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feats, 2)
    return model

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_cnn.py path/to/image.jpg")
        sys.exit(1)

    img_path = sys.argv[1]
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(img_path)

    # Crop to grass ROI (same as training)
    mask = detect_grass_mask(img_bgr)
    ys, xs = np.where(mask > 0)
    if len(xs) > 0 and len(ys) > 0:
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        pad = 5
        y0 = max(y0 - pad, 0)
        x0 = max(x0 - pad, 0)
        y1 = min(y1 + pad, img_bgr.shape[0] - 1)
        x1 = min(x1 + pad, img_bgr.shape[1] - 1)
        img_bgr = img_bgr[y0:y1+1, x0:x1+1]

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    x = transform(pil_img).unsqueeze(0)  # [1,3,224,224]

    model = create_model().to(DEVICE)
    state = torch.load(SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        x = x.to(DEVICE)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred = int(np.argmax(probs))
    label = "grown" if pred == 1 else "trimmed"
    print(f"{img_path} → {label} (P(trimmed)={probs[0]:.2f}, P(grown)={probs[1]:.2f})")
