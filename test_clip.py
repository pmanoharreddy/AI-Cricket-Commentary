"""
test_clip.py

Step 4 — Feed ONE frame into CLIP and print the feature shape.
Expected output: (512,)
"""

import torch
import clip
from PIL import Image

# ── Load CLIP ────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# ── Load ONE real frame ──────────────────────────────────────────────
frame_path = "frames/cover_drive/frame_0001.jpg"
image = Image.open(frame_path).convert("RGB")

# ── Preprocess → Tensor ──────────────────────────────────────────────
image_tensor = preprocess(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

# ── Feed into CLIP ───────────────────────────────────────────────────
with torch.no_grad():
    features = model.encode_image(image_tensor)            # (1, 512)

features = features.squeeze(0)                             # (512,)

# ── Print result ─────────────────────────────────────────────────────
print(f"Feature Shape: {tuple(features.shape)}")
