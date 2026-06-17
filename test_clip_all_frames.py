"""
test_clip_all_frames.py

Step 5 — Feed 10 frames into CLIP → 10 feature vectors
Reference: MatchTime inference_single_video_CLIP.py (jyrao/MatchTime)

Output: (10, 512)
"""

import os
import torch
import clip
from PIL import Image

# ── Load CLIP ────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print(f"CLIP loaded on {device}")

# ── Load 10 frames from your extracted frames ─────────────────────── 
frames_dir = "frames/cover_drive"
num_frames  = 10

frames = []
for i in range(1, num_frames + 1):
    path = os.path.join(frames_dir, f"frame_{i:04d}.jpg")
    img  = Image.open(path).convert("RGB")
    frames.append(preprocess(img))          # each → (3, 224, 224)

# Stack into batch: (10, 3, 224, 224)
batch = torch.stack(frames).to(device)
print(f"Input batch shape : {tuple(batch.shape)}")

# ── Feed all 10 frames into CLIP at once ─────────────────────────────
with torch.no_grad():
    features = model.encode_image(batch)    # (10, 512)

features = features.float()

# ── Print result ──────────────────────────────────────────────────────
print(f"Output shape      : {tuple(features.shape)}")   # (10, 512)
print(f"\n10 frames → CLIP → {features.shape[0]} x {features.shape[1]} feature vectors")
print("✅ Vision encoder working. Now your frames are ready for the Perceiver.")