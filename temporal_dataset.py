"""
temporal_dataset.py

Temporal Dataset — returns CLIP features + commentary
Combines frame_dataset.py + clip_encoder in one class.

What this returns per sample:
    {
        "video_features": tensor(10, 512),   ← CLIP features (not raw frames)
        "commentary":     "Kohli drives for FOUR!"
    }

Checkpoint 1:
    ✅ DataLoader works
    ✅ Batch loading works
    ✅ Shape = (B, N, 512)
"""

import os
import json
import torch
import clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader


# =====================================================================
# TemporalDataset
# =====================================================================
class TemporalDataset(Dataset):
    """
    Loads frames → runs CLIP → returns feature matrix + commentary.

    Args:
        ann_file:   Path to cricket_annotations.json
        num_frames: How many frames to use per clip (default: 10)
        device:     "cuda" or "cpu"
    """

    def __init__(
        self,
        ann_file: str,
        num_frames: int = 10,
        device: str = None,
    ):
        # Load annotations
        with open(ann_file, "r") as f:
            self.annotations = json.load(f)

        self.num_frames = num_frames
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load CLIP once — reused for every sample
        print(f"Loading CLIP on {self.device}...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

        # Freeze CLIP — never update weights
        for param in self.model.parameters():
            param.requires_grad = False

        print(f"✅ CLIP loaded. Dataset size: {len(self.annotations)} samples")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index: int) -> dict:
        ann = self.annotations[index]

        frame_dir  = ann["frame_dir"]
        start      = ann["start_frame"]
        commentary = ann["caption"]

        # ── Load frames ───────────────────────────────────────────────
        frames = []
        for i in range(start, start + self.num_frames):
            path = os.path.join(frame_dir, f"frame_{i:04d}.jpg")

            if not os.path.exists(path):
                # If frame missing, repeat the last loaded frame
                if frames:
                    frames.append(frames[-1])
                continue

            img = Image.open(path).convert("RGB")
            frames.append(self.preprocess(img))     # (3, 224, 224)

        # Stack → (num_frames, 3, 224, 224)
        batch = torch.stack(frames).to(self.device)

        # ── Run CLIP ──────────────────────────────────────────────────
        with torch.no_grad():
            features = self.model.encode_image(batch).float()  # (num_frames, 512)

        # Move to CPU for DataLoader compatibility
        features = features.cpu()

        return {
            "video_features": features,     # tensor(10, 512)
            "commentary":     commentary,   # "Kohli drives for FOUR!"
            "video_id":       ann["video_id"],
        }


# =====================================================================
# Collate function — stacks samples into a batch
# =====================================================================
def collate_fn(batch: list) -> dict:
    """
    Stacks individual samples into a batch.

    Input:  list of dicts, each with video_features (10, 512)
    Output: dict with video_features (B, 10, 512)
    """
    features    = torch.stack([b["video_features"] for b in batch])  # (B, N, 512)
    commentaries = [b["commentary"]  for b in batch]
    video_ids    = [b["video_id"]    for b in batch]

    return {
        "video_features": features,       # (B, N, 512)
        "commentary":     commentaries,
        "video_ids":      video_ids,
    }


# =====================================================================
# Checkpoint 1 — run this file to verify
# =====================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("Temporal Dataset — Checkpoint 1")
    print("=" * 50)

    # Load dataset
    dataset = TemporalDataset(
        ann_file="datasets/cricket_annotations.json",
        num_frames=10,
    )

    # ── Check single sample ───────────────────────────────────────────
    sample = dataset[0]
    print(f"\nSingle sample:")
    print(f"  video_id      : {sample['video_id']}")
    print(f"  video_features: {sample['video_features'].shape}")  # (10, 512)
    print(f"  commentary    : {sample['commentary']}")

    # ── Check DataLoader ──────────────────────────────────────────────
    loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collate_fn,
    )

    batch = next(iter(loader))
    B, N, D = batch["video_features"].shape

    print(f"\nDataLoader batch:")
    print(f"  video_features shape : {tuple(batch['video_features'].shape)}")
    print(f"  commentary           : {batch['commentary']}")

    # ── Checkpoint verification ───────────────────────────────────────
    print("\n" + "=" * 50)
    print("Checkpoint 1 Results:")
    print(f"  ✅ DataLoader works")
    print(f"  ✅ Batch loading works")
    print(f"  ✅ Shape = (B={B}, N={N}, D={D})")
    print("=" * 50)
    print("\n✅ Temporal Dataset complete. Ready for Perceiver Aggregator.")