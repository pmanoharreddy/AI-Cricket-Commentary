"""
datasets/frame_dataset.py

FrameDataset — loads extracted JPG frames → Tensor → Model
Adapted from MatchTime's MatchVoice_Dataset (Rao et al., EMNLP 2024)

What changed from MatchTime:
  - MatchTime loads pre-extracted .npy feature files (baidu/CLIP/C3D embeddings)
  - We load raw .jpg frames extracted by extract_frames.py
  - MatchTime uses SoccerNet Labels-caption.json format
  - We use a simple cricket_annotations.json format
  - MatchTime uses LLaMA-3 tokenizer
  - We use Qwen2.5-VL tokenizer (or LLaMA-3 if preferred)

Flow (matching the image):
    frame_0001.jpg
         ↓
      Tensor  (via CLIP preprocess → (3, 224, 224))
         ↓
      Model   (Perceiver Aggregator → LLM)

Usage:
    dataset = FrameDataset(
        frames_dir="frames/",
        ann_file="datasets/cricket_annotations.json",
    )
    sample = dataset[0]
    print(sample["frames"].shape)   # (n_frames, 3, 224, 224)
    print(sample["caption"])        # "Kohli drives through covers for FOUR!"
"""

import os
import json
import random
import copy

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ── Try to load CLIP preprocessor (preferred) ────────────────────────
# Falls back to standard ImageNet transforms if CLIP not installed
try:
    import clip
    _CLIP_AVAILABLE = True
except ImportError:
    _CLIP_AVAILABLE = False


# =====================================================================
# Transform: frame_0001.jpg  →  Tensor (3, 224, 224)
# =====================================================================
def get_transform(use_clip: bool = True):
    """
    Returns the image transform pipeline.

    If CLIP is available: uses official CLIP preprocess (normalises to
    CLIP's mean/std, which the frozen encoder expects).
    Otherwise: standard ImageNet normalisation.
    """
    if use_clip and _CLIP_AVAILABLE:
        _, preprocess = clip.load("ViT-B/32", device="cpu")
        return preprocess                          # returns PIL Image → Tensor

    # Fallback: standard ImageNet normalisation
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],           # ImageNet mean
            std=[0.229, 0.224, 0.225],            # ImageNet std
        ),
    ])


# =====================================================================
# Annotation helpers
# =====================================================================
def load_cricket_annotations(ann_file: str) -> list:
    """
    Load cricket commentary annotations from a JSON file.

    Expected format (cricket_annotations.json):
    [
        {
            "video_id":   "cover_drive",
            "frame_dir":  "frames/cover_drive/",
            "start_frame": 1,
            "end_frame":   8,
            "caption":    "Kohli drives beautifully through the covers for FOUR!"
        },
        ...
    ]

    Returns:
        List of annotation dicts.
    """
    with open(ann_file, "r") as f:
        data = json.load(f)

    # Validate entries
    valid = []
    for entry in data:
        required = {"video_id", "frame_dir", "start_frame", "end_frame", "caption"}
        if not required.issubset(entry.keys()):
            print(f"  [WARNING] Skipping incomplete annotation: {entry}")
            continue
        valid.append(entry)

    print(f"Loaded {len(valid)} annotations from {ann_file}")
    return valid


def create_sample_annotation_file(output_path: str = "datasets/cricket_annotations.json"):
    """
    Creates a sample annotation JSON so you can see the expected format.
    Replace with your real data.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sample = [
        {
            "video_id":    "cover_drive",
            "frame_dir":   "frames/cover_drive/",
            "start_frame": 1,
            "end_frame":   8,
            "caption":     "Kohli drives beautifully through the covers for FOUR!"
        },
        {
            "video_id":    "wicket_bowled",
            "frame_dir":   "frames/wicket_bowled/",
            "start_frame": 1,
            "end_frame":   8,
            "caption":     "Bumrah bowls a perfect yorker — the batsman is BOWLED!"
        },
        {
            "video_id":    "six_over_midwicket",
            "frame_dir":   "frames/six_over_midwicket/",
            "start_frame": 1,
            "end_frame":   8,
            "caption":     "What a SIX! Dhoni clears the mid-wicket boundary with ease!"
        },
    ]

    with open(output_path, "w") as f:
        json.dump(sample, f, indent=2)

    print(f"Sample annotation file created: {output_path}")
    print("Replace the entries with your real cricket video annotations.")


# =====================================================================
# FrameDataset  (Step 3 from image)
# frame_0001.jpg  →  Tensor  →  Model
# =====================================================================
class FrameDataset(Dataset):
    """
    Loads extracted JPG frames and cricket commentary captions.

    Each sample returns:
        frames:    Tensor of shape (n_frames, 3, 224, 224)
        caption:   String commentary, e.g. "Kohli drives for FOUR!"
        input_ids: Tokenized caption (if tokenizer provided)
        video_id:  String identifier for the clip

    Args:
        frames_root: Root directory where frame folders live.
                     If None, frame_dir in annotation is used as-is.
        ann_file:    Path to cricket_annotations.json
        window:      Number of frames to load per sample (default: 8
                     = 8 seconds at 1 FPS, matches MatchTime's 15s window)
        tokenizer:   HuggingFace tokenizer (optional — for training mode)
        max_token_length: Max caption tokens (default: 128)
        use_clip_transform: Use CLIP preprocessing (default: True)
    """

    def __init__(
        self,
        ann_file: str,
        frames_root: str = None,
        window: int = 8,
        tokenizer=None,
        max_token_length: int = 128,
        use_clip_transform: bool = True,
    ):
        self.annotations = load_cricket_annotations(ann_file)
        self.frames_root = frames_root
        self.window = window
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length
        self.transform = get_transform(use_clip=use_clip_transform)

    # ── Internal helpers ─────────────────────────────────────────────

    def _get_frame_dir(self, ann: dict) -> str:
        """Resolve the frame directory for an annotation entry."""
        if self.frames_root:
            return os.path.join(self.frames_root, ann["frame_dir"])
        return ann["frame_dir"]

    def _load_frames(self, frame_dir: str, start: int, end: int) -> torch.Tensor:
        """
        Load JPG frames from frame_dir in range [start, end].

        frame_0001.jpg, frame_0002.jpg, ... → Tensor (n, 3, 224, 224)
        """
        frame_tensors = []

        for i in range(start, end + 1):
            filename = f"frame_{i:04d}.jpg"
            path = os.path.join(frame_dir, filename)

            if not os.path.exists(path):
                # Skip missing frames gracefully
                continue

            img = Image.open(path).convert("RGB")
            tensor = self.transform(img)     # (3, 224, 224)
            frame_tensors.append(tensor)

        if not frame_tensors:
            raise FileNotFoundError(
                f"No frames found in {frame_dir} for range [{start}, {end}].\n"
                f"Run extract_frames.py first to generate the frames."
            )

        # Pad or truncate to exactly self.window frames
        while len(frame_tensors) < self.window:
            frame_tensors.append(frame_tensors[-1])     # repeat last frame
        frame_tensors = frame_tensors[:self.window]

        return torch.stack(frame_tensors)               # (window, 3, 224, 224)

    def _tokenize(self, caption: str) -> torch.Tensor:
        """Tokenize a caption string. Returns input_ids tensor."""
        if self.tokenizer is None:
            return None

        tokens = self.tokenizer(
            caption,
            return_tensors="pt",
            max_length=self.max_token_length,
            truncation=True,
            padding=False,
        )
        return tokens.input_ids[0]                       # (seq_len,)

    # ── Dataset interface ────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> dict:
        num_retries = 10

        for _ in range(num_retries):
            try:
                ann = self.annotations[index]
                frame_dir = self._get_frame_dir(ann)

                # Load frames → Tensor (window, 3, 224, 224)
                frames = self._load_frames(
                    frame_dir=frame_dir,
                    start=ann["start_frame"],
                    end=ann["end_frame"],
                )

                caption = ann["caption"]
                input_ids = self._tokenize(caption)

                sample = {
                    "frames":   frames,           # (window, 3, 224, 224)
                    "caption":  caption,          # raw string
                    "video_id": ann["video_id"],
                }

                if input_ids is not None:
                    sample["input_ids"] = input_ids

                return sample

            except Exception as e:
                print(f"  [WARNING] Error loading index {index}: {e}. Retrying...")
                index = random.randint(0, len(self) - 1)

        raise RuntimeError(f"Failed to load sample after {num_retries} retries.")

    # ── Collate function (for DataLoader) ───────────────────────────

    def collater(self, batch: list) -> dict:
        """
        Custom collate: stacks frames, pads token sequences.
        Pass this as collate_fn to DataLoader.
        """
        frames   = torch.stack([b["frames"] for b in batch])   # (B, window, 3, 224, 224)
        captions = [b["caption"]  for b in batch]
        video_ids= [b["video_id"] for b in batch]

        result = {
            "frames":    frames,
            "captions":  captions,
            "video_ids": video_ids,
        }

        # Pad input_ids if tokenizer was used
        if "input_ids" in batch[0]:
            input_ids = [b["input_ids"] for b in batch]
            input_ids_padded = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=0,
            )
            result["input_ids"]      = input_ids_padded
            result["attention_mask"] = input_ids_padded.ne(0)

        return result


# =====================================================================
# Quick test — run this file directly to verify everything works
# =====================================================================
if __name__ == "__main__":
    import sys

    print("=" * 55)
    print("FrameDataset — Quick Test")
    print("=" * 55)

    # Step 1: Create sample annotation file
    ann_path = "datasets/cricket_annotations.json"
    if not os.path.exists(ann_path):
        print("\nCreating sample annotation file...")
        create_sample_annotation_file(ann_path)

    # Step 2: Create dummy frames for the first sample (for testing)
    print("\nUsing real frames from frames/cover_drive/")

    # Step 3: Create dataset
    dataset = FrameDataset(
        ann_file=ann_path,
        window=8,
        use_clip_transform=False,   # use ImageNet transforms (no CLIP needed for test)
    )
    print(f"\nDataset size: {len(dataset)} samples")

    # Step 4: Load one sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  video_id : {sample['video_id']}")
    print(f"  frames   : {sample['frames'].shape}")   # (8, 3, 224, 224)
    print(f"  caption  : {sample['caption']}")

    # Step 5: Test DataLoader
    loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=dataset.collater,
    )
    batch = next(iter(loader))
    print(f"\nDataLoader batch:")
    print(f"  frames shape : {batch['frames'].shape}")   # (1, 8, 3, 224, 224)
    print(f"  captions     : {batch['captions']}")

    print("\n✅ FrameDataset working correctly.")
    print("\nNext step: point ann_file to your real cricket_annotations.json")
    print("and frame_dir to your real extracted frames from extract_frames.py")
