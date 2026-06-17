"""
models/vision/clip_encoder.py

Frozen CLIP ViT-B/32 Vision Encoder — adapted from MatchTime
Reference: Radford et al., CLIP (ICML 2021)

This is the FIRST stage of the pipeline:
    Video frames → CLIP encoder → per-frame 512-dim features

IMPORTANT: This encoder is FROZEN during training.
Only the Temporal Aggregator and MLP Projection are trained.
"""

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from typing import List, Union


class CLIPVideoEncoder(nn.Module):
    """
    Wraps CLIP ViT-B/32 to encode video frames.
    Processes frames independently (framewise encoding).
    
    Input:  List of PIL Images or numpy arrays (frames from a video clip)
    Output: Tensor of shape (n_frames, 512) — one 512-dim vector per frame
    
    Usage:
        encoder = CLIPVideoEncoder()
        features = encoder(frames)  # (n_frames, 512)
    """

    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        super().__init__()

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading CLIP {model_name} on {self.device}...")

        try:
            import clip
            self.model, self.preprocess = clip.load(model_name, device=self.device)
        except ImportError:
            raise ImportError(
                "CLIP not installed. Run:\n"
                "  pip install git+https://github.com/openai/CLIP.git"
            )

        # FREEZE all CLIP parameters — never update these
        for param in self.model.parameters():
            param.requires_grad = False

        self.output_dim = 512  # CLIP ViT-B/32 output dimension
        print(f"✅ CLIP encoder loaded. Output dim: {self.output_dim}. All weights FROZEN.")

    @torch.no_grad()
    def encode_frames(self, frames: List[Union[Image.Image, np.ndarray]]) -> torch.Tensor:
        """
        Encode a list of video frames into CLIP features.

        Args:
            frames: List of PIL Images or numpy arrays (H×W×3, RGB)

        Returns:
            features: torch.Tensor of shape (n_frames, 512)
        """
        if len(frames) == 0:
            raise ValueError("Empty frame list provided to CLIP encoder.")

        processed = []
        for frame in frames:
            if isinstance(frame, np.ndarray):
                frame = Image.fromarray(frame.astype(np.uint8))
            processed.append(self.preprocess(frame))

        # Stack into batch: (n_frames, 3, 224, 224)
        batch = torch.stack(processed).to(self.device)

        # Encode with CLIP vision encoder
        features = self.model.encode_image(batch)  # (n_frames, 512)
        features = features.float()                # ensure float32

        return features

    def forward(self, frames: List[Union[Image.Image, np.ndarray]]) -> torch.Tensor:
        return self.encode_frames(frames)


# ---- Quick sanity test ----
if __name__ == "__main__":
    import numpy as np

    print("Testing CLIPVideoEncoder...")

    # Create dummy frames (simulate 16 frames from 8-second clip at 2 FPS)
    dummy_frames = [
        Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        for _ in range(16)
    ]

    encoder = CLIPVideoEncoder()
    features = encoder(dummy_frames)

    print(f"Input:  {len(dummy_frames)} frames")
    print(f"Output: {features.shape}")   # Expected: (16, 512)
    assert features.shape == (16, 512), f"Unexpected shape: {features.shape}"
    print("✅ CLIPVideoEncoder test passed.")
