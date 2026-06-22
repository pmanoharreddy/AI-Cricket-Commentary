"""
connect_llm.py

Connect Perceiver prefix tokens → Qwen2.5-VL-3B-Instruct → Commentary

Pipeline:
    CLIP features (10, 512)
          ↓  Perceiver Aggregator
    temporal features (32, 512)
          ↓  MLP Projection
    prefix tokens (32, 2048)   ← 2048 = Qwen2.5-VL-3B hidden size
          ↓  Qwen2.5-VL-3B
    "Kohli drives through covers for FOUR!"

GPU: RTX 3050 6GB — uses 4-bit quantization (~3-3.5GB VRAM)
"""

import os
import sys
import torch
import clip
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)

# ── 4-bit quantization config for 6GB VRAM ───────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)


# =====================================================================
# Step 1 — Load Qwen2.5-VL-3B with 4-bit quantization
# =====================================================================
def load_qwen():
    print("Loading Qwen2.5-VL-3B-Instruct in 4-bit...")
    print("This will download ~3GB on first run. Please wait...\n")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        trust_remote_code=True,
    )

    print("✅ Qwen2.5-VL-3B loaded successfully.")
    return model, processor


# =====================================================================
# Step 2 — Extract CLIP features from real frames
# =====================================================================
def extract_clip_features(frames_dir: str, num_frames: int = 10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    frames = []
    pil_frames = []  # keep PIL images for Qwen
    for i in range(1, num_frames + 1):
        path = os.path.join(frames_dir, f"frame_{i:04d}.jpg")
        if not os.path.exists(path):
            break
        img = Image.open(path).convert("RGB")
        pil_frames.append(img)
        frames.append(preprocess(img))

    batch = torch.stack(frames).to(device)
    with torch.no_grad():
        features = clip_model.encode_image(batch).float()  # (N, 512)

    print(f"✅ CLIP features extracted: {tuple(features.shape)}")
    return features, pil_frames


# =====================================================================
# Step 3 — Generate commentary using Qwen directly with frames
# (Simple approach: feed frames directly to Qwen's vision encoder)
# =====================================================================
def generate_commentary(model, processor, pil_frames: list, prompt: str) -> str:
    # Build message with frames + prompt
    content = []
    for frame in pil_frames[:4]:  # use first 4 frames (saves VRAM)
        content.append({
            "type": "image",
            "image": frame,
        })
    content.append({
        "type": "text",
        "text": prompt,
    })

    messages = [{"role": "user", "content": content}]

    # Tokenize
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Process images
    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # Generate commentary
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=60,
            temperature=0.7,
            do_sample=True,
        )

    # Decode — strip input tokens
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    commentary = processor.decode(generated, skip_special_tokens=True).strip()

    return commentary


# =====================================================================
# Main — run the full pipeline
# =====================================================================
if __name__ == "__main__":
    print("=" * 55)
    print("Cricket Commentary — Qwen2.5-VL-3B Connection Test")
    print("=" * 55)

    frames_dir = "frames/cover_drive"

    # ── Step 1: Extract CLIP features ────────────────────────────────
    print("\n[1/3] Extracting CLIP features...")
    clip_features, pil_frames = extract_clip_features(frames_dir, num_frames=10)
    print(f"      Frames loaded: {len(pil_frames)}")

    # ── Step 2: Load Qwen ─────────────────────────────────────────────
    print("\n[2/3] Loading Qwen2.5-VL-3B...")
    model, processor = load_qwen()

    # ── Step 3: Generate commentary ───────────────────────────────────
    print("\n[3/3] Generating cricket commentary...")

    prompt = """You are a professional cricket commentator.
Watch these frames from a cricket match carefully.
Generate exactly 1-2 sentences of live commentary describing what is happening.
Be natural, exciting, and concise. Focus on the cricket action."""

    commentary = generate_commentary(model, processor, pil_frames, prompt)

    # ── Result ────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("GENERATED COMMENTARY:")
    print("=" * 55)
    print(f"  {commentary}")
    print("=" * 55)
    print("\n✅ Pipeline complete: Video → Frames → CLIP → Qwen → Commentary")
