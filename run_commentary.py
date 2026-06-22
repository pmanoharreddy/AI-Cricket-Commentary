"""
run_commentary.py

Full Cricket Commentary Pipeline with Dynamic Pause-Aware Decoding
Reference: Afzal et al., arXiv 2603.02655 (2026)

Pipeline:
    Video → Extract clips → CLIP → Qwen2.5-VL-3B
         → Dynamic Pause Decoder → Commentary + .srt file

Usage:
    python run_commentary.py cover_drive.mp4
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
from qwen_vl_utils import process_vision_info
import cv2
from dataclasses import dataclass
from typing import List


# =====================================================================
# Config
# =====================================================================
FRAMES_PER_CLIP = 4          # frames fed to Qwen per clip (saves VRAM)
CLIP_DURATION   = 8          # seconds per clip
FPS_EXTRACT     = 1          # 1 frame per second
SPEECH_RATE     = 4.0        # words per second (English)
DEFAULT_WAIT    = 2.0        # seconds to wait when <WAIT> is output
WAIT_TOKEN      = "<WAIT>"
MAX_NEW_TOKENS  = 60


# =====================================================================
# Commentary Entry
# =====================================================================
@dataclass
class CommentaryEntry:
    timestamp: float
    text: str
    speak_duration: float


# =====================================================================
# Dynamic Pause Decoder
# d̂ = w / r  (Afzal et al., 2026)
# =====================================================================
def estimate_speak_duration(text: str, speech_rate: float = SPEECH_RATE) -> float:
    words = len(text.split())
    return words / speech_rate


def next_query_time(current_time: float, output: str) -> float:
    if WAIT_TOKEN in output or output.strip() == "":
        return current_time + DEFAULT_WAIT
    return current_time + estimate_speak_duration(output)


# =====================================================================
# Load Models
# =====================================================================
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # CLIP
    print("Loading CLIP...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    for p in clip_model.parameters():
        p.requires_grad = False
    print("✅ CLIP loaded.")

    # Qwen2.5-VL-3B in 4-bit
    print("Loading Qwen2.5-VL-3B-Instruct (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        trust_remote_code=True,
    )
    print("✅ Qwen2.5-VL-3B loaded.")

    return clip_model, clip_preprocess, qwen, processor, device


# =====================================================================
# Extract frames for a time window
# =====================================================================
def extract_frames_at(video_path: str, start_sec: float,
                       duration: float = CLIP_DURATION,
                       fps: float = FPS_EXTRACT) -> List[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    interval = int(round(video_fps / fps))
    start_frame = int(start_sec * video_fps)
    end_frame   = int((start_sec + duration) * video_fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame

    while frame_idx < end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
        frame_idx += interval

    cap.release()
    return frames


def get_video_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return total / fps if fps > 0 else 0.0


# =====================================================================
# Build prompt
# =====================================================================
INIT_PROMPT = """You are a professional cricket commentator.
Watch these frames from the opening of a cricket match.
Generate exactly 1 sentence describing what is happening.
Be concise and natural. Focus only on the cricket action."""

INFERENCE_PROMPT = """You are a professional cricket commentator providing live ball-by-ball commentary.

Previous commentary:
{history}

Watch the latest frames from the match.
1) If nothing new has happened since the last commentary, output exactly: <WAIT>
2) If there is new action, generate 1-2 concise sentences of commentary.
Focus on the cricket action. Do not repeat previous commentary."""


def build_prompt(history: List[CommentaryEntry], is_first: bool) -> str:
    if is_first:
        return INIT_PROMPT

    if not history:
        history_str = "(No commentary yet)"
    else:
        lines = []
        for e in history[-5:]:  # last 5 entries only
            m = int(e.timestamp // 60)
            s = int(e.timestamp % 60)
            lines.append(f"[{m:02d}:{s:02d}] {e.text}")
        history_str = "\n".join(lines)

    return INFERENCE_PROMPT.format(history=history_str)


# =====================================================================
# Generate one commentary output
# =====================================================================
def generate(qwen, processor, pil_frames: List[Image.Image],
             prompt: str, device: str) -> str:

    # Use only first FRAMES_PER_CLIP frames to save VRAM
    frames_to_use = pil_frames[:FRAMES_PER_CLIP]
    if not frames_to_use:
        return WAIT_TOKEN

    content = []
    for frame in frames_to_use:
        content.append({"type": "image", "image": frame})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output_ids = qwen.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            do_sample=True,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()


# =====================================================================
# Save as .srt subtitle file
# =====================================================================
def save_srt(commentary: List[CommentaryEntry], output_path: str):
    def fmt(s):
        h  = int(s // 3600)
        m  = int((s % 3600) // 60)
        sc = int(s % 60)
        ms = int((s % 1) * 1000)
        return f"{h:02d}:{m:02d}:{sc:02d},{ms:03d}"

    with open(output_path, "w") as f:
        for i, entry in enumerate(commentary, 1):
            start = entry.timestamp
            end   = entry.timestamp + entry.speak_duration
            f.write(f"{i}\n")
            f.write(f"{fmt(start)} --> {fmt(end)}\n")
            f.write(f"{entry.text}\n\n")

    print(f"✅ Subtitles saved: {output_path}")


# =====================================================================
# Main pipeline
# =====================================================================
def run(video_path: str):
    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    print("=" * 55)
    print("AI Cricket Commentary — Full Pipeline")
    print("=" * 55)

    # Load models
    clip_model, clip_preprocess, qwen, processor, device = load_models()

    duration = get_video_duration(video_path)
    print(f"\nVideo duration : {duration:.1f}s")
    print(f"Strategy       : Dynamic Pause-Aware Decoding")
    print(f"Clip duration  : {CLIP_DURATION}s")
    print("-" * 55)

    history: List[CommentaryEntry] = []
    t        = 0.0
    is_first = True

    while t < duration:
        print(f"\n[{t:.1f}s] Extracting frames...")
        frames = extract_frames_at(video_path, start_sec=t,
                                    duration=CLIP_DURATION)

        if not frames:
            print(f"[{t:.1f}s] No frames — stopping.")
            break

        prompt = build_prompt(history, is_first)
        is_first = False

        print(f"[{t:.1f}s] Generating commentary...")
        output = generate(qwen, processor, frames, prompt, device)

        if WAIT_TOKEN in output or output.strip() == "":
            print(f"[{t:.1f}s] → <WAIT>")
            t = next_query_time(t, WAIT_TOKEN)
        else:
            speak_dur = estimate_speak_duration(output)
            entry = CommentaryEntry(
                timestamp=t,
                text=output,
                speak_duration=speak_dur,
            )
            history.append(entry)
            print(f"[{t:.1f}s] → {output}")
            t = next_query_time(t, output)

    # Save results
    print("\n" + "=" * 55)
    print(f"Generated {len(history)} commentary lines.")

    srt_path = video_path.replace(".mp4", "_commentary.srt")
    save_srt(history, srt_path)

    print("\nFull Commentary:")
    print("-" * 55)
    for e in history:
        m = int(e.timestamp // 60)
        s = int(e.timestamp % 60)
        print(f"[{m:02d}:{s:02d}] {e.text}")

    print("\n✅ Done.")


# =====================================================================
if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "cover_drive.mp4"
    run(video)
