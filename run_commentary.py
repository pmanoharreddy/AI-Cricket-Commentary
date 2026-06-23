"""
run_commentary.py

Full Cricket Commentary Pipeline with Dynamic Pause-Aware Decoding
Reference: Afzal et al., arXiv 2603.02655 (2026)

Usage:
    python run_commentary.py cover_drive.mp4
"""

import os
import sys
import re
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
FRAMES_PER_CLIP = 4
CLIP_DURATION   = 8
FPS_EXTRACT     = 1
SPEECH_RATE     = 4.0
DEFAULT_WAIT    = 2.0
WAIT_TOKEN      = "<WAIT>"
MAX_NEW_TOKENS  = 30


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
# =====================================================================
def estimate_speak_duration(text: str) -> float:
    return len(text.split()) / SPEECH_RATE


def next_query_time(current_time: float, output: str) -> float:
    if WAIT_TOKEN in output or output.strip() == "":
        return current_time + DEFAULT_WAIT
    return current_time + estimate_speak_duration(output)


# =====================================================================
# Load Models
# =====================================================================
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading CLIP...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    for p in clip_model.parameters():
        p.requires_grad = False
    print("✅ CLIP loaded.")

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
# Extract frames
# =====================================================================
def extract_frames_at(video_path, start_sec, duration=CLIP_DURATION,
                       fps=FPS_EXTRACT):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(round(video_fps / fps))
    start_frame = int(start_sec * video_fps)
    end_frame = int((start_sec + duration) * video_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    frame_idx = start_frame
    while frame_idx < end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        frame_idx += interval
    cap.release()
    return frames


def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return total / fps if fps > 0 else 0.0


# =====================================================================
# Prompts — strict, no hallucination
# =====================================================================
INIT_PROMPT = """You are a cricket commentator describing ONLY what you literally see.

STRICT RULES:
- Describe ONLY what is visible in these frames.
- Do NOT guess or invent player names. Say "the batsman" or "the bowler".
- Do NOT mention scores, match results, or team names unless visible on screen.
- Write ONE sentence. Maximum 12 words. English only.
- No lists. No HTML. Stop after one sentence.

What do you see in these cricket frames?"""

INFERENCE_PROMPT = """You are a cricket commentator describing ONLY what you literally see.

Previous commentary:
{history}

STRICT RULES:
- If nothing visually changed since last commentary: output exactly <WAIT>
- If something new is visible: write ONE sentence, maximum 12 words.
- Do NOT invent player names — say "the batsman" or "the bowler".
- Do NOT mention scores unless shown on screen.
- Do NOT repeat previous commentary.
- English only. No lists. No HTML.

What changed in these frames? (one sentence or <WAIT>):"""


def build_prompt(history, is_first):
    if is_first:
        return INIT_PROMPT
    history_str = "(No commentary yet)" if not history else "\n".join(
        f"[{int(e.timestamp//60):02d}:{int(e.timestamp%60):02d}] {e.text}"
        for e in history[-4:]
    )
    return INFERENCE_PROMPT.format(history=history_str)


# =====================================================================
# Clean output
# =====================================================================
def clean_output(text: str) -> str:
    # Return WAIT immediately
    if WAIT_TOKEN in text:
        return WAIT_TOKEN

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Remove non-ASCII
    text = text.encode("ascii", errors="ignore").decode("ascii").strip()

    # Remove numbered list prefixes
    text = re.sub(r"^\d+[\)\.]\s*", "", text).strip()

    # Keep only first sentence
    for punct in [".", "!", "?"]:
        idx = text.find(punct)
        if 0 < idx < 120:
            text = text[:idx + 1].strip()
            break

    # Truncate to 80 chars max if still long
    if len(text) > 100:
        text = text[:97] + "..."

    return text


# =====================================================================
# Generate
# =====================================================================
def generate(qwen, processor, pil_frames, prompt, device):
    frames_to_use = pil_frames[:FRAMES_PER_CLIP]
    if not frames_to_use:
        return WAIT_TOKEN

    content = [{"type": "image", "image": f} for f in frames_to_use]
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
            temperature=0.2,        # very low — less hallucination
            do_sample=True,
            repetition_penalty=1.5, # strong penalty for repetition
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    raw = processor.decode(generated, skip_special_tokens=True).strip()
    return clean_output(raw)


# =====================================================================
# Save SRT
# =====================================================================
def save_srt(commentary, output_path):
    def fmt(s):
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        sc = int(s % 60)
        ms = int((s % 1) * 1000)
        return f"{h:02d}:{m:02d}:{sc:02d},{ms:03d}"

    with open(output_path, "w", encoding="utf-8") as f:
        for i, entry in enumerate(commentary, 1):
            end = entry.timestamp + max(entry.speak_duration, 2.0)
            f.write(f"{i}\n")
            f.write(f"{fmt(entry.timestamp)} --> {fmt(end)}\n")
            f.write(f"{entry.text}\n\n")

    print(f"✅ Subtitles saved: {output_path}")


# =====================================================================
# Main
# =====================================================================
def run(video_path):
    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    print("=" * 55)
    print("AI Cricket Commentary — Full Pipeline")
    print("=" * 55)

    clip_model, clip_preprocess, qwen, processor, device = load_models()

    duration = get_video_duration(video_path)
    print(f"\nVideo duration : {duration:.1f}s")
    print(f"Device         : {device}")
    print("-" * 55)

    history = []
    t = 0.0
    is_first = True

    while t < duration:
        print(f"\n[{t:.1f}s] Generating commentary...")
        frames = extract_frames_at(video_path, start_sec=t)

        if not frames:
            break

        output = generate(qwen, processor, frames,
                          build_prompt(history, is_first), device)
        is_first = False

        if WAIT_TOKEN in output or not output.strip():
            print(f"[{t:.1f}s] → <WAIT>")
            t = next_query_time(t, WAIT_TOKEN)
        else:
            entry = CommentaryEntry(
                timestamp=t,
                text=output,
                speak_duration=estimate_speak_duration(output),
            )
            history.append(entry)
            print(f"[{t:.1f}s] → {output}")
            t = next_query_time(t, output)

    print("\n" + "=" * 55)
    print(f"Generated {len(history)} commentary lines.")

    srt_path = video_path.replace(".mp4", "_commentary.srt")
    save_srt(history, srt_path)

    print("\nFull Commentary:")
    print("-" * 55)
    for e in history:
        print(f"[{int(e.timestamp//60):02d}:{int(e.timestamp%60):02d}] {e.text}")

    print("\n✅ Done.")


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "cover_drive.mp4"
    run(video)
