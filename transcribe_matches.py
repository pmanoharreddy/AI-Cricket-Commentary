"""
transcribe_matches.py

Step 1 — Transcribe all match videos using Whisper
Extracts commentary text with timestamps from each video.

Usage:
    python transcribe_matches.py

Output:
    transcripts/match_01_transcript.json
    transcripts/match_02_transcript.json
    ...
"""

import os
import json
import whisper
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────
MATCHES_DIR    = "raw_matches"       # folder with your 10 match videos
TRANSCRIPTS_DIR = "transcripts"      # output folder for transcripts
WHISPER_MODEL  = "base"              # base = fast, medium = better quality
                                     # options: tiny, base, small, medium, large

def transcribe_all_matches():
    os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)

    # Get all video files
    video_files = sorted([
        f for f in os.listdir(MATCHES_DIR)
        if f.endswith((".mp4", ".mkv", ".avi", ".mov"))
    ])

    if not video_files:
        print(f"No videos found in {MATCHES_DIR}/")
        print("Please put your match videos there first.")
        return

    print(f"Found {len(video_files)} videos to transcribe.")
    print(f"Using Whisper model: {WHISPER_MODEL}")
    print("Loading Whisper model...")

    model = whisper.load_model(WHISPER_MODEL)
    print("✅ Whisper loaded.\n")

    for i, video_file in enumerate(video_files, 1):
        video_path = os.path.join(MATCHES_DIR, video_file)
        transcript_name = video_file.replace(".mp4", "").replace(".mkv", "") + "_transcript.json"
        transcript_path = os.path.join(TRANSCRIPTS_DIR, transcript_name)

        # Skip if already transcribed
        if os.path.exists(transcript_path):
            print(f"[{i}/{len(video_files)}] Already transcribed: {video_file} — skipping.")
            continue

        print(f"[{i}/{len(video_files)}] Transcribing: {video_file}")
        print(f"  This may take 10-30 minutes for a full match...")

        # Run Whisper
        result = model.transcribe(
            video_path,
            language="en",
            verbose=False,
            word_timestamps=False,
        )

        # Save transcript with timestamps
        transcript_data = {
            "video_file": video_file,
            "video_path": video_path,
            "language":   result["language"],
            "segments":   [
                {
                    "start": round(seg["start"], 2),
                    "end":   round(seg["end"], 2),
                    "text":  seg["text"].strip(),
                }
                for seg in result["segments"]
                if len(seg["text"].strip()) > 10   # skip very short segments
            ]
        }

        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)

        print(f"  ✅ Done. {len(transcript_data['segments'])} segments saved.")
        print(f"  Saved: {transcript_path}\n")

    print("=" * 50)
    print("All transcriptions complete.")
    print(f"Transcripts saved in: {TRANSCRIPTS_DIR}/")


if __name__ == "__main__":
    transcribe_all_matches()
