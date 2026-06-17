"""
extract_frames.py

Video → Frame Extraction
Extracts 1 frame every second from a video file.

Usage:
    python extract_frames.py cover_drive.mp4
    python extract_frames.py cover_drive.mp4 --fps 2 --output frames/
    python extract_frames.py cover_drive.mp4 --fps 1 --output frames/ --resize 224

Output:
    frames/frame_0001.jpg
    frames/frame_0002.jpg
    frames/frame_0003.jpg
    ...

Reference: MatchTime (Rao et al., EMNLP 2024)
"""

import os
import cv2
import argparse


def extract_frames(
    video_path: str,
    output_dir: str = "frames",
    fps: float = 1.0,
    resize: int = None,
):
    """
    Extract frames from a video at a given FPS rate.

    Args:
        video_path:  Path to input video (e.g. cover_drive.mp4)
        output_dir:  Folder to save extracted frames
        fps:         How many frames to extract per second (default: 1)
        resize:      If set, resize frames to (resize x resize) pixels
    """

    # ── Validate input ───────────────────────────────────────────────
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    # ── Open video ───────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    video_fps    = cap.get(cv2.CAP_PROP_FPS)           # e.g. 25.0 or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / video_fps

    # How many original frames to skip between each saved frame
    frame_interval = int(round(video_fps / fps))        # e.g. 25 frames → save every 25th

    print(f"Video      : {video_path}")
    print(f"Video FPS  : {video_fps:.1f}")
    print(f"Duration   : {duration_sec:.1f} seconds")
    print(f"Total frames: {total_frames}")
    print(f"Saving every {frame_interval} frames  →  {fps} frame(s)/sec")
    print(f"Output dir : {output_dir}/")
    print("-" * 45)

    # ── Extract and save frames ──────────────────────────────────────
    saved_count  = 0
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save this frame only at the desired interval
        if frame_number % frame_interval == 0:
            saved_count += 1

            # Convert BGR (OpenCV default) → RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Optional resize
            if resize is not None:
                frame_rgb = cv2.resize(frame_rgb, (resize, resize))

            # Convert back to BGR for cv2.imwrite (it writes BGR)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Save as frame_0001.jpg, frame_0002.jpg, ...
            filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(filename, frame_bgr)

            print(f"  Saved: frame_{saved_count:04d}.jpg  (video frame #{frame_number})")

        frame_number += 1

    cap.release()

    print("-" * 45)
    print(f"Done. {saved_count} frames saved to '{output_dir}/'")
    return saved_count


# ── CLI ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract frames from a cricket video (1 frame/sec by default)"
    )
    parser.add_argument(
        "video",
        help="Path to input video file (e.g. cover_drive.mp4)"
    )
    parser.add_argument(
        "--output", "-o",
        default="frames",
        help="Output directory for frames (default: frames/)"
    )
    parser.add_argument(
        "--fps", "-f",
        type=float,
        default=1.0,
        help="Frames to extract per second (default: 1.0)"
    )
    parser.add_argument(
        "--resize", "-r",
        type=int,
        default=None,
        help="Resize frames to NxN pixels, e.g. --resize 224 for CLIP input"
    )

    args = parser.parse_args()

    extract_frames(
        video_path=args.video,
        output_dir=args.output,
        fps=args.fps,
        resize=args.resize,
    )
