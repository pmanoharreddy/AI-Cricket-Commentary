# CricketVoice — AI Cricket Commentary System

**Real-Time Multimodal Cricket Commentary Generation using Vision-Language Models, OCR, Player Recognition, Match Context Memory, and LLMs**

Watch a raw cricket broadcast video → automatically generate spoken audio commentary in real time, like a live broadcast commentator — fully AI-driven.

> **Status:** Week 1 Complete (Research & Architecture). Weeks 2–6 = Implementation.

---

## Novel Contributions (over MatchTime baseline)

1. **Scoreboard OCR Module** — YOLOv8n + PaddleOCR reads live scorecard overlays into structured JSON per delivery
2. **Match Memory Module** — rolling cross-delivery state buffer (runs, wickets, milestones, narrative flags) stored in SQLite
3. **RAG Knowledge Base** — ChromaDB vector index of real cricket commentary; top-3 passages injected per LLM prompt

---

## System Pipeline

```
INPUT       → Frame Extraction (OpenCV, 3 FPS) + Delivery Detection
PERCEPTION  → YOLOv8m Player Detection + ByteTrack + TrackNetV2 Ball + CLIP Events + Scoreboard OCR
CONTEXT     → Match Memory (SQLite) + RAG (ChromaDB) + Historical Stats (CricSheet)
OUTPUT      → Mistral 7B LoRA → Piper TTS → FFmpeg audio-video sync → Streamlit Dashboard
```

---

## Tech Stack (100% Free & Open Source)

| Layer | Tool | License |
|---|---|---|
| Object Detection | YOLOv8m / YOLOv8n | AGPL-3.0 |
| Player Tracking | ByteTrack | MIT |
| Ball Tracking | TrackNetV2 | Academic |
| Visual Encoder | OpenCLIP ViT-B/32 | MIT |
| OCR Engine | PaddleOCR 2.7 | Apache 2.0 |
| LLM | Mistral 7B v0.3 Instruct + QLoRA | Apache 2.0 |
| Vector DB | ChromaDB | Apache 2.0 |
| TTS | Piper TTS (en_US-jenny-medium) | MIT |
| Backend | FastAPI + WebSocket | MIT |
| Frontend | Streamlit | Apache 2.0 |
| Experiment Tracking | MLflow (local) | Apache 2.0 |

---

## Hardware Requirements

- GPU: RTX 4070 8GB VRAM (Mistral 7B 4-bit uses ~5.5–6GB)
- RAM: 32GB
- Storage: ~50GB for models + datasets
- LoRA fine-tuning: Google Colab T4 (free tier)

---

## Project Structure

```
cricket_commentary_system/
├── modules/          # Core pipeline modules
│   ├── scoreboard_ocr.py     # Novel Contribution 1
│   ├── match_memory.py       # Novel Contribution 2
│   └── rag_retriever.py      # Novel Contribution 3
├── training/         # YOLO + LoRA fine-tuning scripts
├── evaluation/       # Metrics: BLEU, METEOR, BERTScore, MOS
├── api/              # FastAPI backend
├── frontend/         # Streamlit dashboard
├── notebooks/        # VSCode Jupyter notebooks
├── data/             # Datasets (gitignored)
└── models/           # Fine-tuned weights (gitignored)
```

---

## Getting Started

```bash
conda create -n cricket python=3.11 -y
conda activate cricket
pip install -r requirements.txt
```

Verify GPU:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Baseline

Primary baseline: **MatchTime** (EMNLP 2024 Oral) — a soccer commentary system adapted to cricket.  
We retain: CLIP visual encoder, Perceiver aggregator, LLaMA-3 style decoder.  
We add: Scoreboard OCR, Match Memory, RAG, TTS audio output.

---

## License

Apache 2.0
