<div align="center">

# 🚦 TrafficLens AI

### Intelligent Dashcam Video Analysis for Indian Roads

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B35?style=flat-square)](https://ultralytics.com)
[![DeepSORT](https://img.shields.io/badge/Tracking-DeepSORT-6C63FF?style=flat-square)](https://github.com/levan92/deep_sort_realtime)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-00C87A?style=flat-square)](LICENSE)

*A real-time ML/AI pipeline that analyses dashcam footage and streams live detections to an interactive analytics dashboard — built for Indian road conditions under the Motor Vehicles Act, 1988.*

---

</div>

## 📸 Dashboard Preview

> Open `dashcam_dashboard.html` via **Live Server** in VS Code while `pipeline.py` runs in the terminal.

```
┌──────────────────────────────────────────────────────────────────┐
│  🟢 LIVE   TRAFFICLENS AI          TIME 0:14:22   PROGRESS 97%  │
│  ████████████████████████████████████████████████████████████░░  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  🚨 59        🔀 20        🚗 391       📹 14       ⚡ 97%       │
│  Violations   Junctions   Vehicles   In Frame   Progress        │
│                                                                  │
├─────────────────────────────────────────────────────────────────-┤
│  VIDEO TIMELINE  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                  │││  ││  │││    │  ││   │     ││  │  ← markers │
│                  ↑ violations    ↑ junctions                    │
├───────────────────┬──────────────────┬───────────────────────────┤
│  VIOLATIONS       │  VIOLATION SPLIT │  VEHICLE CATEGORIES      │
│  🪖 Helmet-less ██│  [Donut chart]   │  [Horizontal bar chart]  │
│  ↩️  Wrong-side  █│                  │  2W  ████████████  55%   │
│  🚦 Signal Jump  █│                  │  LMV ███████       30%   │
│  📱 Phone Use   ██│                  │  HMV ███           10%   │
│  👥 Triple Ride  █│                  │  Oth █              5%   │
├───────────────────┴──────────────────┴───────────────────────────┤
│  LIVE EVENT LOG                              47 EVENTS           │
│  [ALL] [VIOLATIONS] [JUNCTIONS] [HELMET] [PHONE] ...            │
│  ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄   │
│  001  0:01:23  ⚠ VIOLATION  🪖 Helmet-less Riding   81.2%  ▶   │
│  002  0:02:45  ⊕ JUNCTION   ✚ 4-Way / X-Junction    —      ▶   │
│  003  0:03:11  ⚠ VIOLATION  📱 Phone Use             76.8%  ▶   │
└──────────────────────────────────────────────────────────────────┘
```

> **To add real screenshots:** replace this section with images using:
> ```md
> ![Dashboard Screenshot](screenshots/dashboard_live.png)
> ![Event Log](screenshots/event_log.png)
> ```

---

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Architecture Overview](#-architecture-overview)
- [Model Choices](#-model-choices)
- [Setup & Installation](#-setup--installation)
- [Running the Pipeline](#-running-the-pipeline)
- [Live Dashboard](#-live-dashboard)
- [Detection Details](#-detection-details)
- [Output Format](#-output-format)
- [Known Limitations](#-known-limitations)
- [Upgrade Path](#-upgrade-path)

---

## ✨ Features

| Task | What it does |
|------|-------------|
| **Traffic Violation Detection** | Detects 5 violation types: helmet-less riding, wrong-side driving, signal jumping, phone use, triple riding |
| **Junction Detection** | Classifies T-junctions, 4-way crossings, Y-junctions, roundabouts, and flyovers |
| **Vehicle Classification** | Tracks and counts 2W, LMV, HMV, and other vehicles — de-duplicated via DeepSORT |
| **Live Dashboard** | Streams all results to an interactive HTML dashboard updating every 2 seconds |
| **Video Download** | Accepts YouTube links, Google Drive links, direct URLs, or local files |

---

## 📁 Project Structure

```
files/
│
├── pipeline.py              ← ML inference engine (run this)
├── dashcam_dashboard.html   ← Live analytics dashboard (open in browser)
├── dashcam.mp4              ← Your dashcam video
├── yolov8x.pt               ← YOLOv8x model weights (auto-downloaded)
│
├── live_status.json         ← Created by pipeline, polled by dashboard
└── results.json             ← Final output after pipeline completes
```

---

## 🏗 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        pipeline.py                              │
│                                                                 │
│  ┌──────────┐    ┌────────────┐    ┌──────────────────────┐   │
│  │  Video   │───▶│  OpenCV    │───▶│   Frame Sampler      │   │
│  │  Source  │    │  Decoder   │    │   (every 3rd frame)  │   │
│  └──────────┘    └────────────┘    └──────────┬───────────┘   │
│                                               │                │
│                          ┌────────────────────▼─────────────┐  │
│                          │         YOLOv8x Inference        │  │
│                          │  Detects: persons, vehicles,     │  │
│                          │  phones, traffic lights          │  │
│                          └───┬──────────┬────────┬──────────┘  │
│                              │          │        │             │
│                   ┌──────────▼──┐  ┌────▼────┐  ┌▼──────────┐ │
│                   │  DeepSORT   │  │Optical  │  │ Heuristic │ │
│                   │  Tracker    │  │  Flow   │  │  Helmet   │ │
│                   │(vehicle IDs)│  │(Farneb.)│  │ Detector  │ │
│                   └──────┬──────┘  └────┬────┘  └─────┬─────┘ │
│                          │             │              │        │
│              ┌───────────▼─────────────▼──────────────▼──────┐ │
│              │           Violation Logic                      │ │
│              │  • Helmet-less  • Wrong-side  • Signal jump   │ │
│              │  • Phone use   • Triple riding                 │ │
│              └───────────────────────┬────────────────────────┘ │
│                                      │                          │
│              ┌───────────────────────▼────────────────────────┐ │
│              │  Background Writer Thread  (every 2 seconds)   │ │
│              │  Writes live_status.json atomically            │ │
│              └───────────────────────┬────────────────────────┘ │
└──────────────────────────────────────┼─────────────────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │   dashcam_dashboard.html │
                          │   polls live_status.json │
                          │   every 2 s via fetch()  │
                          │                          │
                          │  • KPI cards             │
                          │  • Violation charts      │
                          │  • Vehicle distribution  │
                          │  • Junction breakdown    │
                          │  • Density timeline      │
                          │  • Live event log        │
                          └──────────────────────────┘
```

### Data flow

1. OpenCV decodes every 3rd frame from the video for speed
2. YOLOv8x runs detection — returns bounding boxes + class IDs
3. DeepSORT assigns persistent track IDs across frames
4. Optical flow (Farneback) computes per-pixel motion vectors
5. Violation logic fires based on class combinations and motion
6. A background thread writes `live_status.json` every 2 seconds
7. The dashboard fetches this file and updates only changed DOM nodes

---

## 🤖 Model Choices

### Primary Detector — YOLOv8x

| Property | Value |
|----------|-------|
| Architecture | CSPDarknet + C2f modules + SPPF |
| Training data | COCO 2017 (80 classes) |
| Input resolution | 640 × 640 (auto-scaled) |
| Inference mode | `conf=0.35`, NMS IoU=0.45 |
| Why YOLOv8x? | Highest accuracy variant; `x` over `n/s/m` chosen for dashcam where small objects (helmets, phones) matter |

**COCO classes used:**

| Class ID | Label | Used for |
|----------|-------|----------|
| 0 | person | Rider detection, helmet check, signal jump |
| 1 | bicycle | Others category |
| 2 | car | LMV tracking |
| 3 | motorcycle | Two-wheeler + triple riding |
| 5 | bus | HMV tracking |
| 7 | truck | HMV tracking |
| 9 | traffic_light | Red-light state detection |
| 67 | cell phone | Phone use violation |

### Object Tracker — DeepSORT

DeepSORT (Deep Simple Online and Realtime Tracking) is used exclusively for **vehicle de-duplication** — ensuring the same vehicle isn't counted multiple times as it moves through the frame.

| Property | Value |
|----------|-------|
| Re-ID embedder | MobileNet (lightweight) |
| Max age | 30 frames before track is dropped |
| Min hits | 3 confirmed detections before ID is assigned |
| IoU threshold | 0.7 |

### Motion Analysis — Farneback Optical Flow

Used for two detectors:

- **Wrong-side detection:** vehicles with positive mean horizontal flow (moving right on Indian left-hand-drive roads) are flagged
- **Junction detection:** sudden lateral divergence in optical flow signals opening road space ahead

### Helmet Detector (Heuristic)

Since COCO has no helmet class, a fast HSV-based heuristic is used:

1. Crop the top 28% of each rider's bounding box (head region)
2. Compute mean saturation of the HSV crop
3. Low saturation (< 80) → hard-shell helmet (black/white/silver)
4. High saturation → bare head or hair

> **Production upgrade:** Fine-tune YOLOv8 on a helmet dataset such as [Hard Hat Workers](https://universe.roboflow.com/joseph-nelson/hard-hat-workers) or the IDD dataset for significantly higher accuracy.

---

## ⚙️ Setup & Installation

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10 or 3.11 recommended (3.14 works) |
| pip | latest |
| RAM | 8 GB minimum, 16 GB recommended |
| Disk | ~2 GB free (for model weights) |
| GPU | Optional but strongly recommended (NVIDIA CUDA) |

### Step 1 — Clone or download the project

Place all files in one folder:
```
files/
├── pipeline.py
├── dashcam_dashboard.html
└── dashcam.mp4   ← your video here
```

### Step 2 — Install dependencies

```bash
pip3 install ultralytics deep-sort-realtime opencv-python numpy torch torchvision yt-dlp gdown
```

On Apple Silicon (M1/M2/M3 Mac), PyTorch uses the MPS backend automatically — no extra steps needed.

On NVIDIA GPU systems, install the CUDA version of PyTorch for 5–10× faster inference:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 3 — Fix SSL certificates (Mac only)

```bash
/Applications/Python\ 3.x/Install\ Certificates.command
```

---

## ▶️ Running the Pipeline

### Basic usage

```bash
python3 pipeline.py --video dashcam.mp4
```

### With a YouTube link

```bash
python3 pipeline.py --video "https://www.youtube.com/watch?v=VIDEO_ID"
```

### With a Google Drive link

```bash
python3 pipeline.py --video "https://drive.google.com/file/d/FILE_ID/view"
```

### Custom output path

```bash
python3 pipeline.py --video dashcam.mp4 --output my_results.json
```

### Set source permanently in the script

Edit line 50 of `pipeline.py`:
```python
VIDEO_SOURCE = "dashcam.mp4"   # ← change this
```
Then run without flags:
```bash
python3 pipeline.py
```

### Expected terminal output

```
[INFO] Loading YOLOv8x model…
[INFO] 125814 frames  @  30.0 fps  —  1:09:53
[INFO] Open dashcam_dashboard.html in your browser to see live results!
[INFO]   2.4%  —  0:01:41
[INFO]   4.8%  —  0:03:21
...
==================================================
  DONE  →  results.json
  Violations : 59
  Junctions  : 20
  Vehicles   : 391
==================================================
```

---

## 📊 Live Dashboard

### Opening correctly

The dashboard reads `live_status.json` via `fetch()`, which requires a local HTTP server. **Double-clicking the HTML file will not work.**

**Recommended — VS Code Live Server:**

1. Install the **Live Server** extension by Ritwick Dey
2. Right-click `dashcam_dashboard.html` in the VS Code file explorer
3. Click **"Open with Live Server"**
4. Browser opens at `http://127.0.0.1:5500/dashcam_dashboard.html`

**Alternative — Python HTTP server:**
```bash
cd /path/to/files
python3 -m http.server 5500
# open http://localhost:5500/dashcam_dashboard.html
```

### Dashboard sections

| Section | Updates | Description |
|---------|---------|-------------|
| KPI Cards | Every 2 s | Violations, junctions, vehicles, progress |
| Video Timeline | Every 2 s | Coloured markers for each event; click to seek |
| Violations Table | Every 2 s | Count and percentage per violation type |
| Violation Donut | Every 2 s | Distribution chart |
| Vehicle Categories | Every 2 s | 2W / LMV / HMV / Others bar chart |
| Junction Types | Every 2 s | Grid + pie chart |
| Vehicle Density | Every 2 s | Line chart over time |
| Live Event Log | Every 2 s | All detections, filterable, scroll-stable |

### Reading mode

When you scroll down in the event log to read older entries, the dashboard enters **reading mode**:
- A `📖 Reading mode — new events are queued above` banner appears
- A red `+N new` badge on the log header counts incoming events
- The page never forces you back to the top
- Click **"↑ Jump to latest"** when ready to return to live view

---

## 🔍 Detection Details

### Task 1 — Traffic Violations

| Violation | Detection Method | Legal Basis |
|-----------|-----------------|-------------|
| Helmet-less riding | Person on motorcycle + HSV head-crop analysis | MV Act §129 |
| Wrong-side driving | Positive horizontal optical flow on vehicle bbox | MV Act §112 |
| Signal jumping | Vehicle at stop line when traffic light is red | CMVR Rule 93 |
| Phone use | COCO class 67 (cell phone) detected near driver | MV Act §184 |
| Triple riding | 3+ persons overlapping same motorcycle bbox | MV Act §128 |

### Task 2 — Junction Classification

Junctions are detected when lateral optical flow divergence exceeds a threshold. Classification is by divergence magnitude:

| Type | Divergence | Description |
|------|-----------|-------------|
| T-Junction | 1.8 – 2.5 | Three-arm intersection |
| Y-Junction | 2.0 – 2.5 | Acute-angle fork |
| Roundabout | 2.5 – 3.0 | Circular central island |
| T-Junction (main) | 3.0 – 4.0 | Larger three-arm crossing |
| 4-Way / X-Junction | > 4.0 | Full four-arm crossroad |

A 60-frame cooldown prevents double-counting the same crossing.

### Task 3 — Vehicle Classification

| Category | COCO classes | Indian examples |
|----------|-------------|-----------------|
| Two-Wheeler (2W) | motorcycle (3) | Bikes, scooters, mopeds |
| Light Motor Vehicle (LMV) | car (2) | Cars, jeeps, taxis, autos |
| Heavy Motor Vehicle (HMV) | bus (5), truck (7) | Trucks, buses, tankers |
| Others | bicycle (1) | Cycles, e-rickshaws |

---

## 📄 Output Format

### `live_status.json` (written every 2 seconds)

```json
{
  "status": "processing",
  "progress_pct": 47.3,
  "current_ts": "0:32:41",
  "fps": 30.0,
  "total_frames": 125814,
  "duration": "1:09:53",
  "current_vehicle_count": 11,
  "total_violations": 34,
  "junctions": 12,
  "vehicles_unique": 228,
  "violation_counts": {
    "helmetless": 18,
    "wrong_side": 5,
    "signal_jump": 3,
    "phone_use": 6,
    "triple_ride": 2
  },
  "vehicle_categories": {
    "two_wheeler": 125,
    "lmv": 68,
    "hmv": 24,
    "others": 11
  },
  "junction_breakdown": {
    "T-Junction": 6,
    "4-Way / X-Junction": 4,
    "Roundabout / Traffic Circle": 2
  },
  "density_over_time": [
    { "timestamp": "0:00:00", "count": 4 },
    { "timestamp": "0:00:30", "count": 9 }
  ],
  "recent_events": [
    {
      "type": "violation",
      "subtype": "helmetless",
      "timestamp": "0:32:14",
      "frame": 57720,
      "confidence": 0.812
    }
  ]
}
```

### `results.json` (written on completion)

Same structure as above, plus full `instances` and `top3_detections` arrays per violation type including bounding box coordinates.

---

## ⚠️ Known Limitations

### Detection accuracy

| Limitation | Impact | Mitigation |
|------------|--------|-----------|
| Helmet detector is heuristic (HSV-based) | ~65–70% accuracy on varied lighting | Fine-tune YOLOv8 on a helmet dataset |
| COCO has no auto-rickshaw class | Autos classified as LMV via car class | Add custom class with IDD dataset fine-tuning |
| Phone use requires visible phone object | Missed if hand obscures phone | Train on Indian driver phone-use images |
| Wrong-side uses optical flow threshold | Sensitive to camera shake on bumpy roads | Add IMU data or stabilisation pre-processing |
| Junction detection via flow divergence | May miss junctions at low speed | Combine with lane detection |
| Signal jump assumes bottom-of-frame crossing | Misses distant violations | Use explicit stop-line homography |

### Performance

| Limitation | Detail |
|------------|--------|
| Speed | ~3–8 fps processing on CPU; ~15–25 fps on M1/M2 Mac; ~30+ fps on NVIDIA GPU |
| Memory | YOLOv8x loads ~130 MB weights; DeepSORT adds ~60 MB |
| Long videos | `recent_events` in live JSON is capped at 100 entries; `results.json` stores all |
| Night footage | Detection confidence drops significantly under low light |
| Rain / fog | Occlusion reduces tracking accuracy |

### Legal & ethical

- This tool is for **research and analysis purposes only**
- Detections are probabilistic — not suitable for issuing fines or legal action without human review
- Video footage may contain personal data — ensure compliance with applicable privacy laws before processing

---

## 🚀 Upgrade Path

For production-grade accuracy, consider these improvements:

```
Current (heuristic)              →  Production (fine-tuned)
─────────────────────────────────────────────────────────────
HSV helmet detector              →  YOLOv8 trained on IDD + helmet dataset
Optical flow junction detection  →  Dedicated intersection classifier (ResNet-50)
COCO car class for autos         →  Custom class: auto-rickshaw, e-rickshaw
Flow-based wrong-side detection  →  Lane segmentation + vehicle direction vector
Single-frame phone detection     →  Temporal model (3D CNN / LSTM sequence)
CPU/MPS inference                →  TensorRT optimised ONNX export for NVIDIA GPU
```

**Recommended datasets for fine-tuning:**
- [IDD — Indian Driving Dataset](https://idd.insaan.iiit.ac.in/) — Indian road conditions
- [BDD100K](https://bdd-data.berkeley.edu/) — Diverse dashcam footage
- [Hard Hat Workers](https://universe.roboflow.com/joseph-nelson/hard-hat-workers) — Helmet detection

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `ultralytics` | ≥ 8.0 | YOLOv8 model and inference |
| `deep-sort-realtime` | ≥ 1.3 | Multi-object tracking |
| `opencv-python` | ≥ 4.8 | Video I/O and optical flow |
| `numpy` | ≥ 1.24 | Array operations |
| `torch` + `torchvision` | ≥ 2.0 | Deep learning backend |
| `yt-dlp` | latest | YouTube / web video download |
| `gdown` | latest | Google Drive download |

---

## 📜 License

MIT License — free to use, modify, and distribute with attribution.

---

<div align="center">

Built with YOLOv8 · DeepSORT · OpenCV · Chart.js

*Designed for Indian road conditions under the Motor Vehicles Act, 1988*

</div>
