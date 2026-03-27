<div align="center">

# ⚡ Sport Vision

**AI-powered motion recognition for racket sports**

实时骨骼追踪 · 击球动作识别 · 运动生物力学分析

[![License: MIT](https://img.shields.io/badge/License-MIT-cyan.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-PoseLandmarker-green.svg)](https://ai.google.dev/edge/mediapipe)

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🦴 **Real-time Pose Tracking** | MediaPipe PoseLandmarker with 13 keypoints |
| 🎯 **Action Recognition** | Serve, Smash, Forehand, Backhand, Lob, Drop |
| � **Biomechanics Analysis** | Joint angles, wrist speed, body lean, symmetry |
| 🔥 **Movement Heatmap** | Spatial visualization of player movement |
| ⚡ **Action Timeline** | Sequential recording of all detected actions |
| 🌐 **Real-time WebSocket** | Streaming analysis at ~20 FPS |
| 🎨 **Stunning UI** | Dark neon theme with particle animation |

> **Zero cloud dependency** — All AI inference runs locally on CPU. No API keys needed.

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/MindDock/sport-vision.git
cd sport-vision

# One-click start (creates venv, installs deps, downloads model, launches server)
chmod +x run.sh
./run.sh

# Open browser
open http://localhost:8000
```

### Manual Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download pose model
mkdir -p models
curl -sL -o models/pose_landmarker_lite.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"


# convert tflite to onnx

/home/faith/sport-vision/.venv/bin/python -m tf2onnx.convert --tflite models/pose_landmark_lite.tflite --output models/pose_landmark_lite.onnx

# optional: set the YOLO person detector model path
export SPORT_VISION_PERSON_DETECTOR_MODEL=/home/faith/yolo_c/best.onnx

# optional: disable first-target lock mode in code if you want freer target switching
# Pipeline(lock_first_target=False)


# Start server
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Action Template Validation With ONNX

Use a standard-action video to extract a normalized pose sequence template:

```bash
/home/faith/sport-vision/.venv/bin/python infer_video_action_onnx.py b.mp4 \
  --action-name serve
```

By default this saves the template to `uploads/serve_template.json`. You can also set a custom path
with `--template-json`; if that file does not exist yet, the script will create it as a new template.

Then validate a new video against that template:

```bash
/home/faith/sport-vision/.venv/bin/python infer_video_action_onnx.py b.mp4 \
  --template-json uploads/serve_template.json \
  --match-threshold 0.72


/home/faith/sport-vision/.venv/bin/python infer_video_action_onnx.py b.mp4 --template-json uploads/test_cli_template.json --max-frames 280 --match-threshold 0.72
```

The generated template JSON stores the normalized keypoint-change sequence, joint-angle sequence,
and motion statistics extracted from the reference action. When `--template-json` points to an existing
file, the script compares the new video against that template and reports whether the movement is standard.

当前视频链路已经改为先用轻量级 YOLO 做人体检测，再把检测框裁剪后送入 ONNX pose 模型。
默认开启“只跟踪第一次出现的主目标”模式，多人场景下会优先保持同一个人，降低切人概率。
分析界面支持点击 🎯 后在视频上手动点选主目标，画面也会显示当前检测框和跟踪状态。

## 🎬 Usage

1. **Upload a video** — Click the upload button and select a badminton/tennis match video
2. **Use demo videos** — Place `.mp4` files in the `demo_videos/` directory
3. **Watch the analysis** — Real-time skeleton overlay, action recognition, and biomechanics data
4. **Review the timeline** — All detected actions are recorded in the action timeline
5. **Lock the main athlete** — Click `🎯` in the analysis view, then click the player in the frame to manually lock the target

## 🏗️ Architecture

```
Video Input
    ↓
┌─────────────────────────────────────┐
│  FastAPI Backend                    │
│  ┌──────────────┐                   │
│  │ MediaPipe     │→ 13 Keypoints    │
│  │ PoseLandmarker│                  │
│  └──────────────┘                   │
│         ↓                           │
│  ┌──────────────┐  ┌─────────────┐  │
│  │ Action        │  │ Biomechanics│  │
│  │ Recognizer    │  │ Analyzer    │  │
│  └──────────────┘  └─────────────┘  │
│         ↓                ↓          │
│     WebSocket (JSON + Base64 Frame) │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Frontend (Canvas + Dashboard)      │
│  Skeleton Overlay · Gauges · Heatmap│
└─────────────────────────────────────┘
```

## 📁 Project Structure

```
sport-vision/
├── backend/
│   ├── main.py              # FastAPI server + WebSocket
│   ├── pipeline.py           # Video processing pipeline
│   ├── pose_analyzer.py      # MediaPipe pose estimation + biomechanics
│   ├── action_recognizer.py  # Rule-based action recognition engine
│   └── visualizer.py         # OpenCV skeleton rendering
├── frontend/
│   ├── index.html            # Single-page application
│   ├── css/style.css         # Dark neon design system
│   └── js/
│       ├── app.js            # WebSocket client + UI orchestration
│       ├── dashboard.js      # Gauges, stats, heatmap rendering
│       ├── skeleton-renderer.js  # Canvas skeleton overlay
│       └── particles.js      # Background particle animation
├── models/                   # MediaPipe model (auto-downloaded)
├── demo_videos/              # Place your .mp4 files here
├── requirements.txt
├── run.sh                    # One-click launcher
├── LICENSE
└── README.md
```

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Pose Estimation | [MediaPipe PoseLandmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) |
| Action Recognition | Rule-based keypoint temporal analysis |
| Video Processing | OpenCV |
| Backend | FastAPI + WebSocket |
| Frontend | Vanilla JS + Canvas API |
| Styling | CSS (dark neon + glassmorphism) |

## 🤝 Contributing

Contributions are welcome! Some ideas:

- [ ] **Improve action recognition** — Add ML-based classifier (ST-GCN)
- [ ] **Multi-person tracking** — ByteTrack / BoT-SORT integration
- [ ] **3D pose lifting** — MotionBERT for 2D→3D reconstruction
- [ ] **Action quality assessment** — Score technique quality
- [ ] **Mobile deployment** — CoreML / TFLite export

## 📄 License

[MIT](LICENSE) © [MindDock](https://github.com/MindDock)
