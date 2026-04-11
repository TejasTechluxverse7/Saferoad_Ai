# 🛡️ AegisRoad AI - Intelligent Predictive Road Safety Ecosystem

AegisRoad AI is a **production-ready, demo-ready** modular system for real-time road safety, accident detection, and predictive analytics. Designed for hackathons and high-stakes infrastructure deployments.

---

## 🚀 Key Features

*   **Real-time Detection**: YOLOv8-powered accident detection (`aegis_model.pt`).
*   **Intelligent Tracking**: ByteTrack integration for persistent vehicle IDs.
*   **Near-Miss Prediction**: Velocity-based trajectory analysis with collision point estimation.
*   **Severity Classification**: Categorizes incidents from LOW to CRITICAL based on speed, overlap, and vehicle count.
*   **Forensic Replay**: Automatic generation of slow-motion MP4 reconstructions of accidents.
*   **Explainable AI (XAI)**: Grad-CAM heatmap overlays for transparency in HIGH/CRITICAL events.
*   **Victim Detection**: Heuristic-based fallen person detection.
*   **Multi-Camera ReID**: ResNet-18 based vehicle re-identification across different camera IDs.
*   **Smart Analytics Dashboard**: Glassmorphism UI with MJPEG streaming, real-time map heatmap, and event logs.
*   **Smart Routing**: A* algorithm to route ambulances to the nearest hospital via `city_graph.json`.

---

## 🛠️ Setup & Installation

### 1. Requirements
*   Python 3.10+
*   OpenCV, PyTorch, Ultralytics, Supervision, Flask-SocketIO

### 2. Install Dependencies
```bash
pip install -r requirements_saferoad.txt
```

---

## 🧪 Testing Procedure

### 1. Functional Verification (Quick Start)
Run the main orchestrator with a sample accident video to see the full HUD and logic in action:
```bash
python saferoad_main.py --source sample_videos/acci.mp4
```
*   **Keyboard Controls**: `Q` to Quit, `P` to Pause, `R` to Replay.
*   **What to look for**: Trajectory lines (fading), Speed arrows, Pulsing near-miss circles, Severity banners.

### 2. Automated Feature Check
Verify every internal engine (Detection, Tracking, Prediction, Severity, Routing, etc.) with the built-in validation script:
```bash
python verify_features.py
```
This script runs mock data through every module to ensure architectural integrity.

### 3. Dashboard & Upload Testing
Launch the full API stack and interact via the web browser:
1. Start the system: `python saferoad_main.py --source sample_videos/dashcam.mp4 --port 5050`
2. Open **http://localhost:5050** in your browser.
3. **Live Feed**: Confirm the stream loads and connection shows "LIVE".
4. **Upload Video**: Go to the "Upload Video" tab, drag/drop a local video file, and watch the AI process it in real-time.

### 4. Forensic & Evidence Testing
After an accident detection (or demo trigger):
*   Check the `accident_frames/` folder.
*   Look for `replay_{event_id}.mp4` to see the slow-motion reconstruction.
*   Check the `evidence_{event_id}/` folder for pre/post accident frames and JSON metadata.

---

## 🧠 Training Procedure

To retrain or fine-tune the model using the Roboflow Accident Dataset:
```bash
python model_train.py --epochs 25 --batch 16 --device 0
```
This script will:
1. Automatically fetch the dataset.
2. Train a YOLOv8n model.
3. Save the best weights as `models/aegis_model.pt`.

---

## 📁 Project Structure

*   `saferoad_main.py`: Main system orchestrator.
*   `engine/`: Core logic (Severity, Forensic, GradCAM, Victim Detector, Evidence).
*   `tracking/`: ByteTrack wrapper.
*   `prediction/`: Trajectory and Near-miss engines.
*   `fusion/`: ReID engine for multi-camera tracking.
*   `api/`: Flask-SocketIO server and A* Routing.
*   `dashboard/`: Front-end glassmorphism UI.
*   `model_train.py`: Automated training pipeline.

---

## 👨‍💻 Developer
**Tejas Vilas Kondhalkar**

---
*Built for Hackathon Excellence. Production Ready. Protected by Aegis.*
