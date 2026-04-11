# 🛡️ AegisRoad AI - Intelligent Predictive Road Safety Ecosystem

AegisRoad AI is an advanced, production-grade modular system designed for real-time road safety monitoring, accident detection, and proactive hazard prevention. It integrates state-of-the-art computer vision with predictive analytics to provide a "shield of protection" over urban infrastructure.

---

## 🌟 Comprehensive Feature Suite

### 👁️ Computer Vision & Detection
*   **Aegis Detection Engine**: Utilizes a custom-trained YOLOv8 model (`aegis_model.pt`) optimized for vehicle tracking and accident pattern recognition.
*   **Temporal Verification**: A multi-frame consistency layer that filters out flickering detections, ensuring only persistent and verified accident events trigger alerts.
*   **Sophisticated Tracking**: Integration with **ByteTrack** ensures vehicle identities are maintained across occlusions and high-density traffic.
*   **Victim Detection**: Specialized heuristic engine that analyzes person bounding boxes; specifically flags "fallen" orientations (`W > H`) to identify potential pedestrian or rider injuries.

### 🔮 Predictive Analytics
*   **Near-Miss Prediction**: Real-time trajectory projection that calculates the "Time to Collision" (TTC) between vehicles. Pulsing visual indicators appear on the HUD when a collision is imminent.
*   **Trajectory Visualization**: Dynamic, fading polyline "tails" for every vehicle, showing historical motion paths and directional intent.
*   **Velocity Vector Mapping**: Real-time speed and heading arrows drawn from vehicle centroids, providing instant visual data on traffic flow dynamics.

### 🛡️ Incident Response & Forensics
*   **Dynamic Severity Classification**: A multi-factor scoring engine (LOW, MEDIUM, HIGH, CRITICAL) that evaluates speed, impact overlap, and number of vehicles involved.
*   **Automated Forensic Replay**: Upon a high-severity detection, the system automatically exports a 4x slow-motion MP4 reconstruction of the accident for legal and medical review.
*   **Evidence Package Creation**: Every incident triggers the capture of a "Digital Evidence Package" containing pre/post accident frames and JSON metadata.
*   **Explainable AI (XAI)**: Generates **Grad-CAM heatmaps** for high-severity events, visually highlighting *exactly* why the AI flagged a specific region as an accident.

### 🌐 Smart City Integration
*   **Interactive Glassmorphism Dashboard**: A premium, high-performance web UI featuring live MJPEG analysis streams, real-time event logs, and system health metrics.
*   **Video Upload Analysis**: Allows users to upload any road footage for immediate "Post-Event" AI analysis with the same real-time overlays.
*   **Smart Routing (A*)**: Automated ambulance dispatching that uses the A* search algorithm to find the fastest path to the nearest hospital via a custom city road graph.
*   **VANET / V2X Alerting**: Broadcasts standardized JSON messages over MQTT/HTTP to simulate Vehicle-to-Infrastructure (V2I) communication for upcoming autonomous vehicles.
*   **Multi-Camera ReID**: Uses a ResNet-18 backbone to re-identify vehicles across different camera feeds, maintaining global identity for forensic investigations.

### ⚡ Performance Optimization
*   **Lazy XAI Rendering**: Grad-CAM is only computed for high-severity events to preserve CPU/GPU resources.
*   **Intelligent Frame Skipping**: Processes every 2nd or 3rd frame to ensure 30+ FPS performance on edge devices without losing tracking continuity.
*   **GPU Acceleration**: Native support for CUDA and MPS (Apple Silicon) backends.

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

### 3. Dashboard & Upload Testing
Launch the full API stack and interact via the web browser:
1. Start: `python saferoad_main.py --source sample_videos/dashcam.mp4 --port 5050`
2. Open **http://localhost:5050**
3. **Upload Video**: Drag/drop a file on the "Upload Video" tab to trigger the remote analysis pipeline.

---

## 🧠 Training Procedure

To retrain or fine-tune the model using the Roboflow Accident Dataset:
```bash
python model_train.py --epochs 25 --batch 16 --device 0
```
Outputs trained weights to `models/aegis_model.pt`.

---

## 📁 Project Structure

*   `saferoad_main.py`: Main system orchestrator.
*   `engine/`: Core logic (Severity, Forensic, GradCAM, Victim Detector, Evidence).
*   `tracking/`: ByteTrack wrapper.
*   `prediction/`: Trajectory and Near-miss engines.
*   `api/`: Flask-SocketIO server and A* Routing.
*   `dashboard/`: Front-end glassmorphism UI.

---

## 👨‍💻 Developer
**Tejas Vilas Kondhalkar**

---
*Built for Hackathon Excellence. Production Ready. Protected by Aegis.*
