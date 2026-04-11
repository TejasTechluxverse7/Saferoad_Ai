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

### 🚑 Alerting & Smart City Logistics
*   **Nearest Hospital Dispatch**: Integrated **OLA Maps API** logic that automatically identifies the closest medical facility based on the incident's GPS coordinates.
*   **VANET / V2X Layer**: Implements a dedicated **Vehicular Ad-Hoc Network** protocol that broadcasts standardized JSON alerts over MQTT. This allows Road-Side Units (RSUs) and nearby smart vehicles to receive instant traffic warnings.
*   **Smart Routing (A*)**: Beyond simple distance, the system uses the **A* search algorithm** over a local city road graph (`city_graph.json`) to compute the optimized path for ambulances.
*   **Dynamic Geolocation Engine**: Parses location strings into real-world coordinates, powering the real-time **Leaflet.js Heatmap** on the dashboard.
*   **Image-to-URL Evidence Pipe**: Automatically converts captured accident frames into cloud-accessible URLs, enabling instant visual context for emergency dispatchers via SMS or Chat backends.

### 🌐 Smart City Dashboard
*   **Interactive Glassmorphism UI**: A premium, high-performance web interface for centralized monitoring, featuring MJPEG streams and live incident counts.
*   **Video Upload Analysis**: Allows users to upload and process road footage from their local PC, viewing the same high-fidelity AI overlays in a "Post-Event" forensic mode.
*   **Multi-Camera ReID**: Uses a ResNet-18 backbone to re-identify and track vehicles across different camera viewpoints, maintaining a global "Chain of Custody" for forensic data.

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

### 2. Automated Feature Check
Verify every internal engine (Detection, Tracking, Prediction, Severity, Routing, Hospital Lookup) with the dedicated validation script:
```bash
python verify_features.py
```

### 3. Dashboard, Upload & VANET Testing
1. Start: `python saferoad_main.py --source sample_videos/dashcam.mp4 --port 5050`
2. Open **http://localhost:5050** to view live analysis.
3. Use the **Upload Video** tab to process local footage.
4. Subscribe to the `saferoad/accidents` topic on your MQTT broker to see the VANET alert steam.

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
*   `api/`: Flask-SocketIO server, OLA Maps Hospital Lookup, and A* Routing.
*   `vanet_layer.py`: V2X alert implementation.

---

## 👨‍💻 Developer
**Tejas Vilas Kondhalkar**

---
*Built for Hackathon Excellence. Production Ready. Protected by Aegis.*
