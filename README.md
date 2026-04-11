# SaferoadAI

SaferoadAI is a real-time, edge-based accident detection and smart alerting system designed for smart-city CCTV infrastructure. The system ingests CCTV or live video feeds, runs an optimized YOLOv8 model at the edge, performs temporal accident verification, and broadcasts VANET/V2X-style alerts alongside notifications to the nearest hospitals using the **OLA Maps API**.

## Features

- **Real-time accident detection** using YOLOv8 (Edge AI Detection Module).
- **Temporal accident verification** using a lightweight rule-based temporal module (`temporal_module.py`) approximating tracking + motion reasoning.
- **Integration with CCTV or live video feeds** via `aegis_engine.py` (CLI) and `flask_app.py` (web demo).
- **Automatic alert system** using OLA Maps API to notify the nearest hospital.
- **VANET/V2X-style JSON alerts** via `vanet_layer.py`, with optional MQTT broadcast and HTTP forwarding for RSU/vehicle simulators.
- **Frame capture and image conversion** to URL for message attachments.

## Installation

### Prerequisites

Ensure you have the following installed before proceeding:

- Python 3.8+
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- Streamlit (for deployment, if needed)
- OLA Maps API access

### Clone the Repository

```bash
git clone https://github.com/aayush010904/SaferoadAI.git
cd SaferoadAI
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Dataset

Dataset used for pre-training: [Roboflow dataset URL](https://universe.roboflow.com/accident-detection-model/accident-detection-model/dataset/2). Model is included as `CrashSentinel_Prime.pt`.

## Usage

### Phase 1: Core edge accident pipeline (CLI)

To start the accident detection system on a local video, run:

```bash
python aegis_engine.py --source sample_videos/acci.mp4
```

### How It Works (aligned with the PDF)

- `aegis_engine.py` imports functions from `temporal_module.py` to continuously process frames through the `CrashSentinel_Prime.pt` network.
- For each frame, YOLOv8 produces accident-class detections which are converted into `Detection` objects.
- The temporal verifier aggregates motion and overlap across a short window and triggers an accident event only when the pattern is consistent with a collision and post-impact stillness.
- When an accident is confirmed:
  - A frame is saved to `accident_frames/`.
  - The frame path is converted into a URL by `Image2Url.py` and sent to the nearest hospitals through the demo chat backend.
  - A VANET/V2X-style JSON alert is generated and optionally published over MQTT and/or HTTP.

## Flask Live Stream Demo (uses temporal module + VANET alerts)

The Flask app reuses the same YOLO model and the temporal verifier defined in `temporal_module.py` for live MJPEG streaming. Accident events increment counters and publish VANET-style alerts during streaming.

Run locally:

```bash
pip install -r requirements.txt
python flask_app.py
```

Then open http://localhost:5000, upload or pick a sample video, click **Process Detection** to start the live processed stream. Accident counts appear after the stream completes. The “See live alerts” button is a placeholder—point it to your real alerts/messages URL.

## Deployment

### Quick local server (Waitress)

```bash
pip install waitress
waitress-serve --port=5000 flask_app:app
```

### Container (recommended for hosting)

Example Dockerfile:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=5000
CMD ["gunicorn", "-b", "0.0.0.0:5000", "flask_app:app"]
```

Build and run:

```bash
docker build -t saferoad-ai .
docker run -p 5000:5000 saferoad-ai
```

Host this container on services like Render, Railway, Fly.io, AWS ECS/Fargate, or Azure Web App for Containers. Ensure ffmpeg/H.264 support is available in your base image if you need MP4 output.

### VANET / Vehicle subscriber demo

To simulate a vehicle/RSU listening to accident alerts over MQTT:

```bash
export MQTT_BROKER=localhost
python vanet_subscriber.py
```

This will print each VANET-style JSON alert emitted by `app.py` / `flask_app.py` to the console, approximating the “Vehicle/RSU Response Simulation” part of the project document.

## Project Structure

```
SaferoadAI/
├── aegis_engine.py       # Main application script
├── aegis_test_runner.py  # Automation and video testing script
├── requirements_aegis.txt# Python dependencies
├── README.md             # Project documentation
├── CrashSentinel_Prime.pt# YOLO model optimized for accident detection
├── Image2Url.py          # Converts detected accident frames to image URLs
├── NearestHospital.py    # Fetches nearest hospital using OLA Maps API
├── SendMessage.py        # Sends alert messages with accident details
├── currentLocation.py    # Determines the user's current location
└── temporal_module.py    # Logic checking spatial detection consistency
```

## Future Enhancements

- Improving model accuracy with more training data.
- Expanding API support for other mapping services.
- Implementing real-time traffic management integration.

## Contributions

Feel free to open an issue or submit a pull request if you’d like to contribute!

## License

This project is licensed under the MIT License.

### This Project is developed by Manish Sharma, Aayush Chauhan, Ekansh Dubey, Akhil and Pragyansh Verma for the Hack-4-Viksit Bharat Hackathon
