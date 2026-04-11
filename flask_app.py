import os
import tempfile
from pathlib import Path
from typing import Tuple

import cv2
from flask import (
    Flask,
    Response,
    flash,
    redirect,
    render_template_string,
    request,
    send_file,
    url_for,
)

from app import (
    CONFIDENCE_THRESHOLD,
    NO_ACCIDENT_THRESHOLD,
    detect_accident,
    model as core_model,
    save_accident_frame,
)
from temporal_module import Detection, TemporalAccidentVerifier
from vanet_layer import build_accident_alert, publish_mqtt_alert

model = core_model
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed_videos"
SAMPLE_DIR = BASE_DIR / "sample_videos"
ACCIDENT_DIR = BASE_DIR / "accident_frames"
SAMPLE_VIDEO_PATH = SAMPLE_DIR / "sample_video.mp4"  # default sample video placeholder
SELECTED_SAMPLE_PATH: Path | None = None  # last chosen sample video path

STREAM_SUMMARIES = {}

for d in (UPLOAD_DIR, PROCESSED_DIR, ACCIDENT_DIR):
    d.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "saferoad-secret")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_frames(input_path: Path, summary_key: Tuple[str, str]):
    """Stream processed frames as MJPEG using the temporal module."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = 0
    accident_count = 0
    verifier = TemporalAccidentVerifier()
    no_accident_frames = 0
    accident_active = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model(frame)
        processed = frame.copy()
        temporal_detections = []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0]) if box.cls is not None else 0
                if conf < CONFIDENCE_THRESHOLD or cls_id != 0:
                    continue
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                temporal_detections.append(
                    Detection(bbox=(x1, y1, x2, y2), confidence=conf)
                )
                x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))
                cv2.rectangle(processed, (x1i, y1i), (x2i, y2i), (0, 0, 255), 2)
                cv2.putText(
                    processed,
                    f"Acc: {conf:.2f}",
                    (x1i, max(20, y1i - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

        is_accident, temporal_conf = verifier.update(temporal_detections)

        if is_accident and not accident_active:
            accident_active = True
            accident_count += 1
            frame_filename = save_accident_frame(
                processed, output_dir=str(ACCIDENT_DIR)
            )
            alert = build_accident_alert(
                os.environ.get("LAST_KNOWN_LOCATION", "28.5439375,77.3304876"),
                confidence=temporal_conf,
            )
            publish_mqtt_alert(alert)
            print(
                f"✅ Stream accident trigger (conf={temporal_conf:.2f}) "
                f"frame={frame_count} saved={frame_filename}"
            )

        if temporal_detections:
            no_accident_frames = 0
        else:
            no_accident_frames += 1

        if no_accident_frames >= NO_ACCIDENT_THRESHOLD:
            accident_active = False
            verifier.reset()

        ok, buffer = cv2.imencode(".jpg", processed)
        if not ok:
            continue

        frame_bytes = buffer.tobytes()
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"

    cap.release()
    STREAM_SUMMARIES[summary_key] = {
        "frames": frame_count,
        "accidents": accident_count,
        "fps": fps,
    }


def process_video(input_path: Path) -> Tuple[Path, int, int, float]:
    """
    Run the temporal-aware pipeline and persist a processed MP4.
    Returns (output_path, total_frames, accident_detections, fps).
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    fd, temp_path = tempfile.mkstemp(suffix=".mp4", prefix="processed_")
    os.close(fd)
    output_path = PROCESSED_DIR / Path(temp_path).name
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Video writer could not be opened; missing H.264/avc1 codec")

    accident_count = 0
    frame_idx = 0
    verifier = TemporalAccidentVerifier()
    accident_active = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model(frame)
        processed = frame.copy()
        temporal_detections = []

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                temporal_detections.append(
                    Detection(bbox=(x1, y1, x2, y2), confidence=conf)
                )
                x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))
                cv2.rectangle(processed, (x1i, y1i), (x2i, y2i), (0, 0, 255), 2)
                cv2.putText(
                    processed,
                    f"Acc: {conf:.2f}",
                    (x1i, max(20, y1i - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

        is_accident, temporal_conf = verifier.update(temporal_detections)
        if is_accident and not accident_active:
            accident_active = True
            accident_count += 1
        elif not is_accident:
            accident_active = False

        writer.write(processed)

    cap.release()
    writer.release()

    return output_path, frame_idx, accident_count, fps


# ---------------------------------------------------------------------
# UPDATED TEMPLATE WITH MODERN CSS
# ---------------------------------------------------------------------
TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SafeRoad AI | Accident Detection</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --primary: #4F46E5;
      --primary-hover: #4338ca;
      --bg: #F3F4F6;
      --card-bg: #ffffff;
      --text-main: #111827;
      --text-muted: #6B7280;
      --border: #E5E7EB;
      --shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
      --danger: #EF4444;
      --success: #10B981;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    
    body {
      font-family: 'Inter', sans-serif;
      background: radial-gradient(circle at 20% 20%, #eef2ff 0, #f8fafc 40%, #eef2ff 100%);
      color: var(--text-main);
      display: flex;
      flex-direction: column;
      align-items: center;
      min-height: 100vh;
      padding: 48px 20px;
    }

    .container {
      width: 100%;
      max-width: 980px;
      background: rgba(255,255,255,0.92);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(255,255,255,0.6);
      border-radius: 18px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.12);
      overflow: hidden;
    }

    /* Header */
    .header {
      background: linear-gradient(135deg, #4F46E5 0%, #6366F1 40%, #22d3ee 100%);
      padding: 34px;
      text-align: center;
      color: white;
      position: relative;
    }
    .header h1 { font-weight: 800; font-size: 2rem; margin-bottom: 10px; letter-spacing: -0.5px; }
    .header p { opacity: 0.95; font-weight: 400; font-size: 1rem; }

    /* Content Area */
    .content { padding: 30px; }

    /* Alerts */
    .alert {
      background: #FEF2F2;
      border: 1px solid #FCA5A5;
      color: #991B1B;
      padding: 12px;
      border-radius: 8px;
      margin-bottom: 20px;
      font-size: 0.9rem;
    }

    /* Form Grid */
    form { margin-bottom: 0; }
    .form-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-bottom: 25px;
    }
    
    @media (max-width: 600px) {
      .form-grid { grid-template-columns: 1fr; }
    }

    .form-group label {
      display: block;
      font-weight: 600;
      margin-bottom: 8px;
      color: var(--text-main);
      font-size: 0.9rem;
    }

    /* Custom Input Styling */
    input[type="file"], select {
      width: 100%;
      padding: 10px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #F9FAFB;
      font-size: 0.9rem;
      outline: none;
      transition: border-color 0.2s;
    }
    input[type="file"]:hover, select:hover { border-color: var(--primary); }
    
    /* File input button tweak */
    input[type="file"]::file-selector-button {
      background: var(--text-main);
      color: white;
      border: none;
      padding: 6px 12px;
      border-radius: 4px;
      margin-right: 10px;
      cursor: pointer;
      font-size: 0.8rem;
    }

    /* Submit Button */
    .btn-submit {
      padding: 14px 18px;
      background: linear-gradient(135deg, #4F46E5, #7C3AED);
      color: white;
      border: none;
      border-radius: 10px;
      font-size: 1rem;
      font-weight: 700;
      cursor: pointer;
      transition: transform 0.15s ease, box-shadow 0.2s ease;
      box-shadow: 0 10px 24px rgba(79,70,229,0.25);
    }
    .btn-submit:hover { transform: translateY(-1px); box-shadow: 0 14px 30px rgba(79,70,229,0.32); }

    /* Stats Dashboard */
    .results-area {
      margin-top: 40px;
      border-top: 1px solid var(--border);
      padding-top: 30px;
      animation: fadeIn 0.5s ease-in;
      background: linear-gradient(145deg, #ffffff 0%, #f5f7ff 100%);
      border-radius: 14px;
      padding: 28px;
      box-shadow: 0 12px 28px rgba(0,0,0,0.08);
    }

    .stats-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 15px;
      margin-bottom: 25px;
    }
    .stat-card {
      background: #F3F4F6;
      padding: 15px;
      border-radius: 10px;
      text-align: center;
    }
    .stat-value { font-size: 1.5rem; font-weight: 700; color: var(--primary); display: block; }
    .stat-label { font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }

    /* Live stream section */
    .live-video { width: 100%; background: #000; border-radius: 12px; overflow: hidden; margin-bottom: 16px; }
    .live-video img { width: 100%; display: block; }
    .live-header { font-weight: 700; font-size: 1rem; margin: 18px 0 8px 0; }
    .hint { color: var(--text-muted); font-size: 0.9rem; margin-bottom: 16px; }

    .instructions { margin: 16px 0; padding: 12px; background: #eef2ff; border-radius: 10px; color: #111827; }
    .actions-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; justify-content: center; align-items: center; }
    .btn-secondary { background: linear-gradient(135deg, #111827, #1f2937); color: white; border: none; padding: 12px 16px; border-radius: 10px; text-decoration: none; font-weight: 700; display: inline-block; box-shadow: 0 10px 20px rgba(0,0,0,0.12); }
    .btn-secondary:hover { transform: translateY(-1px); box-shadow: 0 14px 26px rgba(0,0,0,0.16); }

    /* Video Player */
    .video-container {
      background: #000;
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 4px 6px rgba(0,0,0,0.1);
      margin-bottom: 20px;
    }
    video { width: 100%; display: block; max-height: 500px; }

    /* Download Button */
    .btn-download {
      display: inline-block;
      text-decoration: none;
      background: var(--text-main);
      color: white;
      padding: 12px 24px;
      border-radius: 8px;
      font-size: 0.9rem;
      font-weight: 500;
      transition: opacity 0.2s;
    }
    .btn-download:hover { opacity: 0.9; }

    .center-text { text-align: center; }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }
  </style>
</head>
<body>

  <div class="container">
    <div class="header">
      <h1>🚗 SafeRoad AI</h1>
      <p>Autonomous Accident Detection System</p>
    </div>

    <div class="content">
      
      {% with messages = get_flashed_messages() %}
        {% if messages %}
          {% for m in messages %}
            <div class="alert">{{ m }}</div>
          {% endfor %}
        {% endif %}
      {% endwith %}

      <div class="instructions">
        <strong>Demo flow:</strong> Upload or pick a sample, hit "Process Detection" to start live processed streaming. Accident counts will appear after the stream finishes. Use the alerts button to view chat/messages (external system).<br/>
        <strong>Chat login:</strong> email: cctv@gmail.com · password: 123456 (for demo alerts app). Click on hospital chat to see the message alerts.
      </div>

      <div class="actions-row">
        <button class="btn-submit" type="submit" form="process-form">🚀 Process Detection</button>
        <a class="btn-secondary" href="https://tinyurl.com/pepsuchat" target="_blank" rel="noopener">🔔 See live alerts</a>
      </div>

      <form id="process-form" action="{{ url_for('process') }}" method="post" enctype="multipart/form-data">
        <div class="form-grid">
          <div class="form-group">
            <label for="video">📂 Upload New Video</label>
            <input type="file" name="video" accept="video/*" />
          </div>
          <div class="form-group">
            <label for="sample">🎞️ Or Select Sample</label>
            <select name="sample" id="sample">
              <option value="">-- Choose a sample video --</option>
              {% for s in samples %}
                <option value="{{ s }}">{{ s }}</option>
              {% endfor %}
            </select>
          </div>
        </div>
      </form>

      {% if stream_url or processed_url %}
        <div class="results-area">
          <h3 style="margin-bottom: 15px; color: var(--text-main);">Live Analysis</h3>

          {% if total_frames %}
          <div class="stats-grid">
            <div class="stat-card">
              <span class="stat-value">{{ total_frames }}</span>
              <span class="stat-label">Total Frames</span>
            </div>
            <div class="stat-card">
              {% if accident_count is none %}
                <span class="stat-value">--</span>
                <span class="stat-label">Detections (shown after stream ends)</span>
              {% else %}
                <span class="stat-value" style="color: {% if accident_count > 0 %}var(--danger){% else %}var(--success){% endif %};">
                  {{ accident_count }}
                </span>
                <span class="stat-label">Detections (final)</span>
              {% endif %}
            </div>
            <div class="stat-card">
              <span class="stat-value">{{ fps|round(1) }}</span>
              <span class="stat-label">FPS (source)</span>
            </div>
          </div>
          {% endif %}

          {% if stream_url %}
            <div class="live-header">Live processed stream (no file saving)</div>
            <div class="live-video">
              <img src="{{ stream_url }}" alt="Live stream" />
            </div>
            <div class="hint">Frames are processed and streamed directly while the video is read.</div>
          {% endif %}

          {% if processed_url %}
            <div class="hint">A processed MP4 is available for download if needed.</div>
            <div class="center-text">
              <a class="btn-download" href="{{ processed_url }}" download>⬇ Download Processed Video</a>
            </div>
          {% endif %}
        </div>
      {% endif %}

    </div>
  </div>

</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    samples = [
        f.name
        for f in SAMPLE_DIR.glob("*")
        if f.suffix.lower()[1:] in ALLOWED_EXTENSIONS
    ]
    return render_template_string(
        TEMPLATE,
        samples=samples,
        processed_url=None,
        stream_url=None,
        total_frames=None,
        accident_count=None,
        fps=None,
    )


@app.route("/process", methods=["POST"])
def process():
    global SELECTED_SAMPLE_PATH
    file = request.files.get("video")
    sample_name = request.form.get("sample", "").strip()

    stream_kind = None
    stream_name = None

    if file and file.filename and allowed_file(file.filename):
        filename = file.filename.rsplit("/", 1)[-1]
        save_path = UPLOAD_DIR / filename
        file.save(save_path)
        input_path = save_path
        stream_kind, stream_name = "upload", filename
        SELECTED_SAMPLE_PATH = None
    elif sample_name:
        candidate = SAMPLE_DIR / sample_name
        if not candidate.exists():
            flash("Sample video not found.")
            return redirect(url_for("index"))
        input_path = candidate
        stream_kind, stream_name = "sample", sample_name
        SELECTED_SAMPLE_PATH = candidate
    elif SAMPLE_VIDEO_PATH.exists():
      # Fallback to default sample video if available
        input_path = SAMPLE_VIDEO_PATH
        stream_kind, stream_name = "sample", SAMPLE_VIDEO_PATH.name
        SELECTED_SAMPLE_PATH = SAMPLE_VIDEO_PATH
    else:
        flash("Please upload a video or choose a sample.")
        return redirect(url_for("index"))

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        flash("Could not open video for streaming.")
        return redirect(url_for("index"))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    summary_key = (stream_kind, stream_name)
    STREAM_SUMMARIES.pop(summary_key, None)  # reset old summary if any
    summary = STREAM_SUMMARIES.get(summary_key)
    accident_count = summary["accidents"] if summary else None

    samples = [
        f.name
        for f in SAMPLE_DIR.glob("*")
        if f.suffix.lower()[1:] in ALLOWED_EXTENSIONS
    ]
    return render_template_string(
        TEMPLATE,
        samples=samples,
        processed_url=None,
        stream_url=url_for("stream_video", kind=stream_kind, filename=stream_name),
        total_frames=total_frames,
        accident_count=accident_count,
        fps=fps,
    )


@app.route("/video/<path:filename>")
def serve_video(filename: str):
    path = PROCESSED_DIR / filename
    if not path.exists():
        flash("Video not found.")
        return redirect(url_for("index"))
    return send_file(path, mimetype="video/mp4", as_attachment=False)


@app.route("/stream/<kind>/<path:filename>")
def stream_video(kind: str, filename: str):
    if kind == "upload":
        source_path = UPLOAD_DIR / filename
    elif kind == "sample":
        source_path = SAMPLE_DIR / filename
    else:
        flash("Invalid stream source.")
        return redirect(url_for("index"))

    if not source_path.exists():
        flash("Stream source not found.")
        return redirect(url_for("index"))

    return Response(
        generate_frames(source_path, (kind, filename)),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
