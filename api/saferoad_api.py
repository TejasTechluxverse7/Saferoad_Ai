"""
AegisRoad AI – Flask-SocketIO Real-Time API
============================================
Provides:
  REST:
    GET  /                          → main dashboard (serves dashboard/index.html)
    GET  /stream/<camera_id>        → MJPEG live stream
    POST /upload                    → upload a local video file, get stream URL
    GET  /stream_upload/<filename>  → MJPEG stream of uploaded+processed video
    GET  /uploads                   → list uploaded video files
    GET  /events                    → paginated JSON accident events
    GET  /replay/<event_id>         → serve forensic replay MP4
    GET  /heatmap                   → aggregated accident locations JSON
    GET  /route/<event_id>          → ambulance route for event
    GET  /health                    → health check

  WebSocket (Socket.IO):
    emit 'accident_event'   → full accident event dict
    emit 'near_miss_event'  → near-miss alert dict
    emit 'stats_update'     → per-second frame stats
"""

from __future__ import annotations

import json
import os
import time
import threading
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional
from werkzeug.utils import secure_filename

from flask import Flask, Response, jsonify, send_file, request
from flask_socketio import SocketIO

from api.routing import AmbulanceRouter


# ─── Setup ────────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).resolve().parent.parent   # project root
DASHBOARD_DIR = BASE_DIR / "dashboard"
ACCIDENT_DIR  = BASE_DIR / "accident_frames"
UPLOAD_DIR    = BASE_DIR / "uploads"

UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTS = {"mp4", "avi", "mov", "mkv", "webm"}

app = Flask(__name__, static_folder=str(DASHBOARD_DIR))
app.config["SECRET_KEY"]      = os.environ.get("FLASK_SECRET_KEY", "saferoad-ai-2024")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024   # 500 MB upload limit

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet",
                    logger=False, engineio_logger=False)

# In-memory event store (bounded)
_event_store: deque[Dict[str, Any]] = deque(maxlen=500)
_event_lock = threading.Lock()

# Active camera streams: camera_id → generator function
_camera_streams: Dict[str, Any] = {}

# Shared YOLO model reference (set by saferoad_main.py via set_pipeline_model)
_yolo_model = None
_model_lock = threading.Lock()

# Router instance (lazy)
_router: Optional[AmbulanceRouter] = None

# Track active upload stream threads (filename → bool)
_upload_streams_active: Dict[str, bool] = {}


def get_router() -> AmbulanceRouter:
    global _router
    if _router is None:
        _router = AmbulanceRouter()
    return _router


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


# ─── Upload Video Processor ───────────────────────────────────────────────────

def _process_upload_stream(video_path: Path):
    """
    Generator that opens an uploaded video, runs YOLO on each frame,
    draws basic overlays, and yields MJPEG bytes.
    Loops until the stream is stopped.
    """
    import cv2
    import numpy as np

    CONF = 0.40

    # Colour map for severity-like annotations
    _COL = {"accident": (40, 40, 255), "vehicle": (80, 200, 255), "person": (0, 200, 80)}

    def _annotate(frame, results):
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < CONF:
                    continue
                cls_id = int(box.cls[0]) if box.cls is not None else 0
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label_map = getattr(r, "names", {})
                label = label_map.get(cls_id, f"cls{cls_id}") if label_map else f"cls{cls_id}"
                color = _COL.get(label.lower(), (100, 200, 255))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # Corner accents
                for cx1, cy1, cx2, cy2 in [(x1,y1,x1+12,y1),(x1,y1,x1,y1+12),
                                            (x2,y2,x2-12,y2),(x2,y2,x2,y2-12)]:
                    cv2.line(frame, (cx1,cy1),(cx2,cy2),(255,255,255),2)

                tag = f"{label}  {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, 0.48, 1)
                # Label bg
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, max(0,y1-th-10)),
                              (x1+tw+10, y1), (10,10,30), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                cv2.putText(frame, tag, (x1+4, max(14,y1-5)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.48, color, 1, cv2.LINE_AA)
        return frame

    def _draw_hud(frame, fn: int, fps_val: float):
        h, w = frame.shape[:2]
        # Top strip
        overlay = frame.copy()
        cv2.rectangle(overlay,(0,0),(w,44),(12,14,24),-1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.putText(frame, "AEGISROAD AI  |  UPLOADED VIDEO ANALYSIS",
                    (14,28), cv2.FONT_HERSHEY_DUPLEX, 0.60,(255,165,40),1,cv2.LINE_AA)
        cv2.putText(frame, f"FPS:{fps_val:.1f}  FRAME:{fn}",
                    (w-200,28), cv2.FONT_HERSHEY_DUPLEX, 0.46,(200,200,200),1,cv2.LINE_AA)
        # Bottom hint
        cv2.putText(frame, video_path.name,
                    (14, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.42,(120,120,160),1,cv2.LINE_AA)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return

    with _model_lock:
        model = _yolo_model

    frame_n = 0
    t_last   = time.time()
    fps_val  = 0.0
    key = video_path.name

    _upload_streams_active[key] = True

    try:
        while _upload_streams_active.get(key, False):
            ret, frame = cap.read()
            if not ret:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_n = 0
                continue

            frame_n += 1

            # Run YOLO if model is available
            if model is not None:
                try:
                    results = model(frame, verbose=False, conf=CONF, imgsz=640)
                    frame = _annotate(frame, results)
                except Exception:
                    pass

            # Compute FPS
            now = time.time()
            fps_val = 0.9 * fps_val + 0.1 * (1.0 / max(now - t_last, 0.001))
            t_last = now

            _draw_hud(frame, frame_n, fps_val)

            # Resize to 1280×720 for streaming
            disp = cv2.resize(frame, (1280, 720))
            ok, buf = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 78])
            if not ok:
                continue

            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")
    finally:
        cap.release()
        _upload_streams_active.pop(key, None)


# ─── REST Endpoints ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    idx = DASHBOARD_DIR / "index.html"
    if idx.exists():
        return send_file(str(idx))
    return "<h1>AegisRoad AI API Online</h1><p>Dashboard not found.</p>", 200


@app.route("/health")
def health():
    return jsonify({
        "status":  "ok",
        "events":  len(_event_store),
        "uploads": [f.name for f in UPLOAD_DIR.iterdir() if _allowed_file(f.name)],
        "timestamp": time.time(),
    })


@app.route("/events")
def get_events():
    page     = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 20))
    with _event_lock:
        events = list(_event_store)
    events.sort(key=lambda e: e.get("timestamp", 0), reverse=True)
    start = (page - 1) * per_page
    return jsonify({
        "total":  len(events),
        "page":   page,
        "events": events[start: start + per_page],
    })


@app.route("/heatmap")
def heatmap():
    with _event_lock:
        points = [
            [e["location"]["lat"], e["location"]["lon"], e.get("severity_score", 0.5)]
            for e in _event_store if "location" in e
        ]
    return jsonify({"points": points})


@app.route("/replay/<event_id>")
def serve_replay(event_id: str):
    safe_id = event_id.replace("..", "").replace("/", "")
    replay_path = ACCIDENT_DIR / f"replay_{safe_id}.mp4"
    if replay_path.exists():
        return send_file(str(replay_path), mimetype="video/mp4",
                         as_attachment=False, conditional=True)
    return jsonify({"error": "Replay not ready yet"}), 202


@app.route("/evidence/<event_id>")
def serve_evidence_meta(event_id: str):
    safe_id = event_id.replace("..", "").replace("/", "")
    meta_path = ACCIDENT_DIR / f"evidence_{safe_id}" / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Evidence package not found"}), 404


@app.route("/route/<event_id>")
def get_route(event_id: str):
    with _event_lock:
        ev = next((e for e in _event_store if e.get("event_id") == event_id), None)
    if ev is None:
        return jsonify({"error": "Event not found"}), 404
    loc = ev.get("location", {})
    result = get_router().route(loc.get("lat", 28.5439), loc.get("lon", 77.3305))
    return jsonify(result)


@app.route("/stream/<camera_id>")
def stream_video(camera_id: str):
    gen = _camera_streams.get(camera_id)
    if gen is None:
        return jsonify({"error": f"No stream for camera '{camera_id}'"}), 404
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ─── Upload Endpoints ─────────────────────────────────────────────────────────

@app.route("/upload", methods=["POST"])
def upload_video():
    """
    Accept a multipart/form-data file upload.
    Returns JSON with the stream URL for the uploaded video.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file field in request"}), 400

    f = request.files["file"]
    if not f or not f.filename:
        return jsonify({"error": "Empty filename"}), 400
    if not _allowed_file(f.filename):
        return jsonify({"error": f"Unsupported format. Allowed: {ALLOWED_EXTS}"}), 415

    filename = secure_filename(f.filename)
    save_path = UPLOAD_DIR / filename
    f.save(str(save_path))

    stream_url = f"/stream_upload/{filename}"
    return jsonify({
        "status":     "ok",
        "filename":   filename,
        "size_mb":    round(save_path.stat().st_size / 1_048_576, 2),
        "stream_url": stream_url,
        "message":    f"Upload complete. Stream at {stream_url}",
    })


@app.route("/uploads")
def list_uploads():
    """Return list of previously uploaded video files."""
    files = [
        {"filename": f.name,
         "size_mb":  round(f.stat().st_size / 1_048_576, 2),
         "stream_url": f"/stream_upload/{f.name}"}
        for f in sorted(UPLOAD_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
        if _allowed_file(f.name)
    ]
    return jsonify({"uploads": files})


@app.route("/stream_upload/<path:filename>")
def stream_upload(filename: str):
    """
    MJPEG stream of an uploaded video processed through YOLO.
    Loops the video until the client disconnects.
    """
    safe_name = secure_filename(filename)
    video_path = UPLOAD_DIR / safe_name

    if not video_path.exists():
        return jsonify({"error": "File not found in uploads"}), 404

    # Stop any existing stream for this file before starting a new one
    _upload_streams_active[safe_name] = False
    time.sleep(0.15)

    return Response(
        _process_upload_stream(video_path),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/stop_upload/<path:filename>", methods=["POST"])
def stop_upload_stream(filename: str):
    """Stop an active upload stream."""
    safe_name = secure_filename(filename)
    _upload_streams_active[safe_name] = False
    return jsonify({"status": "stopped", "filename": safe_name})


# ─── Socket.IO Helpers (called from saferoad_main.py) ────────────────────────

def emit_accident_event(event: Dict[str, Any]) -> None:
    with _event_lock:
        _event_store.appendleft(event)
    socketio.emit("accident_event", event)


def emit_near_miss_event(alert: Dict[str, Any]) -> None:
    socketio.emit("near_miss_event", alert)


def emit_stats_update(stats: Dict[str, Any]) -> None:
    socketio.emit("stats_update", stats)


def register_camera_stream(camera_id: str, generator_fn) -> None:
    _camera_streams[camera_id] = generator_fn


def set_pipeline_model(model) -> None:
    """Share the loaded YOLO model with this module for upload stream processing."""
    global _yolo_model
    with _model_lock:
        _yolo_model = model


def start_api(host: str = "0.0.0.0", port: int = 5050, debug: bool = False) -> None:
    print(f"🌐 AegisRoad AI API starting on http://{host}:{port}")
    socketio.run(app, host=host, port=port, debug=debug, use_reloader=False)
