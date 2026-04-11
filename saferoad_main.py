"""
AegisRoad AI – Main Orchestrator
==================================
Production-ready, hackathon-demo-ready intelligent road safety pipeline.

  python saferoad_main.py --source sample_videos/video1.mp4
  python saferoad_main.py --source 0                         # webcam
  python saferoad_main.py --source rtsp://...                # RTSP stream

Keyboard controls:
  q  – quit
  p  – pause / resume
  r  – replay (loop video back to start)

Pipeline stages:
  Input (video / RTSP / webcam)
    → Frame resize (640×640) + optional frame-skip (every 2nd frame)
    → YOLO Detection        (aegis_model.pt)
    → ByteTrack Tracking    (persistent IDs)
    → Trajectory Prediction (speed, near-miss)
    → Temporal Verifier     (confirm real accident)
    → Severity Classifier   (LOW / MEDIUM / HIGH / CRITICAL)
    → Evidence Buffer       (saves pre/post frames)
    → Forensics Replay      (slow-motion MP4 export)
    → Victim Detector       (fallen persons)
    → Grad-CAM XAI          (HIGH/CRITICAL only)
    → ReID Fusion           (cross-camera global IDs)
    → Rich OpenCV HUD       (hackathon overlay system)
    → Flask-SocketIO API    (dashboard + MJPEG stream)
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# ── AegisRoad AI modules ──────────────────────────────────────────────────────
from tracking.bytetrack_wrapper import ByteTrackWrapper
from prediction.trajectory_engine import TrajectoryEngine
from engine.temporal_module import Detection, TemporalAccidentVerifier
from engine.severity_classifier import SeverityClassifier, SeverityLevel
from engine.evidence_buffer import EvidenceBuffer
from engine.forensics_engine import ForensicsEngine
from engine.gradcam import GradCAMExplainer
from engine.victim_detector import VictimDetector
from fusion.reid_engine import ReIDEngine, GlobalTracker
from vanet.vanet_layer import build_accident_alert, publish_mqtt_alert
from api.saferoad_api import (
    emit_accident_event,
    emit_near_miss_event,
    emit_stats_update,
    register_camera_stream,
    set_pipeline_model,
    start_api,
)


# ─── Constants ────────────────────────────────────────────────────────────────

# Primary model – falls back to CrashSentinel_Prime.pt if not found
_CANDIDATE_MODELS = [
    "models/aegis_model.pt",
    "aegis_model.pt",
    "CrashSentinel_Prime.pt",
]
MODEL_PATH = next((m for m in _CANDIDATE_MODELS if Path(m).exists()), _CANDIDATE_MODELS[-1])

CONFIDENCE_THRESH  = 0.40
NO_ACCIDENT_RESET  = 30        # frames of no detection → reset state
STATS_EMIT_EVERY   = 15        # frames between WebSocket stats pushes
DISPLAY_W          = 1280      # OpenCV window width
DISPLAY_H          = 720       # OpenCV window height
PROCESS_W          = 640       # YOLO inference width
PROCESS_H          = 640       # YOLO inference height
DEMO_TRIGGER_EVERY = 100       # demo-mode: inject accident every N frames

# ─── Colour palette (BGR) ────────────────────────────────────────────────────

COL_ACCENT     = (255, 165,  40)   # Aegis orange-blue
COL_ACCENT2    = (220, 100, 255)   # purple (ReID labels)
COL_TRACK      = (255, 200,  80)   # track boxes
COL_NEAR_MISS  = (0,   230, 255)   # cyan
COL_WHITE      = (255, 255, 255)
COL_BLACK      = (0,     0,   0)
COL_OK         = (80,  220,  80)
COL_WARN       = (0,   165, 255)
COL_DANGER     = (40,   40, 255)
COL_CRITICAL   = (128,   0, 200)

SEV_COLORS = {
    "LOW":      COL_OK,
    "MEDIUM":   COL_WARN,
    "HIGH":     COL_DANGER,
    "CRITICAL": COL_CRITICAL,
}

# ─── HUD Drawing Helpers ──────────────────────────────────────────────────────

def _alpha_rect(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                color: Tuple, alpha: float = 0.55) -> None:
    """Draw a semi-transparent filled rectangle."""
    roi = img[y1:y2, x1:x2]
    solid = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(solid, alpha, roi, 1 - alpha, 0, roi)
    img[y1:y2, x1:x2] = roi


def _text(img: np.ndarray, msg: str, x: int, y: int,
          color=COL_WHITE, scale: float = 0.55, thickness: int = 1,
          font=cv2.FONT_HERSHEY_DUPLEX) -> None:
    cv2.putText(img, msg, (x, y), font, scale, COL_BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, msg, (x, y), font, scale, color,     thickness,     cv2.LINE_AA)


def _badge(img: np.ndarray, label: str, x: int, y: int,
           color: Tuple, w: int = 0) -> None:
    """Draw a pill-shaped badge."""
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.48, 1)
    pad = 8
    bw = max(w, tw + pad * 2)
    bh = th + pad
    _alpha_rect(img, x, y - th - pad // 2, x + bw, y + pad // 2, color, 0.85)
    cv2.rectangle(img, (x, y - th - pad // 2), (x + bw, y + pad // 2),
                  tuple(min(255, c + 80) for c in color), 1)
    cv2.putText(img, label, (x + pad, y),
                cv2.FONT_HERSHEY_DUPLEX, 0.48, COL_WHITE, 1, cv2.LINE_AA)


def draw_status_panel(frame: np.ndarray, stats: dict, accident_active: bool,
                      severity: str, demo_mode: bool) -> None:
    """Top-left info panel."""
    h, w = frame.shape[:2]
    pw, ph = 300, 190
    _alpha_rect(frame, 10, 10, 10 + pw, 10 + ph, (12, 14, 24), 0.75)
    cv2.rectangle(frame, (10, 10), (10 + pw, 10 + ph), COL_ACCENT, 1)

    # Branding
    _text(frame, "AEGISROAD AI", 22, 36, COL_ACCENT, 0.72, 2)
    if demo_mode:
        _badge(frame, "DEMO MODE", 10 + pw - 100, 36, (0, 80, 200), 90)

    y = 60
    items = [
        ("FPS",      f"{stats.get('fps', 0):.1f}"),
        ("FRAME",    f"{stats.get('frame', 0)}"),
        ("TRACKS",   f"{stats.get('tracks', 0)}"),
        ("ACCIDENTS",f"{stats.get('accidents', 0)}"),
        ("NEAR-MISS",f"{stats.get('near_misses', 0)}"),
        ("VICTIMS",  f"{stats.get('victims', 0)}"),
    ]
    for label, val in items:
        _text(frame, label, 22, y, (160, 160, 180), 0.42, 1)
        _text(frame, val,  155, y, COL_WHITE, 0.48, 1)
        y += 20

    # Severity indicator
    sev_col = SEV_COLORS.get(severity, COL_OK)
    _alpha_rect(frame, 22, y + 4, 22 + pw - 24, y + 22, sev_col, 0.4)
    _text(frame, f"SEVERITY: {severity}", 30, y + 18, COL_WHITE, 0.50, 1)


def draw_alert_banner(frame: np.ndarray, severity: str, conf: float,
                      flash_frame: int) -> None:
    """Full-width alert banner at the bottom with flashing border."""
    h, w = frame.shape[:2]
    sev_col = SEV_COLORS.get(severity, COL_DANGER)

    # Animated border (flash every 15 frames)
    border_col = sev_col if (flash_frame // 15) % 2 == 0 else COL_WHITE
    thickness = 5
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_col, thickness)

    # Bottom banner
    bh = 52
    _alpha_rect(frame, 0, h - bh, w, h, sev_col, 0.80)
    icon = {"LOW": "●", "MEDIUM": "▲", "HIGH": "▲▲", "CRITICAL": "⬛ CRITICAL"}.get(severity, "!")
    _text(frame, f"  {icon}  ACCIDENT DETECTED  [{severity}]   Confidence: {conf:.1%}",
          16, h - 16, COL_WHITE, 0.80, 2, cv2.FONT_HERSHEY_DUPLEX)


def draw_tracks(frame: np.ndarray, tracked_vehicles, traj_engine,
                track_histories: dict) -> None:
    """Draw bounding boxes, IDs, speed, and trajectory polylines."""
    for tv in tracked_vehicles:
        x1, y1, x2, y2 = map(int, tv.bbox)

        # Trajectory polyline
        hist = track_histories.get(tv.track_id, [])
        if len(hist) > 1:
            pts = np.array(hist[-30:], dtype=np.int32)
            n = len(pts)
            for i in range(1, n):
                alpha = i / n
                col = tuple(int(c * alpha) for c in COL_TRACK)
                cv2.line(frame, pts[i - 1], pts[i], col, 2, cv2.LINE_AA)

        # Box
        spd = traj_engine.get_speed_kmh(tv.track_id)
        sev_col = COL_DANGER if spd > 80 else COL_WARN if spd > 50 else COL_TRACK
        cv2.rectangle(frame, (x1, y1), (x2, y2), sev_col, 2)

        # Corner accents
        cs = 10  # corner size
        cv2.line(frame, (x1, y1), (x1 + cs, y1), COL_WHITE, 2)
        cv2.line(frame, (x1, y1), (x1, y1 + cs), COL_WHITE, 2)
        cv2.line(frame, (x2, y2), (x2 - cs, y2), COL_WHITE, 2)
        cv2.line(frame, (x2, y2), (x2, y2 - cs), COL_WHITE, 2)

        # Label background
        label = f"#{tv.track_id}  {spd:.0f}km/h"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.48, 1)
        _alpha_rect(frame, x1, max(0, y1 - lh - 8), x1 + lw + 10, y1, (10, 10, 30), 0.7)
        _text(frame, label, x1 + 4, y1 - 5, sev_col, 0.48)

        # Speed bar under box
        bar_len = min(int((spd / 120) * (x2 - x1)), x2 - x1)
        cv2.rectangle(frame, (x1, y2 + 1), (x1 + bar_len, y2 + 5), sev_col, -1)


def draw_near_misses(frame: np.ndarray, alerts) -> None:
    """Animated near-miss circles and labels."""
    for alert in alerts:
        cx, cy = int(alert.collision_x), int(alert.collision_y)
        for r in [20, 36, 52]:
            cv2.circle(frame, (cx, cy), r, COL_NEAR_MISS, 1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 6, COL_NEAR_MISS, -1, cv2.LINE_AA)
        _text(frame, "NEAR-MISS", cx - 42, cy - 60, COL_NEAR_MISS, 0.62, 2)
        _text(frame, f"d={alert.distance_px:.0f}px  {alert.angle_deg:.0f}deg",
              cx - 50, cy - 42, (200, 240, 255), 0.40)


def draw_speed_arrows(frame: np.ndarray, tracked_vehicles, traj_engine) -> None:
    """Draw velocity arrows on each vehicle."""
    for tv in tracked_vehicles:
        state = traj_engine._tracks.get(tv.track_id)
        if not state:
            continue
        vx, vy = state.vx * 6, state.vy * 6
        if math.hypot(vx, vy) < 2:
            continue
        cx, cy = int(tv.cx), int(tv.cy)
        ex, ey = int(cx + vx), int(cy + vy)
        cv2.arrowedLine(frame, (cx, cy), (ex, ey), COL_ACCENT, 2,
                        cv2.LINE_AA, tipLength=0.35)


def draw_victims(frame: np.ndarray, victims) -> None:
    """Highlight fallen persons detected near the accident."""
    for v in victims:
        x1, y1, x2, y2 = map(int, v.bbox)
        col = (0, 80, 255) if v.fallen else (0, 200, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        label = "FALLEN PERSON" if v.fallen else "PERSON"
        _text(frame, label, x1, max(15, y1 - 8), col, 0.50, 2)


def draw_demo_watermark(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    _text(frame, "[ DEMO MODE - SIMULATED EVENT ]",
          w // 2 - 180, h - 70, (80, 200, 255), 0.62, 1)


def draw_reid_labels(frame: np.ndarray, tracked_vehicles, global_ids: dict) -> None:
    for tv in tracked_vehicles:
        gid = global_ids.get(tv.track_id, "?")
        x1, y2 = int(tv.bbox[0]), int(tv.bbox[3])
        _text(frame, f"G:{gid}", x1 + 2, y2 + 14, COL_ACCENT2, 0.36, 1)


def draw_compass_hud(frame: np.ndarray, frame_count: int) -> None:
    """Miniature rotating compass in bottom-right corner (purely cosmetic)."""
    h, w = frame.shape[:2]
    cx, cy, r = w - 50, h - 50, 28
    _alpha_rect(frame, cx - r - 4, cy - r - 4, cx + r + 4, cy + r + 4, (10, 12, 24), 0.65)
    cv2.circle(frame, (cx, cy), r, COL_ACCENT, 1, cv2.LINE_AA)
    angle = math.radians(frame_count * 1.5 % 360)
    nx = int(cx + (r - 6) * math.sin(angle))
    ny = int(cy - (r - 6) * math.cos(angle))
    cv2.line(frame, (cx, cy), (nx, ny), (0, 60, 255), 2, cv2.LINE_AA)
    _text(frame, "N", cx - 5, cy - r + 12, COL_WHITE, 0.35, 1)


# ─── MJPEG frame generator (for Flask streaming) ─────────────────────────────

def make_frame_generator(pipeline: "AegisPipeline"):
    def gen():
        while True:
            frame = pipeline.get_latest_frame()
            if frame is None:
                time.sleep(0.033)
                continue
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
            if not ok:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")
    return gen


# ─── AegisPipeline ───────────────────────────────────────────────────────────

class AegisPipeline:
    """
    AegisRoad AI – Full inference + display pipeline.

    Call run() to start. Displays annotated frames in an OpenCV window
    while streaming via Flask-SocketIO on a background thread.
    """

    def __init__(
        self,
        source,
        camera_id: str = "cam1",
        location_str: str = "28.5439375,77.3304876",
        gradcam_enabled: bool = True,
        demo_mode: bool = True,
        skip_frames: bool = True,
        port: int = 5050,
    ) -> None:
        self.source       = source
        self.camera_id    = camera_id
        self.location_str = location_str
        self.demo_mode    = demo_mode
        self.skip_frames  = skip_frames
        self.port         = port

        self._frame_count  = 0
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock   = threading.Lock()
        self._paused       = False

        # Track history for trajectory polylines {track_id → deque[(cx,cy)]}
        self._track_hist: dict = {}
        # Global ReID mapping {track_id → global_id_short}
        self._global_ids: dict = {}

        print("\n🛡️  AegisRoad AI – Intelligent Road Safety Ecosystem")
        print("=" * 54)

        # ── Model ──────────────────────────────────────────────────
        print(f"⏳ Loading model: {MODEL_PATH}")
        self.yolo = YOLO(MODEL_PATH)
        print(f"✅ Model ready  ({MODEL_PATH})")
        set_pipeline_model(self.yolo)   # share with upload stream processor

        # ── Modules ────────────────────────────────────────────────
        self.tracker           = ByteTrackWrapper(frame_rate=30, track_threshold=0.35)
        self.traj_engine       = TrajectoryEngine(fps=30)
        self.temporal_verifier = TemporalAccidentVerifier()
        self.severity_clf      = SeverityClassifier()
        self.evidence_buf      = EvidenceBuffer(fps=30, pre_seconds=5)
        self.forensics         = ForensicsEngine(fps=30)
        self.victim_det        = VictimDetector()
        self.reid_engine       = ReIDEngine()
        self.global_tracker    = GlobalTracker(self.reid_engine)
        self.gradcam           = GradCAMExplainer(model_path=MODEL_PATH) if gradcam_enabled else None

        # ── Session state ──────────────────────────────────────────
        self.accident_active    = False
        self.no_accident_frames = 0
        self.last_severity      = SeverityLevel.LOW
        self.last_conf          = 0.0
        self.total_accidents    = 0
        self.total_near_misses  = 0
        self._flash_frame       = 0
        self._demo_accident     = False
        self._demo_severity     = "HIGH"

        # FPS smoothing
        self._fps_buf = deque(maxlen=30)

        print("✅ All modules ready\n")

    # ── Thread-safe frame sharing ─────────────────────────────────

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def _push_frame(self, frame: np.ndarray) -> None:
        with self._frame_lock:
            self._latest_frame = frame

    # ── Main display loop ─────────────────────────────────────────

    def run(self):
        source = self.source if isinstance(self.source, str) else int(self.source)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"❌ Cannot open source: {self.source}")
            return

        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.traj_engine.fps = fps_src
        total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1

        print(f"▶  Source: {self.source}")
        print(f"▶  FPS:    {fps_src:.1f}  |  Frames: {total_frames}")
        print(f"▶  Demo mode: {self.demo_mode}  |  Frame-skip: {self.skip_frames}")
        print(f"▶  Controls:  Q=quit  P=pause  R=replay\n")

        cv2.namedWindow("🛡️ AegisRoad AI", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("🛡️ AegisRoad AI", DISPLAY_W, DISPLAY_H)

        t_last_stats = time.time()
        replay_requested = False

        while cap.isOpened():
            if replay_requested:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                replay_requested = False
                self._frame_count = 0
                self.temporal_verifier.reset()
                self.tracker.reset()
                self.accident_active = False
                print("🔁 Replay: back to frame 0")

            if self._paused:
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('p'):
                    self._paused = False
                continue

            ret, raw_frame = cap.read()
            if not ret:
                if isinstance(self.source, str):
                    print("ℹ  End of video – replaying…")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._frame_count = 0
                    self.temporal_verifier.reset()
                    continue
                break

            t0 = time.time()
            self._frame_count += 1
            self._flash_frame += 1

            # ── Frame-skip (every 2nd frame → ~2× faster inference) ────
            if self.skip_frames and self._frame_count % 2 == 0:
                # Still display the frame, just skip YOLO
                display = self._make_display_frame(raw_frame, [], [], [], [], False, "LOW", 0.0)
                self._push_frame(display)
                cv2.imshow("🛡️ AegisRoad AI", display)
                key = cv2.waitKey(1) & 0xFF
                if   key == ord('q'): break
                elif key == ord('p'): self._paused = True
                elif key == ord('r'): replay_requested = True
                continue

            # ── 1. YOLO (pass full-res frame; YOLO resizes internally) ───
            # imgsz=640 controls inference resolution; output boxes are in
            # raw_frame coordinate space — no manual scaling needed.
            results = self.yolo(raw_frame, verbose=False,
                                conf=CONFIDENCE_THRESH, imgsz=PROCESS_W)

            # ── 2. ByteTrack ───────────────────────────────────────────
            tracked = self.tracker.update(results, raw_frame)

            # Update trajectory histories
            for tv in tracked:
                if tv.track_id not in self._track_hist:
                    self._track_hist[tv.track_id] = deque(maxlen=45)
                self._track_hist[tv.track_id].append((int(tv.cx), int(tv.cy)))

            # ── 3. Trajectory + Near-Miss ──────────────────────────────
            near_misses = self.traj_engine.update(tracked, t0)
            for nm in near_misses:
                self.total_near_misses += 1
                emit_near_miss_event(nm.to_dict())

            # ── 4. Temporal Verifier ───────────────────────────────────
            temporal_dets, accident_bboxes = [], []
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0]) if box.cls is not None else 0
                    if cls_id != 0 or conf < CONFIDENCE_THRESH:
                        continue
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    temporal_dets.append(Detection(bbox=(x1, y1, x2, y2), confidence=conf))
                    accident_bboxes.append((x1, y1, x2, y2))

            is_accident, temporal_conf = self.temporal_verifier.update(temporal_dets)

            # ── 5. Demo mode injection ─────────────────────────────────
            if self.demo_mode and (self._frame_count % DEMO_TRIGGER_EVERY == 0):
                is_accident     = True
                temporal_conf   = 0.88
                self._demo_accident  = True
                self._demo_severity  = "HIGH"
                # Fake bbox at frame centre
                h0, w0 = raw_frame.shape[:2]
                fx1, fy1 = w0 // 3, h0 // 3
                fx2, fy2 = 2 * w0 // 3, 2 * h0 // 3
                accident_bboxes = accident_bboxes or [(fx1, fy1, fx2, fy2)]
                print(f"[DEMO] Injecting simulated accident at frame {self._frame_count}")

            # ── 6. Evidence buffer ─────────────────────────────────────
            self.evidence_buf.push(raw_frame,
                                   [{"bbox": list(d.bbox)} for d in temporal_dets], t0)

            # ── 7. Forensics ingest ────────────────────────────────────
            vs = self.forensics.build_vehicle_states_from_tracked(tracked, self.traj_engine)
            col_pt = (near_misses[0].collision_x, near_misses[0].collision_y) if near_misses else None
            self.forensics.ingest_frame(vs, col_pt, t0)

            # ── 8. Accident confirmed ──────────────────────────────────
            if is_accident and not self.accident_active:
                self.accident_active = True
                self.last_conf       = temporal_conf
                self.total_accidents += 1

                max_spd = max(
                    (self.traj_engine._tracks[tv.track_id].smoothed_speed * fps_src
                     for tv in tracked if tv.track_id in self.traj_engine._tracks),
                    default=60.0,
                ) * fps_src

                sev = self.severity_clf.classify(
                    speed_px_s=max_spd,
                    bbox_a=accident_bboxes[0] if accident_bboxes else None,
                    bbox_b=accident_bboxes[1] if len(accident_bboxes) > 1 else None,
                    vehicle_count=max(len(tracked), 1),
                    temporal_conf=temporal_conf,
                )
                self.last_severity = sev.level

                event_id = str(uuid.uuid4())
                self.evidence_buf.trigger({
                    "event_id": event_id, "camera_id": self.camera_id,
                    "severity": sev.level.value, "confidence": temporal_conf,
                    "demo": self._demo_accident,
                })
                self.forensics.generate_replay(event_id=event_id,
                                               involved_ids={tv.track_id for tv in tracked})
                vanet = build_accident_alert(self.location_str, temporal_conf, self.camera_id)
                publish_mqtt_alert(vanet)

                from vanet.vanet_layer import parse_location_string
                loc = parse_location_string(self.location_str) or {"lat": 0.0, "lon": 0.0}
                emit_accident_event({
                    "event_id":       event_id,
                    "type":           "ACCIDENT",
                    "timestamp":      t0,
                    "camera_id":      self.camera_id,
                    "location":       loc,
                    "confidence":     round(temporal_conf, 3),
                    "severity":       sev.level.value,
                    "severity_score": round(sev.score, 3),
                    "breakdown":      sev.to_dict()["breakdown"],
                    "vehicle_count":  len(tracked),
                    "demo":           self._demo_accident,
                })
                print(f"🚨 ACCIDENT [{sev.level.value}] "
                      f"conf={temporal_conf:.2f}  vehicles={len(tracked)}"
                      + ("  [DEMO]" if self._demo_accident else ""))

            if temporal_dets:
                self.no_accident_frames = 0
            else:
                self.no_accident_frames += 1
            if self.no_accident_frames >= NO_ACCIDENT_RESET:
                self.accident_active   = False
                self._demo_accident    = False
                self.temporal_verifier.reset()

            # ── 9. Victim Detection ────────────────────────────────────
            victims = self.victim_det.detect(results, accident_bboxes)

            # ── 10. ReID Fusion ────────────────────────────────────────
            for tv in tracked:
                gid = self.global_tracker.match(raw_frame, tv.bbox, self.camera_id, tv.track_id)
                self._global_ids[tv.track_id] = gid[:6]

            # ── 11. Grad-CAM (HIGH/CRITICAL + accident only) ───────────
            if (self.gradcam and self.accident_active and
                    self.last_severity in (SeverityLevel.HIGH, SeverityLevel.CRITICAL)):
                raw_frame = self.gradcam.explain_frame(raw_frame,
                                                       accident_bboxes[0] if accident_bboxes else None)

            # ── 12. Compute FPS ────────────────────────────────────────
            self._fps_buf.append(time.time() - t0)
            actual_fps = 1.0 / (sum(self._fps_buf) / max(len(self._fps_buf), 1))

            # ── 13. Build display frame ────────────────────────────────
            display = self._make_display_frame(
                raw_frame, tracked, near_misses, victims,
                accident_bboxes, self.accident_active,
                self.last_severity.value, self.last_conf,
            )

            # ── 14. Stats emit ─────────────────────────────────────────
            if self._frame_count % STATS_EMIT_EVERY == 0:
                emit_stats_update({
                    "fps":         round(actual_fps, 1),
                    "tracks":      len(tracked),
                    "victims":     len(victims),
                    "accidents":   self.total_accidents,
                    "near_misses": self.total_near_misses,
                    "frame":       self._frame_count,
                    "camera_id":   self.camera_id,
                })

            self._push_frame(display)
            cv2.imshow("🛡️ AegisRoad AI", display)

            # ── Keyboard ───────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if   key == ord('q'): break
            elif key == ord('p'):
                self._paused = True
                print("⏸  Paused. Press P to resume.")
            elif key == ord('r'):
                replay_requested = True

        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ AegisRoad AI pipeline finished.")

    # ── Frame renderer ─────────────────────────────────────────────

    def _make_display_frame(
        self,
        raw: np.ndarray,
        tracked,
        near_misses,
        victims,
        accident_bboxes: list,
        accident_active: bool,
        severity: str,
        conf: float,
    ) -> np.ndarray:
        """Compose the full annotated display frame."""
        display = cv2.resize(raw, (DISPLAY_W, DISPLAY_H))
        sx = DISPLAY_W / raw.shape[1]
        sy = DISPLAY_H / raw.shape[0]

        # Scale trajectories to display coords
        scaled_hist = {}
        for tid, pts in self._track_hist.items():
            scaled_hist[tid] = [(int(p[0] * sx), int(p[1] * sy)) for p in pts]

        # Scale tracked vehicles
        class _Scaled:
            def __init__(self, tv):
                x1, y1, x2, y2 = tv.bbox
                self.bbox = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
                self.cx = tv.cx * sx; self.cy = tv.cy * sy
                self.track_id = tv.track_id; self.class_name = tv.class_name

        scaled_tv = [_Scaled(tv) for tv in tracked]

        # Trajectory polylines + boxes
        draw_tracks(display, scaled_tv, self.traj_engine, scaled_hist)

        # Speed arrows
        draw_speed_arrows(display, scaled_tv, self.traj_engine)

        # Near-miss circles
        class _ScaledNM:
            def __init__(self, nm):
                self.collision_x  = nm.collision_x  * sx
                self.collision_y  = nm.collision_y  * sy
                self.distance_px  = nm.distance_px
                self.angle_deg    = nm.angle_deg
        draw_near_misses(display, [_ScaledNM(nm) for nm in near_misses])

        # Victims
        class _ScaledVic:
            def __init__(self, v):
                x1, y1, x2, y2 = v.bbox
                self.bbox   = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
                self.fallen = v.fallen
        draw_victims(display, [_ScaledVic(v) for v in victims])

        # Accident overlay
        if accident_active and accident_bboxes:
            draw_alert_banner(display, severity, conf, self._flash_frame)
            # Highlight accident bbox(es)
            for bb in accident_bboxes[:2]:
                bx1 = int(bb[0] * sx); by1 = int(bb[1] * sy)
                bx2 = int(bb[2] * sx); by2 = int(bb[3] * sy)
                sc  = SEV_COLORS.get(severity, COL_DANGER)
                cv2.rectangle(display, (bx1, by1), (bx2, by2), sc, 3)
                # Crosshair
                mx, my = (bx1 + bx2) // 2, (by1 + by2) // 2
                cv2.line(display, (mx - 20, my), (mx + 20, my), sc, 2)
                cv2.line(display, (mx, my - 20), (mx, my + 20), sc, 2)

        if self._demo_accident:
            draw_demo_watermark(display)

        # ReID labels
        draw_reid_labels(display, scaled_tv, self._global_ids)

        # Status panel
        fps_val = (1.0 / (sum(self._fps_buf) / max(len(self._fps_buf), 1))
                   if self._fps_buf else 0.0)
        draw_status_panel(display, {
            "fps":        fps_val,
            "frame":      self._frame_count,
            "tracks":     len(tracked),
            "accidents":  self.total_accidents,
            "near_misses":self.total_near_misses,
            "victims":    len(victims),
        }, accident_active, severity, self.demo_mode)

        draw_compass_hud(display, self._flash_frame)

        # Severity badge top-right
        _badge(display, f"SEVERITY: {severity}", DISPLAY_W - 180, 40,
               SEV_COLORS.get(severity, COL_OK), 165)

        # Controls hint bottom-left
        _text(display, "Q=Quit  P=Pause  R=Replay",
              12, DISPLAY_H - 10, (120, 120, 140), 0.38)

        return display


# ─── Entry Point ─────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="🛡️ AegisRoad AI – Intelligent Road Safety System"
    )
    ap.add_argument("--source",      type=str,  default="sample_videos/video1.mp4",
                    help="Video file, RTSP URL, or webcam index")
    ap.add_argument("--camera-id",   type=str,  default="cam1")
    ap.add_argument("--location",    type=str,  default="28.5439375,77.3304876")
    ap.add_argument("--port",        type=int,  default=5050)
    ap.add_argument("--host",        type=str,  default="0.0.0.0")
    ap.add_argument("--no-demo",     action="store_true",
                    help="Disable demo-mode accident simulation")
    ap.add_argument("--no-skip",     action="store_true",
                    help="Process every frame (slower but more accurate)")
    ap.add_argument("--no-gradcam",  action="store_true",
                    help="Disable Grad-CAM XAI overlay (saves memory)")
    ap.add_argument("--no-api",      action="store_true",
                    help="Skip starting the Flask-SocketIO API server")
    return ap.parse_args()


def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    pipeline = AegisPipeline(
        source        = source,
        camera_id     = args.camera_id,
        location_str  = args.location,
        gradcam_enabled = not args.no_gradcam,
        demo_mode     = not args.no_demo,
        skip_frames   = not args.no_skip,
        port          = args.port,
    )

    if not args.no_api:
        register_camera_stream(args.camera_id, make_frame_generator(pipeline))
        api_thread = threading.Thread(
            target=start_api,
            kwargs={"host": args.host, "port": args.port, "debug": False},
            daemon=True,
        )
        api_thread.start()
        time.sleep(1.2)
        print(f"🌐 Dashboard:  http://localhost:{args.port}/")
        print(f"📡 Stream:     http://localhost:{args.port}/stream/{args.camera_id}")
        print(f"📊 Events:     http://localhost:{args.port}/events\n")

    pipeline.run()


if __name__ == "__main__":
    main()
