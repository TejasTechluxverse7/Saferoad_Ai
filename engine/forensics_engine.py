"""
SaferoadAI – Forensics Replay Engine  🏆 HACKATHON WOW FEATURE
===============================================================
Reconstructs the accident in slow-motion animated video, showing:
  • Colour-coded vehicle path polylines (with fading opacity)
  • Speed arrows (scaled velocity vectors per vehicle)
  • Pulsing collision zone
  • Frame timestamp overlay
  • Slow-motion (4× frame duplication)

Output: `accident_frames/replay_<event_id>.mp4`

Usage:
    fe = ForensicsEngine(output_dir="accident_frames", fps=30)

    # During processing – feed trajectory snapshots every frame
    fe.ingest_frame(track_trajectories, involved_ids, accident_point)

    # On accident event – generate replay asynchronously
    replay_path = fe.generate_replay(event_id)   # blocks briefly then returns path
"""

from __future__ import annotations

import math
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np


# ─── Config ───────────────────────────────────────────────────────────────────

CANVAS_W        = 960
CANVAS_H        = 540
REPLAY_FPS      = 30
SLOWMO_FACTOR   = 4       # each frame rendered N times
PATH_FADE_STEPS = 20      # older path segments get more transparent
MAX_TRAIL_LEN   = 60      # max trajectory points to draw per vehicle
ARROW_SCALE     = 1.8     # scale factor for speed arrows
PULSE_FRAMES    = 12      # frames per pulse animation cycle


# Vehicle palette – track_id % len → BGR color
_PALETTE = [
    (86, 180, 233), (230, 159, 0), (0, 158, 115),
    (204, 121, 167),(213, 94, 0),  (0, 114, 178),
    (240, 228, 66), (0, 204, 153),
]


@dataclass
class _FrameSnapshot:
    """Single-frame data for replay reconstruction."""
    timestamp: float
    vehicles: Dict[int, dict]  # track_id → {cx, cy, vx, vy, speed_px_s}
    collision_point: Optional[Tuple[float, float]]


# ─── ForensicsEngine ──────────────────────────────────────────────────────────

class ForensicsEngine:
    """
    Records per-frame vehicle trajectories and renders animated
    slow-motion forensic replay videos on demand.

    Parameters
    ----------
    output_dir    : Directory where replay MP4s are written.
    fps           : Source video frame rate.
    buffer_seconds: How many seconds of history to keep in RAM.
    canvas_size   : (width, height) of the replay canvas in pixels.
    """

    def __init__(
        self,
        output_dir: str = "accident_frames",
        fps: float = 30.0,
        buffer_seconds: float = 12.0,
        canvas_size: Tuple[int, int] = (CANVAS_W, CANVAS_H),
    ) -> None:
        self.output_dir = output_dir
        self.fps = max(fps, 1.0)
        self.canvas_w, self.canvas_h = canvas_size

        buf_len = int(buffer_seconds * fps) + 1
        self._ring: deque[_FrameSnapshot] = deque(maxlen=buf_len)
        self._active_events: Dict[str, dict] = {}
        self._lock = threading.Lock()

        os.makedirs(output_dir, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest_frame(
        self,
        vehicle_states: Dict[int, dict],
        collision_point: Optional[Tuple[float, float]] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Record one frame of vehicle states.

        Parameters
        ----------
        vehicle_states : {track_id: {"cx": float, "cy": float,
                                     "vx": float, "vy": float,
                                     "speed_px_s": float}}
        collision_point : Pixel coords of detected collision centre.
        timestamp       : Epoch time (defaults to time.time()).
        """
        snap = _FrameSnapshot(
            timestamp=timestamp or time.time(),
            vehicles={int(k): dict(v) for k, v in vehicle_states.items()},
            collision_point=collision_point,
        )
        with self._lock:
            self._ring.append(snap)

    def generate_replay(
        self,
        event_id: Optional[str] = None,
        involved_ids: Optional[Set[int]] = None,
        post_seconds: float = 3.0,
    ) -> str:
        """
        Snapshot the current buffer and render a slow-motion replay MP4.

        Parameters
        ----------
        event_id    : Unique ID (auto-generated if None).
        involved_ids: Track IDs to highlight (all drawn if None).
        post_seconds: Additional seconds to wait for post-event footage.

        Returns
        -------
        str : Absolute path to the written MP4.
        """
        if event_id is None:
            event_id = str(uuid.uuid4())

        with self._lock:
            frames = list(self._ring)

        out_path = os.path.join(self.output_dir, f"replay_{event_id}.mp4")

        # Render in background thread; write sentinel file when done
        t = threading.Thread(
            target=self._render,
            args=(frames, out_path, involved_ids or set()),
            daemon=True,
        )
        t.start()
        print(f"🎬 Forensics replay rendering → {out_path}")
        return out_path

    # ── Internal – Renderer ───────────────────────────────────────────────────

    def _render(
        self,
        frames: List[_FrameSnapshot],
        out_path: str,
        highlight_ids: Set[int],
    ) -> None:
        if not frames:
            return

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            out_path, fourcc, REPLAY_FPS,
            (self.canvas_w, self.canvas_h)
        )

        # Normalise vehicle coordinates to canvas space
        all_cx = [v["cx"] for snap in frames for v in snap.vehicles.values()]
        all_cy = [v["cy"] for snap in frames for v in snap.vehicles.values()]
        if not all_cx:
            writer.release()
            return

        src_min_x, src_max_x = min(all_cx), max(all_cx)
        src_min_y, src_max_y = min(all_cy), max(all_cy)
        scale_x, offset_x = _fit_scale(src_min_x, src_max_x, self.canvas_w, margin=60)
        scale_y, offset_y = _fit_scale(src_min_y, src_max_y, self.canvas_h, margin=60)

        def to_canvas(cx: float, cy: float) -> Tuple[int, int]:
            px = int((cx - src_min_x) * scale_x + offset_x)
            py = int((cy - src_min_y) * scale_y + offset_y)
            return px, py

        # Build per-vehicle path history
        path_history: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        collision_pts: List[Tuple[int, int]] = []

        for frame_idx, snap in enumerate(frames):
            # ── Build canvas ──────────────────────────────────────────────────
            canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
            canvas[:] = (15, 17, 25)   # dark background

            # Grid lines
            _draw_grid(canvas)

            # Update path histories
            for tid, vstate in snap.vehicles.items():
                pt = to_canvas(vstate["cx"], vstate["cy"])
                path_history[tid].append(pt)

            # Collision point
            if snap.collision_point:
                cp = to_canvas(*snap.collision_point)
                collision_pts.append(cp)

            # ── Draw paths (fading polylines) ─────────────────────────────────
            for tid, pts in path_history.items():
                color = _get_color(tid)
                trail = pts[-MAX_TRAIL_LEN:]
                n = len(trail)
                for seg_i in range(1, n):
                    alpha = (seg_i / n) ** 1.5          # fade older segments
                    thickness = max(1, int(2 * alpha))
                    col = tuple(int(c * alpha) for c in color)
                    cv2.line(canvas, trail[seg_i - 1], trail[seg_i], col, thickness, cv2.LINE_AA)

            # ── Draw vehicles ─────────────────────────────────────────────────
            for tid, vstate in snap.vehicles.items():
                pt = to_canvas(vstate["cx"], vstate["cy"])
                color = _get_color(tid)
                is_highlight = (not highlight_ids) or (tid in highlight_ids)
                radius = 12 if is_highlight else 7

                # Vehicle circle
                cv2.circle(canvas, pt, radius, color, -1, cv2.LINE_AA)
                cv2.circle(canvas, pt, radius + 2, (255, 255, 255), 1, cv2.LINE_AA)

                # Speed arrow
                vx = vstate.get("vx", 0.0)
                vy = vstate.get("vy", 0.0)
                spd = vstate.get("speed_px_s", 0.0)
                if abs(vx) + abs(vy) > 0.5:
                    mag = math.hypot(vx, vy)
                    arrow_len = min(int(spd * ARROW_SCALE * scale_x / self.fps), 80) + 20
                    ex = pt[0] + int(vx / mag * arrow_len)
                    ey = pt[1] + int(vy / mag * arrow_len)
                    cv2.arrowedLine(canvas, pt, (ex, ey), color, 2, cv2.LINE_AA, tipLength=0.35)

                # ID label
                label = f"#{tid}"
                cv2.putText(canvas, label, (pt[0] + 14, pt[1] + 5),
                            cv2.FONT_HERSHEY_DUPLEX, 0.45, color, 1, cv2.LINE_AA)
                # Speed label
                kmh_approx = spd / 18.0 * 3.6   # rough PX_PER_METER=18
                cv2.putText(canvas, f"{kmh_approx:.0f}km/h",
                            (pt[0] + 14, pt[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

            # ── Collision zone (pulsing circle) ───────────────────────────────
            if collision_pts:
                cp = collision_pts[-1]
                pulse_r = 28 + int(10 * math.sin(frame_idx * math.pi * 2 / PULSE_FRAMES))
                cv2.circle(canvas, cp, pulse_r, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.circle(canvas, cp, pulse_r - 10, (0, 60, 200), -1, cv2.LINE_AA)
                cv2.putText(canvas, "IMPACT", (cp[0] - 28, cp[1] - pulse_r - 6),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 100, 255), 1, cv2.LINE_AA)

            # ── HUD ──────────────────────────────────────────────────────────
            elapsed = snap.timestamp - frames[0].timestamp
            _draw_hud(canvas, frame_idx, len(frames), elapsed, len(snap.vehicles))

            # ── Slow-motion: write each frame N times ─────────────────────────
            for _ in range(SLOWMO_FACTOR):
                writer.write(canvas)

        writer.release()
        print(f"✅ Replay video written → {out_path}")

    # ── Utility access ────────────────────────────────────────────────────────

    def build_vehicle_states_from_tracked(
        self,
        tracked_vehicles,
        trajectory_engine,
    ) -> Dict[int, dict]:
        """
        Convenience builder for `ingest_frame()`.
        Combines TrackedVehicle list with TrajectoryEngine velocity data.
        """
        states = {}
        for tv in tracked_vehicles:
            te_state = trajectory_engine._tracks.get(tv.track_id)
            vx = te_state.vx if te_state else 0.0
            vy = te_state.vy if te_state else 0.0
            spd = te_state.smoothed_speed * trajectory_engine.fps if te_state else 0.0
            states[tv.track_id] = {
                "cx": tv.cx, "cy": tv.cy,
                "vx": vx, "vy": vy,
                "speed_px_s": spd,
            }
        return states


# ─── Drawing Helpers ──────────────────────────────────────────────────────────

def _get_color(track_id: int) -> Tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


def _fit_scale(v_min: float, v_max: float, canvas_dim: int, margin: int) -> Tuple[float, float]:
    span = max(v_max - v_min, 1.0)
    scale = (canvas_dim - 2 * margin) / span
    offset = margin - v_min * scale
    return scale, offset


def _draw_grid(canvas: np.ndarray, spacing: int = 60) -> None:
    h, w = canvas.shape[:2]
    for x in range(0, w, spacing):
        cv2.line(canvas, (x, 0), (x, h), (30, 33, 45), 1)
    for y in range(0, h, spacing):
        cv2.line(canvas, (0, y), (w, y), (30, 33, 45), 1)


def _draw_hud(
    canvas: np.ndarray,
    frame_idx: int,
    total_frames: int,
    elapsed: float,
    vehicle_count: int,
) -> None:
    h, w = canvas.shape[:2]

    # Top-left: branding
    cv2.putText(canvas, "SaferoadAI  |  FORENSIC REPLAY",
                (16, 28), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    # Top-right: SLOW MOTION badge
    badge = "4x SLOW MOTION"
    (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(canvas, (w - bw - 24, 10), (w - 8, 36), (0, 60, 200), -1)
    cv2.putText(canvas, badge, (w - bw - 16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Bottom bar
    bar_h = 36
    cv2.rectangle(canvas, (0, h - bar_h), (w, h), (20, 22, 32), -1)

    # Progress bar
    progress = frame_idx / max(total_frames - 1, 1)
    cv2.rectangle(canvas, (0, h - bar_h), (int(w * progress), h), (30, 100, 200), -1)

    cv2.putText(canvas,
                f"T+{elapsed:.2f}s    Vehicles: {vehicle_count}    Frame {frame_idx + 1}/{total_frames}",
                (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
