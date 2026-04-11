"""
SaferoadAI – Trajectory Prediction Engine
==========================================
Maintains per-vehicle trajectory histories, computes speed,
and detects near-miss events between pairs of vehicles.

All coordinates are in pixel space. Callers must convert to
real-world units using camera calibration if needed.

Usage:
    engine = TrajectoryEngine(fps=30)
    near_misses = engine.update(tracked_vehicles, timestamp)
    for nm in near_misses:
        print(nm.to_dict())
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Deque

from tracking.bytetrack_wrapper import TrackedVehicle


# ─── Configuration ────────────────────────────────────────────────────────────

HISTORY_LEN        = 30      # frames of trajectory to retain per vehicle
NEAR_MISS_DIST_PX  = 120     # pixel distance threshold for near-miss check
NEAR_MISS_ANGLE    = 60.0    # degrees – divergence angle to consider conflict
SPEED_ALPHA        = 0.3     # EMA smoothing factor for speed
STALE_TTL          = 60      # frames before dropping an unseen track
PIXELS_PER_METER   = 18.0    # calibration constant (tune per camera)


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class TrajectoryPoint:
    cx: float
    cy: float
    timestamp: float   # epoch seconds
    speed_px_s: float  # smoothed pixel-speed


@dataclass
class NearMissAlert:
    id_a: int
    id_b: int
    distance_px: float
    relative_speed_px_s: float
    collision_x: float    # estimated collision point x
    collision_y: float    # estimated collision point y
    angle_deg: float      # angle between velocity vectors
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "type": "NEAR_MISS",
            "id_a": self.id_a,
            "id_b": self.id_b,
            "distance_px": round(self.distance_px, 1),
            "relative_speed_px_s": round(self.relative_speed_px_s, 1),
            "collision_point": [round(self.collision_x, 1), round(self.collision_y, 1)],
            "angle_deg": round(self.angle_deg, 1),
            "timestamp": self.timestamp,
        }


# ─── Internal track state ─────────────────────────────────────────────────────

@dataclass
class _TrackState:
    history: Deque[TrajectoryPoint] = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    smoothed_speed: float = 0.0
    last_seen_frame: int = 0
    vx: float = 0.0   # velocity vector x component (px/frame)
    vy: float = 0.0   # velocity vector y component (px/frame)


# ─── TrajectoryEngine ─────────────────────────────────────────────────────────

class TrajectoryEngine:
    """
    Per-vehicle trajectory tracker with near-miss detection.

    Parameters
    ----------
    fps : float
        Video frame rate – used to convert frame-displacement to speed.
    near_miss_dist_px : float
        Distance threshold (pixels) below which near-miss is checked.
    near_miss_angle : float
        Min angle (degrees) between velocity vectors to flag conflict.
    pixels_per_meter : float
        Camera calibration: pixels per real-world meter.
    """

    def __init__(
        self,
        fps: float = 30.0,
        near_miss_dist_px: float = NEAR_MISS_DIST_PX,
        near_miss_angle: float = NEAR_MISS_ANGLE,
        pixels_per_meter: float = PIXELS_PER_METER,
    ) -> None:
        self.fps = max(fps, 1.0)
        self.near_miss_dist_px = near_miss_dist_px
        self.near_miss_angle = near_miss_angle
        self.pixels_per_meter = pixels_per_meter

        self._tracks: Dict[int, _TrackState] = {}
        self._frame_count: int = 0
        # Throttle: pairs already alerted this second
        self._alerted_pairs: set[frozenset] = set()
        self._alert_reset_frame: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        vehicles: List[TrackedVehicle],
        timestamp: Optional[float] = None,
    ) -> List[NearMissAlert]:
        """
        Call once per frame. Returns list of NearMissAlert (may be empty).

        Parameters
        ----------
        vehicles : List[TrackedVehicle]
            Output from ByteTrackWrapper.update()
        timestamp : float | None
            Epoch time for this frame; defaults to time.time()
        """
        ts = timestamp or time.time()
        self._frame_count += 1

        # Reset alert throttle every 30 frames (~1 second)
        if self._frame_count - self._alert_reset_frame >= int(self.fps):
            self._alerted_pairs.clear()
            self._alert_reset_frame = self._frame_count

        # Update each tracked vehicle
        seen_ids: set[int] = set()
        for v in vehicles:
            self._update_track(v, ts)
            seen_ids.add(v.track_id)

        # Prune stale tracks
        stale = [tid for tid, st in self._tracks.items()
                 if (self._frame_count - st.last_seen_frame) > STALE_TTL]
        for tid in stale:
            del self._tracks[tid]

        # Detect near-miss events
        return self._detect_near_misses(vehicles)

    def get_trajectory(self, track_id: int) -> List[TrajectoryPoint]:
        """Return the full trajectory history for a given track ID."""
        state = self._tracks.get(track_id)
        return list(state.history) if state else []

    def get_speed_kmh(self, track_id: int) -> float:
        """Return smoothed speed in km/h for a track."""
        state = self._tracks.get(track_id)
        if not state:
            return 0.0
        px_per_s = state.smoothed_speed * self.fps
        m_per_s = px_per_s / self.pixels_per_meter
        return m_per_s * 3.6

    def get_all_trajectories(self) -> Dict[int, List[dict]]:
        """Snapshot all track histories for the forensics engine."""
        return {
            tid: [
                {"cx": p.cx, "cy": p.cy, "t": p.timestamp, "spx": p.speed_px_s}
                for p in state.history
            ]
            for tid, state in self._tracks.items()
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _update_track(self, v: TrackedVehicle, ts: float) -> None:
        if v.track_id not in self._tracks:
            self._tracks[v.track_id] = _TrackState()

        state = self._tracks[v.track_id]
        state.last_seen_frame = self._frame_count

        if state.history:
            prev = state.history[-1]
            dx = v.cx - prev.cx
            dy = v.cy - prev.cy
            frame_disp = math.hypot(dx, dy)  # pixels/frame

            # EMA speed smoothing
            state.smoothed_speed = (
                SPEED_ALPHA * frame_disp + (1 - SPEED_ALPHA) * state.smoothed_speed
            )
            state.vx = SPEED_ALPHA * dx + (1 - SPEED_ALPHA) * state.vx
            state.vy = SPEED_ALPHA * dy + (1 - SPEED_ALPHA) * state.vy
        else:
            frame_disp = 0.0

        pt = TrajectoryPoint(
            cx=v.cx,
            cy=v.cy,
            timestamp=ts,
            speed_px_s=state.smoothed_speed * self.fps,
        )
        state.history.append(pt)

    def _detect_near_misses(self, vehicles: List[TrackedVehicle]) -> List[NearMissAlert]:
        alerts: List[NearMissAlert] = []

        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                a, b = vehicles[i], vehicles[j]
                pair = frozenset({a.track_id, b.track_id})
                if pair in self._alerted_pairs:
                    continue

                dist = math.hypot(a.cx - b.cx, a.cy - b.cy)
                if dist >= self.near_miss_dist_px:
                    continue

                # Check velocity divergence angle
                sa = self._tracks.get(a.track_id)
                sb = self._tracks.get(b.track_id)
                if not sa or not sb:
                    continue

                angle = _angle_between(sa.vx, sa.vy, sb.vx, sb.vy)
                if angle < self.near_miss_angle:
                    continue  # Vehicles moving in same direction – not a conflict

                # Collision point estimate: midpoint
                col_x = (a.cx + b.cx) / 2.0
                col_y = (a.cy + b.cy) / 2.0
                rel_speed = abs(sa.smoothed_speed - sb.smoothed_speed) * self.fps

                alert = NearMissAlert(
                    id_a=a.track_id,
                    id_b=b.track_id,
                    distance_px=dist,
                    relative_speed_px_s=rel_speed,
                    collision_x=col_x,
                    collision_y=col_y,
                    angle_deg=angle,
                )
                alerts.append(alert)
                self._alerted_pairs.add(pair)

        return alerts


# ─── Utility ──────────────────────────────────────────────────────────────────

def _angle_between(vx1: float, vy1: float, vx2: float, vy2: float) -> float:
    """Angle in degrees between two 2-D velocity vectors."""
    mag1 = math.hypot(vx1, vy1)
    mag2 = math.hypot(vx2, vy2)
    if mag1 < 1e-6 or mag2 < 1e-6:
        return 0.0
    cos_a = (vx1 * vx2 + vy1 * vy2) / (mag1 * mag2)
    cos_a = max(-1.0, min(1.0, cos_a))
    return math.degrees(math.acos(cos_a))
