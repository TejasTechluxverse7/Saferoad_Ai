"""
SaferoadAI – Evidence Buffer
=============================
Maintains a circular frame buffer and automatically saves a
time-stamped evidence package (pre-event, at-event, post-event frames
+ JSON metadata sidecar) on accident confirmation.

Design
------
* Stores last `pre_seconds * fps` frames in a deque (circular).
* On `trigger()`: forks a background thread that waits `post_seconds`
  more frames, then writes the full evidence package to disk.
* Evidence directory layout:

    accident_frames/
      evidence_<uuid>/
        frame_pre_000.jpg … frame_pre_N.jpg
        frame_event_000.jpg
        frame_post_000.jpg … frame_post_M.jpg
        metadata.json

Usage:
    buf = EvidenceBuffer(output_dir="accident_frames", fps=30)
    buf.push(frame, detections=[...])           # call every frame
    if accident_confirmed:
        pkg_dir = buf.trigger(event_meta)       # non-blocking
"""

from __future__ import annotations

import json
import os
import time
import uuid
import threading
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Deque, Optional, Tuple

import cv2
import numpy as np


# ─── Stored Frame ─────────────────────────────────────────────────────────────

@dataclass
class _StoredFrame:
    frame: np.ndarray
    timestamp: float
    detections: List[dict]   # serialisable detection dicts


# ─── EvidenceBuffer ───────────────────────────────────────────────────────────

class EvidenceBuffer:
    """
    Circular evidence buffer with background evidence-package writer.

    Parameters
    ----------
    output_dir  : Directory where evidence folders are written.
    fps         : Approximate video FPS (used to compute buffer sizes).
    pre_seconds : Seconds of video to capture before the event.
    post_seconds: Seconds of video to capture after the event.
    jpeg_quality: JPEG compression quality for saved frames (0-100).
    """

    def __init__(
        self,
        output_dir: str = "accident_frames",
        fps: float = 30.0,
        pre_seconds: float = 5.0,
        post_seconds: float = 5.0,
        jpeg_quality: int = 90,
    ) -> None:
        self.output_dir = output_dir
        self.fps = max(fps, 1.0)
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.jpeg_quality = jpeg_quality

        pre_cap = int(pre_seconds * self.fps) + 1
        self._ring: Deque[_StoredFrame] = deque(maxlen=pre_cap)
        self._lock = threading.Lock()

        # Post-event frames accumulated per active trigger
        self._active_triggers: Dict[str, dict] = {}   # event_id → context

        os.makedirs(output_dir, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def push(
        self,
        frame: np.ndarray,
        detections: Optional[List[dict]] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Ingest a frame into the circular buffer.
        Must be called every frame (before trigger decisions).
        """
        ts = timestamp or time.time()
        sf = _StoredFrame(
            frame=frame.copy(),
            timestamp=ts,
            detections=detections or [],
        )
        with self._lock:
            self._ring.append(sf)
            # Feed post-event frames to any active triggers
            for ev_id, ctx in list(self._active_triggers.items()):
                ctx["post_frames"].append(sf)
                ctx["remaining"] -= 1
                if ctx["remaining"] <= 0:
                    self._finalize_trigger(ev_id)

    def trigger(self, event_meta: Optional[Dict[str, Any]] = None) -> str:
        """
        Snapshot the current pre-event buffer and start collecting
        post-event frames. Returns the unique event_id string.

        Parameters
        ----------
        event_meta : Optional extra metadata to embed in metadata.json.
        """
        event_id = str(uuid.uuid4())
        post_cap = int(self.post_seconds * self.fps) + 1

        with self._lock:
            pre_snap = list(self._ring)   # snapshot of current ring

        self._active_triggers[event_id] = {
            "pre_frames":  pre_snap,
            "event_frame": pre_snap[-1] if pre_snap else None,
            "post_frames": deque(maxlen=post_cap),
            "remaining":   post_cap,
            "meta":        event_meta or {},
            "event_id":    event_id,
            "started_at":  time.time(),
        }

        print(f"🔴 Evidence trigger: {event_id} ({len(pre_snap)} pre-frames captured)")
        return event_id

    def get_buffer_snapshot(self) -> List[np.ndarray]:
        """Return the current pre-event frames as raw ndarray list."""
        with self._lock:
            return [sf.frame for sf in self._ring]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _finalize_trigger(self, event_id: str) -> None:
        ctx = self._active_triggers.pop(event_id, None)
        if ctx is None:
            return

        # Write evidence package in a background thread
        t = threading.Thread(
            target=self._write_package, args=(ctx,), daemon=True
        )
        t.start()

    def _write_package(self, ctx: dict) -> None:
        """Write frames + metadata to disk."""
        event_id = ctx["event_id"]
        pkg_dir = os.path.join(self.output_dir, f"evidence_{event_id}")
        os.makedirs(pkg_dir, exist_ok=True)

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]

        # --- Pre-event frames ------------------------------------------------
        pre_paths: List[str] = []
        for i, sf in enumerate(ctx["pre_frames"]):
            fn = os.path.join(pkg_dir, f"frame_pre_{i:03d}.jpg")
            cv2.imwrite(fn, sf.frame, encode_params)
            pre_paths.append(fn)

        # --- Post-event frames -----------------------------------------------
        post_paths: List[str] = []
        for i, sf in enumerate(ctx["post_frames"]):
            fn = os.path.join(pkg_dir, f"frame_post_{i:03d}.jpg")
            cv2.imwrite(fn, sf.frame, encode_params)
            post_paths.append(fn)

        # --- Metadata sidecar ------------------------------------------------
        meta = {
            "event_id":    event_id,
            "timestamp":   ctx["started_at"],
            "pre_count":   len(pre_paths),
            "post_count":  len(post_paths),
            "package_dir": pkg_dir,
            **ctx["meta"],
        }
        meta_path = os.path.join(pkg_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"✅ Evidence package saved → {pkg_dir} "
              f"({len(pre_paths)} pre + {len(post_paths)} post frames)")
