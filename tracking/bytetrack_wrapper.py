"""
SaferoadAI – ByteTrack Wrapper
================================
Converts YOLO Results → supervision Detections → ByteTrack tracked objects.

Usage:
    tracker = ByteTrackWrapper()
    tracked = tracker.update(yolo_results, frame)
    for tv in tracked:
        print(tv.track_id, tv.bbox, tv.class_name)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import supervision as sv
    _SV_AVAILABLE = True
except ImportError:
    _SV_AVAILABLE = False


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class TrackedVehicle:
    """Single tracked object output from ByteTrack."""
    track_id: int
    bbox: tuple[float, float, float, float]   # (x1, y1, x2, y2) pixels
    class_id: int
    class_name: str
    confidence: float
    # Derived
    cx: float = field(init=False)
    cy: float = field(init=False)
    width: float = field(init=False)
    height: float = field(init=False)

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.cx = (x1 + x2) / 2.0
        self.cy = (y1 + y2) / 2.0
        self.width = x2 - x1
        self.height = y2 - y1

    @property
    def aspect_ratio(self) -> float:
        """width/height — >1.5 indicates fallen/horizontal object."""
        return self.width / max(self.height, 1.0)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "bbox": list(self.bbox),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 3),
            "cx": round(self.cx, 1),
            "cy": round(self.cy, 1),
        }


# ─── ByteTrack Wrapper ────────────────────────────────────────────────────────

# YOLO class names for standard COCO model (index → label)
_COCO_NAMES: dict[int, str] = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    5: "bus", 7: "truck", 9: "traffic light",
}

# For accident-specific models the class map is simpler
_ACCIDENT_NAMES: dict[int, str] = {0: "accident", 1: "vehicle", 2: "person"}


class ByteTrackWrapper:
    """
    Wraps supervision.ByteTrack to assign persistent track IDs to YOLO detections.

    Parameters
    ----------
    frame_rate : int
        Source video FPS (used for ByteTrack lost-track patience).
    track_threshold : float
        Min confidence to enter tracker.
    class_names : dict[int, str]
        Maps class integer IDs to human-readable labels.
    filter_classes : list[int] | None
        If set, only track objects with these class IDs.
    """

    def __init__(
        self,
        frame_rate: int = 30,
        track_threshold: float = 0.35,
        class_names: Optional[dict[int, str]] = None,
        filter_classes: Optional[List[int]] = None,
    ) -> None:
        self.class_names = class_names or _COCO_NAMES
        self.filter_classes = set(filter_classes) if filter_classes else None
        self.track_threshold = track_threshold

        if _SV_AVAILABLE:
            self._tracker = sv.ByteTrack(
                frame_rate=frame_rate,
                track_activation_threshold=track_threshold,
            )
        else:
            self._tracker = None
            self._next_id = 1
            self._simple_tracks: dict[int, dict] = {}
            print("⚠️  supervision not installed – using fallback ID assignment")

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, yolo_results, frame: np.ndarray) -> List[TrackedVehicle]:
        """
        Ingest a single frame's YOLO Results and return tracked vehicles.

        Parameters
        ----------
        yolo_results : ultralytics Results
            Direct output of `model(frame)`.
        frame : np.ndarray
            The raw BGR frame (used only if supervision needs it).

        Returns
        -------
        List[TrackedVehicle]
        """
        if _SV_AVAILABLE and self._tracker is not None:
            return self._track_with_supervision(yolo_results, frame)
        return self._track_simple(yolo_results)

    def reset(self) -> None:
        """Clear tracker state (e.g., on new video source)."""
        if _SV_AVAILABLE and self._tracker:
            self._tracker = sv.ByteTrack()
        else:
            self._next_id = 1
            self._simple_tracks.clear()

    # ── Internal – supervision path ───────────────────────────────────────────

    def _track_with_supervision(self, yolo_results, frame: np.ndarray) -> List[TrackedVehicle]:
        """Convert YOLO Results → sv.Detections → ByteTrack → TrackedVehicle list."""
        sv_dets = sv.Detections.from_ultralytics(yolo_results[0])

        # Filter by confidence
        mask = sv_dets.confidence >= self.track_threshold
        sv_dets = sv_dets[mask]

        # Filter by class
        if self.filter_classes and sv_dets.class_id is not None:
            cls_mask = np.isin(sv_dets.class_id, list(self.filter_classes))
            sv_dets = sv_dets[cls_mask]

        if len(sv_dets) == 0:
            return []

        tracked = self._tracker.update_with_detections(sv_dets)
        return self._sv_to_tracked_vehicles(tracked)

    def _sv_to_tracked_vehicles(self, sv_tracked) -> List[TrackedVehicle]:
        vehicles: List[TrackedVehicle] = []
        if sv_tracked.tracker_id is None:
            return vehicles

        for i in range(len(sv_tracked)):
            tid = int(sv_tracked.tracker_id[i])
            bbox = tuple(float(v) for v in sv_tracked.xyxy[i])
            cls_id = int(sv_tracked.class_id[i]) if sv_tracked.class_id is not None else 0
            conf = float(sv_tracked.confidence[i]) if sv_tracked.confidence is not None else 1.0
            cls_name = self.class_names.get(cls_id, f"class_{cls_id}")

            vehicles.append(TrackedVehicle(
                track_id=tid,
                bbox=bbox,
                class_id=cls_id,
                class_name=cls_name,
                confidence=conf,
            ))
        return vehicles

    # ── Internal – simple fallback path ──────────────────────────────────────

    def _track_simple(self, yolo_results) -> List[TrackedVehicle]:
        """
        Dead-simple nearest-centre matching when supervision is not available.
        Not production-grade but allows the system to run without supervision.
        """
        current_dets: List[TrackedVehicle] = []

        for r in yolo_results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < self.track_threshold:
                    continue
                cls_id = int(box.cls[0]) if box.cls is not None else 0
                if self.filter_classes and cls_id not in self.filter_classes:
                    continue
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                # Nearest existing track
                best_id, best_dist = None, float("inf")
                for tid, state in self._simple_tracks.items():
                    d = ((state["cx"] - cx) ** 2 + (state["cy"] - cy) ** 2) ** 0.5
                    if d < best_dist and d < 80:
                        best_dist, best_id = d, tid

                if best_id is None:
                    best_id = self._next_id
                    self._next_id += 1

                self._simple_tracks[best_id] = {"cx": cx, "cy": cy}
                cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
                current_dets.append(TrackedVehicle(
                    track_id=best_id,
                    bbox=(x1, y1, x2, y2),
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                ))
        return current_dets
