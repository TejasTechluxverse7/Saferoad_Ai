"""
SaferoadAI – Victim Detector
=============================
Detects fallen/injured persons using YOLO's `person` class and an
aspect-ratio heuristic:

    If bbox_width / bbox_height > FALLEN_RATIO → person is horizontal → fallen

Also checks if a person is in proximity to an accident bounding box.

Usage:
    vd = VictimDetector(fallen_ratio=1.5, accident_proximity_px=80)
    victims = vd.detect(yolo_results, accident_bboxes)
    for v in victims:
        print(v)  # {"id": ..., "bbox": ..., "fallen": True}
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ─── Config ───────────────────────────────────────────────────────────────────

PERSON_CLASS_ID      = 0      # COCO class 0 = person
FALLEN_RATIO         = 1.5    # width/height > this → likely fallen/horizontal
ACCIDENT_PROX_PX     = 100    # pixels within accident bbox centroid → at-risk


# ─── Data ─────────────────────────────────────────────────────────────────────

@dataclass
class DetectedVictim:
    person_index: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    fallen: bool
    near_accident: bool

    def to_dict(self) -> dict:
        x1, y1, x2, y2 = self.bbox
        return {
            "person_index": self.person_index,
            "bbox": list(self.bbox),
            "confidence": round(self.confidence, 3),
            "fallen": self.fallen,
            "near_accident": self.near_accident,
            "cx": round((x1 + x2) / 2, 1),
            "cy": round((y1 + y2) / 2, 1),
        }


# ─── VictimDetector ───────────────────────────────────────────────────────────

class VictimDetector:
    """
    Detects fallen persons within YOLO results.

    Parameters
    ----------
    person_class_id     : YOLO class ID for 'person'.
    fallen_ratio        : Min width/height ratio to classify as fallen.
    accident_proximity_px: Pixel radius to check person-accident proximity.
    min_confidence      : Min confidence to accept a person detection.
    """

    def __init__(
        self,
        person_class_id: int = PERSON_CLASS_ID,
        fallen_ratio: float = FALLEN_RATIO,
        accident_proximity_px: float = ACCIDENT_PROX_PX,
        min_confidence: float = 0.30,
    ) -> None:
        self.person_class_id = person_class_id
        self.fallen_ratio = fallen_ratio
        self.accident_proximity_px = accident_proximity_px
        self.min_confidence = min_confidence

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(
        self,
        yolo_results,
        accident_bboxes: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> List[DetectedVictim]:
        """
        Detect fallen / at-risk persons in a frame.

        Parameters
        ----------
        yolo_results  : ultralytics Results for the current frame.
        accident_bboxes: List of confirmed accident bounding boxes (xyxy).
                         Used for proximity check.

        Returns
        -------
        List[DetectedVictim]
        """
        victims: List[DetectedVictim] = []
        acc_centers = [_bbox_center(b) for b in (accident_bboxes or [])]

        person_idx = 0
        for r in yolo_results:
            for box in r.boxes:
                cls_id = int(box.cls[0]) if box.cls is not None else -1
                conf = float(box.conf[0])

                if cls_id != self.person_class_id or conf < self.min_confidence:
                    continue

                x1, y1, x2, y2 = map(float, box.xyxy[0])
                w = x2 - x1
                h = max(y2 - y1, 1.0)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                fallen = (w / h) > self.fallen_ratio

                near_acc = any(
                    math.hypot(cx - ac[0], cy - ac[1]) < self.accident_proximity_px
                    for ac in acc_centers
                )

                victims.append(DetectedVictim(
                    person_index=person_idx,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    fallen=fallen,
                    near_accident=near_acc,
                ))
                person_idx += 1

        return victims

    def annotate_frame(
        self,
        frame: np.ndarray,
        victims: List[DetectedVictim],
    ) -> np.ndarray:
        """Draw victim annotations on frame (in-place copy)."""
        out = frame.copy()
        for v in victims:
            x1, y1, x2, y2 = map(int, v.bbox)
            color = (0, 80, 255) if v.fallen else (0, 200, 255)
            label_parts = []
            if v.fallen:
                label_parts.append("⚠ FALLEN")
            if v.near_accident:
                label_parts.append("AT-RISK")
            label = " | ".join(label_parts) if label_parts else "Person"
            label += f" ({v.confidence:.2f})"

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        return out


# ─── Utility ──────────────────────────────────────────────────────────────────

def _bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2
