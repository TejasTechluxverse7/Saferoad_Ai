"""
SaferoadAI – Rule-Based Severity Classifier
============================================
Classifies accident severity into LOW / MEDIUM / HIGH / CRITICAL
using physics-informed heuristics. No model training required.

Inputs
------
* speed_px_s  – smoothed pixel-speed of the detected vehicle(s)
* bbox_A, bbox_B – bounding boxes of colliding objects (xyxy format)
* vehicle_count  – number of vehicles involved
* temporal_conf  – confidence from TemporalAccidentVerifier

Output
------
SeverityLevel enum + numeric score in [0, 1]

Design
------
severity_raw = w_speed * speed_score
             + w_overlap * overlap_score
             + w_count * count_score
             + w_conf * conf_score

Thresholds:
  LOW      [0.00 – 0.30)
  MEDIUM   [0.30 – 0.55)
  HIGH     [0.55 – 0.75)
  CRITICAL [0.75 – 1.00]
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple


# ─── Severity Levels ──────────────────────────────────────────────────────────

class SeverityLevel(str, Enum):
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def color_bgr(self) -> Tuple[int, int, int]:
        """OpenCV BGR color for on-frame annotation."""
        return {
            "LOW":      (0, 200, 100),
            "MEDIUM":   (0, 165, 255),
            "HIGH":     (0, 60, 255),
            "CRITICAL": (0, 0, 220),
        }[self.value]

    @property
    def css_color(self) -> str:
        return {
            "LOW":      "#22c55e",
            "MEDIUM":   "#f97316",
            "HIGH":     "#ef4444",
            "CRITICAL": "#7c3aed",
        }[self.value]


@dataclass
class SeverityResult:
    level: SeverityLevel
    score: float           # raw score in [0, 1]
    speed_score: float
    overlap_score: float
    count_score: float
    conf_score: float

    def to_dict(self) -> dict:
        return {
            "severity": self.level.value,
            "score": round(self.score, 3),
            "breakdown": {
                "speed":   round(self.speed_score, 3),
                "overlap": round(self.overlap_score, 3),
                "count":   round(self.count_score, 3),
                "temporal_conf": round(self.conf_score, 3),
            },
        }


# ─── Classifier ───────────────────────────────────────────────────────────────

class SeverityClassifier:
    """
    Stateless rule-based severity scorer.

    Parameters
    ----------
    max_speed_px_s : float
        Speed (px/s) considered "maximum dangerous" → maps to score 1.0.
        Default 300 px/s ≈ ~60 km/h at typical highway camera resolution.
    w_speed, w_overlap, w_count, w_conf : float
        Feature weights (must sum to 1.0).
    """

    def __init__(
        self,
        max_speed_px_s: float = 300.0,
        w_speed: float  = 0.35,
        w_overlap: float = 0.30,
        w_count: float  = 0.15,
        w_conf: float   = 0.20,
    ) -> None:
        assert abs(w_speed + w_overlap + w_count + w_conf - 1.0) < 1e-6, \
            "Weights must sum to 1.0"
        self.max_speed_px_s = max_speed_px_s
        self.w_speed   = w_speed
        self.w_overlap = w_overlap
        self.w_count   = w_count
        self.w_conf    = w_conf

    def classify(
        self,
        speed_px_s: float = 0.0,
        bbox_a: Optional[Tuple[float, float, float, float]] = None,
        bbox_b: Optional[Tuple[float, float, float, float]] = None,
        vehicle_count: int = 1,
        temporal_conf: float = 0.5,
    ) -> SeverityResult:
        """
        Classify severity from raw measurements.

        Parameters
        ----------
        speed_px_s     : Dominant vehicle speed in pixels/second.
        bbox_a, bbox_b : Bounding boxes of the two primary vehicles (xyxy).
                         If only one vehicle, pass bbox_a only.
        vehicle_count  : Total vehicles visible at the time of incident.
        temporal_conf  : Confidence score from TemporalAccidentVerifier.
        """
        speed_score   = _clamp(speed_px_s / self.max_speed_px_s)
        overlap_score = _bbox_overlap(bbox_a, bbox_b) if bbox_a and bbox_b else 0.0
        count_score   = _clamp((vehicle_count - 1) / 4.0)   # 1 vehicle → 0, 5+ → 1
        conf_score    = _clamp(temporal_conf)

        raw = (
            self.w_speed   * speed_score
            + self.w_overlap * overlap_score
            + self.w_count   * count_score
            + self.w_conf    * conf_score
        )
        raw = _clamp(raw)

        level = _score_to_level(raw)
        return SeverityResult(
            level=level,
            score=raw,
            speed_score=speed_score,
            overlap_score=overlap_score,
            count_score=count_score,
            conf_score=conf_score,
        )

    def classify_from_event(self, event: dict) -> SeverityResult:
        """Convenience wrapper accepting an event dict."""
        return self.classify(
            speed_px_s    = event.get("speed_px_s", 0.0),
            bbox_a        = event.get("bbox_a"),
            bbox_b        = event.get("bbox_b"),
            vehicle_count = event.get("vehicle_count", 1),
            temporal_conf = event.get("temporal_conf", 0.5),
        )


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _bbox_overlap(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    """Intersection-over-Union between two boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _score_to_level(score: float) -> SeverityLevel:
    if score < 0.30:
        return SeverityLevel.LOW
    if score < 0.55:
        return SeverityLevel.MEDIUM
    if score < 0.75:
        return SeverityLevel.HIGH
    return SeverityLevel.CRITICAL
