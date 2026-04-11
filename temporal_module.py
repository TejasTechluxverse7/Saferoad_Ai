import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np


@dataclass
class Detection:
    """
    Lightweight detection representation for temporal reasoning.

    Assumes axis-aligned boxes in (x1, y1, x2, y2) pixel coordinates and a
    single "accident" class coming from YOLO.
    """

    bbox: Tuple[float, float, float, float]
    confidence: float

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    """Compute IoU between two boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h

    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


class TemporalAccidentVerifier:
    """
    Rule-based temporal accident verifier.

    This approximates the "tracking + temporal verification" stage mentioned in
    the project document by:
    - Maintaining a short sliding window of accident-class detections
    - Measuring box overlap and motion magnitude across the window
    - Emitting an accident event only if motion abruptly drops while overlap stays high
    """

    def __init__(
        self,
        window_size: int = 12,
        min_frames_with_detections: int = 4,
        min_iou_for_same_region: float = 0.3,
        min_motion_pixels: float = 15.0,
        min_still_iou: float = 0.5,
    ) -> None:
        self.window_size = window_size
        self.min_frames_with_detections = min_frames_with_detections
        self.min_iou_for_same_region = min_iou_for_same_region
        self.min_motion_pixels = min_motion_pixels
        self.min_still_iou = min_still_iou

        self._history: Deque[List[Detection]] = deque(maxlen=window_size)
        self._last_accident_score: float = 0.0

    def reset(self) -> None:
        self._history.clear()
        self._last_accident_score = 0.0

    def update(self, detections: List[Detection]) -> Tuple[bool, float]:
        """
        Ingest detections for the current frame and return (is_accident, confidence).

        Confidence is a soft score in [0, 1] based on motion + overlap heuristics.
        """
        self._history.append(detections)

        # Not enough temporal context yet
        if len(self._history) < self.min_frames_with_detections:
            return False, 0.0

        # Collect boxes that consistently appear in roughly the same region
        representative_boxes: List[Tuple[float, float, float, float]] = []
        for frame_dets in self._history:
            for det in frame_dets:
                matched = False
                for i, rep in enumerate(representative_boxes):
                    if iou(det.bbox, rep) >= self.min_iou_for_same_region:
                        # Merge into representative region
                        rx1, ry1, rx2, ry2 = rep
                        dx1, dy1, dx2, dy2 = det.bbox
                        merged = (min(rx1, dx1), min(ry1, dy1), max(rx2, dx2), max(ry2, dy2))
                        representative_boxes[i] = merged
                        matched = True
                        break
                if not matched:
                    representative_boxes.append(det.bbox)

        if not representative_boxes:
            self._last_accident_score *= 0.9
            return False, self._last_accident_score

        # For each region, compute motion and "stillness" after the motion
        scores: List[float] = []
        history_list = list(self._history)

        for region in representative_boxes:
            centers = []
            active_frames = 0
            for frame_dets in history_list:
                best_det = None
                best_iou = 0.0
                for det in frame_dets:
                    val = iou(det.bbox, region)
                    if val > best_iou:
                        best_iou = val
                        best_det = det
                if best_det and best_iou >= self.min_iou_for_same_region:
                    centers.append(best_det.center)
                    active_frames += 1

            if active_frames < self.min_frames_with_detections or len(centers) < 2:
                continue

            # Motion magnitude (approximate speed proxy)
            motion = 0.0
            for (x1, y1), (x2, y2) in zip(centers[:-1], centers[1:]):
                motion += math.hypot(x2 - x1, y2 - y1)

            # IoU between first and last box in this region
            first_box = None
            last_box = None
            for frame_dets in history_list:
                for det in frame_dets:
                    if iou(det.bbox, region) >= self.min_iou_for_same_region:
                        if first_box is None:
                            first_box = det.bbox
                        last_box = det.bbox
            if first_box is None or last_box is None:
                continue

            still_iou = iou(first_box, last_box)

            # Heuristic: an "accident-like" event is when we observe
            # non-trivial motion followed by the object becoming still in place.
            motion_score = np.clip(motion / (self.min_motion_pixels * active_frames), 0.0, 2.0)
            still_score = np.clip((still_iou - self.min_still_iou) / (1.0 - self.min_still_iou + 1e-6), 0.0, 1.0)

            region_score = 0.6 * min(1.0, motion_score) + 0.4 * still_score
            scores.append(region_score)

        if not scores:
            self._last_accident_score *= 0.9
            return False, self._last_accident_score

        # Aggregate region scores into a single confidence
        score = float(np.clip(max(scores), 0.0, 1.0))

        # Small temporal smoothing so confidence does not jump abruptly
        self._last_accident_score = 0.7 * self._last_accident_score + 0.3 * score

        is_accident = self._last_accident_score >= 0.6
        return is_accident, float(self._last_accident_score)

