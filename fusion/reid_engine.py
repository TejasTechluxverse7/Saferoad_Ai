"""
SaferoadAI – Re-ID Engine & Global Multi-Camera Tracker
========================================================
Assigns globally consistent vehicle IDs across multiple camera feeds
using lightweight embedding-based Re-Identification.

Approach
--------
1. Crop vehicle from frame using bounding box.
2. Pass through a lightweight ResNet-18 feature extractor (no GPU required).
3. Compare embedding against global gallery using cosine similarity.
4. If similarity > threshold → same vehicle → assign existign global ID.
5. Otherwise → new global ID assigned.

Falls back to a colour-histogram descriptor if torchvision is not installed.

Usage:
    reid = ReIDEngine()
    tracker = GlobalTracker(reid, similarity_threshold=0.80)

    # Per frame, per camera:
    global_id = tracker.match(frame, bbox, camera_id="cam1", local_track_id=3)
"""

from __future__ import annotations

import math
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
    import torchvision.models as models
    import torchvision.transforms as T
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ─── Config ───────────────────────────────────────────────────────────────────

SIMILARITY_THRESHOLD = 0.80   # cosine similarity to declare same vehicle
GALLERY_TTL          = 120    # seconds before gallery entry expires
EMBEDDING_DIM        = 512    # ResNet-18 avgpool output dim


# ─── Gallery Entry ────────────────────────────────────────────────────────────

@dataclass
class GalleryEntry:
    global_id: str
    embedding: np.ndarray
    last_seen: float = field(default_factory=time.time)
    camera_ids: List[str] = field(default_factory=list)

    def update(self, new_embedding: np.ndarray, camera_id: str) -> None:
        """Online EMA update of the stored embedding."""
        self.embedding = 0.7 * self.embedding + 0.3 * new_embedding
        self.embedding /= np.linalg.norm(self.embedding) + 1e-12
        self.last_seen = time.time()
        if camera_id not in self.camera_ids:
            self.camera_ids.append(camera_id)


# ─── ReIDEngine ───────────────────────────────────────────────────────────────

class ReIDEngine:
    """
    Lightweight vehicle ReID using ResNet-18 (falls back to colour histogram).

    Parameters
    ----------
    device : 'cpu' or 'cuda'. Autodetected if None.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        self._use_torch = _TORCH_AVAILABLE
        self._device = device

        if self._use_torch:
            self._init_torch(device)
        else:
            print("⚠️  torchvision not installed – ReID uses colour histogram fallback")

    def _init_torch(self, device: Optional[str]) -> None:
        import torch
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = dev

        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove final FC layer → use global average pool output (512-d)
        self._model = torch.nn.Sequential(*list(net.children())[:-1])
        self._model = self._model.to(dev).eval()

        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def get_embedding(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """
        Extract a normalised embedding vector for a vehicle crop.

        Parameters
        ----------
        frame : BGR frame
        bbox  : (x1, y1, x2, y2) in pixels

        Returns
        -------
        np.ndarray of shape (EMBEDDING_DIM,), L2-normalised.
        """
        crop = _safe_crop(frame, bbox, padding=5)
        if crop is None or crop.size == 0:
            return np.zeros(EMBEDDING_DIM)

        if self._use_torch:
            return self._torch_embed(crop)
        return self._hist_embed(crop)

    def _torch_embed(self, crop: np.ndarray) -> np.ndarray:
        import torch
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb).unsqueeze(0).to(self._device)
        with torch.no_grad():
            feat = self._model(tensor)        # (1, 512, 1, 1)
        feat = feat.squeeze().cpu().numpy()   # (512,)
        norm = np.linalg.norm(feat) + 1e-12
        return feat / norm

    def _hist_embed(self, crop: np.ndarray) -> np.ndarray:
        """HSV colour histogram as fallback 48-d descriptor."""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
        feat = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
        norm = np.linalg.norm(feat) + 1e-12
        # Pad to EMBEDDING_DIM
        padded = np.zeros(EMBEDDING_DIM)
        padded[:len(feat)] = feat / norm
        return padded


# ─── GlobalTracker ────────────────────────────────────────────────────────────

class GlobalTracker:
    """
    Maintains a cross-camera gallery and matches new detections to global IDs.

    Parameters
    ----------
    reid_engine          : ReIDEngine instance.
    similarity_threshold : Cosine similarity threshold for ID matching.
    """

    def __init__(
        self,
        reid_engine: Optional[ReIDEngine] = None,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ) -> None:
        self.reid = reid_engine or ReIDEngine()
        self.threshold = similarity_threshold
        self._gallery: List[GalleryEntry] = []
        # (camera_id, local_track_id) → global_id
        self._local_to_global: Dict[Tuple[str, int], str] = {}

    def match(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        camera_id: str,
        local_track_id: int,
    ) -> str:
        """
        Match a vehicle to the global gallery.

        Returns
        -------
        str : global_id (UUID string)
        """
        key = (camera_id, local_track_id)

        # If we've already matched this (camera, local_id) pair recently, reuse
        if key in self._local_to_global:
            gid = self._local_to_global[key]
            # Update gallery entry
            emb = self.reid.get_embedding(frame, bbox)
            for entry in self._gallery:
                if entry.global_id == gid:
                    entry.update(emb, camera_id)
                    return gid

        # New detection – get embedding and search gallery
        emb = self.reid.get_embedding(frame, bbox)
        self._prune_stale()

        best_entry: Optional[GalleryEntry] = None
        best_sim: float = -1.0

        for entry in self._gallery:
            sim = float(np.dot(emb, entry.embedding))   # both L2-normalised → cosine
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_entry and best_sim >= self.threshold:
            # Match found → update existing entry
            best_entry.update(emb, camera_id)
            self._local_to_global[key] = best_entry.global_id
            return best_entry.global_id
        else:
            # New vehicle
            new_id = str(uuid.uuid4())[:8]
            entry = GalleryEntry(global_id=new_id, embedding=emb, camera_ids=[camera_id])
            self._gallery.append(entry)
            self._local_to_global[key] = new_id
            return new_id

    def get_gallery_size(self) -> int:
        return len(self._gallery)

    def _prune_stale(self) -> None:
        now = time.time()
        self._gallery = [e for e in self._gallery if now - e.last_seen < GALLERY_TTL]


# ─── Utility ──────────────────────────────────────────────────────────────────

def _safe_crop(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    padding: int = 0,
) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
    x2, y2 = min(w, x2 + padding), min(h, y2 + padding)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()
