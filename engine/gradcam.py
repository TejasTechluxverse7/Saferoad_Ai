"""
SaferoadAI – Grad-CAM Explainability Module
============================================
Generates heatmap overlays on YOLO-processed frames to show
*why* the model flagged a region as an accident.

Uses pytorch-grad-cam (pip install pytorch-grad-cam).
If not installed, the module degrades gracefully (returns original frame).

Only activated for HIGH and CRITICAL severity events to avoid
per-frame performance overhead.

Usage:
    explainer = GradCAMExplainer(model_path="CrashSentinel_Prime.pt")
    overlay = explainer.explain_frame(frame, target_bbox=(x1, y1, x2, y2))
    cv2.imshow("Grad-CAM", overlay)
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import torch
    import torchvision.transforms as T
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    _GRADCAM_AVAILABLE = True
except ImportError:
    _GRADCAM_AVAILABLE = False



warnings.filterwarnings("ignore")


# ─── GradCAMExplainer ─────────────────────────────────────────────────────────

class GradCAMExplainer:
    """
    Wraps a YOLO backbone to produce Grad-CAM visualisations.

    The YOLO model's backbone is extracted and used as a standard
    PyTorch ClassifierOutputTarget for Grad-CAM.

    Parameters
    ----------
    model_path  : Path to the YOLO .pt weights file.
    device      : 'cuda' or 'cpu' (auto-detected if None).
    input_size  : Resize input to this square size before GradCAM pass.
    alpha       : Heatmap blending alpha (0=original frame, 1=pure heatmap).
    """

    def __init__(
        self,
        model_path: str = "CrashSentinel_Prime.pt",
        device: Optional[str] = None,
        input_size: int = 320,
        alpha: float = 0.55,
    ) -> None:
        self._available = _TORCH_AVAILABLE and _GRADCAM_AVAILABLE
        self.alpha = alpha
        self.input_size = input_size
        self._cam = None

        if not self._available:
            print("⚠️  Grad-CAM unavailable: pytorch-grad-cam or torch not installed.")
            return

        try:
            from ultralytics import YOLO as _YOLO
            yolo = _YOLO(model_path)
            self._backbone = yolo.model.model  # raw nn.Sequential backbone

            # Device
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = device
            self._backbone = self._backbone.to(device).eval()

            # Target layer: second-to-last conv block of backbone
            target_layers = self._get_target_layers()

            self._cam = GradCAM(
                model=self._backbone,
                target_layers=target_layers,
            )

            self._transform = T.Compose([
                T.ToTensor(),
                T.Resize((input_size, input_size)),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
            print(f"✅ Grad-CAM explainer ready on {device}")

        except Exception as e:
            print(f"⚠️  Grad-CAM init failed: {e}")
            self._available = False

    # ── Public API ────────────────────────────────────────────────────────────

    def explain_frame(
        self,
        frame: np.ndarray,
        target_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> np.ndarray:
        """
        Produce Grad-CAM overlay for a frame.

        Parameters
        ----------
        frame       : BGR numpy frame from OpenCV.
        target_bbox : (x1, y1, x2, y2) crop region to explain.
                      If None, uses the full frame.

        Returns
        -------
        np.ndarray : BGR frame with heatmap overlay drawn.
                     Returns original frame if Grad-CAM is unavailable.
        """
        if not self._available or self._cam is None:
            return frame

        try:
            return self._run_gradcam(frame, target_bbox)
        except Exception as e:
            print(f"⚠️  Grad-CAM inference error: {e}")
            return frame

    def is_available(self) -> bool:
        return self._available

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_target_layers(self):
        """Find the last few conv-style layers in the YOLO backbone."""
        target = []
        for m in self._backbone.modules():
            # C2f and SPPF from YOLOv8 contain Conv submodules
            if hasattr(m, "cv2"):   # C2f block
                target = [m.cv2]
        if not target:
            # Fallback: last module with weight params
            for m in self._backbone.modules():
                if hasattr(m, "weight") and hasattr(m, "bias"):
                    target = [m]
        return target

    def _run_gradcam(
        self,
        frame: np.ndarray,
        target_bbox: Optional[Tuple[float, float, float, float]],
    ) -> np.ndarray:
        import torch

        h, w = frame.shape[:2]
        result = frame.copy()

        # ── Determine ROI ──────────────────────────────────────────────────
        if target_bbox:
            x1, y1, x2, y2 = (int(v) for v in target_bbox)
            x1, y1 = max(0, x1 - 20), max(0, y1 - 20)
            x2, y2 = min(w, x2 + 20), min(h, y2 + 20)
            roi = frame[y1:y2, x1:x2]
        else:
            roi = frame
            x1, y1 = 0, 0

        if roi.size == 0:
            return result

        # ── Prepare tensor ────────────────────────────────────────────────
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        rgb_norm = rgb.astype(np.float32) / 255.0
        tensor = self._transform(rgb_norm).unsqueeze(0).to(self._device)

        # ── Grad-CAM pass ─────────────────────────────────────────────────
        grayscale_cam = self._cam(
            input_tensor=tensor,
            targets=None,   # highest-scoring class
        )[0]   # shape: (H, W)

        # ── Overlay on ROI ────────────────────────────────────────────────
        cam_resized = cv2.resize(grayscale_cam, (roi.shape[1], roi.shape[0]))
        cam_overlay = show_cam_on_image(rgb_norm, cam_resized, use_rgb=True)
        cam_bgr = cv2.cvtColor(cam_overlay, cv2.COLOR_RGB2BGR)

        # Blend with original ROI
        blended = cv2.addWeighted(roi, 1 - self.alpha, cam_bgr, self.alpha, 0)
        result[y1:y1 + blended.shape[0], x1:x1 + blended.shape[1]] = blended

        # ── Label ────────────────────────────────────────────────────────
        cv2.putText(result, "Grad-CAM XAI",
                    (x1 + 4, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)
        return result
