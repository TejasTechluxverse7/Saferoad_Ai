"""
SaferoadAI – Engine Package
Contains severity classifier, evidence buffer, forensics engine,
Grad-CAM explainer, and victim detector.
"""
from .severity_classifier import SeverityClassifier, SeverityLevel
from .evidence_buffer import EvidenceBuffer
from .forensics_engine import ForensicsEngine
from .gradcam import GradCAMExplainer
from .victim_detector import VictimDetector
from .temporal_module import Detection, TemporalAccidentVerifier

__all__ = [
    "SeverityClassifier", "SeverityLevel",
    "EvidenceBuffer",
    "ForensicsEngine",
    "GradCAMExplainer",
    "VictimDetector",
    "Detection", "TemporalAccidentVerifier",
]
