import os
import cv2
import sys
import time
import numpy as np
from pathlib import Path

# Fix paths
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

errors = []
results = []

def log(feature, status, msg=""):
    mark = "[OK]  " if status else "[FAIL]"
    out = f"{mark} {feature:25} | {msg}"
    print(out)
    results.append(out)
    if not status:
        errors.append((feature, msg))

def run_tests():
    print("AegisRoad AI - Feature Verification\n" + "="*45)

    base_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.putText(base_frame, "TEST", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    
    # ── 1. Model Loading ──
    try:
        from ultralytics import YOLO
        model_path = "models/aegis_model.pt" if os.path.exists("models/aegis_model.pt") else "CrashSentinel_Prime.pt"
        if not os.path.exists(model_path):
            model_path = "yolov8n.pt"  # Fallback
        model = YOLO(model_path)
        log("1. YOLO Detection", True, f"Loaded {model_path}")
    except Exception as e:
        log("1. YOLO Detection", False, str(e))
        return

    # ── 2. ByteTrack ──
    try:
        from tracking.bytetrack_wrapper import ByteTrackWrapper
        tracker = ByteTrackWrapper()
        # Mocking detection
        res = model(base_frame, verbose=False)
        tracked = tracker.update(res, base_frame)
        log("2. ByteTrack Tracking", True, f"Tracker initialized, processed 1 frame")
    except Exception as e:
        log("2. ByteTrack Tracking", False, str(e))

    # ── 3. Predictions / Near-Miss ──
    try:
        from prediction.trajectory_engine import TrajectoryEngine
        traj = TrajectoryEngine()
        # Update with empty tracks
        nm = traj.update([], time.time())
        log("3. Near-Miss Prediction", True, "Engine idle run successful")
    except Exception as e:
        log("3. Near-Miss Prediction", False, str(e))

    # ── 4. Severity Classifier ──
    try:
        from engine.severity_classifier import SeverityClassifier, SeverityLevel
        sev = SeverityClassifier()
        res = sev.classify(speed_px_s=45.0, bbox_a=(10,10,50,50), bbox_b=(10,10,30,30), vehicle_count=2, temporal_conf=0.8)
        log("4. Severity Classifier", True, f"Evaluated mock overlap -> Score: {res.score:.2f}, {res.level.value}")
    except Exception as e:
        log("4. Severity Classifier", False, str(e))

    # ── 5. Evidence Buffer ──
    try:
        from engine.evidence_buffer import EvidenceBuffer
        buf = EvidenceBuffer()
        buf.push(base_frame, [{"bbox": [10,10,50,50]}], time.time())
        log("5. Evidence Buffer", True, "Added frame successfully")
    except Exception as e:
        log("5. Evidence Buffer", False, str(e))

    # ── 6. Grad-CAM XAI ──
    try:
        from engine.gradcam import GradCAMExplainer
        cam = GradCAMExplainer(model_path=model_path, device="cpu")
        out = cam.explain_frame(base_frame, target_bbox=(10,10,100,100))
        log("6. Grad-CAM XAI", True, "Generated heatmap successfully")
    except Exception as e:
        log("6. Grad-CAM XAI", False, str(e))

    # ── 7. Victim Detector ──
    try:
        from engine.victim_detector import VictimDetector
        vd = VictimDetector()
        res = model(base_frame, verbose=False) # Get YOLO results wrapper
        victims = vd.detect(res, [(10,10,100,100)]) 
        log("7. Victim Detection", True, f"Ran detection logic safely. Victims: {len(victims)}")
    except Exception as e:
        log("7. Victim Detection", False, str(e))

    # ── 8. Ambulance Router ──
    try:
        from api.routing import AmbulanceRouter
        router = AmbulanceRouter()
        route = router.route(28.5439, 77.3305) # dummy coords
        log("8. Smart Routing (A*)", True, f"Found route to {route.get('hospital_name', 'Unknown')} in {route.get('eta_minutes',0)} min")
    except Exception as e:
        log("8. Smart Routing (A*)", False, str(e))

    # ── 9. VANET Alerts ──
    try:
        from vanet.vanet_layer import build_accident_alert
        alert_payload = build_accident_alert("28.5439, 77.3305", 0.95, "cam1")
        log("9. VANET Data Sync", True, "Alert payload generated successfully")
    except Exception as e:
        log("9. VANET Data Sync", False, str(e))

    # ── 10. Dashboard API (Basic check) ──
    try:
        import requests
        # We don't want to block, just check if requests is there and we can import flask stuff
        from api.saferoad_api import app, _allowed_file
        log("10. Web Dashboard API", True, "Flask modules imported safely")
    except Exception as e:
        log("10. Web Dashboard API", False, str(e))
        
    print("\n" + "="*45)
    print(f"Total passing: {len(results)-len(errors)}/{len(results)}")
    
    with open("feature_check.md", "w", encoding="utf-8") as f:
        f.write("# AegisRoad AI Verification Results\n\n")
        for r in results:
            f.write(f"- {r}\n")

if __name__ == '__main__':
    run_tests()
