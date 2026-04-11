import cv2
import os
import argparse
from typing import List
from ultralytics import YOLO

from temporal_module import Detection, TemporalAccidentVerifier

class AegisVisionPipeline:
    def __init__(self, source):
        print("🛡️ Initializing AegisVision: Dedicated Vehicle Accident Intelligence...")
        
        # Vehicle Accident Spatial Model (CityGuard trained model)
        try:
            self.model_spatial = YOLO("CrashSentinel_Prime.pt")
        except Exception as e:
            print(f"❌ Error loading accident model: {e}")
            self.model_spatial = None
            
        self.accident_verifier = TemporalAccidentVerifier()
        self.source = source
        self.CONFIDENCE_THRESHOLD = 0.50 
        self.NO_ACCIDENT_THRESHOLD = 30
        self.accident_active = False
        self.no_accident_frames = 0

    def _yolo_to_temporal_detections(self, results) -> List[Detection]:
        detections: List[Detection] = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0]) if box.cls is not None else 0
                conf = float(box.conf[0])
                if cls != 0 or conf < self.CONFIDENCE_THRESHOLD:
                    continue 
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                detections.append(Detection(bbox=(x1, y1, x2, y2), confidence=conf))
        return detections

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened(): return
            
        print("✅ Engine active. Running Accident Verification loop...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            alerts = []
            results_spatial = None
            
            if self.model_spatial is not None:
                try:
                    results_spatial = self.model_spatial(frame, verbose=False)
                except: pass

            annotated_frame = frame.copy()

            if results_spatial:
                temporal_detections = self._yolo_to_temporal_detections(results_spatial)
                is_accident, temporal_conf = self.accident_verifier.update(temporal_detections)
                accident_detected_this_frame = False

                for det in temporal_detections:
                    x1, y1, x2, y2 = map(int, det.bbox)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3) 
                    cv2.putText(annotated_frame, f"Accident ({det.confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    accident_detected_this_frame = True

                if is_accident:
                    self.accident_active = True
                    bbox_to_alert = temporal_detections[-1].bbox if temporal_detections else (0,0,0,0)
                    alerts.append({
                        "type": "VEHICLE_ACCIDENT_VERIFIED", 
                        "risk": temporal_conf, 
                        "bbox": bbox_to_alert
                    })

                if accident_detected_this_frame:
                    self.no_accident_frames = 0
                else:
                    self.no_accident_frames += 1

                if self.no_accident_frames >= self.NO_ACCIDENT_THRESHOLD:
                    self.accident_active = False
                    self.accident_verifier.reset()

            y_offset = 50
            for alert in alerts:
                alert_type = alert['type']
                color = (0, 0, 255) # Red 
                
                text = f"[PRIORITY ALERT] {alert_type} (Risk: {alert['risk']:.2f})"
                cv2.putText(annotated_frame, text, (30, y_offset), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
                y_offset += 35
                
                x1, y1, x2, y2 = map(int, alert['bbox'])
                if sum(alert['bbox']) > 0: 
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 5)
                    cv2.putText(annotated_frame, alert_type, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
            
            cv2.imshow("🛡️ AegisVision Threat Intelligence Platform", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        print("Shutting down AegisVision Engine.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="sample_videos/video1.mp4", help="Path to video or RTSP stream")
    args = parser.parse_args()
    
    pipeline = AegisVisionPipeline(args.source)
    pipeline.run()
