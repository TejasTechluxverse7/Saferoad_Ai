import cv2
import os
import time
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import glob
from ultralytics import YOLO

from temporal_module import Detection, TemporalAccidentVerifier

OUTPUT_DIR = "evaluation_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class AegisTestRunner:
    def __init__(self):
        print("🛠️ Initializing Aegis Automated Tester...")
        try:
            self.model_spatial = YOLO("CrashSentinel_Prime.pt")
        except:
            self.model_spatial = None
            
        self.CONFIDENCE_THRESHOLD = 0.50

    def _yolo_to_temporal_detections(self, results):
        detections = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0]) if box.cls is not None else 0
                conf = float(box.conf[0])
                if cls != 0 or conf < self.CONFIDENCE_THRESHOLD:
                    continue 
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                detections.append(Detection(bbox=(x1, y1, x2, y2), confidence=conf))
        return detections

    def test_video(self, video_path):
        vid_name = os.path.basename(video_path).split('.')[0]
        out_video = os.path.join(OUTPUT_DIR, f"{vid_name}_evaluated.mp4")
        
        accident_verifier = TemporalAccidentVerifier()
        no_accident_frames = 0

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open {video_path}")
            return None
            
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_video = int(cap.get(cv2.CAP_PROP_FPS))
        if fps_video == 0: fps_video = 30
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Lock max resolution for testing
        max_w, max_h = 1280, 720
        scale = 1.0
        if w > max_w: scale = max_w / w
        target_w = int(w * scale)
        target_h = int(h * scale)
        
        writer = cv2.VideoWriter(out_video, fourcc, fps_video, (target_w, target_h))

        metrics = {
            "video": vid_name,
            "total_frames": 0,
            "alerts_triggered": [],
            "processing_times_ms": []
        }

        print(f"\n▶️ Testing [{vid_name}]...")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 600:
            print(f"⚠️ Video is > 20 seconds. Capping analysis at 500 frames to save time.")
            frame_count = 500

        while cap.isOpened() and metrics["total_frames"] < frame_count:
            ret, frame = cap.read()
            if not ret: break
            
            if scale != 1.0: frame = cv2.resize(frame, (target_w, target_h))
                
            start_time = time.time()
            alerts = []

            # PHASE 1: AI
            results_spatial = None
            if self.model_spatial:
                try: results_spatial = self.model_spatial(frame, verbose=False)
                except: pass

            # PHASE 2: CANVAS
            annotated_frame = frame.copy()

            # PHASE 3: THREATS
            if results_spatial:
                filtered_detections = self._yolo_to_temporal_detections(results_spatial)
                
                is_accident, temporal_conf = accident_verifier.update(filtered_detections)
                accident_detected_this_frame = False

                for det in filtered_detections:
                    x1, y1, x2, y2 = map(int, det.bbox)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3) 
                    cv2.putText(annotated_frame, f"Accident ({det.confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    accident_detected_this_frame = True

                if is_accident:
                    bbox_to_alert = filtered_detections[-1].bbox if filtered_detections else (0,0,0,0)
                    alerts.append({"type": "VEHICLE_ACCIDENT_VERIFIED", "risk": temporal_conf, "bbox": bbox_to_alert})

                if accident_detected_this_frame: no_accident_frames = 0
                else: no_accident_frames += 1
                if no_accident_frames >= 30: accident_verifier.reset()

            # PHASE 4: VISUALS
            y_offset = 50
            for alert in alerts:
                alert_type = alert['type']
                color = (0, 0, 255) if "ACCIDENT" in alert_type else (255, 255, 255)
                
                txt = f"{alert_type} ({alert['risk']:.2f})"
                cv2.putText(annotated_frame, txt, (30, y_offset), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
                y_offset += 35
                
                metrics["alerts_triggered"].append({
                    "frame": metrics["total_frames"],
                    "type": alert_type,
                    "risk": alert['risk']
                })

            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            metrics["processing_times_ms"].append(latency_ms)
            metrics["total_frames"] += 1
            
            writer.write(annotated_frame)
            
            if metrics["total_frames"] % 50 == 0:
                print(f"   ... Analyzed {metrics['total_frames']}/{frame_count} frames")

        cap.release()
        writer.release()
        
        avg_latency = sum(metrics["processing_times_ms"]) / max(1, len(metrics["processing_times_ms"]))
        metrics["avg_latency_ms"] = avg_latency
        metrics["avg_fps"] = 1000.0 / avg_latency if avg_latency > 0 else 0
        
        print(f"✅ Finished [{vid_name}]. Avg Inference FPS: {metrics['avg_fps']:.1f}")
        return metrics

if __name__ == "__main__":
    runner = AegisTestRunner()
    all_results = []
    
    videos = glob.glob("test_videos/*.mp4")
    if not videos:
        print("❌ No videos found in test_videos directory! Run fetch script first.")
    
    for vid in videos:
        res = runner.test_video(vid)
        if res: all_results.append(res)
        
    with open(os.path.join(OUTPUT_DIR, "test_summary.json"), 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print("\n🏁 Full Test Suite Completed!")
    print(f"Results saved to {OUTPUT_DIR}/")
