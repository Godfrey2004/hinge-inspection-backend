import os
import cv2
import asyncio
from ultralytics import YOLO

class HingeDetector:
    def __init__(self, model_path="best.pt"):
        """
        Initialize the AI Hinge Detector and load the YOLO model.
        """
        if not os.path.exists(model_path):
            print(f"Warning: Model file {model_path} not found. Ensure it is placed in the backend folder.")
        
        self.model = YOLO(model_path) if os.path.exists(model_path) else None

    def process_video_sync(self, input_path: str, output_path: str) -> dict:
        """
        Synchronous video processing logic using OpenCV and YOLO.
        """
        if not self.model:
            raise Exception("YOLO model not loaded. Please provide 'best.pt'.")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception(f"Cannot open input video {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30
            
        # Use mp4v codec for standard .mp4 output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Overall analytics state
        left_hinge_detected = False
        right_hinge_detected = False
        overall_max_conf = 0.0
        # Cache last known detections to draw on skipped frames
        last_boxes = []
        
        # Timeline for live UI sync
        timeline = []
        last_state = None

        green = (0, 255, 0)
        red = (0, 0, 255)
        blue = (255, 0, 0)

        # ── Speed optimisation ────────────────────────────────────────────
        # Process only 1 in every FRAME_SKIP frames through YOLO.
        # Skipped frames still get the HUD drawn using the cached detections.
        FRAME_SKIP = 5
        frame_idx = 0
        # Inference at a smaller size is much faster on CPU (imgsz=320 vs 640)
        INFER_SIZE = 320
        # Camera offset calibration: the gap between hinges is at ~62% of the frame width
        SPLIT_RATIO = 0.62
        # ──────────────────────────────────────────────────────────────────

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_left_detected = False
            frame_right_detected = False

            if frame_idx % FRAME_SKIP == 0:
                # Run YOLO inference on this key frame
                results = self.model(frame, verbose=False, imgsz=INFER_SIZE)
                last_boxes = []

                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])

                        if conf > overall_max_conf:
                            overall_max_conf = conf

                        x_center = (x1 + x2) / 2
                        is_left = x_center < (width * SPLIT_RATIO)

                        if is_left:
                            frame_left_detected = True
                            left_hinge_detected = True
                        else:
                            frame_right_detected = True
                            right_hinge_detected = True

                        last_boxes.append({
                            "coords": (x1, y1, x2, y2),
                            "label": f"{'LEFT' if is_left else 'RIGHT'} {conf:.2f}",
                            "color": green,
                        })
            else:
                # Skipped frame — reuse cached detections
                for b in last_boxes:
                    if "LEFT" in b["label"]:
                        frame_left_detected = True
                    else:
                        frame_right_detected = True

            # Draw bounding boxes (cached or freshly detected)
            for b in last_boxes:
                x1, y1, x2, y2 = b["coords"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), b["color"], 3)
                cv2.putText(frame, b["label"], (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, b["color"], 2)

            # Draw dividing line
            split_x = int(width * SPLIT_RATIO)
            cv2.line(frame, (split_x, 0), (split_x, height), blue, 2)

            # HUD overlay
            l_status = "OK" if frame_left_detected else "MISSING"
            r_status = "OK" if frame_right_detected else "MISSING"
            l_color  = green if frame_left_detected else red
            r_color  = green if frame_right_detected else red
            overall_status = "PASS" if (frame_left_detected and frame_right_detected) else "FAIL"
            o_color = green if overall_status == "PASS" else red

            cv2.putText(frame, f"LEFT: {l_status}",         (30, 50),          cv2.FONT_HERSHEY_SIMPLEX, 1, l_color, 2)
            cv2.putText(frame, f"RIGHT: {r_status}",        (width - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, r_color, 2)
            cv2.putText(frame, f"INSPECTION: {overall_status}", (30, 100),     cv2.FONT_HERSHEY_SIMPLEX, 1, o_color, 2)

            out.write(frame)
            
            # Record state changes for live UI timeline
            current_state = (frame_left_detected, frame_right_detected)
            if current_state != last_state:
                timeline.append({
                    "time": round(frame_idx / fps, 2),
                    "left": frame_left_detected,
                    "right": frame_right_detected
                })
                last_state = current_state

            frame_idx += 1

        cap.release()
        out.release()
        
        # --- Convert video to Web-safe H.264 using FFmpeg ---
        try:
            import imageio_ffmpeg
            import subprocess
            import shutil
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            temp_output = output_path.replace(".mp4", "_temp.mp4")
            shutil.move(output_path, temp_output)
            subprocess.run([
                ffmpeg_path, "-y", "-i", temp_output, 
                "-vcodec", "libx264", "-f", "mp4", output_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.remove(temp_output)
        except Exception as e:
            print(f"FFmpeg conversion failed: {e}")
        # ----------------------------------------------------

        
        # Calculate final analytics based on the entire video
        total_hinges = 0
        if left_hinge_detected: total_hinges += 1
        if right_hinge_detected: total_hinges += 1
        
        final_inspection = "PASS" if total_hinges == 2 else "FAIL"

        return {
            "left_hinge": "OK" if left_hinge_detected else "MISSING",
            "right_hinge": "OK" if right_hinge_detected else "MISSING",
            "inspection": final_inspection,
            "total_hinges": total_hinges,
            "confidence": int(overall_max_conf * 100),
            "output_video": os.path.basename(output_path),
            "timeline": timeline
        }

    async def process_video(self, input_path: str, output_path: str) -> dict:
        """
        Asynchronously process the video by running the synchronous OpenCV logic in a thread pool.
        """
        return await asyncio.to_thread(self.process_video_sync, input_path, output_path)

# Singleton instance to be used by the router
detector = HingeDetector()
