"""
Video output script — draws bounding boxes around vehicles and plates,
overlays detected plate numbers, and saves annotated video to output folder.

Usage:
  python scripts/video_output.py --source ALPR.mp4
  python scripts/video_output.py --source mycarplate.mp4 --output output/annotated.mp4
"""

import argparse
import sys
import os
from pathlib import Path
from collections import defaultdict
import csv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="ALPR Video Output with Bounding Boxes")
    parser.add_argument("--source", required=True, help="Input video file")
    parser.add_argument("--output", default=None, help="Output video path")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    from src.utils.config import load_config
    from src.utils.logger import get_logger
    from src.detection.vehicle_detector import VehicleDetector
    from src.detection.plate_detector import PlateDetector
    from src.ocr import create_ocr_engine
    from src.validation.plate_validator import PlateValidator
    from src.ocr.ocr_postprocessor import remove_noise_characters

    config = load_config(args.config)
    log = get_logger("video_output", config=config)
    det_cfg = config["detection"]

    # Output path
    source_name = Path(args.source).stem
    output_path = args.output or f"output/{source_name}_annotated.mp4"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading models...")
    # Use from_config() (not the old positional 2-arg constructor) so
    # imgsz/device/half stay in sync with config.yaml's detection section --
    # vehicle_model_path/plate_model_path may point at pre-exported
    # .engine/.onnx plans whose expected imgsz is config-driven, not the
    # class defaults. Confidence thresholds here are intentionally more
    # permissive than config.yaml for this visualization tool, so override
    # them after construction.
    vehicle_detector = VehicleDetector.from_config(config)
    vehicle_detector.confidence_threshold = 0.2
    plate_detector = PlateDetector.from_config(config)
    plate_detector.confidence_threshold = 0.15
    ocr_backend      = str(config.get("ocr", {}).get("backend", "paddleocr"))
    ocr_engine       = create_ocr_engine(ocr_backend, config)
    plate_validator  = PlateValidator()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        log.error("Cannot open: %s", args.source)
        sys.exit(1)

    # Get video properties
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    log.info("Processing %s (%dx%d @ %.1f fps, %d frames)", args.source, width, height, fps, total)

    # Cache: vehicle bbox -> last known plate text
    plate_cache: dict[tuple, str] = {}
    frame_idx = 0
    
    # CSV data collection
    detections_data = []

    # Colors
    COLOR_VEHICLE = (0, 255, 0)      # Green for vehicle
    COLOR_PLATE   = (0, 165, 255)    # Orange for plate
    COLOR_TEXT_BG = (0, 0, 0)        # Black background for text
    COLOR_TEXT    = (255, 255, 255)  # White text

    def draw_label(img, text, x, y, color_bg, color_text, font_scale=0.6, thickness=2):
        """Draw text with background box."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(img, (x, y - th - baseline - 4), (x + tw + 4, y + baseline), color_bg, -1)
        cv2.putText(img, text, (x + 2, y - 2), font, font_scale, color_text, thickness)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % 50 == 0:
            log.info("Frame %d / %d", frame_idx, total)

        # Detect vehicles
        detections = vehicle_detector.detect(frame)

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            h, w = frame.shape[:2]

            # Draw vehicle bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_VEHICLE, 2)
            draw_label(frame, f"{det.class_label} {det.confidence:.2f}",
                       x1, y1, COLOR_VEHICLE, COLOR_TEXT_BG)

            # Crop vehicle
            vehicle_crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
            if vehicle_crop.size == 0:
                continue

            # Detect plates
            plate_boxes = []
            try:
                from ultralytics import YOLO
                if not hasattr(main, '_plate_model'):
                    main._plate_model = YOLO(det_cfg["plate_model_path"])
                results = main._plate_model(vehicle_crop, verbose=False, conf=0.15)
                for r in results:
                    if r.boxes is not None:
                        for box in r.boxes:
                            px1, py1, px2, py2 = [int(v) for v in box.xyxy[0]]
                            plate_boxes.append((px1, py1, px2, py2))
            except Exception:
                pass

            # Fallback: use bottom 35% of vehicle
            if not plate_boxes:
                hv = vehicle_crop.shape[0]
                plate_boxes = [(0, int(hv * 0.65), vehicle_crop.shape[1], hv)]

            for px1, py1, px2, py2 in plate_boxes:
                # Draw plate box on original frame (offset by vehicle position)
                abs_px1 = x1 + px1
                abs_py1 = y1 + py1
                abs_px2 = x1 + px2
                abs_py2 = y1 + py2
                cv2.rectangle(frame, (abs_px1, abs_py1), (abs_px2, abs_py2), COLOR_PLATE, 2)

                # Crop plate
                plate_crop = vehicle_crop[max(0,py1):min(vehicle_crop.shape[0],py2),
                                           max(0,px1):min(vehicle_crop.shape[1],px2)]
                if plate_crop.size == 0:
                    continue

                # Upscale for OCR
                ph, pw = plate_crop.shape[:2]
                if pw < 200:
                    scale = max(200 / pw, 2.0)
                    ocr_input = cv2.resize(plate_crop, None, fx=scale, fy=scale,
                                           interpolation=cv2.INTER_CUBIC)
                else:
                    ocr_input = plate_crop

                # OCR
                raw_text, confidence = ocr_engine.recognize(ocr_input)
                if raw_text and confidence >= 0.40:
                    raw_text = remove_noise_characters(raw_text)
                    plate_number, _ = plate_validator.validate(raw_text)
                    if plate_number:
                        plate_cache[(x1, y1, x2, y2)] = plate_number
                        draw_label(frame, plate_number,
                                   abs_px1, abs_py1 - 5,
                                   COLOR_PLATE, COLOR_TEXT, font_scale=0.7, thickness=2)
                        
                        # Collect detection data for CSV
                        detections_data.append({
                            'frame': frame_idx,
                            'timestamp': frame_idx / fps,
                            'vehicle_x1': x1,
                            'vehicle_y1': y1,
                            'vehicle_x2': x2,
                            'vehicle_y2': y2,
                            'vehicle_class': det.class_label,
                            'vehicle_confidence': f"{det.confidence:.2f}",
                            'plate_x1': abs_px1,
                            'plate_y1': abs_py1,
                            'plate_x2': abs_px2,
                            'plate_y2': abs_py2,
                            'plate_number': plate_number,
                            'ocr_confidence': f"{confidence:.2f}"
                        })

        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        writer.write(frame)

    cap.release()
    writer.release()
    log.info("✅ Annotated video saved to: %s", output_path)
    
    # Save detections to CSV
    csv_path = Path(output_path).with_suffix('.csv')
    if detections_data:
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['frame', 'timestamp', 'vehicle_x1', 'vehicle_y1', 'vehicle_x2', 'vehicle_y2',
                         'vehicle_class', 'vehicle_confidence', 'plate_x1', 'plate_y1', 'plate_x2', 'plate_y2',
                         'plate_number', 'ocr_confidence']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detections_data)
        log.info("✅ CSV file saved to: %s", csv_path)
        print(f"CSV file saved to: {csv_path}")
    else:
        log.warning("No detections found")
        print("No detections found")
    
    print(f"\nDone! Video saved to: {output_path}")


if __name__ == "__main__":
    main()
