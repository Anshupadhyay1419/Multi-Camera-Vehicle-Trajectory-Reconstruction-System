"""
Test the plate detection model on test_2_images folder.
For each image:
  - Detects license plates using YOLOv8m
  - Runs OCR to read plate number
  - Saves annotated image with bounding box
  - Saves JSON with bbox coordinates and plate number

Usage:
  python scripts/test_model.py
  python scripts/test_model.py --images test_2_images/test/images --output output/test_2_output
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def deskew_plate(plate_crop: np.ndarray) -> np.ndarray:
    """Detect and correct tilt in a license plate crop."""
    try:
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY) if len(plate_crop.shape) == 3 else plate_crop.copy()
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=max(30, min(plate_crop.shape[:2]) // 3))

        if lines is None or len(lines) == 0:
            return plate_crop

        angles = []
        for line in lines[:20]:
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            if -45 <= angle <= 45:
                angles.append(angle)

        if not angles:
            return plate_crop

        median_angle = float(np.median(angles))
        if abs(median_angle) < 1.0:
            return plate_crop

        h, w = plate_crop.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)

        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        rotated = cv2.warpAffine(plate_crop, M, (new_w, new_h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        return rotated

    except Exception:
        return plate_crop


def preprocess_plate(plate_crop: np.ndarray, target_width: int = 400) -> np.ndarray:
    """Full preprocessing pipeline for a plate crop."""
    if plate_crop is None or plate_crop.size == 0:
        return plate_crop

    h, w = plate_crop.shape[:2]

    if w > 800:
        scale = 800 / w
        plate_crop = cv2.resize(plate_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    elif w < target_width:
        scale = target_width / w
        plate_crop = cv2.resize(plate_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    plate_crop = deskew_plate(plate_crop)

    lab = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    plate_crop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return plate_crop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="test_2_images/test/images", help="Input images folder")
    parser.add_argument("--output", default="output/test_2_output", help="Output folder")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    from src.utils.config import load_config
    from src.utils.logger import get_logger
    from src.detection.plate_detector import PlateDetector
    from src.ocr.paddle_ocr_engine import PaddleOCREngine
    from src.validation.plate_validator import PlateValidator
    from src.ocr.ocr_postprocessor import remove_noise_characters

    config = load_config(args.config)
    log = get_logger("test_model", config=config)
    det_cfg = config["detection"]

    # Create output folder
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading models...")
    # from_config() (not the old positional 2-arg constructor) so
    # imgsz/device/half stay in sync with config.yaml's detection section --
    # plate_model_path may point at a pre-exported .engine/.onnx plan whose
    # expected imgsz is config-driven, not the class default. This tool's
    # confidence threshold is intentionally more permissive than
    # config.yaml, so override it after construction.
    plate_detector = PlateDetector.from_config(config)
    plate_detector.confidence_threshold = 0.15
    ocr_engine     = PaddleOCREngine(use_angle_cls=True, lang="en")
    plate_validator = PlateValidator()

    # Get all images
    images_dir = Path(args.images)
    image_files = sorted(
        list(images_dir.glob("*.jpg")) +
        list(images_dir.glob("*.jpeg")) +
        list(images_dir.glob("*.png")),
        key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem
    )

    log.info("Found %d images in %s", len(image_files), images_dir)

    total_detected = 0
    total_ocr_success = 0

    for idx, img_path in enumerate(image_files, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            log.warning("Cannot read: %s", img_path)
            continue

        h, w = img.shape[:2]
        annotated = img.copy()

        # Detect plates
        plate_crops = plate_detector.detect(img)

        # Fallback: use bottom 40% of image if no plate detected
        if not plate_crops:
            bottom = img[int(h * 0.6):, :]
            if bottom.size > 0:
                plate_crops_fallback = [bottom]
                fallback_bboxes = [(0, int(h * 0.6), w, h)]
            else:
                plate_crops_fallback = []
                fallback_bboxes = []
        else:
            plate_crops_fallback = None
            fallback_bboxes = None

        detections = []

        # Process plate detector results
        if plate_crops:
            from ultralytics import YOLO
            if not hasattr(main, '_model'):
                main._model = YOLO(det_cfg["plate_model_path"])
            results = main._model(img, verbose=False, conf=0.15)
            for r in results:
                if r.boxes is None:
                    continue
                for i, box in enumerate(r.boxes):
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                    conf = float(box.conf[0])
                    plate_crop = img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                    if plate_crop.size == 0:
                        continue

                    ocr_input = preprocess_plate(plate_crop, target_width=400)

                    raw_text, ocr_conf = ocr_engine.recognize(ocr_input)
                    plate_number = None
                    if raw_text and ocr_conf >= 0.40:
                        raw_text = remove_noise_characters(raw_text)
                        raw_text = raw_text.replace("IND", "").replace("INDIA", "").strip()
                        plate_number, _ = plate_validator.validate(raw_text)
                        if not plate_number and len(raw_text) >= 8:
                            for length in range(11, 7, -1):
                                for start in range(len(raw_text) - length + 1):
                                    candidate = raw_text[start:start+length]
                                    p, _ = plate_validator.validate(candidate)
                                    if p:
                                        plate_number = p
                                        break
                                if plate_number:
                                    break

                    detections.append({
                        "bbox": {
                            "xmin": x1, "ymin": y1,
                            "xmax": x2, "ymax": y2,
                            "x_center": round((x1 + x2) / 2 / w, 4),
                            "y_center": round((y1 + y2) / 2 / h, 4),
                            "width_norm": round((x2 - x1) / w, 4),
                            "height_norm": round((y2 - y1) / h, 4),
                        },
                        "plate_number": plate_number or raw_text or "",
                        "ocr_confidence": round(ocr_conf, 3),
                        "detection_confidence": round(conf, 3),
                    })

                    # Draw on annotated image
                    color = (0, 165, 255)  # Orange
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = plate_number or raw_text or "?"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
                    cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(annotated, label, (x1 + 2, y1 - 4), font, 0.6, (0, 0, 0), 2)

                    if plate_number:
                        total_ocr_success += 1
                    total_detected += 1

        # Use fallback if no plate detector results
        elif plate_crops_fallback:
            for (fx1, fy1, fx2, fy2), crop in zip(fallback_bboxes, plate_crops_fallback):
                ocr_input = preprocess_plate(crop, target_width=400)
                raw_text, ocr_conf = ocr_engine.recognize(ocr_input)
                plate_number = None
                if raw_text and ocr_conf >= 0.40:
                    raw_text = remove_noise_characters(raw_text)
                    raw_text = raw_text.replace("IND", "").replace("INDIA", "").strip()
                    plate_number, _ = plate_validator.validate(raw_text)
                    if not plate_number and len(raw_text) >= 8:
                        for length in range(11, 7, -1):
                            for start in range(len(raw_text) - length + 1):
                                candidate = raw_text[start:start+length]
                                p, _ = plate_validator.validate(candidate)
                                if p:
                                    plate_number = p
                                    break
                            if plate_number:
                                break

                if plate_number or (raw_text and ocr_conf >= 0.40):
                    detections.append({
                        "bbox": {
                            "xmin": fx1, "ymin": fy1,
                            "xmax": fx2, "ymax": fy2,
                            "x_center": round((fx1 + fx2) / 2 / w, 4),
                            "y_center": round((fy1 + fy2) / 2 / h, 4),
                            "width_norm": round((fx2 - fx1) / w, 4),
                            "height_norm": round((fy2 - fy1) / h, 4),
                        },
                        "plate_number": plate_number or raw_text or "",
                        "ocr_confidence": round(ocr_conf, 3),
                        "detection_confidence": 0.0,
                        "note": "fallback_detection",
                    })
                    total_detected += 1
                    if plate_number:
                        total_ocr_success += 1

        # Build JSON result
        result = {
            "image": img_path.name,
            "image_width": w,
            "image_height": h,
            "plates_detected": len(detections),
            "detections": detections,
        }

        # Save JSON
        json_path = output_dir / f"{img_path.stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        # Save annotated image
        out_img_path = output_dir / img_path.name
        cv2.imwrite(str(out_img_path), annotated)

        if idx % 10 == 0 or idx == len(image_files):
            log.info("Processed %d/%d | Detected: %d | OCR success: %d",
                     idx, len(image_files), total_detected, total_ocr_success)

    # Summary JSON
    summary = {
        "total_images": len(image_files),
        "total_plates_detected": total_detected,
        "total_ocr_success": total_ocr_success,
        "ocr_success_rate": f"{total_ocr_success/max(total_detected,1)*100:.1f}%",
        "output_folder": str(output_dir),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"  Images processed : {len(image_files)}")
    print(f"  Plates detected  : {total_detected}")
    print(f"  OCR success      : {total_ocr_success}")
    print(f"  Output folder    : {output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
