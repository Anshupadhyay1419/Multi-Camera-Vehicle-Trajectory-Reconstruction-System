#!/usr/bin/env python
"""
Semi-automated plate annotation tool for Phase 1 dataset collection.

Usage:
  python scripts/annotate_plates.py --input data/raw/ --output data/annotated/

Generates JSON annotations for each plate crop using PaddleOCR + operator corrections.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Try to import PIL for image quality assessment
try:
    from PIL import ImageFilter, ImageStat
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class PlateAnnotator:
    """Semi-automated plate annotation using OCR + operator input."""

    def __init__(self, ocr_engine=None):
        """Initialize annotator with optional OCR engine."""
        self.ocr_engine = ocr_engine
        if ocr_engine is None:
            try:
                from src.ocr.paddle_ocr_engine import PaddleOCREngine
                self.ocr_engine = PaddleOCREngine()
            except ImportError:
                print("Warning: PaddleOCR not available. Using manual entry only.")

    def assess_image_quality(self, image_path: str) -> dict:
        """Assess image quality (blur, contrast, brightness)."""
        img = cv2.imread(image_path)
        if img is None:
            return {"blur_score": 0.0, "contrast": 0.0, "brightness": 0.0}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Laplacian variance (blur detection)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(1.0, laplacian_var / 500.0)  # Normalize to 0-1

        # Contrast (standard deviation of intensity)
        contrast = gray.std() / 128.0  # Normalize

        # Brightness (mean intensity)
        brightness = gray.mean() / 255.0

        return {
            "blur_score": float(blur_score),
            "contrast": float(contrast),
            "brightness": float(brightness),
        }

    def run_ocr(self, image_path: str) -> tuple[str, float]:
        """Run OCR on image, return (text, confidence)."""
        if self.ocr_engine is None:
            return ("", 0.0)

        try:
            result = self.ocr_engine.recognize(cv2.imread(image_path))
            return (result[0] if result else "", result[1] if len(result) > 1 else 0.0)
        except Exception as e:
            print(f"OCR error: {e}")
            return ("", 0.0)

    def create_annotation(
        self,
        image_path: str,
        vehicle_id: str,
        plate_number: str = None,
        plate_series: str = "normal",
        plate_quality: str = "A",
        weather: str = "clear",
        lighting: str = "daylight",
        vehicle_type: str = "Car",
        vehicle_color: str = "Unknown",
        annotator: str = "operator",
    ) -> dict:
        """Create annotation JSON for a plate image."""

        # Auto-detect if not provided
        if plate_number is None or not plate_number.strip():
            ocr_text, ocr_conf = self.run_ocr(image_path)
            plate_number = ocr_text
            ocr_confidence = ocr_conf
        else:
            ocr_text = plate_number
            ocr_confidence = 0.0

        # Assess quality
        quality_metrics = self.assess_image_quality(image_path)

        return {
            "image_file": Path(image_path).name,
            "vehicle_id": vehicle_id,
            "plate_number": plate_number.upper(),
            "plate_series": plate_series,
            "plate_color": "Yellow" if self._is_commercial(plate_number) else "White",
            "plate_quality": plate_quality,
            "weather": weather,
            "lighting": lighting,
            "vehicle_type": vehicle_type,
            "vehicle_color": vehicle_color,
            "image_quality": quality_metrics.get("blur_score", 0.0),
            "ocr_confidence": float(ocr_confidence),
            "ocr_raw_text": ocr_text,
            "annotation_date": datetime.now().isoformat().split("T")[0],
            "annotator": annotator,
            "notes": "",
        }

    @staticmethod
    def _is_commercial(plate: str) -> bool:
        """Heuristic: commercial plates often have specific prefixes."""
        # Indian commercial plates often have specific patterns
        # This is a placeholder — adjust based on actual observations
        return False


def batch_annotate(
    input_dir: str,
    output_dir: str,
    ocr_backend: str = "paddleocr",
    interactive: bool = False,
):
    """Batch annotate all plate images in input_dir."""

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize annotator
    ocr_engine = None
    if ocr_backend == "paddleocr":
        try:
            from src.ocr.paddle_ocr_engine import PaddleOCREngine
            ocr_engine = PaddleOCREngine()
            print("✓ PaddleOCR loaded")
        except ImportError:
            print("⚠ PaddleOCR not available, using OCR text suggestion only")

    annotator = PlateAnnotator(ocr_engine)

    # Process all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [
        f
        for f in input_path.rglob("*")
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    print(f"Found {len(image_files)} images to annotate")

    for idx, image_file in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] Processing: {image_file.name}")

        vehicle_id = f"V_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx:03d}"
        ocr_text, ocr_conf = annotator.run_ocr(str(image_file))

        # Interactive mode: prompt operator for corrections
        if interactive:
            print(f"  OCR suggestion: {ocr_text} (confidence: {ocr_conf:.2f})")
            plate_number = input("  Enter correct plate number (or press Enter to accept): ").strip() or ocr_text
        else:
            plate_number = ocr_text

        plate_series = input("  Plate series (normal/commercial/temporary) [normal]: ").strip() or "normal"
        plate_quality = input("  Plate quality (A/B/C) [A]: ").strip() or "A"
        weather = input("  Weather (clear/cloudy/rain/fog) [clear]: ").strip() or "clear"
        lighting = input("  Lighting (daylight/evening/night) [daylight]: ").strip() or "daylight"
        vehicle_type = input("  Vehicle type (Car/SUV/Commercial/2-wheeler/Auto) [Car]: ").strip() or "Car"
        notes = input("  Notes (optional): ").strip()

        # Create annotation
        annotation = annotator.create_annotation(
            image_path=str(image_file),
            vehicle_id=vehicle_id,
            plate_number=plate_number,
            plate_series=plate_series,
            plate_quality=plate_quality,
            weather=weather,
            lighting=lighting,
            vehicle_type=vehicle_type,
            annotator="operator",
        )

        if notes:
            annotation["notes"] = notes

        # Save annotation
        annotation_file = output_path / f"{image_file.stem}.json"
        with open(annotation_file, "w") as f:
            json.dump(annotation, f, indent=2)

        print(f"  ✓ Saved: {annotation_file.name}")


def main():
    parser = argparse.ArgumentParser(description="Semi-automated plate annotation tool")
    parser.add_argument("--input", required=True, help="Input directory with plate images")
    parser.add_argument("--output", required=True, help="Output directory for annotations")
    parser.add_argument("--ocr-backend", default="paddleocr", help="OCR backend (paddleocr/trocr)")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode (prompt for each image)")

    args = parser.parse_args()

    batch_annotate(
        input_dir=args.input,
        output_dir=args.output,
        ocr_backend=args.ocr_backend,
        interactive=args.interactive,
    )


if __name__ == "__main__":
    main()
