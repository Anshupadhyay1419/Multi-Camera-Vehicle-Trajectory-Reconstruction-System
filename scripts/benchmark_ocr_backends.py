"""Compare RapidOCR and PARSeq TensorRT OCR backends on a directory of plate crops."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.ocr import create_ocr_engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark OCR backends on plate crops")
    parser.add_argument("--image-dir", type=str, default="OCR_data/images")
    parser.add_argument("--label-dir", type=str, default="OCR_data/labels")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    return parser.parse_args()


def _load_config(path: str) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _collect_images(image_dir: Path) -> list[Path]:
    return sorted(
        [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)

    image_paths = _collect_images(image_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    rapid_cfg = dict(config)
    rapid_cfg["ocr"] = dict(config.get("ocr", {}))
    rapid_cfg["ocr"]["backend"] = "rapidocr"
    rapid_engine = create_ocr_engine("rapidocr", rapid_cfg)

    parseq_cfg = dict(config)
    parseq_cfg["ocr"] = dict(config.get("ocr", {}))
    parseq_cfg["ocr"]["backend"] = "parseq_tensorrt"
    parseq_engine = create_ocr_engine("parseq_tensorrt", parseq_cfg)

    timings: dict[str, list[float]] = {"rapidocr": [], "parseq_tensorrt": []}
    for image_path in image_paths[:50]:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        label = label_path.read_text(encoding="utf-8").strip() if label_path.exists() else ""

        for name, engine in (("rapidocr", rapid_engine), ("parseq_tensorrt", parseq_engine)):
            start = time.perf_counter()
            _ = engine.recognize(image)
            elapsed = time.perf_counter() - start
            timings[name].append(elapsed)

    summary = {
        "rapidocr": {
            "mean_ms": round(statistics.mean(timings["rapidocr"]) * 1000.0, 2),
            "p95_ms": round(np.percentile(np.array(timings["rapidocr"]), 95) * 1000.0, 2),
            "samples": len(timings["rapidocr"]),
        },
        "parseq_tensorrt": {
            "mean_ms": round(statistics.mean(timings["parseq_tensorrt"]) * 1000.0, 2),
            "p95_ms": round(np.percentile(np.array(timings["parseq_tensorrt"]), 95) * 1000.0, 2),
            "samples": len(timings["parseq_tensorrt"]),
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
