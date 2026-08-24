"""Export a trained OCR model to ONNX for TensorRT conversion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a parseq-style OCR checkpoint to ONNX")
    parser.add_argument("--checkpoint", type=str, default="models/ocr/best_parseq.pt")
    parser.add_argument("--output", type=str, default="models/ocr/parseq.onnx")
    parser.add_argument("--image-size", type=str, default="32,128")
    return parser.parse_args()


def _parse_image_size(value: str) -> tuple[int, int]:
    parts = [int(item.strip()) for item in value.split(",")]
    if len(parts) != 2:
        raise ValueError("image-size must be in H,W form")
    return parts[0], parts[1]


def main() -> None:
    args = parse_args()
    image_size = _parse_image_size(args.image_size)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from training.parseq_model import PARSeqOCRModel

    model = PARSeqOCRModel(image_size=image_size, seq_len=10)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)

    model.eval()
    dummy = torch.randn(1, 3, image_size[0], image_size[1])
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print(f"Saved ONNX model to {output_path}")


if __name__ == "__main__":
    main()
