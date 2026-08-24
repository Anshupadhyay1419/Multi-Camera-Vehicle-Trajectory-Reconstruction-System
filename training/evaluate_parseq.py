"""Evaluate a trained OCR checkpoint on a test subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.dataset import build_dataloaders
from training.parseq_model import PARSeqOCRModel, decode_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a parseq-style OCR checkpoint")
    parser.add_argument("--checkpoint", type=str, default="models/ocr/best_parseq.pt")
    parser.add_argument("--image-dir", type=str, default="OCR_data/images")
    parser.add_argument("--label-dir", type=str, default="OCR_data/labels")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataloaders = build_dataloaders(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        batch_size=args.batch_size,
        split_ratios=(0.7, 0.15, 0.15),
        workers=0,
    )
    state = torch.load(args.checkpoint, map_location="cpu")
    seq_len = 10
    model = PARSeqOCRModel(image_size=(32, 128), seq_len=seq_len)
    model.load_state_dict(state)
    model.eval()

    predictions: list[str] = []
    references: list[str] = []
    with torch.no_grad():
        for images, labels in dataloaders["test"]:
            outputs = model(images)
            pred_indices = outputs.argmax(dim=2)
            predicted_texts = decode_indices(pred_indices)
            for label, text in zip(labels, predicted_texts):
                predictions.append(text)
                references.append(label)

    report = {
        "num_examples": len(references),
        "predictions": predictions[:10],
        "references": references[:10],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
