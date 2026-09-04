"""Export the official baudm/parseq-tiny checkpoint for TensorRT deployment.

This intentionally does not use ``training.parseq_model``: that file is the
project's legacy lightweight CNN, not the reference PARSeq architecture.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

OFFICIAL_CHARSET = (
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
)
WEIGHTS_URL = "https://github.com/baudm/parseq/releases/download/v1.0.0/parseq_tiny-e7a21b54.pt"


class PARSeqTensorRTExport(nn.Module):
    """Export the official PARSeq-Tiny non-autoregressive inference graph."""

    def __init__(self, model: nn.Module, tokenizer: object) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # NAR inference is a supported PARSeq mode. It avoids unrolling eleven
        # transformer-decoder passes in ONNX, which exceeds the Orin's build
        # memory even though the runtime engine itself is small.
        return self.model(self.tokenizer, images, max_length=10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export official PARSeq-Tiny to ONNX")
    parser.add_argument("--source-dir", default="/tmp/baudm-parseq", help="Clone of https://github.com/baudm/parseq")
    parser.add_argument("--checkpoint", default="models/ocr/parseq_official_tiny.pt")
    parser.add_argument(
        "--local-checkpoint", default=None,
        help="Path to a locally fine-tuned bare-model state_dict (e.g. the output of "
             "training/finetune_parseq_official.py) to export instead of downloading "
             "the stock pretrained weights.",
    )
    parser.add_argument("--output", default="models/ocr/parseq_official_tiny.onnx")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    if not (source_dir / "strhub").is_dir():
        raise FileNotFoundError(f"Official PARSeq source is not available: {source_dir}")
    sys.path.insert(0, str(source_dir))

    from strhub.data.utils import Tokenizer
    from strhub.models.parseq.model import PARSeq

    tokenizer = Tokenizer(OFFICIAL_CHARSET)
    model = PARSeq(
        num_tokens=len(tokenizer), max_label_length=25, img_size=(32, 128), patch_size=(4, 8),
        embed_dim=192, enc_num_heads=3, enc_mlp_ratio=4, enc_depth=12,
        dec_num_heads=6, dec_mlp_ratio=4, dec_depth=1, decode_ar=False,
        refine_iters=1, dropout=0.1,
    ).eval()
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if args.local_checkpoint:
        local_path = Path(args.local_checkpoint)
        state = torch.load(local_path, map_location="cpu")
        print(f"Loading fine-tuned checkpoint: {local_path}")
    else:
        state = torch.hub.load_state_dict_from_url(
            WEIGHTS_URL, model_dir=str(checkpoint_path.parent), map_location="cpu"
        )
    model.load_state_dict(state, strict=True)
    # Keep an explicit, reproducible checkpoint name next to the ONNX
    # artefact -- but only for the stock-download path. --checkpoint's
    # default is the *original* pretrained weights file; if a
    # --local-checkpoint was given, that file already exists on disk, and
    # saving over --checkpoint's default here would silently destroy the
    # only copy of the un-fine-tuned baseline.
    if not args.local_checkpoint:
        torch.save(state, checkpoint_path)

    export_model = PARSeqTensorRTExport(model, tokenizer).eval()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((1, 3, 32, 128), dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(
            export_model, dummy, output_path, input_names=["input"], output_names=["output"],
            opset_version=17, dynamo=False,
        )
    if args.local_checkpoint:
        print(f"Exported official PARSeq-Tiny checkpoint (source unchanged): {local_path}")
    else:
        print(f"Exported official PARSeq-Tiny checkpoint: {checkpoint_path}")
    print(f"Exported ONNX: {output_path}")


if __name__ == "__main__":
    main()
