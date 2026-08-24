"""Minimal PARSeq fine-tuning entry point for plate OCR."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.dataset import build_dataloaders
from training.parseq_model import PARSeqOCRModel, encode_plate, PAD_INDEX

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("training.train_parseq")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a lightweight PARSeq-style OCR model")
    parser.add_argument("--image-dir", type=str, default="OCR_data/images")
    parser.add_argument("--label-dir", type=str, default="OCR_data/labels")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=str, default="32,128")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="models/ocr")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the checkpoint at output_dir/last_parseq.ckpt",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path to save/resume training state. Defaults to output_dir/last_parseq.ckpt",
    )
    parser.add_argument(
        "--save-epoch-checkpoints",
        action="store_true",
        help="Save a separate checkpoint after each epoch in addition to last_parseq.ckpt",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save an epoch checkpoint every N epochs when --save-epoch-checkpoints is enabled.",
    )
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Load model weights from a .pt file before training starts.",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="Set the starting epoch number for resumed training.",
    )
    return parser.parse_args()


def _parse_image_size(value: str) -> tuple[int, int]:
    parts = [int(item.strip()) for item in value.split(",")]
    if len(parts) != 2:
        raise ValueError("image-size must be in H,W form")
    return parts[0], parts[1]


def main() -> None:
    args = parse_args()
    image_size = _parse_image_size(args.image_size)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataloaders = build_dataloaders(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        workers=args.workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device %s", device)

    seq_len = 10
    model = PARSeqOCRModel(image_size=image_size, seq_len=seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_INDEX)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output_dir / "last_parseq.ckpt"
    start_epoch = args.start_epoch
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        checkpoint_epoch = checkpoint.get("epoch", 0)
        if args.start_epoch <= 0:
            start_epoch = checkpoint_epoch
        logger.info(
            "Resuming training from epoch %d using checkpoint %s",
            start_epoch,
            checkpoint_path,
        )
    elif args.resume:
        logger.warning("Resume requested but checkpoint not found: %s. Starting anew.", checkpoint_path)

    if args.load_model is not None:
        model_path = Path(args.load_model)
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info("Loaded weights from %s", model_path)
        else:
            raise FileNotFoundError(f"Requested model file not found: {model_path}")

    best_loss = float("inf")
    last_checkpoint_path = output_dir / "last_parseq.ckpt"
    best_checkpoint_path = output_dir / "best_parseq.pt"

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in dataloaders["train"]:
            images = images.to(device)
            targets = torch.tensor([
                encode_plate(label, seq_len=seq_len)
                for label in labels
            ], dtype=torch.long, device=device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.transpose(1, 2), targets)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

        avg_loss = running_loss / max(1, len(dataloaders["train"]))
        logger.info("epoch %d loss %.4f", epoch + 1, avg_loss)

        checkpoint = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        torch.save(checkpoint, last_checkpoint_path)
        logger.info("Saved resume checkpoint to %s", last_checkpoint_path)

        if args.save_epoch_checkpoints and (
            (epoch + 1) % args.checkpoint_interval == 0
            or epoch + 1 == args.epochs
        ):
            epoch_checkpoint = output_dir / f"epoch_{epoch + 1:03d}_parseq.ckpt"
            torch.save(checkpoint, epoch_checkpoint)
            logger.info("Saved epoch checkpoint to %s", epoch_checkpoint)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_checkpoint_path)
            logger.info("Saved best model to %s", best_checkpoint_path)

    torch.save(model.state_dict(), output_dir / "last_parseq.pt")
    logger.info("Training complete. Final weights saved to %s", output_dir / "last_parseq.pt")
    logger.info("Best model saved to %s", best_checkpoint_path)


if __name__ == "__main__":
    main()
