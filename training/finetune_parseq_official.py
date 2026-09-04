"""Fine-tune the official baudm/parseq-tiny transformer on this project's
own plate crops (OCR_data/), instead of deploying it zero-shot.

Why this exists: config/config.yaml currently points the pipeline at
parseq-tiny's *pretrained* checkpoint, downloaded as-is from GitHub. That
model was trained on general English scene-text (signage, documents), never
on Indian plates, which is the likely source of systematic OCR confusions
(O/C, O/0, ...). This script continues training from those same pretrained
weights on the project's 3,700+ labeled plate crops, so the model keeps its
strong transformer architecture but adapts to this exact domain.

This deliberately reuses strhub's own PARSeq LightningModule (via Hydra, the
same way the upstream repo's own train.py instantiates it) rather than
reimplementing PARSeq's permutation-language-model training objective by
hand -- that objective is genuinely intricate (multiple permutations of the
label sequence per step, iterative refinement) and worth getting from the
reference implementation, not guessed at.

Requires a local clone of https://github.com/baudm/parseq (see --source-dir,
same convention as scripts/export_official_parseq.py) plus:
    pip install hydra-core omegaconf pytorch-lightning nltk timm
(nltk/timm are hard imports inside strhub/models/base.py even though this
script never exercises the code paths that use them.)

Verified against a real clone: "tiny" is configs/experiment/parseq-tiny.yaml
(a Hydra experiment overlay, not a model= group -- selected via
+experiment=parseq-tiny), training_step()/validation_step() both unpack
`images, labels = batch` (so a plain (image_tensor_batch, list_of_label_str)
tuple is the right shape), and configure_optimizers()'s automatic
batch-size/256 LR scaling is bypassed here in favor of a fixed --lr AdamW.
Still run --print-config-only first when trying a new --source-dir clone --
if the repo's structure has moved since, that's where it'll show up, fast
and without burning a training run.

Usage:
    python training/finetune_parseq_official.py --print-config-only
    python training/finetune_parseq_official.py --epochs 10 --lr 1e-5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

WEIGHTS_URL = "https://github.com/baudm/parseq/releases/download/v1.0.0/parseq_tiny-e7a21b54.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune official PARSeq-tiny on plate crops")
    parser.add_argument("--source-dir", default="/tmp/baudm-parseq",
                         help="Local clone of https://github.com/baudm/parseq")
    parser.add_argument("--image-dir", default="OCR_data/images")
    parser.add_argument("--label-dir", default="OCR_data/lables")  # this repo's actual dir name
    parser.add_argument("--model-override", default="parseq-tiny",
                         help="Hydra experiment overlay under <source-dir>/configs/experiment/, "
                              "selected as +experiment=<value> (list with: "
                              "ls <source-dir>/configs/experiment/)")
    parser.add_argument("--charset-override", default="94_full",
                         help="Hydra config group under <source-dir>/configs/charset/ -- "
                              "must match the pretrained checkpoint's tokenizer "
                              "(list with: ls <source-dir>/configs/charset/)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5,
                         help="Fine-tuning LR. Keep this low (1e-5 to 5e-5) -- the "
                              "model's from-scratch training LR (~7e-4) will wreck "
                              "already-converged pretrained weights.")
    parser.add_argument("--output", default="models/ocr/parseq_official_tiny_finetuned.pt",
                         help="Where to save the fine-tuned bare-model state_dict, "
                              "ready for scripts/export_official_parseq.py --local-checkpoint")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--augment", action="store_true",
        help="Apply degradation augmentation (motion blur, glare, low-light "
             "noise, brightness/angle jitter) to the training split only, "
             "via training.dataset.PlateOCRDataset. Recommended when the "
             "deployment camera will see real-world conditions this crop "
             "set doesn't cover (night, glare, motion blur) -- OCR_data/ "
             "as-is is likely all well-lit, static crops.",
    )
    parser.add_argument("--print-config-only", action="store_true",
                         help="Compose the Hydra config, load pretrained weights, and "
                              "print a model summary, then exit without training.")
    return parser.parse_args()


def _build_model(source_dir: Path, model_override: str, charset_override: str, lr: float):
    """Compose strhub's Hydra config and instantiate the LightningModule the
    same way the upstream repo's train.py does, so we don't have to guess
    the (many, version-sensitive) constructor keyword arguments by hand.
    """
    sys.path.insert(0, str(source_dir))
    import hydra
    from hydra import compose, initialize_config_dir

    config_dir = source_dir / "configs"
    if not config_dir.is_dir():
        raise FileNotFoundError(
            f"No configs/ directory at {config_dir}. Is --source-dir a full clone "
            f"of https://github.com/baudm/parseq (not just strhub/)?"
        )

    with initialize_config_dir(config_dir=str(config_dir), version_base="1.2"):
        cfg = compose(
            config_name="main",
            overrides=[
                # "tiny" is an experiment overlay (configs/experiment/parseq-tiny.yaml),
                # not a model= group -- it sets embed_dim/enc_num_heads/dec_num_heads on
                # top of the base model=parseq group it also selects. Needs the "+"
                # since "experiment" isn't in main.yaml's own defaults list.
                f"+experiment={model_override}",
                f"charset={charset_override}",
                f"model.lr={lr}",
                "model.warmup_pct=0.0",  # fine-tuning: no LR warmup, we're already converged
            ],
        )
        model = hydra.utils.instantiate(cfg.model)

    return model


def _load_pretrained_into(model, checkpoint_path: Path | None) -> None:
    """Load parseq-tiny's pretrained weights into model.model (the bare
    strhub.models.parseq.model.PARSeq inner module), same source used by
    scripts/export_official_parseq.py.
    """
    if checkpoint_path is not None and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        print(f"Loaded pretrained weights from local file: {checkpoint_path}")
    else:
        cache_dir = Path("models/ocr")
        cache_dir.mkdir(parents=True, exist_ok=True)
        state = torch.hub.load_state_dict_from_url(
            WEIGHTS_URL, model_dir=str(cache_dir), map_location="cpu"
        )
        print(f"Downloaded pretrained weights from {WEIGHTS_URL}")
    model.model.load_state_dict(state, strict=True)


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    if not (source_dir / "strhub").is_dir():
        raise FileNotFoundError(
            f"Official PARSeq source is not available at {source_dir}. Clone it first:\n"
            f"  git clone https://github.com/baudm/parseq {source_dir}\n"
            f"  pip install -r {source_dir}/requirements.txt  # or at least: hydra-core omegaconf pytorch-lightning"
        )

    model = _build_model(source_dir, args.model_override, args.charset_override, args.lr)
    _load_pretrained_into(model, Path("models/ocr/parseq_official_tiny.pt"))

    # BaseSystem.configure_optimizers() scales lr by (batch_size / 256), a
    # convention for large-batch distributed *pretraining* -- at our small
    # fine-tuning batch size that divides args.lr down to near zero. Replace
    # it with a plain fixed-LR AdamW so `--lr` means what it says.
    def _configure_optimizers():
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    model.configure_optimizers = _configure_optimizers

    print(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    if args.print_config_only:
        print("--print-config-only set: model built and pretrained weights loaded "
              "successfully. Exiting without training.")
        return

    from training.dataset import build_dataloaders

    loaders = build_dataloaders(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        image_size=(32, 128),
        batch_size=args.batch_size,
        workers=args.workers,
        augment_train=args.augment,
    )

    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_path.parent),
        filename="parseq_finetune_lightning_best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=20,
        callbacks=[checkpoint_callback],
        logger=False,
    )
    trainer.fit(model, train_dataloaders=loaders["train"], val_dataloaders=loaders["val"])

    best_ckpt_path = checkpoint_callback.best_model_path
    if not best_ckpt_path:
        print("No checkpoint was saved (training may not have completed an epoch).")
        return

    # Lightning's checkpoint holds the whole system's state_dict, with every
    # bare-model parameter prefixed "model." (PARSeq system's self.model
    # attribute). Strip that prefix so the result loads directly into
    # strhub.models.parseq.model.PARSeq / scripts/export_official_parseq.py,
    # exactly like the stock pretrained checkpoint does.
    lightning_state = torch.load(best_ckpt_path, map_location="cpu")["state_dict"]
    prefix = "model."
    bare_state = {k[len(prefix):]: v for k, v in lightning_state.items() if k.startswith(prefix)}
    torch.save(bare_state, output_path)

    print(f"Done. Best val_loss={checkpoint_callback.best_model_score:.4f}")
    print(f"Bare-model checkpoint saved -> {output_path}")
    print(f"Next: python scripts/export_official_parseq.py --local-checkpoint {output_path}")


if __name__ == "__main__":
    main()
