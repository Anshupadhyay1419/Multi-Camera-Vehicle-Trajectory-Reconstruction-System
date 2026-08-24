"""Dataset utilities for PARSeq OCR fine-tuning on cropped plate images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


class PlateOCRDataset(Dataset):
    """Simple image/label dataset for plate OCR training."""

    def __init__(
        self,
        image_dir: str | Path,
        label_dir: str | Path,
        image_size: tuple[int, int] = (32, 128),
        transform: transforms.Compose | None = None,
    ) -> None:
        self.image_dir = self._resolve_path(image_dir)
        self.label_dir = self._resolve_path(label_dir, kind="label")
        self.image_size = image_size
        self.transform = transform or self._default_transform(image_size)
        self.samples = self._collect_samples()

    @staticmethod
    def _resolve_path(path: str | Path, kind: str | None = None) -> Path:
        candidate = Path(path)
        if candidate.exists():
            return candidate

        repo_root = Path(__file__).resolve().parents[1]
        repo_candidate = repo_root / candidate
        if repo_candidate.exists():
            return repo_candidate

        if kind == "label":
            alternate_name = None
            if candidate.name == "labels":
                alternate_name = "lables"
            elif candidate.name == "lables":
                alternate_name = "labels"

            if alternate_name is not None:
                alternate_candidate = candidate.parent / alternate_name
                repo_alternate = repo_root / alternate_candidate
                if repo_alternate.exists():
                    return repo_alternate
                if alternate_candidate.exists():
                    return alternate_candidate

            for label_name in ("labels", "lables"):
                repo_fallback = repo_root / "OCR_data" / label_name
                if repo_fallback.exists():
                    return repo_fallback

        return candidate

    @staticmethod
    def _default_transform(image_size: tuple[int, int]) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def _collect_samples(self) -> list[tuple[Path, str]]:
        if not self.image_dir.exists() or not self.label_dir.exists():
            raise FileNotFoundError(f"Image or label directory not found: {self.image_dir}, {self.label_dir}")

        image_paths = sorted(self.image_dir.glob("*.jpg")) + sorted(self.image_dir.glob("*.jpeg")) + sorted(self.image_dir.glob("*.png"))
        samples: list[tuple[Path, str]] = []
        label_files = {path.stem: path for path in self.label_dir.glob("*.txt")}
        for image_path in image_paths:
            label_path = None
            for candidate_name in {image_path.stem, image_path.stem.replace("_", ""), image_path.stem.lstrip("0") }:
                if candidate_name in label_files:
                    label_path = label_files[candidate_name]
                    break
            if label_path is None:
                stem_with_prefix = image_path.stem if image_path.stem.startswith("image") else f"image{image_path.stem}"
                if stem_with_prefix in label_files:
                    label_path = label_files[stem_with_prefix]
            if label_path is None:
                continue
            text = label_path.read_text(encoding="utf-8").strip()
            if text:
                samples.append((image_path, text))

        if not samples:
            raise ValueError(f"No labeled image samples found in {self.image_dir}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0
        return image_tensor, label


def build_dataloaders(
    image_dir: str | Path,
    label_dir: str | Path,
    image_size: tuple[int, int] = (32, 128),
    batch_size: int = 8,
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    workers: int = 4,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """Create train/val/test DataLoaders with deterministic splits."""
    dataset = PlateOCRDataset(image_dir=image_dir, label_dir=label_dir, image_size=image_size)

    if not (abs(sum(split_ratios) - 1.0) < 1e-6):
        raise ValueError("split_ratios must sum to 1.0")

    generator = torch.Generator().manual_seed(seed)
    n_total = len(dataset)
    n_train = int(n_total * split_ratios[0])
    n_val = int(n_total * split_ratios[1])
    n_test = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=generator)

    def _loader(ds: torch.utils.data.Dataset[Any]) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
        )

    return {
        "train": _loader(train_ds),
        "val": _loader(val_ds),
        "test": _loader(test_ds),
    }
