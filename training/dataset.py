"""Dataset utilities for PARSeq OCR fine-tuning on cropped plate images."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


class PlateOCRDataset(Dataset):
    """Simple image/label dataset for plate OCR training."""

    def __init__(
        self,
        image_dir: str | Path,
        label_dir: str | Path,
        image_size: tuple[int, int] = (32, 128),
        transform: transforms.Compose | None = None,
        augment: bool = False,
    ) -> None:
        self.image_dir = self._resolve_path(image_dir)
        self.label_dir = self._resolve_path(label_dir, kind="label")
        self.image_size = image_size
        self.augment = augment
        if transform is not None:
            self.transform = transform
        elif augment:
            self.transform = self._augment_transform(image_size)
        else:
            self.transform = self._default_transform(image_size)
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

    @staticmethod
    def _augment_transform(image_size: tuple[int, int]) -> transforms.Compose:
        """Training-only transform: simulates the real-world degradation a
        live camera feed sees that this repo's crop dataset otherwise
        doesn't cover -- motion blur, glare, low-light noise, and small
        viewpoint/angle variance. Applied before the deterministic
        resize/normalize every split shares, so it never touches val/test.
        """
        return transforms.Compose(
            [
                transforms.Lambda(PlateOCRDataset._random_motion_blur),
                transforms.Lambda(PlateOCRDataset._random_glare),
                transforms.Lambda(PlateOCRDataset._random_noise),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.03),
                transforms.RandomAffine(degrees=6, translate=(0.03, 0.03), scale=(0.92, 1.08), shear=3),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    @staticmethod
    def _random_motion_blur(img: Image.Image, p: float = 0.25) -> Image.Image:
        """Directional (horizontal/vertical) blur — a closer proxy for a
        moving vehicle than an isotropic Gaussian blur.
        """
        if random.random() > p:
            return img
        kernel_size = random.choice([3, 5, 7])
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        if random.random() < 0.5:
            kernel[kernel_size // 2, :] = 1.0
        else:
            kernel[:, kernel_size // 2] = 1.0
        kernel /= kernel_size
        arr = cv2.filter2D(np.array(img), -1, kernel)
        return Image.fromarray(arr)

    @staticmethod
    def _random_glare(img: Image.Image, p: float = 0.15) -> Image.Image:
        """Synthetic localized bright spot — headlight glare / direct sun."""
        if random.random() > p:
            return img
        arr = np.array(img).astype(np.float32)
        h, w = arr.shape[:2]
        cx, cy = random.randint(0, w), random.randint(0, h)
        radius = max(1, random.randint(w // 4, max(w // 4, w // 2)))
        yy, xx = np.mgrid[0:h, 0:w]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        glare = (np.clip(1.0 - dist / radius, 0.0, 1.0) ** 2) * random.uniform(80, 180)
        arr = np.clip(arr + glare[..., None], 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    @staticmethod
    def _random_noise(img: Image.Image, p: float = 0.2) -> Image.Image:
        """Sensor noise, as seen on a low-light/high-ISO night capture."""
        if random.random() > p:
            return img
        arr = np.array(img).astype(np.float32)
        sigma = random.uniform(5.0, 20.0)
        noise = np.random.normal(0.0, sigma, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

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
    augment_train: bool = False,
) -> dict[str, DataLoader]:
    """Create train/val/test DataLoaders with deterministic splits.

    When augment_train is True, the train split gets the degradation
    augmentation from PlateOCRDataset._augment_transform (motion blur,
    glare, noise, brightness/angle jitter) while val/test stay on the
    plain deterministic transform -- evaluation must see real, unmodified
    crops to be a meaningful accuracy signal.
    """
    # A plain (non-augmented) instance determines sample order/count for
    # every split, and is reused directly for val/test. Augmentation is
    # applied via a *second* instance over the same directories (same
    # sorted glob -> same sample order) so an index computed against one
    # lines up with the other.
    eval_dataset = PlateOCRDataset(image_dir=image_dir, label_dir=label_dir, image_size=image_size, augment=False)
    train_dataset = (
        PlateOCRDataset(image_dir=image_dir, label_dir=label_dir, image_size=image_size, augment=True)
        if augment_train else eval_dataset
    )
    if train_dataset is not eval_dataset and train_dataset.samples != eval_dataset.samples:
        # The two instances glob image_dir/label_dir independently. If the
        # directory contents change between the two _collect_samples() calls
        # (e.g. a labeling pipeline adding/removing crops concurrently),
        # indices computed against eval_dataset would silently pair the
        # wrong image/label when applied to train_dataset via Subset below.
        # Fail loudly instead of training on mismatched pairs.
        raise RuntimeError(
            "train_dataset and eval_dataset sample lists diverged -- "
            "image_dir/label_dir changed between dataset construction calls. "
            "Re-run once the directory contents are stable."
        )

    if not (abs(sum(split_ratios) - 1.0) < 1e-6):
        raise ValueError("split_ratios must sum to 1.0")

    generator = torch.Generator().manual_seed(seed)
    n_total = len(eval_dataset)
    n_train = int(n_total * split_ratios[0])
    n_val = int(n_total * split_ratios[1])

    shuffled = torch.randperm(n_total, generator=generator).tolist()
    train_indices = shuffled[:n_train]
    val_indices = shuffled[n_train:n_train + n_val]
    test_indices = shuffled[n_train + n_val:]

    train_ds = Subset(train_dataset, train_indices)
    val_ds = Subset(eval_dataset, val_indices)
    test_ds = Subset(eval_dataset, test_indices)

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
