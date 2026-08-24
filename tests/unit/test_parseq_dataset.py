from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from training.dataset import PlateOCRDataset, build_dataloaders


def _write_plate_sample(path: Path, text: str) -> None:
    image = np.zeros((24, 64, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    Image.fromarray(image).save(path)
    label_path = path.with_suffix('.txt')
    label_path.write_text(text, encoding='utf-8')


def test_build_dataloaders_creates_batches(tmp_path: Path) -> None:
    images_dir = tmp_path / 'images'
    labels_dir = tmp_path / 'labels'
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    for index in range(8):
        image_path = images_dir / f'{index:04d}.jpg'
        _write_plate_sample(image_path, 'AB1234')

    dataloaders = build_dataloaders(
        image_dir=str(images_dir),
        label_dir=str(labels_dir),
        image_size=(32, 128),
        batch_size=2,
        split_ratios=(0.5, 0.25, 0.25),
        workers=0,
        seed=7,
    )

    assert 'train' in dataloaders
    assert 'val' in dataloaders
    assert 'test' in dataloaders

    train_batch = next(iter(dataloaders['train']))
    images, labels = train_batch
    assert images.shape[0] == 2
    assert labels[0].startswith('AB1234')
