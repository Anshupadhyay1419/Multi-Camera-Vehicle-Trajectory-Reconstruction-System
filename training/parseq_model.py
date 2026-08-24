"""Lightweight sequence OCR model for fine-tuning and TensorRT export."""

from __future__ import annotations

import torch
from torch import nn

CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PAD_TOKEN = "<pad>"
PAD_INDEX = len(CHARSET)
VOCAB_SIZE = len(CHARSET) + 1
DEFAULT_SEQ_LEN = 10


class PARSeqOCRModel(nn.Module):
    """Compact fixed-length OCR model for plate crops."""

    def __init__(
        self,
        image_size: tuple[int, int] = (32, 128),
        seq_len: int = DEFAULT_SEQ_LEN,
        vocab_size: int = VOCAB_SIZE,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 8)),
        )

        flattened = 128 * 2 * 8
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, seq_len * vocab_size),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            images: (B, 3, H, W)

        Returns:
            logits: (B, seq_len, vocab_size)
        """
        features = self.encoder(images)
        logits = self.head(features)
        return logits.view(-1, self.seq_len, self.vocab_size)


def decode_indices(indices: torch.Tensor) -> list[str]:
    """Convert a batch of predicted token indices into strings."""
    strings: list[str] = []
    for row in indices.tolist():
        chars: list[str] = []
        for idx in row:
            if idx == PAD_INDEX:
                break
            if 0 <= idx < len(CHARSET):
                chars.append(CHARSET[idx])
            else:
                break
        strings.append("".join(chars))
    return strings


def encode_plate(text: str, seq_len: int = DEFAULT_SEQ_LEN) -> list[int]:
    """Encode a plate string into fixed-length token indices."""
    normalized = "".join(ch for ch in text.upper() if ch.isalnum())
    encoded: list[int] = []
    for ch in normalized:
        if ch in CHARSET:
            encoded.append(CHARSET.index(ch))
        if len(encoded) >= seq_len:
            break
    if len(encoded) < seq_len:
        encoded.extend([PAD_INDEX] * (seq_len - len(encoded)))
    return encoded
