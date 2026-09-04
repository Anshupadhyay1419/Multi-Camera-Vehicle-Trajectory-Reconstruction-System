"""Publishes the pipeline's current annotated frame to a JPEG file that the
API server's /stream endpoint reads and serves as an MJPEG feed to the
dashboard. A file (not shared memory or a socket) is deliberately the
simplest thing that works here: the pipeline and the API server are
separate processes, this needs no IPC setup, and one small JPEG rewritten
a few times a second is negligible disk I/O.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np


class LiveFramePublisher:
    """Rate-limited, atomic JPEG writer for the live dashboard feed.

    Args:
        path:           Where to write the current frame.
        max_fps:        Upper bound on write rate -- the pipeline may run
                        much faster than this; there is no reason to
                        re-encode/rewrite a JPEG on every single frame for
                        a feed a human is just watching.
        jpeg_quality:   0-100.
    """

    def __init__(self, path: str, max_fps: float = 8.0, jpeg_quality: int = 80) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.min_interval_s = 1.0 / max(max_fps, 0.1)
        self.jpeg_quality = int(jpeg_quality)
        # Salt with our own PID so two LiveFramePublisher instances (e.g. two
        # pipeline processes pointed at the same live_frame_path) never
        # write/rename the same temp file out from under each other.
        self._tmp_path = self.path.with_suffix(f"{self.path.suffix}.{os.getpid()}.tmp")
        self._last_write = 0.0

    def publish(self, frame: np.ndarray) -> None:
        """Write `frame` as the current live frame, subject to max_fps."""
        now = time.monotonic()
        if now - self._last_write < self.min_interval_s:
            return

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return

        # Write to a temp file then rename over the target: a reader that
        # opens the file mid-write would otherwise occasionally get a
        # truncated/corrupt JPEG. os.replace is atomic on the same filesystem.
        self._tmp_path.write_bytes(buf.tobytes())
        os.replace(self._tmp_path, self.path)
        self._last_write = now
