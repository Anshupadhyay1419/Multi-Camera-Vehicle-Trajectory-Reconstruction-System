"""Publishes the pipeline's current annotated frame to a JPEG file that the
API server's /stream endpoint reads and serves as an MJPEG feed to the
dashboard. A file (not shared memory or a socket) is deliberately the
simplest thing that works here: the pipeline and the API server are
separate processes, this needs no IPC setup, and one small JPEG rewritten
a few times a second is negligible disk I/O.
"""

from __future__ import annotations

import os
import threading
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

    def __init__(
        self,
        path: str,
        max_fps: float = 8.0,
        jpeg_quality: int = 80,
        max_width: int | None = None,
        background: bool = False,
    ) -> None:
        self.path = Path(path)
        # Downscale wider frames before encoding. A preview panel on a
        # dashboard is a few hundred pixels wide, so encoding a full 1080p
        # frame there costs several times the work of a 960px one for no
        # visible gain. None keeps the original size (the single-gate
        # /stream behaviour).
        self.max_width = int(max_width) if max_width else None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.min_interval_s = 1.0 / max(max_fps, 0.1)
        self.jpeg_quality = int(jpeg_quality)
        # Salt with our own PID so two LiveFramePublisher instances (e.g. two
        # pipeline processes pointed at the same live_frame_path) never
        # write/rename the same temp file out from under each other.
        self._tmp_path = self.path.with_suffix(f"{self.path.suffix}.{os.getpid()}.tmp")

        # Rate limiting by SCHEDULE rather than by "time since last write".
        # With the old rule a frame was accepted only once a full interval
        # had passed since the previous one, which rounds the rate DOWN to a
        # multiple of the source's frame period: at a 15fps target and a
        # 23fps camera it waited for every second frame and published 11.6fps
        # (measured 11.0). Advancing a schedule lets the occasional shorter
        # gap make up for the longer ones, so the average meets the target.
        self._next_due = 0.0

        # Optional background encoder. Resizing and JPEG-encoding a frame
        # takes several milliseconds; done on the pipeline thread that is
        # time taken from ALPR itself (measured: a recorded clip processed at
        # 13fps with previews on, against 15-17fps without). With
        # background=True the pipeline only hands over a reference to the
        # frame; a worker thread does the rest, and OpenCV releases the GIL
        # while it works, so it runs on another core.
        self._background = bool(background)
        self._pending: np.ndarray | None = None
        self._pending_lock = threading.Lock()
        self._wake = threading.Event()
        self._stopping = False
        self._worker: threading.Thread | None = None
        if self._background:
            self._worker = threading.Thread(
                target=self._run_worker, name="live-frame-writer", daemon=True
            )
            self._worker.start()

    def is_due(self) -> bool:
        """True if a frame published now would actually be written.

        Lets the caller skip preparing a frame that publish() would discard.
        The pipeline draws its tracking overlay (a full frame copy plus
        drawing) on every processed frame; at a capped preview rate most of
        those used to be thrown away unseen.
        """
        return time.monotonic() >= self._next_due

    def publish(self, frame: np.ndarray) -> None:
        """Publish `frame` as the current live frame, subject to max_fps.

        The frame must not be modified afterwards by the caller: in
        background mode it is encoded a moment later on another thread. The
        pipeline satisfies this -- it publishes either a freshly drawn
        overlay copy or a raw frame it is about to discard.
        """
        now = time.monotonic()
        if now < self._next_due:
            return
        # Advance the schedule, but never let it fall far behind real time:
        # after a long pause (a slow frame, a reconnect) the preview should
        # simply resume at the target rate, not burst to catch up.
        # At most one interval of catch-up credit: a single slightly early
        # frame evens out the rate; anything more would be a burst.
        self._next_due = max(self._next_due, now - self.min_interval_s) + self.min_interval_s

        if not self._background:
            self._write(frame)
            return

        with self._pending_lock:
            # Only the newest frame matters; an older one still waiting is
            # replaced rather than queued, so latency cannot build up.
            self._pending = frame
        self._wake.set()

    def close(self) -> None:
        """Write any frame still waiting, then stop the background worker.

        Safe to call more than once, and a no-op for a synchronous publisher.
        Called at the end of each camera's run, so a queue of cameras does
        not accumulate one idle writer thread per camera.
        """
        if self._worker is None:
            return
        self._stopping = True
        self._wake.set()
        self._worker.join(timeout=5.0)
        self._worker = None

    # ── internals ────────────────────────────────────────────────────────

    def _run_worker(self) -> None:
        while True:
            self._wake.wait()
            self._wake.clear()
            with self._pending_lock:
                frame, self._pending = self._pending, None
            if frame is not None:
                try:
                    self._write(frame)
                except Exception:
                    # A failed preview write must never take anything down;
                    # the next frame simply tries again.
                    pass
            if self._stopping and self._pending is None:
                return

    def _write(self, frame: np.ndarray) -> None:
        if self.max_width and frame.shape[1] > self.max_width:
            scale = self.max_width / frame.shape[1]
            frame = cv2.resize(
                frame, (self.max_width, int(round(frame.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return

        # Write to a temp file then rename over the target: a reader that
        # opens the file mid-write would otherwise occasionally get a
        # truncated/corrupt JPEG. os.replace is atomic on the same filesystem.
        self._tmp_path.write_bytes(buf.tobytes())
        os.replace(self._tmp_path, self.path)
