"""
Frame capture module for the ALPR University Gate system.

Wraps cv2.VideoCapture to support RTSP streams and local video files.
Implements frame-skip for performance and automatic reconnection on failure.

Live sources (RTSP/RTMP/HTTP/webcam) use a background reader thread that
always keeps only the most recently captured frame. Without this, a
synchronous read_frame() ties the pipeline's frame rate to the camera's
frame rate: if processing ever falls behind (a slow frame, a GC pause, a
network hiccup), frames queue up in OpenCV's internal buffer and the
pipeline starts silently processing increasingly stale video, with latency
that grows without bound. Always reading only the latest frame means the
pipeline instead drops frames it can't keep up with and stays close to
real time -- the standard approach for live video, as opposed to a
recorded file where every frame should be processed in order.

Local video files keep the original synchronous, frame-skip-respecting
behavior: deterministic, full coverage, no background thread.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np

from src.utils.logger import get_logger

_logger = get_logger("capture.frame_capture")


class FrameCapture:
    """Capture frames from an RTSP stream, webcam, or local video file.

    Args:
        source:       RTSP/RTMP/HTTP URL, webcam index (as a string), or a
                      local video file path.
        frame_skip:   Process every Nth frame. Applies to local video files
                      only -- a live source already only ever yields the
                      newest frame, which supersedes frame-skipping.
        max_retries:  Number of reconnect attempts before halting.
        stall_timeout: For live sources, how long read_frame() waits for a
                      new frame before treating the connection as stalled
                      and forcing a reconnect. Guards against a camera/
                      network fault that hangs cv2.VideoCapture.read()
                      instead of returning False.
    """

    def __init__(
        self,
        source: str,
        frame_skip: int = 2,
        max_retries: int = 5,
        stall_timeout: float = 10.0,
    ) -> None:
        self.source = source
        self.frame_skip = max(1, frame_skip)
        self.max_retries = max_retries
        self.stall_timeout = stall_timeout
        self.is_live = self._is_live_source(source)

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count: int = 0

        # Live-mode background reader state.
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_reader = threading.Event()
        self._frame_cond = threading.Condition()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_id: int = 0
        self._consumed_frame_id: int = 0
        self._reader_failed = threading.Event()
        # Bumped every time we abandon a reader thread after a stall. A
        # Python thread blocked inside a hung C-extension call (like a
        # network read with no OS-level timeout) cannot be force-killed,
        # and cap.release() unblocking it is a backend-dependent trick, not
        # a guarantee. So instead of waiting on the old thread, we abandon
        # it and start a fresh capture + reader thread; the old thread (if
        # it ever wakes up) checks its captured epoch against the current
        # one before touching shared state, so it can't clobber the frames
        # its replacement is producing.
        self._epoch: int = 0

    @classmethod
    def from_config(cls, config: dict) -> "FrameCapture":
        """Construct from the full config dict."""
        video_cfg = config.get("video", {})
        return cls(
            source=str(video_cfg.get("source", "")),
            frame_skip=int(video_cfg.get("frame_skip", 2)),
            max_retries=int(video_cfg.get("max_retries", 5)),
            stall_timeout=float(video_cfg.get("stall_timeout", 10.0)),
        )

    def open(self) -> bool:
        """Open the video source with retry logic.

        Returns:
            True if opened successfully.

        Raises:
            RuntimeError: If the source cannot be opened after all retries.
        """
        self._cap = self._open_capture_with_retries()

        if self.is_live:
            # Ask the backend to keep as little of its own internal frame
            # queue as possible -- our reader thread already discards
            # everything except the newest frame, but a smaller backend
            # buffer means less latency before that frame even reaches us.
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._stop_reader.clear()
            self._reader_failed.clear()
            self._latest_frame = None
            self._latest_frame_id = 0
            self._consumed_frame_id = 0
            self._start_reader_thread()

        self._frame_count = 0
        _logger.info("Video source opened: '%s' (live=%s)", self.source, self.is_live)
        return True

    def read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read the next frame to be processed.

        For local video files: reads frame_skip frames and returns the
        last one, in order, matching the original synchronous behavior.

        For live sources: blocks until the background reader thread has
        produced a frame newer than the last one consumed (or until
        stall_timeout elapses, in which case a reconnect is forced).

        Returns:
            (success, frame) — frame is None when success is False.
        """
        if self.is_live:
            # `self._cap` is None whenever a reader thread exhausted its own
            # reconnect attempts, because _reconnect() clears it before
            # retrying. For a live camera that must NOT be terminal: a stream
            # that is down for longer than max_retries*5s almost always comes
            # back, and a camera is supposed to run until STOP. Returning
            # early here left the capture permanently dead -- the pipeline
            # spun on `continue` forever while the dashboard still showed the
            # camera as "running". _read_live_frame() reopens instead.
            # _epoch guards against a read before open(); _stop_reader
            # against one after release().
            if self._epoch == 0 or self._stop_reader.is_set():
                return False, None
            return self._read_live_frame()

        if self._cap is None:
            return False, None
        return self._read_file_frame()

    def stream_time_seconds(self) -> float:
        """Seconds elapsed *within the stream*, for time-windowed logic.

        A live source runs at real time, so wall clock is its stream clock.
        A recorded file does not: it decodes as fast (or as slowly) as the
        host manages, so anything that means "within the last N seconds of
        footage" -- deduplication above all -- has to measure the video's own
        timeline or its behaviour changes with the machine it runs on. This
        gate video is 10.7s long but takes 28-51s to process here, which was
        ageing a plate out of a 30s dedup window mid-video and storing the
        same vehicle twice.
        """
        if self.is_live or self._cap is None:
            return time.time()
        position_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
        if position_ms and position_ms > 0:
            return float(position_ms) / 1000.0
        # Backends that don't report position still need a monotonic video
        # clock; derive one from the decoded frame count.
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0
        return self._frame_count * self.frame_skip / fps

    def release(self) -> None:
        """Release the video capture resource."""
        self._stop_reader.set()
        with self._frame_cond:
            self._frame_cond.notify_all()
        reader_stopped = True
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=self.stall_timeout + 2.0)
            reader_stopped = not self._reader_thread.is_alive()
            self._reader_thread = None

        if self._cap is not None:
            if reader_stopped:
                self._cap.release()
                _logger.info("Video source released.")
            else:
                # The reader thread didn't stop within the join timeout --
                # it's most likely still blocked inside a hung cap.read()
                # on this exact object. Calling .release() on it now would
                # race a concurrent read from another thread (the same
                # hazard _read_live_frame()'s stall-recovery path goes out
                # of its way to avoid). Leak the handle rather than risk a
                # crash; the daemon thread will exit harmlessly if it ever
                # unblocks.
                _logger.warning(
                    "Reader thread for '%s' did not stop in time; leaving "
                    "its capture unreleased instead of risking a concurrent "
                    "release() while it may still be reading.",
                    self.source,
                )
            self._cap = None

    def __enter__(self) -> "FrameCapture":
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.release()

    # ------------------------------------------------------------------
    # Local video file path (synchronous, deterministic)
    # ------------------------------------------------------------------

    def _read_file_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        if not self._cap.isOpened():
            return False, None

        frame = None
        for _ in range(self.frame_skip):
            ret, frame = self._cap.read()
            if not ret:
                return False, None

        self._frame_count += 1
        return True, frame

    # ------------------------------------------------------------------
    # Live source path (threaded, always-latest-frame)
    # ------------------------------------------------------------------

    def _start_reader_thread(self) -> None:
        self._epoch += 1
        my_epoch = self._epoch
        self._reader_thread = threading.Thread(
            target=self._reader_loop, args=(my_epoch,),
            name=f"frame-capture-reader-{my_epoch}", daemon=True,
        )
        self._reader_thread.start()

    def _reader_loop(self, my_epoch: int) -> None:
        """Background thread: continuously read frames, keep only the
        newest. Handles reconnects internally so read_frame() never blocks
        on network I/O beyond stall_timeout.

        `my_epoch` is captured once at start. A thread whose epoch no
        longer matches self._epoch has been abandoned (superseded by a
        fresh reconnect after a stall) and must not touch shared state —
        it may still be alive, blocked inside a hung read() with no way
        to force it to stop.
        """
        cap = self._cap
        while not self._stop_reader.is_set() and my_epoch == self._epoch:
            ret, frame = cap.read()
            if my_epoch != self._epoch or self._stop_reader.is_set():
                return  # Abandoned or stopped while this read() was blocked.

            if not ret:
                _logger.warning(
                    "Frame read failed on live source '%s'. Attempting reconnect...",
                    self.source,
                )
                reconnected = self._reconnect(my_epoch)
                if my_epoch != self._epoch or self._stop_reader.is_set():
                    return
                if not reconnected:
                    self._reader_failed.set()
                    with self._frame_cond:
                        self._frame_cond.notify_all()
                    return
                cap = self._cap
                continue

            with self._frame_cond:
                if my_epoch != self._epoch:
                    return
                self._latest_frame = frame
                self._latest_frame_id += 1
                self._frame_cond.notify_all()

    def _read_live_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        with self._frame_cond:
            got_new = self._frame_cond.wait_for(
                lambda: self._latest_frame_id != self._consumed_frame_id
                or self._reader_failed.is_set(),
                timeout=self.stall_timeout,
            )
            if got_new and not self._reader_failed.is_set():
                frame = self._latest_frame
                self._consumed_frame_id = self._latest_frame_id
                self._frame_count += 1
                return True, frame

        # Either the reader gave up after exhausting its own reconnect
        # retries (_reader_failed), or no frame arrived within
        # stall_timeout at all -- meaning the reader thread is most likely
        # blocked inside a hung read() that may never return. Don't wait
        # on it: abandon it (it will exit harmlessly on its own if it ever
        # wakes up, via the epoch check) and open a fresh connection.
        _logger.warning(
            "No usable frame from '%s' within %.1fs — abandoning stalled "
            "reader and reconnecting.",
            self.source, self.stall_timeout,
        )
        try:
            self._cap = self._open_capture_with_retries()
        except RuntimeError:
            _logger.critical("Reconnect to '%s' exhausted all retries.", self.source)
            return False, None

        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._reader_failed.clear()
        with self._frame_cond:
            self._latest_frame = None
            self._latest_frame_id = 0
            self._consumed_frame_id = 0
        self._start_reader_thread()
        return False, None

    # ------------------------------------------------------------------
    # Shared connect/reconnect
    # ------------------------------------------------------------------

    @staticmethod
    def _is_live_source(source: str) -> bool:
        """Return True if the source is a live stream/camera, not a file."""
        src = str(source).lower()
        return (src.startswith("rtsp://") or src.startswith("rtmp://") or
                src.startswith("http://") or src.startswith("https://") or
                src.isdigit())  # webcam index

    def _open_capture_with_retries(self) -> cv2.VideoCapture:
        for attempt in range(1, self.max_retries + 1):
            cap = cv2.VideoCapture(self.source)
            if cap.isOpened():
                return cap

            _logger.warning(
                "Failed to open video source '%s' (attempt %d/%d). "
                "Retrying in 5s...",
                self.source, attempt, self.max_retries,
            )
            cap.release()
            # Interruptible wait: a live camera whose stream is gone retries
            # indefinitely, so an uninterruptible sleep here would make STOP
            # take up to max_retries*5s to be noticed.
            if self._stop_reader.wait(timeout=5):
                break

        msg = (
            f"Could not open video source '{self.source}' "
            f"after {self.max_retries} attempts."
        )
        _logger.critical(msg)
        raise RuntimeError(msg)

    def _reconnect(self, my_epoch: Optional[int] = None) -> bool:
        """Attempt to reconnect to the video source.

        `my_epoch` (when given, i.e. called from the background reader loop)
        is rechecked right before publishing the new capture to `self._cap`.
        Without this, a reader thread whose epoch was already superseded by
        `_read_live_frame()`'s own stall-timeout reconnect could finish this
        method AFTER the new epoch's reader thread has started, and clobber
        `self._cap` with its own (now-abandoned) capture -- orphaning the
        actually-active one.
        """
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        for attempt in range(1, self.max_retries + 1):
            if self._stop_reader.is_set():
                return False
            if my_epoch is not None and my_epoch != self._epoch:
                return False
            _logger.warning(
                "Reconnect attempt %d/%d for '%s'...",
                attempt, self.max_retries, self.source,
            )
            cap = cv2.VideoCapture(self.source)
            if cap.isOpened():
                if self.is_live:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if my_epoch is not None and my_epoch != self._epoch:
                    # Superseded while this cv2.VideoCapture() call was in
                    # flight. Release our own capture instead of publishing
                    # it -- self._cap already belongs to the newer epoch.
                    cap.release()
                    return False
                self._cap = cap
                _logger.info("Reconnected to '%s'", self.source)
                return True
            cap.release()
            if self._stop_reader.wait(timeout=5):
                return False

        _logger.critical(
            "Could not reconnect to '%s' after %d attempts. Halting.",
            self.source, self.max_retries,
        )
        return False
