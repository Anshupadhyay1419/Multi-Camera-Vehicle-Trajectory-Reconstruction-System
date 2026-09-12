"""
VideoSource abstraction -- the seam between "where frames come from" and
everything that consumes them.

    VideoSource (ABC)
      ├── UploadSource   recorded file, finite, has a known frame count
      └── RTSPSource     live stream, unbounded, reconnects on failure

The ALPR pipeline never constructs either directly: it asks
`create_video_source(camera, config)` and gets whichever one the camera's
`source_type` names. Migrating a demo site to a real camera is therefore a
config edit (`source_type: upload` -> `rtsp`, fill in `rtsp_url`) and
nothing else -- no pipeline change, no dashboard change, no new code path.

Both subclasses delegate the actual capture to the existing
`capture.FrameCapture`, which already handles RTSP reconnection, live-source
frame dropping and file frame-skip. This layer adds only what the
multi-camera scheduler needs on top and FrameCapture has no opinion about:
a total frame count for progress reporting, a source-kind flag, and
open-time validation with messages that name the camera rather than the URI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from src.cameras.models import CameraConfig, SourceType
from src.utils.logger import get_logger

_logger = get_logger("cameras.sources")


class VideoSourceError(Exception):
    """Raised when a source cannot be opened or is misconfigured."""


class VideoSource(ABC):
    """A camera's stream of frames, open to close.

    Implementations wrap `FrameCapture` and are used as context managers:

        with create_video_source(camera, config) as source:
            while True:
                ok, frame = source.read()
                if not ok:
                    break

    `read()` returning False means *this source is finished* -- for a file,
    the end; for a live stream, a failure FrameCapture could not recover
    from. A transient live stall is handled inside FrameCapture and never
    surfaces here, so the consumer's loop stays identical for both kinds.
    """

    def __init__(self, camera: CameraConfig, uri: str, config: dict) -> None:
        self.camera = camera
        self.uri = uri
        self._config = config
        self._capture = None  # lazily built in open(), so __init__ never does I/O

    # ── contract ──────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def source_type(self) -> SourceType:
        """Which kind of source this is."""

    @property
    @abstractmethod
    def is_live(self) -> bool:
        """True for an unbounded stream, False for a finite recording.

        Consumers use this to decide whether "no frame" means retry or stop,
        and whether a progress percentage is meaningful at all.
        """

    @abstractmethod
    def total_frames(self) -> int:
        """Frame count for progress reporting, or 0 when unknowable."""

    @abstractmethod
    def validate(self) -> None:
        """Check the URI is usable before any capture is attempted.

        Raises:
            VideoSourceError: with a message naming the camera, so an
                operator reading the dashboard log knows which of four
                identical-looking failures is theirs.
        """

    # ── shared behaviour ──────────────────────────────────────────────────

    def open(self) -> None:
        """Validate, then open the underlying capture.

        Raises:
            VideoSourceError: The source is invalid or could not be opened.
        """
        self.validate()

        # Imported here rather than at module scope: FrameCapture pulls in
        # OpenCV, and the API process imports this module purely for the
        # CameraConfig-to-source mapping without ever opening a stream.
        from src.capture.frame_capture import FrameCapture

        video_cfg = dict(self._config.get("video", {}))
        try:
            self._capture = FrameCapture(
                source=self.uri,
                frame_skip=int(video_cfg.get("frame_skip", 1)),
                max_retries=int(video_cfg.get("max_retries", 5)),
                stall_timeout=float(video_cfg.get("stall_timeout", 10.0)),
            )
            self._capture.open()
        except Exception as exc:
            self._capture = None
            raise VideoSourceError(
                f"Camera {self.camera.camera_id} ({self.camera.camera_name}): "
                f"cannot open {self.source_type.value} source {self.uri!r}: {exc}"
            ) from exc

        _logger.info(
            "Camera %s: opened %s source %r (live=%s, frames=%d)",
            self.camera.camera_id, self.source_type.value, self.uri,
            self.is_live, self.total_frames(),
        )

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Return the next frame, or (False, None) when the source is done."""
        if self._capture is None:
            raise VideoSourceError(
                f"Camera {self.camera.camera_id}: read() before open()"
            )
        return self._capture.read_frame()

    def release(self) -> None:
        """Release the capture. Safe to call more than once."""
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception as exc:
                # A failure to release must never mask the real error that
                # is usually already unwinding the stack around it.
                _logger.warning(
                    "Camera %s: error releasing source: %s",
                    self.camera.camera_id, exc,
                )
            finally:
                self._capture = None

    @property
    def capture(self):
        """The underlying FrameCapture, or None before open()."""
        return self._capture

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, *exc_info) -> None:
        self.release()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(camera={self.camera.camera_id!r}, "
            f"uri={self.uri!r})"
        )


class UploadSource(VideoSource):
    """A recorded video file.

    Finite and seekable, so progress is a real percentage and the consumer
    can trust that read() returning False means the clip ended rather than
    that something broke.
    """

    @property
    def source_type(self) -> SourceType:
        return SourceType.UPLOAD

    @property
    def is_live(self) -> bool:
        return False

    def validate(self) -> None:
        path = Path(self.uri)
        if not path.exists():
            raise VideoSourceError(
                f"Camera {self.camera.camera_id} ({self.camera.camera_name}): "
                f"video file not found: {self.uri!r}. Upload a video for this "
                f"camera, or set video_path in camera_config.yaml."
            )
        if not path.is_file():
            raise VideoSourceError(
                f"Camera {self.camera.camera_id} ({self.camera.camera_name}): "
                f"video_path {self.uri!r} is a directory, not a file"
            )
        if path.stat().st_size == 0:
            raise VideoSourceError(
                f"Camera {self.camera.camera_id} ({self.camera.camera_name}): "
                f"video file is empty: {self.uri!r}"
            )

        # Confirm the file is actually decodable, not merely present. Without
        # this, a mis-named or truncated file passes validation and only
        # fails inside FrameCapture.open(), which retries five times at five
        # second intervals first -- so a wrong upload costs 25 seconds per
        # camera and looks like processing rather than like a mistake. The
        # probe below opens the container and reads one frame; on a real clip
        # it is milliseconds, and it lets the dashboard reject the file the
        # moment it is uploaded.
        if not self._is_decodable(path):
            raise VideoSourceError(
                f"Camera {self.camera.camera_id} ({self.camera.camera_name}): "
                f"{path.name} is not a video this system can read. It may be "
                f"the wrong file, an unsupported codec, or a truncated upload."
            )

    def _is_decodable(self, path: Path) -> bool:
        """True if OpenCV can open the container and read one frame."""
        try:
            import cv2

            probe = cv2.VideoCapture(str(path))
            try:
                if not probe.isOpened():
                    return False
                ok, frame = probe.read()
                return bool(ok) and frame is not None
            finally:
                probe.release()
        except Exception as exc:
            # An unexpected probe failure must not block a file that might
            # be perfectly good -- let FrameCapture be the judge.
            _logger.debug(
                "Camera %s: could not probe %s (%s); allowing it through",
                self.camera.camera_id, path, exc,
            )
            return True

    def total_frames(self) -> int:
        """Frame count from the container metadata, or 0 if unavailable.

        Read straight from the file rather than from the open capture so the
        scheduler can size a progress bar before processing starts. Some
        containers report a wrong or absent count; 0 is returned for those
        and the dashboard falls back to an indeterminate bar rather than
        showing a percentage it cannot honour.
        """
        try:
            import cv2

            probe = cv2.VideoCapture(self.uri)
            try:
                count = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
            finally:
                probe.release()
            return max(0, count)
        except Exception as exc:
            _logger.debug(
                "Camera %s: could not probe frame count for %r: %s",
                self.camera.camera_id, self.uri, exc,
            )
            return 0


class RTSPSource(VideoSource):
    """A live RTSP/RTMP/HTTP stream, or a local webcam index.

    Unbounded: there is no total frame count and no natural end, so a run
    against one of these is stopped by the operator (or by the scheduler's
    own limits), not by the source running out. FrameCapture's live path
    already keeps only the newest frame and reconnects on failure, so
    nothing extra is needed here beyond URI validation.
    """

    # Schemes FrameCapture's live path recognises. A bare integer (webcam
    # index) is accepted too and checked separately.
    _LIVE_SCHEMES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://")

    @property
    def source_type(self) -> SourceType:
        return SourceType.RTSP

    @property
    def is_live(self) -> bool:
        return True

    def validate(self) -> None:
        uri = (self.uri or "").strip()
        if not uri:
            raise VideoSourceError(
                f"Camera {self.camera.camera_id} ({self.camera.camera_name}): "
                f"source_type is 'rtsp' but rtsp_url is not set in camera_config.yaml"
            )
        if uri.isdigit():
            return  # webcam index, e.g. "0"
        if not uri.lower().startswith(self._LIVE_SCHEMES):
            raise VideoSourceError(
                f"Camera {self.camera.camera_id} ({self.camera.camera_name}): "
                f"rtsp_url {uri!r} is not a recognised stream URL. Expected one "
                f"of {', '.join(self._LIVE_SCHEMES)} or a webcam index."
            )

    def total_frames(self) -> int:
        return 0  # a live stream has no end to measure against


# Registry of source_type -> implementation. Adding a third kind of source
# (an HLS playlist, a directory of images, a file-backed replay of a stream)
# means adding a VideoSource subclass and one entry here; no consumer of
# create_video_source() changes.
_SOURCE_TYPES: dict[SourceType, type[VideoSource]] = {
    SourceType.UPLOAD: UploadSource,
    SourceType.RTSP: RTSPSource,
}


def create_video_source(camera: CameraConfig, config: dict) -> VideoSource:
    """Build the VideoSource this camera's configuration calls for.

    Args:
        camera: The camera whose source is wanted.
        config: The full ALPR config dict (for the `video:` tuning block --
                frame_skip, max_retries, stall_timeout).

    Returns:
        An unopened VideoSource. Call open(), or use it as a context manager.

    Raises:
        VideoSourceError: The camera names no usable source.
    """
    uri = camera.source_uri
    if not uri:
        field = "rtsp_url" if camera.source_type is SourceType.RTSP else "video_path"
        raise VideoSourceError(
            f"Camera {camera.camera_id} ({camera.camera_name}): "
            f"source_type is {camera.source_type.value!r} but {field} is not set"
        )

    source_cls = _SOURCE_TYPES.get(camera.source_type, UploadSource)
    return source_cls(camera, uri, config)
