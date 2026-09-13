"""
Continuous MJPEG preview streams for the dashboard's camera wall.

Why this exists: Streamlit can only change what is on screen by re-running the
page. The camera wall used to show each camera's latest annotated frame as an
st.image, so the picture could only change when the whole page re-ran --
every two seconds while processing. The pipeline was publishing up to eight
frames a second; the operator saw one every two seconds, with the page
rebuilding around it each time. That is the "not running smoothly".

An MJPEG stream (multipart/x-mixed-replace) is what a browser plays natively
and continuously in a plain <img> tag -- the standard way to show a Python
vision pipeline's output live on a web page, short of WebRTC. The image
element updates on its own as frames arrive, independently of Streamlit's
page re-runs, so the video is as smooth as the pipeline produces it.

Design:
  * A small threaded HTTP server inside the dashboard process, started once.
  * GET /stream/<camera_id>  multipart MJPEG of data/live_frames/<id>.jpg,
                             pushing a new part whenever the file changes.
  * GET /frame/<camera_id>   the current frame as a single JPEG.
  * Serves ONLY <camera_id>.jpg from the configured frames directory; the id
    is validated, so no path can be requested.
  * Until a camera has produced a frame, the stream sends a small
    placeholder instead of failing, so the <img> stays connected and starts
    playing by itself the moment frames appear.
"""

from __future__ import annotations

import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

_logger = get_logger("cameras.preview_server")

_BOUNDARY = "alprpreview"
_CAMERA_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# How often a stream checks for a new frame. Kept well below the publish
# interval: at 40ms, deliveries snapped to 80ms gaps (two ticks) and frames
# published close together were merged -- measured 13.5fps delivered against
# a 15fps publisher. A stat() every 10ms per open stream is negligible.
_POLL_SECONDS = 0.01
# A JPEG caught mid-rename can be missing its end-of-image marker.
_JPEG_EOI = b"\xff\xd9"


def _placeholder_jpeg(text: str = "Waiting for video") -> bytes:
    """A small dark JPEG with a message, shown before a camera has frames."""
    try:
        import cv2
        import numpy as np

        image = np.full((270, 480, 3), 32, dtype=np.uint8)
        (width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
        cv2.putText(image, text, ((480 - width) // 2, 142),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1, cv2.LINE_AA)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            return encoded.tobytes()
    except Exception as exc:
        _logger.debug("Could not render the placeholder frame: %s", exc)
    return b""


class _PreviewHandler(BaseHTTPRequestHandler):
    frames_dir: Path = Path("data/live_frames")
    placeholder: bytes = b""
    stop_event: threading.Event = threading.Event()

    # Silence the default per-request stderr logging; a stream is one long
    # request and the dashboard polls nothing, so there is nothing useful.
    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        return

    def _frame_path(self, camera_id: str) -> Optional[Path]:
        if not _CAMERA_ID.match(camera_id):
            return None
        return self.frames_dir / f"{camera_id}.jpg"

    def _read_frame(self, path: Path) -> Optional[bytes]:
        try:
            data = path.read_bytes()
        except OSError:
            return None
        return data if data.endswith(_JPEG_EOI) else None

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        parts = self.path.split("?", 1)[0].strip("/").split("/")
        if len(parts) != 2 or parts[0] not in ("stream", "frame"):
            self.send_error(404, "Use /stream/<camera_id> or /frame/<camera_id>")
            return

        path = self._frame_path(parts[1])
        if path is None:
            self.send_error(400, "Invalid camera id")
            return

        if parts[0] == "frame":
            self._send_single(path)
        else:
            self._send_stream(path)

    def _send_single(self, path: Path) -> None:
        data = self._read_frame(path) or self.placeholder
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_stream(self, path: Path) -> None:
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY}")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()

        last_mtime: Optional[float] = None
        sent_placeholder = False
        last_send = 0.0
        try:
            while not self.stop_event.is_set():
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    mtime = None

                data = None
                if mtime is not None and mtime != last_mtime:
                    data = self._read_frame(path)
                    if data is not None:
                        last_mtime = mtime
                        sent_placeholder = False
                elif mtime is None and not sent_placeholder:
                    data = self.placeholder
                    sent_placeholder = True

                now = time.monotonic()
                # Re-send the current frame now and then even when nothing
                # changed. Some browsers only paint a multipart part once the
                # NEXT one begins, which would otherwise hold a paused
                # camera's last frame back indefinitely.
                if data is None and now - last_send > 2.0:
                    data = self._read_frame(path) if mtime is not None else self.placeholder

                if data:
                    self.wfile.write(
                        (f"--{_BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                         f"Content-Length: {len(data)}\r\n\r\n").encode("ascii")
                        + data + b"\r\n"
                    )
                    self.wfile.flush()
                    last_send = now

                time.sleep(_POLL_SECONDS)
        except (BrokenPipeError, ConnectionResetError, OSError):
            # The browser navigated away or the panel was removed -- normal.
            return


class PreviewServer:
    """Owns the background HTTP server. One per process."""

    def __init__(self, frames_dir: Path, host: str = "0.0.0.0", port: int = 8765) -> None:
        self.frames_dir = Path(frames_dir)
        self.host = host
        self.port = int(port)
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.error: Optional[str] = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        """Start serving. Returns False (with self.error set) if it could not.

        Never raises: a port that is already taken must cost the dashboard its
        smooth previews, not the whole page. The dashboard falls back to
        still frames in that case and says why.
        """
        if self.running:
            return True
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        handler = type("BoundPreviewHandler", (_PreviewHandler,), {
            "frames_dir": self.frames_dir,
            "placeholder": _placeholder_jpeg(),
            "stop_event": self._stop,
        })
        try:
            self._server = ThreadingHTTPServer((self.host, self.port), handler)
        except OSError as exc:
            self.error = f"port {self.port} is unavailable ({exc.strerror or exc})"
            _logger.warning("Camera preview server not started: %s", self.error)
            return False

        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="camera-preview-server", daemon=True
        )
        self._thread.start()
        _logger.info("Camera preview streams on http://%s:%d/stream/<camera_id>",
                     self.host, self.port)
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


_server: Optional[PreviewServer] = None
_server_lock = threading.Lock()


def get_preview_server(frames_dir: Path, host: str, port: int) -> PreviewServer:
    """This process's preview server, started on first use.

    A module-level singleton for the same reason as the camera manager:
    Streamlit re-runs its script constantly, and a server created per rerun
    would fight itself for the port.
    """
    global _server
    with _server_lock:
        if _server is None:
            _server = PreviewServer(frames_dir, host=host, port=port)
            _server.start()
        return _server
