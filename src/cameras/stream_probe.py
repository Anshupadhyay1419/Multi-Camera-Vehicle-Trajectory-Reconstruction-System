"""
Fast, bounded checks for live stream URLs.

Why this exists: opening an unreachable RTSP URL through OpenCV blocks for
about 30 seconds per attempt, and FrameCapture retries five times -- so a
camera on a network this device cannot reach froze the processing queue for
roughly three minutes while the dashboard showed "Connecting" / "No frames
yet" and gave no reason. Measured on this Jetson against
rtsp://…@192.168.1.100:554, which is not on any of its networks.

Two checks, both bounded:

  probe_stream()        TCP connect to the stream's host and port. Seconds,
                        never minutes. Answers "can this device reach the
                        camera at all?", which is the question that was
                        taking three minutes to answer.
  grab_preview_frame()  Actually open the stream and read one frame, for the
                        dashboard's "Test stream" button. Only attempted
                        after probe_stream() succeeds, and abandoned after a
                        timeout rather than blocking the page.

Neither raises. Both return a result the UI can show as-is, with any
password in the URL masked.
"""

from __future__ import annotations

import socket
import threading
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from src.utils.logger import get_logger

_logger = get_logger("cameras.stream_probe")

# Port a scheme uses when the URL does not name one.
_DEFAULT_PORTS = {
    "rtsp": 554,
    "rtsps": 322,
    "rtmp": 1935,
    "http": 80,
    "https": 443,
}


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of a reachability or preview check."""

    ok: bool
    message: str
    host: Optional[str] = None
    port: Optional[int] = None


def mask_credentials(url: Optional[str]) -> str:
    """Return *url* with any password replaced by ***.

    Camera URLs routinely embed credentials (rtsp://admin:123456@host/...),
    and the dashboard is a page other people look at. The username is kept,
    because "which account is this using?" is useful when a login fails.
    """
    if not url:
        return ""
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if not parts.password:
        return url
    user = parts.username or ""
    host = parts.hostname or ""
    netloc = f"{user}:***@{host}"
    if parts.port:
        netloc += f":{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def stream_endpoint(url: str) -> Optional[tuple[str, int]]:
    """Extract (host, port) from a stream URL, or None if there isn't one.

    None for a webcam index ("0") and for anything that is not a network
    URL -- there is nothing to connect to, so nothing to probe.
    """
    text = (url or "").strip()
    if not text or text.isdigit():
        return None
    try:
        parts = urlsplit(text)
        port = parts.port
    except ValueError:
        return None
    scheme = (parts.scheme or "").lower()
    if not parts.hostname or scheme not in _DEFAULT_PORTS:
        return None
    return parts.hostname, port or _DEFAULT_PORTS[scheme]


def probe_stream(url: str, timeout: float = 3.0) -> ProbeResult:
    """Check that this device can open a TCP connection to the stream.

    This does not prove the stream plays (wrong path, wrong password and
    wrong codec all pass it), but it cleanly separates the most common
    failure -- a camera on a network this device is not on -- and it does so
    in at most *timeout* seconds instead of OpenCV's ~30 per attempt.
    """
    endpoint = stream_endpoint(url)
    if endpoint is None:
        return ProbeResult(ok=True, message="Local device -- nothing to probe")

    host, port = endpoint
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
    except socket.timeout:
        return ProbeResult(
            ok=False, host=host, port=port,
            message=(
                f"No response from {host}:{port} within {timeout:.0f}s. The camera "
                f"is probably on a network this device is not connected to, or "
                f"it is switched off."
            ),
        )
    except socket.gaierror:
        return ProbeResult(
            ok=False, host=host, port=port,
            message=f"Cannot resolve the camera address {host!r}.",
        )
    except ConnectionRefusedError:
        return ProbeResult(
            ok=False, host=host, port=port,
            message=(
                f"{host} refused the connection on port {port}. The device is "
                f"reachable, but nothing is serving a stream on that port."
            ),
        )
    except OSError as exc:
        return ProbeResult(
            ok=False, host=host, port=port,
            message=f"Cannot reach {host}:{port}: {exc.strerror or exc}.",
        )

    return ProbeResult(ok=True, host=host, port=port,
                       message=f"{host}:{port} is reachable.")


def grab_preview_frame(
    url: str, timeout: float = 12.0
) -> tuple[ProbeResult, Optional[np.ndarray]]:
    """Open the stream and read one frame, giving up after *timeout* seconds.

    Runs the OpenCV read on a daemon thread and waits for it with a deadline.
    A hung network read cannot be interrupted from Python, so on timeout the
    thread is abandoned rather than joined -- the page gets its answer on
    time, and the thread ends on its own when OpenCV finally gives up.

    Returns:
        (result, frame) where frame is a BGR array on success, else None.
    """
    reachable = probe_stream(url)
    if not reachable.ok:
        return reachable, None

    outcome: dict = {}

    def read_one() -> None:
        try:
            import cv2

            source = int(url) if url.strip().isdigit() else url
            capture = cv2.VideoCapture(source)
            try:
                if not capture.isOpened():
                    outcome["error"] = "the stream would not open"
                    return
                ok, frame = capture.read()
                if not ok or frame is None:
                    outcome["error"] = "the stream opened but sent no frame"
                    return
                outcome["frame"] = frame
            finally:
                capture.release()
        except Exception as exc:  # never let the probe thread die noisily
            outcome["error"] = str(exc)

    worker = threading.Thread(target=read_one, name="stream-preview", daemon=True)
    worker.start()
    worker.join(timeout)

    if worker.is_alive():
        return ProbeResult(
            ok=False, host=reachable.host, port=reachable.port,
            message=(
                f"{reachable.host}:{reachable.port} answers, but no video arrived "
                f"within {timeout:.0f}s. Check the stream path and the login."
            ),
        ), None

    if "frame" in outcome:
        height, width = outcome["frame"].shape[:2]
        return ProbeResult(
            ok=True, host=reachable.host, port=reachable.port,
            message=f"Stream is live ({width}x{height}).",
        ), outcome["frame"]

    return ProbeResult(
        ok=False, host=reachable.host, port=reachable.port,
        message=(
            f"{reachable.host}:{reachable.port} answers, but "
            f"{outcome.get('error', 'no frame was received')}. Check the stream "
            f"path and the login."
        ),
    ), None
