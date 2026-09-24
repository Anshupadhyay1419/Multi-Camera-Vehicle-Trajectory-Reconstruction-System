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
import time
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


# ── bounded multi-frame sampling ──────────────────────────────────────────
#
# grab_preview_frame() answers "is there a picture?" in one frame. Measuring
# a stream needs several: a single frame gives no frame interval, so no
# achieved frame rate, and no way to tell a stream that delivers one frame
# and stalls from one that runs. This reads a short burst instead and
# reports what it measured.
#
# It lives here, beside probe_stream(), rather than in a module of its own,
# because it is the same job under the same constraints -- a bounded check
# against a stream URL that must never block its caller for minutes and must
# never raise. FrameCapture is deliberately NOT used: its five reconnect
# attempts at roughly 30s each are right for a pipeline that must survive a
# flaky camera all day, and wrong for an operator waiting on a button.


@dataclass(frozen=True)
class StreamSample:
    """What a short read of a stream measured.

    Every field is optional except `ok` and `message`: a stream can open and
    then deliver nothing, and half-measured is a real outcome that the
    caller has to be able to report rather than a reason to raise.
    """

    ok: bool
    message: str
    host: Optional[str] = None
    port: Optional[int] = None
    #: TCP connect time. The network's contribution, isolated from decoding.
    connect_latency_ms: Optional[float] = None
    #: Time for OpenCV to open the stream: negotiation, not transport.
    open_latency_ms: Optional[float] = None
    #: Open to first decoded frame. What an operator experiences as "lag".
    first_frame_latency_ms: Optional[float] = None
    frames_read: int = 0
    #: Measured from the gaps BETWEEN frames, not frames/total-time, so the
    #: one-off cost of opening the stream does not drag the rate down.
    measured_fps: Optional[float] = None
    #: What the stream CLAIMS, which is often 0 or a nominal 25 over RTSP.
    declared_fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    #: Four-character code, e.g. "hevc"/"avc1". Frequently absent on RTSP.
    fourcc: Optional[str] = None
    frame: Optional[np.ndarray] = None

    @property
    def resolution(self) -> Optional[str]:
        if not self.width or not self.height:
            return None
        return f"{self.width}x{self.height}"


def _decode_fourcc(value: float) -> Optional[str]:
    """Turn OpenCV's CAP_PROP_FOURCC float into its four characters.

    Returns None rather than a string of control characters when the backend
    does not know -- which is common over RTSP, where the container never
    carries a FOURCC and OpenCV reports 0.
    """
    try:
        code = int(value)
    except (TypeError, ValueError):
        return None
    if code <= 0:
        return None
    text = "".join(chr((code >> shift) & 0xFF) for shift in (0, 8, 16, 24))
    cleaned = text.strip().strip("\x00")
    return cleaned if cleaned.isprintable() and cleaned.strip() else None


def sample_stream(
    url: str, *, frames: int = 5, timeout: float = 12.0
) -> StreamSample:
    """Open a stream, read a few frames, measure it, and close it.

    Bounded by *timeout* end to end and never raises: every failure comes
    back as `ok=False` with a message written for an operator rather than a
    stack trace.

    The OpenCV work runs on a daemon thread that is abandoned, not joined, if
    it overruns -- a network read blocked inside a C extension cannot be
    interrupted from Python, so waiting for it would reintroduce exactly the
    multi-minute hang this module exists to avoid. The capture is released in
    that thread's `finally`, so an abandoned thread still frees its socket
    and decoder when OpenCV eventually returns.
    """
    started = time.perf_counter()
    reachable = probe_stream(url)
    connect_ms = (time.perf_counter() - started) * 1000.0
    if not reachable.ok:
        return StreamSample(
            ok=False, message=reachable.message,
            host=reachable.host, port=reachable.port,
            connect_latency_ms=round(connect_ms, 1),
        )

    wanted = max(1, int(frames))
    outcome: dict = {}

    def read_many() -> None:
        capture = None
        try:
            import cv2

            source = int(url) if url.strip().isdigit() else url
            opened_at = time.perf_counter()
            capture = cv2.VideoCapture(source)
            outcome["open_ms"] = (time.perf_counter() - opened_at) * 1000.0
            if not capture.isOpened():
                outcome["error"] = "the stream would not open"
                return

            stamps: list[float] = []
            first_frame: Optional[np.ndarray] = None
            for _ in range(wanted):
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                stamps.append(time.perf_counter())
                if first_frame is None:
                    first_frame = frame

            if not stamps:
                outcome["error"] = "the stream opened but sent no frames"
                return

            outcome["first_frame_ms"] = (stamps[0] - opened_at) * 1000.0
            outcome["stamps"] = stamps
            outcome["frame"] = first_frame
            outcome["width"] = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            outcome["height"] = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            outcome["declared_fps"] = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            outcome["fourcc"] = _decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC))
        except Exception as exc:  # never let the sampling thread die noisily
            outcome["error"] = str(exc)
        finally:
            if capture is not None:
                try:
                    capture.release()
                except Exception:
                    pass

    worker = threading.Thread(target=read_many, name="stream-sample", daemon=True)
    worker.start()
    worker.join(timeout)

    if worker.is_alive():
        return StreamSample(
            ok=False,
            message=(
                f"{reachable.host}:{reachable.port} answers, but a sample of "
                f"{wanted} frame(s) did not finish within {timeout:.0f}s. Check "
                f"the stream path and the login."
            ),
            host=reachable.host, port=reachable.port,
            connect_latency_ms=round(connect_ms, 1),
        )

    if "error" in outcome:
        return StreamSample(
            ok=False,
            message=(
                f"{reachable.host}:{reachable.port} answers, but "
                f"{outcome['error']}. Check the stream path and the login."
            ),
            host=reachable.host, port=reachable.port,
            connect_latency_ms=round(connect_ms, 1),
            open_latency_ms=_rounded(outcome.get("open_ms")),
        )

    stamps = outcome["stamps"]
    measured_fps = None
    if len(stamps) >= 2:
        gaps = [later - earlier for earlier, later in zip(stamps, stamps[1:])]
        average = sum(gaps) / len(gaps)
        if average > 0:
            measured_fps = round(1.0 / average, 1)

    width, height = outcome.get("width") or 0, outcome.get("height") or 0
    if not width or not height:
        # The backend did not report a size; take it from the pixels, which
        # cannot be wrong.
        frame = outcome.get("frame")
        if frame is not None and getattr(frame, "shape", None):
            height, width = frame.shape[:2]

    return StreamSample(
        ok=True,
        message=f"Read {len(stamps)} frame(s) at {width}x{height}.",
        host=reachable.host, port=reachable.port,
        connect_latency_ms=round(connect_ms, 1),
        open_latency_ms=_rounded(outcome.get("open_ms")),
        first_frame_latency_ms=_rounded(outcome.get("first_frame_ms")),
        frames_read=len(stamps),
        measured_fps=measured_fps,
        declared_fps=round(outcome["declared_fps"], 1) if outcome.get("declared_fps") else None,
        width=width or None,
        height=height or None,
        fourcc=outcome.get("fourcc"),
        frame=outcome.get("frame"),
    )


def _rounded(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value, 1)
