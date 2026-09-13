"""
Unit tests for live-stream reachability checks (cameras/stream_probe.py).

Background: an RTSP camera that dropped off the network froze a processing
run for about three minutes -- OpenCV blocks ~30s per open attempt and
FrameCapture retries five times -- while the dashboard showed only
"No frames yet". These checks answer "can this device reach the camera?" in
seconds and say so in words, and the dashboard's Test stream button uses
them to show a real preview frame.

Everything here runs against local sockets, so no camera is needed.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest

from src.cameras import stream_probe
from src.cameras.stream_probe import (
    grab_preview_frame,
    mask_credentials,
    probe_stream,
    stream_endpoint,
)


@pytest.fixture()
def listening_port():
    """A real local TCP port that accepts connections."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(5)
    port = server.getsockname()[1]
    stop = threading.Event()

    def accept_loop():
        server.settimeout(0.2)
        while not stop.is_set():
            try:
                connection, _ = server.accept()
                connection.close()
            except OSError:
                continue

    thread = threading.Thread(target=accept_loop, daemon=True)
    thread.start()
    yield port
    stop.set()
    server.close()


@pytest.fixture()
def closed_port():
    """A local port with nothing listening on it."""
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


class TestMaskCredentials:
    def test_the_password_is_hidden_and_the_user_kept(self):
        assert (
            mask_credentials("rtsp://admin:123456@192.168.1.100:554/11")
            == "rtsp://admin:***@192.168.1.100:554/11"
        )

    def test_a_url_without_credentials_is_unchanged(self):
        url = "rtsp://192.168.1.50:554/stream1"
        assert mask_credentials(url) == url

    def test_empty_and_none_are_safe(self):
        assert mask_credentials("") == ""
        assert mask_credentials(None) == ""


class TestStreamEndpoint:
    @pytest.mark.parametrize("url,expected", [
        ("rtsp://cam.local/live", ("cam.local", 554)),
        ("rtsp://admin:pw@10.0.0.5:8554/a", ("10.0.0.5", 8554)),
        ("http://cam/video.mjpg", ("cam", 80)),
        ("https://cam/video.mjpg", ("cam", 443)),
        ("rtmp://host/app", ("host", 1935)),
    ])
    def test_host_and_default_ports(self, url, expected):
        assert stream_endpoint(url) == expected

    @pytest.mark.parametrize("url", ["0", "", "   ", "/local/file.mp4", "ftp://x/y"])
    def test_nothing_to_probe(self, url):
        assert stream_endpoint(url) is None


class TestProbeStream:
    def test_a_listening_port_is_reachable(self, listening_port):
        result = probe_stream(f"rtsp://127.0.0.1:{listening_port}/live")
        assert result.ok is True
        assert result.port == listening_port

    def test_a_refused_port_is_reported_in_words(self, closed_port):
        result = probe_stream(f"rtsp://127.0.0.1:{closed_port}/live")
        assert result.ok is False
        assert "refused" in result.message

    def test_an_unroutable_address_fails_within_the_timeout(self):
        """The bug being fixed: this used to take ~3 minutes. TEST-NET-1 is
        reserved for documentation and never routed."""
        started = time.monotonic()
        result = probe_stream("rtsp://192.0.2.1:554/live", timeout=1.0)
        assert result.ok is False
        assert time.monotonic() - started < 5.0

    def test_an_unresolvable_host_is_reported(self):
        result = probe_stream("rtsp://no-such-camera.invalid:554/x", timeout=1.0)
        assert result.ok is False

    def test_a_webcam_index_needs_no_probe(self):
        assert probe_stream("0").ok is True

    def test_the_message_never_contains_the_password(self, closed_port):
        result = probe_stream(f"rtsp://admin:secret99@127.0.0.1:{closed_port}/x")
        assert "secret99" not in result.message


class TestGrabPreviewFrame:
    def test_an_unreachable_stream_returns_no_frame_quickly(self, closed_port):
        started = time.monotonic()
        result, frame = grab_preview_frame(f"rtsp://127.0.0.1:{closed_port}/x")
        assert result.ok is False and frame is None
        assert time.monotonic() - started < 5.0

    def test_a_port_that_answers_but_sends_no_video_is_reported(self, listening_port):
        """Reachable is not the same as streaming: wrong path or wrong login
        must still come back as a failure, not a blank preview."""
        result, frame = grab_preview_frame(
            f"rtsp://127.0.0.1:{listening_port}/x", timeout=8.0
        )
        assert result.ok is False
        assert frame is None

    def test_a_hung_read_is_abandoned_at_the_deadline(self, monkeypatch, listening_port):
        """A network read that never returns must not freeze the page."""
        import cv2

        class Hangs:
            def __init__(self, *args): pass
            def isOpened(self):
                time.sleep(30)
                return False
            def release(self): pass

        monkeypatch.setattr(cv2, "VideoCapture", Hangs)
        started = time.monotonic()
        result, frame = grab_preview_frame(
            f"rtsp://127.0.0.1:{listening_port}/x", timeout=1.0
        )
        assert result.ok is False and frame is None
        assert "within 1s" in result.message
        assert time.monotonic() - started < 5.0

    def test_a_recorded_file_through_the_same_path_yields_a_frame(self):
        """The success path, using a real decodable source."""
        result, frame = grab_preview_frame("ALPR.mp4")
        # A file path has no endpoint, so it is opened directly.
        assert result.ok is True
        assert frame is not None and frame.ndim == 3
