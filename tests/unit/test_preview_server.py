"""
Unit tests for the camera wall's MJPEG preview server.

The camera wall used to be choppy because video could only change when the
Streamlit page re-ran (every 2s). These streams let the browser play frames
continuously. The key property tested here is throughput: a stream must
deliver frames at the rate the pipeline publishes them, not at page-refresh
rate.
"""

from __future__ import annotations

import socket
import threading
import time
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

from src.cameras.preview_server import PreviewServer


def _free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]; s.close()
    return port


def _write_frame(path, value: int) -> None:
    image = np.full((120, 160, 3), value % 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(encoded.tobytes())
    tmp.replace(path)


@pytest.fixture()
def server(tmp_path):
    srv = PreviewServer(tmp_path, host="127.0.0.1", port=_free_port())
    assert srv.start(), srv.error
    yield srv
    srv.stop()


def _count_stream_parts(port: int, camera_id: str, seconds: float) -> int:
    """Read the MJPEG stream for a while and count complete JPEG parts."""
    parts = 0
    deadline = time.monotonic() + seconds
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/stream/{camera_id}", timeout=5
    ) as response:
        assert response.headers["Content-Type"].startswith("multipart/x-mixed-replace")
        buffer = b""
        while time.monotonic() < deadline:
            chunk = response.read1(65536)
            if not chunk:
                break
            buffer += chunk
            while True:
                start = buffer.find(b"\xff\xd8")
                end = buffer.find(b"\xff\xd9", start + 2)
                if start == -1 or end == -1:
                    break
                parts += 1
                buffer = buffer[end + 2:]
    return parts


class TestSingleFrame:
    def test_serves_the_current_frame(self, server, tmp_path):
        _write_frame(tmp_path / "CAM001.jpg", 100)
        with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/frame/CAM001") as r:
            data = r.read()
        assert r.headers["Content-Type"] == "image/jpeg"
        assert data.startswith(b"\xff\xd8") and data.endswith(b"\xff\xd9")

    def test_a_camera_with_no_frames_gets_a_placeholder_not_an_error(self, server):
        with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/frame/CAM009") as r:
            data = r.read()
        assert data.startswith(b"\xff\xd8")

    @pytest.mark.parametrize("path", ["/frame/..%2F..%2Fetc", "/frame/a.b", "/stream/x%20y"])
    def test_invalid_camera_ids_are_rejected(self, server, path):
        """Only <camera_id>.jpg from the frames directory is ever served."""
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(f"http://127.0.0.1:{server.port}{path}")
        assert exc.value.code in (400, 404)

    def test_unknown_routes_are_404(self, server):
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(f"http://127.0.0.1:{server.port}/secrets")
        assert exc.value.code == 404


class TestStreamThroughput:
    def test_the_stream_keeps_up_with_a_15fps_publisher(self, server, tmp_path):
        """The actual fix: frames reach the browser at publish rate (~15/s),
        not at the old page-refresh rate of one every two seconds."""
        frame_path = tmp_path / "CAM001.jpg"
        _write_frame(frame_path, 0)
        stop = threading.Event()

        def publisher():
            value = 1
            while not stop.is_set():
                _write_frame(frame_path, value)
                value += 1
                time.sleep(1 / 15)

        writer = threading.Thread(target=publisher, daemon=True)
        writer.start()
        try:
            parts = _count_stream_parts(server.port, "CAM001", seconds=3.0)
        finally:
            stop.set(); writer.join()

        fps = parts / 3.0
        assert fps >= 10, f"stream delivered only {fps:.1f} fps"

    def test_a_stream_opened_before_frames_exist_starts_playing_when_they_do(
        self, server, tmp_path
    ):
        """The panel is on screen before the camera's turn in the queue."""
        frame_path = tmp_path / "CAM002.jpg"

        def publish_later():
            time.sleep(0.8)
            for value in range(20):
                _write_frame(frame_path, value)
                time.sleep(0.05)

        threading.Thread(target=publish_later, daemon=True).start()
        parts = _count_stream_parts(server.port, "CAM002", seconds=2.5)
        # 1 placeholder, then the real frames.
        assert parts >= 5

    def test_a_paused_camera_still_repaints_periodically(self, server, tmp_path):
        """Some browsers only paint a part once the next begins; an idle
        camera must still send its frame again so it does not look blank."""
        _write_frame(tmp_path / "CAM003.jpg", 50)
        parts = _count_stream_parts(server.port, "CAM003", seconds=4.5)
        assert parts >= 2


class TestServerLifecycle:
    def test_a_taken_port_is_reported_not_raised(self, tmp_path):
        """Costs the dashboard its smooth video, never the whole page."""
        blocker = socket.socket()
        blocker.bind(("127.0.0.1", 0)); blocker.listen(1)
        port = blocker.getsockname()[1]
        try:
            srv = PreviewServer(tmp_path, host="127.0.0.1", port=port)
            assert srv.start() is False
            assert "port" in (srv.error or "")
            assert srv.running is False
        finally:
            blocker.close()

    def test_start_is_idempotent(self, server):
        assert server.start() is True
