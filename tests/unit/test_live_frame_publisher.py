"""
Unit tests for LiveFramePublisher's rate control and background encoding.

Both properties came from measuring a real run:

* The old limiter accepted a frame only once a full interval had passed
  since the last write, which rounds the rate DOWN to a multiple of the
  source's frame period -- a 15fps target against a 23fps camera published
  11fps.
* Encoding previews on the pipeline thread slowed ALPR itself.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import pytest

from src.utils.live_frame import LiveFramePublisher


class _FakeClock:
    """A controllable stand-in for time.monotonic.

    The rate control is pure arithmetic over the clock, so it is tested on a
    simulated one. Real sleeps made these tests fail on a busy machine
    (a browser and an IDE open on the Jetson were enough), which says
    nothing about the code.
    """

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


@pytest.fixture()
def clock(monkeypatch):
    import src.utils.live_frame as module

    fake = _FakeClock()
    monkeypatch.setattr(module, "time", fake)
    return fake


def _accepted(publisher: LiveFramePublisher, clock: _FakeClock,
              source_fps: float, seconds: float) -> int:
    """Feed frames at source_fps on the simulated clock; count acceptances."""
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    accepted = 0
    for _ in range(int(round(source_fps * seconds))):
        if publisher.is_due():
            accepted += 1
        publisher.publish(frame)
        clock.now += 1.0 / source_fps
    return accepted


class TestRateControl:
    def test_the_average_rate_meets_the_target_against_a_faster_source(self, tmp_path, clock):
        """Measured regression: 15fps target + 23fps camera used to give 11fps."""
        publisher = LiveFramePublisher(str(tmp_path / "f.jpg"), max_fps=15)
        accepted = _accepted(publisher, clock, source_fps=23, seconds=10)
        assert 148 <= accepted <= 151, f"{accepted / 10:.1f} fps over 10 simulated seconds"

    def test_a_slower_source_publishes_every_frame(self, tmp_path, clock):
        publisher = LiveFramePublisher(str(tmp_path / "f.jpg"), max_fps=15)
        assert _accepted(publisher, clock, source_fps=10, seconds=5) == 50

    def test_after_a_long_pause_it_does_not_burst(self, tmp_path, clock):
        """A stall must not be 'made up' with a burst of back-to-back frames."""
        publisher = LiveFramePublisher(str(tmp_path / "f.jpg"), max_fps=10)
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        publisher.publish(frame)
        clock.now += 5.0                       # the source stalls
        accepted = 0
        for _ in range(20):                    # then 20 frames 1ms apart
            if publisher.is_due():
                accepted += 1
            publisher.publish(frame)
            clock.now += 0.001
        assert accepted <= 2

    def test_the_old_rule_would_have_failed_this(self, tmp_path, clock):
        """Pin the specific failure: a limiter that waits a full interval
        after each write gets ~11.5fps from a 23fps source at a 15fps cap."""
        interval = 1 / 15
        last_write, naive = -1.0, 0
        for i in range(230):
            now = i / 23
            if now - last_write >= interval:
                naive += 1
                last_write = now
        publisher = LiveFramePublisher(str(tmp_path / "f.jpg"), max_fps=15)
        assert naive < 120 < _accepted(publisher, clock, source_fps=23, seconds=10)


class TestWriting:
    def test_frames_are_downscaled_to_max_width(self, tmp_path):
        path = tmp_path / "f.jpg"
        publisher = LiveFramePublisher(str(path), max_fps=30, max_width=960)
        publisher.publish(np.zeros((1080, 1920, 3), dtype=np.uint8))
        image = cv2.imread(str(path))
        assert image.shape[1] == 960
        assert image.shape[0] == 540          # aspect ratio kept

    def test_without_max_width_the_size_is_unchanged(self, tmp_path):
        """The single-gate feed keeps full-size frames."""
        path = tmp_path / "f.jpg"
        LiveFramePublisher(str(path), max_fps=30).publish(
            np.zeros((720, 1280, 3), dtype=np.uint8)
        )
        assert cv2.imread(str(path)).shape[:2] == (720, 1280)


class TestBackgroundEncoding:
    def test_publish_returns_without_doing_the_encode(self, tmp_path, monkeypatch):
        """The pipeline thread must not wait for resizing and encoding."""
        import src.utils.live_frame as module

        real_imencode = cv2.imencode
        encoding_started = threading.Event()

        def slow_imencode(*args, **kwargs):
            encoding_started.set()
            time.sleep(0.3)
            return real_imencode(*args, **kwargs)

        monkeypatch.setattr(module.cv2, "imencode", slow_imencode)
        publisher = LiveFramePublisher(str(tmp_path / "f.jpg"), max_fps=30, background=True)
        try:
            started = time.monotonic()
            publisher.publish(np.zeros((480, 640, 3), dtype=np.uint8))
            assert time.monotonic() - started < 0.1, "publish() blocked on the encode"
            assert encoding_started.wait(2.0), "the background worker never encoded"
        finally:
            publisher.close()

    def test_close_flushes_the_last_frame(self, tmp_path):
        """The camera wall should keep a finished camera's final frame."""
        path = tmp_path / "f.jpg"
        publisher = LiveFramePublisher(str(path), max_fps=30, background=True)
        publisher.publish(np.full((48, 64, 3), 200, dtype=np.uint8))
        publisher.close()
        assert path.exists()
        assert cv2.imread(str(path)).mean() > 150

    def test_close_stops_the_worker_thread(self, tmp_path):
        """One camera run per queue position must not leave threads behind."""
        before = {t.name for t in threading.enumerate() if t.name == "live-frame-writer"}
        publisher = LiveFramePublisher(str(tmp_path / "f.jpg"), background=True)
        publisher.close()
        publisher.close()                       # idempotent
        after = [t for t in threading.enumerate() if t.name == "live-frame-writer"]
        assert len(after) <= len(before)

    def test_only_the_newest_frame_is_written_when_the_encoder_falls_behind(
        self, tmp_path, monkeypatch
    ):
        """No queue can build up, so preview latency cannot grow."""
        import src.utils.live_frame as module

        written_values = []
        real_imencode = cv2.imencode

        def slow_imencode(ext, frame, *args, **kwargs):
            written_values.append(int(frame[0, 0, 0]))
            time.sleep(0.2)
            return real_imencode(ext, frame, *args, **kwargs)

        monkeypatch.setattr(module.cv2, "imencode", slow_imencode)
        publisher = LiveFramePublisher(str(tmp_path / "f.jpg"), max_fps=1000, background=True)
        try:
            for value in range(1, 11):
                publisher.publish(np.full((8, 8, 3), value, dtype=np.uint8))
                time.sleep(0.005)
        finally:
            publisher.close()
        assert len(written_values) < 10
        assert written_values[-1] == 10, "the newest frame was not the last one written"
