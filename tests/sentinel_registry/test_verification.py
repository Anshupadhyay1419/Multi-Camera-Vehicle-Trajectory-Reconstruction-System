"""Camera verification (M1.3): the service and the stream sampler.

No camera and no network. The sampler is injected, so every branch an
operator can hit -- unreachable, timeout, no frames, a stream that opens
and works -- is driven directly, deterministically, in milliseconds.

`sample_stream` itself is exercised against a fake `cv2` module, which is
the only honest way to assert things like "the capture is released even
when the read throws" without a camera that can be made to misbehave.
"""

from __future__ import annotations

import sys
import types
import uuid
from datetime import datetime, timezone

import numpy as np
import pytest

from src.cameras.stream_probe import ProbeResult, StreamSample, sample_stream
from sentinel_system.registry.exceptions import CameraNotFoundError
from sentinel_system.registry.models import Camera
from sentinel_system.verification import ConnectionStatus
from sentinel_system.verification.models import CameraVerification
from sentinel_system.verification.service import CameraVerificationService
from tests.sentinel_registry.conftest import camera_payload


@pytest.fixture()
def camera(session) -> Camera:
    row = Camera(**camera_payload().model_dump())
    session.add(row)
    session.commit()
    return row


def working_sample(**overrides) -> StreamSample:
    data = {
        "ok": True,
        "message": "Read 5 frame(s) at 1920x1080.",
        "host": "10.20.4.11",
        "port": 554,
        "connect_latency_ms": 3.1,
        "open_latency_ms": 12.0,
        "first_frame_latency_ms": 26.9,
        "frames_read": 5,
        "measured_fps": 24.9,
        "declared_fps": 25.0,
        "width": 1920,
        "height": 1080,
        "fourcc": "hevc",
        "frame": np.zeros((1080, 1920, 3), np.uint8),
    }
    data.update(overrides)
    return StreamSample(**data)


def service_with(session, sample: StreamSample, tmp_path=None) -> CameraVerificationService:
    from sentinel_system.core.config import Settings

    settings = Settings()
    if tmp_path is not None:
        settings = Settings(thumbnail_directory=str(tmp_path / "cameras"))
    return CameraVerificationService(
        session, settings=settings, sampler=lambda url, **kw: sample
    )


class TestSuccessfulVerification:
    def test_a_working_camera_verifies(self, session, camera, tmp_path):
        result = service_with(session, working_sample(), tmp_path).verify(camera.id)
        assert result.verified is True
        assert result.connection_status is ConnectionStatus.REACHABLE
        assert result.errors == []

    def test_the_measurements_are_recorded(self, session, camera, tmp_path):
        result = service_with(session, working_sample(), tmp_path).verify(camera.id)
        assert result.measured_resolution == "1920x1080"
        assert result.measured_fps == 24.9
        assert result.frames_sampled == 5
        assert result.codec_detected == "hevc"

    def test_headline_latency_sums_connect_open_and_first_frame(
        self, session, camera, tmp_path
    ):
        """A fast TCP connect to a camera that then takes seconds to show a
        picture is not a fast camera."""
        result = service_with(session, working_sample(), tmp_path).verify(camera.id)
        assert result.measured_latency_ms == pytest.approx(3.1 + 12.0 + 26.9, abs=0.1)
        assert result.connect_latency_ms == 3.1
        assert result.first_frame_latency_ms == 26.9

    def test_the_result_is_persisted(self, session, camera, tmp_path):
        service_with(session, working_sample(), tmp_path).verify(camera.id)
        rows = session.query(CameraVerification).all()
        assert len(rows) == 1 and rows[0].camera_id == camera.id

    def test_the_camera_record_is_never_touched(self, session, camera, tmp_path):
        """Registered spec and measured reality must stay distinguishable:
        a camera commissioned 1080p25 that delivers 704x576 IS the finding."""
        before = (camera.resolution, camera.fps, camera.codec, camera.status)
        service_with(
            session,
            working_sample(width=704, height=576, measured_fps=8.0),
            tmp_path,
        ).verify(camera.id)
        session.refresh(camera)
        assert (camera.resolution, camera.fps, camera.codec, camera.status) == before


class TestThumbnail:
    def test_a_thumbnail_is_written_and_addressable(self, session, camera, tmp_path):
        result = service_with(session, working_sample(), tmp_path).verify(camera.id)
        assert result.thumbnail_path == f"/thumbnails/cameras/{camera.id}.jpg"
        assert (tmp_path / "cameras" / f"{camera.id}.jpg").is_file()

    def test_repeated_verification_reuses_one_file(self, session, camera, tmp_path):
        """One file per camera, so clicking Verify repeatedly cannot fill
        the disk; the history table still records every attempt."""
        service = service_with(session, working_sample(), tmp_path)
        service.verify(camera.id)
        service.verify(camera.id)
        assert len(list((tmp_path / "cameras").glob("*.jpg"))) == 1
        assert session.query(CameraVerification).count() == 2

    def test_a_failed_stream_produces_no_thumbnail(self, session, camera, tmp_path):
        sample = StreamSample(ok=False, message="Cannot resolve the camera address.")
        result = service_with(session, sample, tmp_path).verify(camera.id)
        assert result.thumbnail_path is None

    def test_an_unwritable_thumbnail_does_not_fail_the_verification(
        self, session, camera, tmp_path, monkeypatch
    ):
        """The stream demonstrably delivered frames; a disk problem is
        reported beside that measurement, not instead of it."""
        import cv2

        monkeypatch.setattr(cv2, "imwrite", lambda *a, **k: False)
        result = service_with(session, working_sample(), tmp_path).verify(camera.id)
        assert result.verified is True
        assert result.thumbnail_path is None
        assert any("thumbnail" in e.lower() for e in result.errors)


class TestFailureCases:
    @pytest.mark.parametrize(
        "message,expected",
        [
            ("Cannot resolve the camera address 'cam.local'.", ConnectionStatus.UNREACHABLE),
            ("10.0.0.5 refused the connection on port 554.", ConnectionStatus.UNREACHABLE),
            ("Cannot reach 10.0.0.5:554: No route to host.", ConnectionStatus.UNREACHABLE),
            ("No response from 10.0.0.5:554 within 3s.", ConnectionStatus.TIMEOUT),
            ("10.0.0.5:554 answers, but a sample of 5 frame(s) did not finish within 12s.",
             ConnectionStatus.TIMEOUT),
            ("10.0.0.5:554 answers, but the stream opened but sent no frames.",
             ConnectionStatus.NO_FRAMES),
            ("10.0.0.5:554 answers, but the stream would not open.",
             ConnectionStatus.STREAM_ERROR),
        ],
    )
    def test_each_failure_is_classified_for_the_operator(
        self, session, camera, message, expected
    ):
        result = service_with(session, StreamSample(ok=False, message=message)).verify(
            camera.id
        )
        assert result.connection_status is expected
        assert result.verified is False
        assert result.errors == [message]

    def test_an_unclassifiable_failure_is_still_a_result_not_a_crash(
        self, session, camera
    ):
        sample = StreamSample(ok=False, message="something nobody anticipated")
        result = service_with(session, sample).verify(camera.id)
        assert result.verified is False
        assert result.connection_status is ConnectionStatus.STREAM_ERROR

    def test_a_failure_is_recorded_not_discarded(self, session, camera):
        service_with(session, StreamSample(ok=False, message="boom")).verify(camera.id)
        assert session.query(CameraVerification).count() == 1

    def test_a_failure_has_no_invented_measurements(self, session, camera):
        """Storing zeroes would be a lie that later averages would consume."""
        result = service_with(session, StreamSample(ok=False, message="boom")).verify(
            camera.id
        )
        assert result.measured_fps is None
        assert result.measured_resolution is None
        assert result.frames_sampled == 0

    def test_an_unknown_camera_raises(self, session):
        with pytest.raises(CameraNotFoundError):
            service_with(session, working_sample()).verify(uuid.uuid4())

    def test_a_sampler_that_explodes_is_contained(self, session, camera):
        """A bug in the sampler must not become a 500 on the endpoint."""
        def boom(url, **kwargs):
            raise RuntimeError("driver exploded")

        service = CameraVerificationService(session, sampler=boom)
        result = service.verify(camera.id)
        assert result.verified is False
        assert "driver exploded" in result.verification_message


class TestHistory:
    def test_latest_returns_the_most_recent(self, session, camera, tmp_path):
        service_with(session, StreamSample(ok=False, message="first"), tmp_path).verify(
            camera.id
        )
        service_with(session, working_sample(), tmp_path).verify(camera.id)
        latest = service_with(session, working_sample(), tmp_path).latest(camera.id)
        assert latest.verified is True

    def test_latest_is_none_when_never_verified(self, session, camera):
        assert service_with(session, working_sample()).latest(camera.id) is None

    def test_history_is_newest_first(self, session, camera, tmp_path):
        for n in range(3):
            service_with(
                session, StreamSample(ok=False, message=f"attempt {n}"), tmp_path
            ).verify(camera.id)
        history = service_with(session, working_sample(), tmp_path).history(camera.id)
        assert [h.verification_message for h in history] == [
            "attempt 2", "attempt 1", "attempt 0"
        ]


class TestLogging:
    @pytest.fixture()
    def captured(self):
        import logging

        records: list[logging.LogRecord] = []

        class Collector(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = Collector()
        logger = logging.getLogger("sentinel.verification")
        logger.addHandler(handler)
        try:
            yield records
        finally:
            logger.removeHandler(handler)

    def test_start_and_completion_are_logged(self, session, camera, captured, tmp_path):
        service_with(session, working_sample(), tmp_path).verify(camera.id)
        text = " ".join(r.getMessage() for r in captured)
        assert "Verification started" in text
        assert "Verification completed" in text

    def test_a_failure_is_logged_as_such(self, session, camera, captured):
        service_with(session, StreamSample(ok=False, message="nope")).verify(camera.id)
        text = " ".join(r.getMessage() for r in captured)
        assert "Verification failed" in text

    def test_the_stream_password_is_never_logged(self, session, captured, tmp_path):
        row = Camera(
            **camera_payload(
                camera_code="AHM-SEC-0009",
                stream_url="rtsp://admin:hunter2@10.20.4.11:554/live",
            ).model_dump()
        )
        session.add(row)
        session.commit()
        service_with(session, working_sample(), tmp_path).verify(row.id)
        text = " ".join(r.getMessage() for r in captured)
        assert "hunter2" not in text
        assert "admin:***@10.20.4.11" in text


# ── the sampler itself, against a fake OpenCV ────────────────────────────


class _FakeCapture:
    """Stands in for cv2.VideoCapture, and records that it was released."""

    def __init__(self, frames=5, opened=True, raise_on_read=False, props=None):
        self._frames = frames
        self._opened = opened
        self._raise = raise_on_read
        self._read = 0
        self.released = False
        self._props = props or {}

    def isOpened(self):
        return self._opened

    def read(self):
        if self._raise:
            raise RuntimeError("decoder blew up")
        if self._read >= self._frames:
            return False, None
        self._read += 1
        return True, np.zeros((1080, 1920, 3), np.uint8)

    def get(self, prop):
        return self._props.get(prop, 0)

    def release(self):
        self.released = True


@pytest.fixture()
def fake_cv2(monkeypatch):
    """Install a fake `cv2` and a reachable probe, and hand back the capture."""
    made: list[_FakeCapture] = []
    module = types.ModuleType("cv2")
    module.CAP_PROP_FRAME_WIDTH = 3
    module.CAP_PROP_FRAME_HEIGHT = 4
    module.CAP_PROP_FPS = 5
    module.CAP_PROP_FOURCC = 6

    def factory(*_args, **_kwargs):
        capture = made[0]
        return capture

    module.VideoCapture = factory
    monkeypatch.setitem(sys.modules, "cv2", module)
    monkeypatch.setattr(
        "src.cameras.stream_probe.probe_stream",
        lambda url, timeout=3.0: ProbeResult(
            ok=True, message="reachable", host="10.0.0.5", port=554
        ),
    )
    return made


class TestSampleStream:
    def test_it_measures_a_working_stream(self, fake_cv2):
        import cv2

        fake_cv2.append(
            _FakeCapture(
                frames=5,
                props={
                    cv2.CAP_PROP_FRAME_WIDTH: 1920,
                    cv2.CAP_PROP_FRAME_HEIGHT: 1080,
                    cv2.CAP_PROP_FPS: 25.0,
                    cv2.CAP_PROP_FOURCC: 1748121139,  # 'secv' -> printable
                },
            )
        )
        sample = sample_stream("rtsp://10.0.0.5:554/live", frames=5, timeout=5)
        assert sample.ok is True
        assert sample.frames_read == 5
        assert sample.resolution == "1920x1080"
        assert sample.declared_fps == 25.0
        assert sample.measured_fps is not None and sample.measured_fps > 0

    def test_the_capture_is_always_released(self, fake_cv2):
        """Resource cleanup: the whole point of a bounded check."""
        fake_cv2.append(_FakeCapture(frames=3))
        sample_stream("rtsp://10.0.0.5:554/live", frames=3, timeout=5)
        assert fake_cv2[0].released is True

    def test_the_capture_is_released_even_when_reading_throws(self, fake_cv2):
        fake_cv2.append(_FakeCapture(raise_on_read=True))
        sample = sample_stream("rtsp://10.0.0.5:554/live", frames=3, timeout=5)
        assert sample.ok is False
        assert fake_cv2[0].released is True

    def test_a_stream_that_will_not_open_is_reported(self, fake_cv2):
        fake_cv2.append(_FakeCapture(opened=False))
        sample = sample_stream("rtsp://10.0.0.5:554/live", frames=3, timeout=5)
        assert sample.ok is False and "would not open" in sample.message

    def test_a_stream_that_sends_nothing_is_reported(self, fake_cv2):
        fake_cv2.append(_FakeCapture(frames=0))
        sample = sample_stream("rtsp://10.0.0.5:554/live", frames=3, timeout=5)
        assert sample.ok is False and "sent no frames" in sample.message

    def test_resolution_falls_back_to_the_pixels(self, fake_cv2):
        """Backends that report 0x0 still hand over frames that cannot lie."""
        fake_cv2.append(_FakeCapture(frames=2, props={}))
        sample = sample_stream("rtsp://10.0.0.5:554/live", frames=2, timeout=5)
        assert sample.resolution == "1920x1080"

    def test_an_unreachable_host_never_opens_a_capture(self, monkeypatch):
        """probe_stream is the cheap gate: it exists so an unreachable
        camera costs seconds, not OpenCV's ~30s per attempt."""
        monkeypatch.setattr(
            "src.cameras.stream_probe.probe_stream",
            lambda url, timeout=3.0: ProbeResult(
                ok=False, message="Cannot resolve the camera address 'x'."
            ),
        )
        sample = sample_stream("rtsp://nowhere/live", frames=5, timeout=5)
        assert sample.ok is False
        assert sample.frames_read == 0
        assert sample.connect_latency_ms is not None

    def test_a_single_frame_yields_no_fps(self, fake_cv2):
        """One frame gives no interval, so no rate -- and saying otherwise
        would be inventing a number."""
        fake_cv2.append(_FakeCapture(frames=1))
        sample = sample_stream("rtsp://10.0.0.5:554/live", frames=1, timeout=5)
        assert sample.ok is True and sample.measured_fps is None
