"""
Unit tests for the VideoSource abstraction (cameras/sources.py).

This is the seam that makes the RTSP migration a config edit: the factory
must pick the right implementation from `source_type` alone, and each
implementation must reject a bad URI with a message that names the camera
(one of four identical-looking failures otherwise).
"""

from __future__ import annotations

import dataclasses

import pytest

from src.cameras.models import CameraConfig, SourceType
from src.cameras.sources import (
    RTSPSource,
    UploadSource,
    VideoSourceError,
    create_video_source,
)


@pytest.fixture()
def camera():
    return CameraConfig(
        camera_id="CAM001",
        camera_name="India Gate",
        latitude=28.6129,
        longitude=77.2295,
        order=1,
        source_type=SourceType.UPLOAD,
        video_path="ALPR.mp4",
    )


@pytest.fixture(scope="session")
def video_file(tmp_path_factory):
    """A real, decodable video file.

    Genuinely encoded rather than a few stub bytes, because UploadSource now
    probes that a file can actually be decoded before accepting it -- a check
    that exists so a mis-named upload is rejected in milliseconds instead of
    after 25 seconds of FrameCapture retries. A stub file would (correctly)
    be refused here too, so the fixture has to produce the real thing.

    Tiny and short: these tests never run it through detection, they only
    need something the container parser accepts.
    """
    import cv2
    import numpy as np

    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48)
    )
    if not writer.isOpened():
        pytest.skip("no usable OpenCV video encoder available")
    try:
        for index in range(10):
            frame = np.full((48, 64, 3), index * 20 % 255, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()

    if not path.is_file() or path.stat().st_size == 0:
        pytest.skip("OpenCV produced no video file")
    return path


class TestFactory:
    def test_upload_camera_gets_an_upload_source(self, camera):
        assert isinstance(create_video_source(camera, {}), UploadSource)

    def test_rtsp_camera_gets_an_rtsp_source(self, camera):
        live = dataclasses.replace(
            camera, source_type=SourceType.RTSP, rtsp_url="rtsp://host:554/stream"
        )
        source = create_video_source(live, {})
        assert isinstance(source, RTSPSource)
        assert source.uri == "rtsp://host:554/stream"

    def test_switching_source_type_is_the_only_change_needed(self, camera, video_file):
        """The migration contract: same camera, same consumer, one field."""
        recorded = dataclasses.replace(camera, video_path=str(video_file))
        live = dataclasses.replace(
            recorded, source_type=SourceType.RTSP, rtsp_url="rtsp://host/stream"
        )
        assert create_video_source(recorded, {}).is_live is False
        assert create_video_source(live, {}).is_live is True

    def test_no_source_configured_raises_naming_the_missing_field(self, camera):
        with pytest.raises(VideoSourceError, match="video_path is not set"):
            create_video_source(dataclasses.replace(camera, video_path=None), {})

        live = dataclasses.replace(camera, source_type=SourceType.RTSP, video_path=None)
        with pytest.raises(VideoSourceError, match="rtsp_url is not set"):
            create_video_source(live, {})


class TestUploadValidation:
    def test_a_real_file_validates(self, camera):
        """The repository's sample clip is a genuine video and must pass."""
        create_video_source(camera, {}).validate()

    def test_missing_file_names_the_camera(self, camera):
        source = create_video_source(
            dataclasses.replace(camera, video_path="/nope/missing.mp4"), {}
        )
        with pytest.raises(VideoSourceError) as exc:
            source.validate()
        message = str(exc.value)
        assert "CAM001" in message and "India Gate" in message
        assert "not found" in message

    def test_empty_file_is_rejected(self, camera, tmp_path):
        empty = tmp_path / "empty.mp4"
        empty.touch()
        source = create_video_source(dataclasses.replace(camera, video_path=str(empty)), {})
        with pytest.raises(VideoSourceError, match="empty"):
            source.validate()

    def test_directory_is_rejected(self, camera, tmp_path):
        source = create_video_source(dataclasses.replace(camera, video_path=str(tmp_path)), {})
        with pytest.raises(VideoSourceError, match="directory"):
            source.validate()

    def test_an_undecodable_file_is_rejected_immediately(self, camera, tmp_path):
        """A mis-named or truncated upload must be caught at validation, not
        inside FrameCapture -- which retries five times at five second
        intervals first, so a wrong file otherwise costs 25 seconds per
        camera and looks like processing rather than a mistake."""
        import time

        bogus = tmp_path / "notes.mp4"
        bogus.write_bytes(b"this is plainly not a video" * 200)
        source = create_video_source(
            dataclasses.replace(camera, video_path=str(bogus)), {}
        )

        started = time.monotonic()
        with pytest.raises(VideoSourceError, match="not a video this system can read"):
            source.validate()
        assert time.monotonic() - started < 5.0, "validation should be immediate"

    def test_a_real_video_passes_the_decode_probe_quickly(self, camera):
        import time

        started = time.monotonic()
        create_video_source(camera, {}).validate()
        assert time.monotonic() - started < 5.0

    def test_reading_before_open_is_an_error_not_a_crash(self, camera, video_file):
        source = create_video_source(dataclasses.replace(camera, video_path=str(video_file)), {})
        with pytest.raises(VideoSourceError, match="read\\(\\) before open\\(\\)"):
            source.read()

    def test_release_before_open_is_safe(self, camera, video_file):
        create_video_source(
            dataclasses.replace(camera, video_path=str(video_file)), {}
        ).release()

    def test_total_frames_of_a_real_video(self, camera):
        """The sample clip is a real container, so the count is a real number
        -- this is what sizes the dashboard's progress bar."""
        assert create_video_source(camera, {}).total_frames() > 0

    def test_total_frames_of_an_undecodable_file_is_zero_not_an_error(
        self, camera, tmp_path
    ):
        """0 rather than a raise: the count only sizes a progress bar, so an
        unreadable container costs a percentage, never the run."""
        bogus = tmp_path / "broken.mp4"
        bogus.write_bytes(b"not a container" * 100)
        assert create_video_source(
            dataclasses.replace(camera, video_path=str(bogus)), {}
        ).total_frames() == 0


class TestRTSPValidation:
    @pytest.mark.parametrize(
        "url",
        [
            "rtsp://192.168.1.50:554/stream1",
            "rtsps://cam.example.org/live",
            "rtmp://host/live",
            "http://cam.local/video.mjpg",
            "https://cam.local/video.mjpg",
            "0",          # webcam index
        ],
    )
    def test_accepts_every_supported_stream_form(self, camera, url):
        live = dataclasses.replace(camera, source_type=SourceType.RTSP, rtsp_url=url)
        create_video_source(live, {}).validate()

    def test_rejects_something_that_is_not_a_stream_url(self, camera):
        live = dataclasses.replace(
            camera, source_type=SourceType.RTSP, rtsp_url="/home/ansh/clip.mp4"
        )
        with pytest.raises(VideoSourceError, match="not a recognised stream URL"):
            create_video_source(live, {}).validate()

    def test_blank_url_names_the_camera_and_the_field(self, camera):
        live = dataclasses.replace(camera, source_type=SourceType.RTSP, rtsp_url="   ")
        source = RTSPSource(live, "   ", {})
        with pytest.raises(VideoSourceError) as exc:
            source.validate()
        assert "CAM001" in str(exc.value) and "rtsp_url" in str(exc.value)

    def test_a_live_stream_has_no_frame_count(self, camera):
        """Unbounded: the dashboard must show an indeterminate bar, not 0%."""
        live = dataclasses.replace(
            camera, source_type=SourceType.RTSP, rtsp_url="rtsp://host/s"
        )
        assert create_video_source(live, {}).total_frames() == 0
        assert create_video_source(live, {}).is_live is True


class TestContextManager:
    def test_open_and_release_a_real_video(self, camera):
        source = create_video_source(camera, {"video": {"frame_skip": 1}})
        with source as opened:
            ok, frame = opened.read()
            assert ok and frame is not None
            assert frame.shape[0] > 0
        assert source.capture is None    # released on exit

    def test_exit_releases_even_when_the_body_raises(self, camera):
        source = create_video_source(camera, {"video": {}})
        with pytest.raises(ValueError):
            with source:
                raise ValueError("boom")
        assert source.capture is None

    def test_opening_a_missing_file_raises_before_any_capture(self, camera):
        source = create_video_source(
            dataclasses.replace(camera, video_path="/nope/missing.mp4"), {}
        )
        with pytest.raises(VideoSourceError):
            source.open()
        assert source.capture is None
