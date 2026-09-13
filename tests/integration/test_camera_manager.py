"""
Integration tests for the sequential CameraManager (cameras/manager.py).

The one property this module exists to guarantee is that cameras are
processed STRICTLY ONE AT A TIME, in queue order -- the demo depends on it
for monotonic timestamps, and the Jetson depends on it because the ALPR
pipeline already saturates the single GPU. Everything else here protects
that guarantee's usefulness: a failing camera must not take the queue down,
and a stop must be cooperative so buffered plates are still flushed.

A fake runner stands in for the real pipeline, so these run in under a
second with no GPU, no models and no video decoding.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from pathlib import Path

import pytest

from src.cameras.manager import CameraManager, get_camera_manager, reset_camera_manager
from src.cameras.models import CameraState, SessionState, SourceType
from src.cameras.registry import load_camera_registry
from src.cameras.sources import VideoSourceError
from src.cameras.status_store import StatusStore


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
    return str(path)


@pytest.fixture()
def registry(video_file):
    """The shipped four-camera registry, every camera pointed at a real file."""
    registry = load_camera_registry("config/camera_config.yaml")
    for camera in registry.all:
        registry.replace(dataclasses.replace(camera, video_path=video_file))
    return registry


@pytest.fixture()
def store(tmp_path):
    return StatusStore(str(tmp_path / "status.json"))


@pytest.fixture(autouse=True)
def _streams_are_reachable(monkeypatch, request):
    """Treat every test stream URL as reachable unless a test opts out.

    These tests use made-up URLs (rtsp://host/stream) to exercise scheduling
    and source selection; the real network check would turn every one of
    them into a DNS failure. Tests about reachability itself are marked
    `real_probe` and get the genuine check.
    """
    if request.node.get_closest_marker("real_probe"):
        return
    from src.cameras import stream_probe

    monkeypatch.setattr(
        stream_probe, "probe_stream",
        lambda url, timeout=3.0: stream_probe.ProbeResult(ok=True, message="stubbed"),
    )


class RecordingRunner:
    """Fake pipeline that records how it was called and when it overlapped."""

    def __init__(self, detections_per_camera: int = 2, work_seconds: float = 0.02):
        self.calls: list[dict] = []
        self.order: list[str] = []
        self.overlaps: list[list[str]] = []
        self._active: list[str] = []
        self._lock = threading.Lock()
        self._detections = detections_per_camera
        self._work = work_seconds

    def __call__(self, config, camera_meta, session_context, progress,
                 source_override, max_duration_seconds=None, models=None):
        camera_id = camera_meta["camera_id"]
        with self._lock:
            self._active.append(camera_id)
            self.order.append(camera_id)
            if len(self._active) > 1:
                self.overlaps.append(list(self._active))
        self.calls.append({
            "camera_meta": camera_meta,
            "session_context": session_context,
            "source_override": source_override,
            "max_duration_seconds": max_duration_seconds,
        })

        progress.on_start(total_frames=100)
        for frame in range(1, 101, 25):
            if progress.should_stop():
                break
            progress.on_frame(frames_processed=frame, fps=30.0,
                              avg_ocr_ms=4.0, avg_detection_ms=12.0)
            time.sleep(self._work)
        for index in range(self._detections):
            progress.on_detection(
                plate_number=f"DL8CA123{index}", confidence=0.9, plate_image="p.jpg"
            )

        with self._lock:
            self._active.remove(camera_id)
        return {"frames_processed": 100, "events_stored": self._detections,
                "avg_ocr_ms": 4.0, "avg_plate_detection_ms": 12.0}


class TestSequentialExecution:
    def test_cameras_run_one_at_a_time_in_queue_order(self, registry, store):
        """The core guarantee. Any overlap here means two TensorRT contexts
        competing for one GPU, and non-monotonic demo timestamps."""
        runner = RecordingRunner()
        manager = CameraManager({"video": {}}, registry, runner, store)

        manager.start()
        assert manager.wait(timeout=30), "session did not finish"

        assert runner.order == ["CAM001", "CAM002", "CAM003", "CAM004"]
        assert runner.overlaps == [], f"cameras overlapped: {runner.overlaps}"

    def test_each_camera_finishes_before_the_next_starts(self, registry, store):
        """Stronger than 'no overlap': completion must precede the next
        start, since the next camera's events must timestamp later."""
        timeline: list[tuple[str, str, float]] = []

        def runner(config, camera_meta, session_context, progress,
                   source_override, max_duration_seconds=None, models=None):
            camera_id = camera_meta["camera_id"]
            timeline.append((camera_id, "start", time.monotonic()))
            time.sleep(0.03)
            timeline.append((camera_id, "end", time.monotonic()))
            return {"frames_processed": 1, "events_stored": 0}

        manager = CameraManager({"video": {}}, registry, runner, store)
        manager.start()
        assert manager.wait(timeout=30)

        for index in range(0, len(timeline) - 2, 2):
            _, _, this_end = timeline[index + 1]
            _, _, next_start = timeline[index + 2]
            assert this_end <= next_start

    def test_each_camera_gets_its_own_metadata_and_source(self, registry, store):
        """Four independent cameras, even when replaying the same footage:
        no camera may inherit another's identity or queue position."""
        runner = RecordingRunner()
        manager = CameraManager({"video": {}}, registry, runner, store)
        manager.start()
        manager.wait(timeout=30)

        assert [c["camera_meta"]["camera_id"] for c in runner.calls] == [
            "CAM001", "CAM002", "CAM003", "CAM004"
        ]
        assert [c["camera_meta"]["camera_name"] for c in runner.calls] == [
            "India Gate", "Connaught Place", "Karol Bagh", "Kashmere Gate"
        ]
        # Real, distinct coordinates -- nothing hardcoded downstream.
        assert len({c["camera_meta"]["latitude"] for c in runner.calls}) == 4
        # Queue position frozen onto every event this camera stores.
        assert [c["session_context"]["trajectory_order"] for c in runner.calls] == [1, 2, 3, 4]

    def test_one_session_id_spans_the_whole_run(self, registry, store):
        runner = RecordingRunner()
        manager = CameraManager({"video": {}}, registry, runner, store)
        session_id = manager.start()
        manager.wait(timeout=30)

        sessions = {c["session_context"]["processing_session"] for c in runner.calls}
        assert sessions == {session_id}

    def test_only_the_requested_cameras_run(self, registry, store):
        runner = RecordingRunner()
        manager = CameraManager({"video": {}}, registry, runner, store)
        manager.start(camera_ids=["CAM003", "CAM001"])
        manager.wait(timeout=30)
        # Restricted, but still in registry order rather than request order.
        assert runner.order == ["CAM001", "CAM003"]

    def test_disabled_cameras_are_never_queued(self, registry, store):
        registry.replace(dataclasses.replace(registry.require("CAM002"), enabled=False))
        runner = RecordingRunner()
        manager = CameraManager({"video": {}}, registry, runner, store)
        manager.start()
        manager.wait(timeout=30)
        assert "CAM002" not in runner.order


class TestFailureIsolation:
    def test_a_camera_with_no_video_fails_alone(self, registry, store):
        registry.replace(
            dataclasses.replace(registry.require("CAM002"), video_path="/nope/missing.mp4")
        )
        runner = RecordingRunner()
        manager = CameraManager({"video": {}}, registry, runner, store)
        manager.start()
        manager.wait(timeout=30)

        # The bad source is caught before the pipeline is ever invoked.
        assert runner.order == ["CAM001", "CAM003", "CAM004"]
        states = {c["camera_id"]: c["state"] for c in manager.get_status()["cameras"]}
        assert states["CAM002"] == CameraState.FAILED.value
        assert states["CAM001"] == states["CAM003"] == CameraState.COMPLETED.value

    @pytest.mark.real_probe
    def test_an_unreachable_stream_fails_in_seconds_with_a_reason(self, registry, store):
        """Reported bug: an offline RTSP camera froze the queue for ~3 minutes
        (5 OpenCV opens x ~30s) while the dashboard showed "No frames yet"."""
        import socket

        closed = socket.socket(); closed.bind(("127.0.0.1", 0))
        port = closed.getsockname()[1]; closed.close()

        runner = RecordingRunner()
        manager = CameraManager({"video": {}}, registry, runner, store)
        manager.set_rtsp_url("CAM001", f"rtsp://admin:pw123@127.0.0.1:{port}/11")

        started = time.monotonic()
        manager.start(camera_ids=["CAM001", "CAM002"])
        assert manager.wait(timeout=30)

        assert time.monotonic() - started < 10.0
        assert "CAM001" not in runner.order, "the pipeline was started on a dead stream"
        assert runner.order == ["CAM002"], "the queue did not move on"
        camera = next(c for c in manager.get_status()["cameras"] if c["camera_id"] == "CAM001")
        assert camera["state"] == CameraState.FAILED.value
        assert "refused" in camera["error"]
        logs = " ".join(manager.get_status()["logs"])
        assert "pw123" not in logs, "the stream password leaked into the log"

    def test_a_pipeline_crash_fails_one_camera_and_the_queue_continues(self, registry, store):
        def runner(config, camera_meta, session_context, progress,
                   source_override, max_duration_seconds=None, models=None):
            if camera_meta["camera_id"] == "CAM002":
                raise RuntimeError("TensorRT context lost")
            return {"frames_processed": 10, "events_stored": 1}

        manager = CameraManager({"video": {}}, registry, runner, store)
        manager.start()
        manager.wait(timeout=30)

        status = manager.get_status()
        states = {c["camera_id"]: c["state"] for c in status["cameras"]}
        assert states["CAM002"] == CameraState.FAILED.value
        assert states["CAM004"] == CameraState.COMPLETED.value
        assert status["state"] == SessionState.FAILED.value    # reported, not hidden
        error = next(c["error"] for c in status["cameras"] if c["camera_id"] == "CAM002")
        assert "TensorRT context lost" in error

    def test_continue_on_error_false_stops_the_queue(self, registry, store):
        registry._settings["processing"] = {"continue_on_error": False}
        runner = RecordingRunner()

        def failing(config, camera_meta, session_context, progress,
                    source_override, max_duration_seconds=None, models=None):
            if camera_meta["camera_id"] == "CAM002":
                raise RuntimeError("stop everything")
            return runner(config, camera_meta, session_context, progress, source_override)

        manager = CameraManager({"video": {}}, registry, failing, store)
        manager.start()
        manager.wait(timeout=30)
        assert "CAM004" not in runner.order


class TestSessionControl:
    def test_cannot_start_two_sessions_at_once(self, registry, store):
        def slow(config, camera_meta, session_context, progress,
                 source_override, max_duration_seconds=None, models=None):
            time.sleep(0.4)
            return {}

        manager = CameraManager({"video": {}}, registry, slow, store)
        manager.start()
        with pytest.raises(RuntimeError, match="already running"):
            manager.start()
        manager.stop()
        manager.wait(timeout=30)

    def test_start_with_no_eligible_cameras_raises(self, registry, store):
        for camera in registry.all:
            registry.replace(dataclasses.replace(camera, enabled=False))
        manager = CameraManager({"video": {}}, registry, RecordingRunner(), store)
        with pytest.raises(RuntimeError, match="No cameras to process"):
            manager.start()

    def test_stop_is_cooperative_and_reaches_the_pipeline(self, registry, store):
        """The pipeline must SEE the stop (so its final flush still runs),
        not be killed."""
        saw_stop = threading.Event()

        def runner(config, camera_meta, session_context, progress,
                   source_override, max_duration_seconds=None, models=None):
            for _ in range(400):
                if progress.should_stop():
                    saw_stop.set()
                    break
                time.sleep(0.01)
            return {"frames_processed": 1, "events_stored": 0}

        manager = CameraManager({"video": {}}, registry, runner, store)
        manager.start()
        time.sleep(0.15)
        assert manager.stop() is True
        assert manager.wait(timeout=30)
        assert saw_stop.is_set(), "the pipeline never observed the stop request"
        assert manager.get_status()["state"] == SessionState.CANCELLED.value

    def test_stop_when_idle_reports_nothing_to_stop(self, registry, store):
        manager = CameraManager({"video": {}}, registry, RecordingRunner(), store)
        assert manager.stop() is False

    def test_cancelled_run_marks_the_untouched_cameras_skipped(self, registry, store):
        def runner(config, camera_meta, session_context, progress,
                   source_override, max_duration_seconds=None, models=None):
            for _ in range(200):
                if progress.should_stop():
                    break
                time.sleep(0.01)
            return {}

        manager = CameraManager({"video": {}}, registry, runner, store)
        manager.start()
        time.sleep(0.1)
        manager.stop()
        manager.wait(timeout=30)
        states = [c["state"] for c in manager.get_status()["cameras"]]
        assert CameraState.PENDING.value not in states, "a queued camera was left pending"


class TestStatusReporting:
    def test_progress_is_published_for_another_process_to_read(self, registry, tmp_path):
        """The dashboard and the API are usually not the process running the
        queue, so status has to survive the process boundary."""
        path = tmp_path / "status.json"
        manager = CameraManager({"video": {}}, registry, RecordingRunner(), StatusStore(str(path)))
        session_id = manager.start()
        manager.wait(timeout=30)

        # A completely independent reader, as another process would be.
        published = StatusStore(str(path)).read()
        assert published is not None
        assert published["session_id"] == session_id
        assert published["state"] == SessionState.COMPLETED.value
        assert published["total_detections"] == 8       # 4 cameras x 2
        assert [c["camera_id"] for c in published["cameras"]] == [
            "CAM001", "CAM002", "CAM003", "CAM004"
        ]

    def test_detections_and_plates_are_counted_per_camera(self, registry, store):
        manager = CameraManager({"video": {}}, registry, RecordingRunner(detections_per_camera=3), store)
        manager.start()
        manager.wait(timeout=30)
        for camera in manager.get_status()["cameras"]:
            assert camera["detections"] == 3
            assert camera["unique_plates"] == 3
            assert camera["percent"] == 100.0

    def test_idle_status_before_anything_runs(self, registry, store):
        manager = CameraManager({"video": {}}, registry, RecordingRunner(), store)
        assert manager.get_status()["state"] == SessionState.IDLE.value


class TestUploads:
    def test_upload_gives_each_camera_its_own_file(self, registry, store, tmp_path):
        """The demo uploads ONE video four times; the cameras must still be
        independent, exactly as four real streams would be."""
        registry._settings["processing"] = {"upload_dir": str(tmp_path / "uploads")}
        manager = CameraManager({"video": {}}, registry, RecordingRunner(), store)

        payload = b"\x00" * 1024
        paths = [
            manager.save_upload(camera_id, "same_recording.mp4", payload)
            for camera_id in ("CAM001", "CAM002", "CAM003", "CAM004")
        ]
        assert len({str(p) for p in paths}) == 4, "cameras shared a file"
        assert all(p.exists() for p in paths)
        assert manager.registry.require("CAM003").video_path == str(paths[2])

    def test_reupload_replaces_rather_than_accumulates(self, registry, store, tmp_path):
        registry._settings["processing"] = {"upload_dir": str(tmp_path / "uploads")}
        manager = CameraManager({"video": {}}, registry, RecordingRunner(), store)
        manager.save_upload("CAM001", "a.mp4", b"x" * 100)
        manager.save_upload("CAM001", "b.mp4", b"y" * 200)
        assert len(list((tmp_path / "uploads").iterdir())) == 1

    def test_empty_upload_is_rejected(self, registry, store):
        manager = CameraManager({"video": {}}, registry, RecordingRunner(), store)
        with pytest.raises(ValueError, match="empty"):
            manager.save_upload("CAM001", "a.mp4", b"")

    def test_upload_for_an_unknown_camera_is_rejected(self, registry, store):
        manager = CameraManager({"video": {}}, registry, RecordingRunner(), store)
        with pytest.raises(KeyError):
            manager.save_upload("NOPE", "a.mp4", b"x" * 10)

    # ── bulk upload ───────────────────────────────────────────────────────

    def test_one_video_is_broadcast_to_every_camera(self, registry, store, tmp_path):
        """The demo case: drop in ONE recording and every virtual camera gets
        it -- but each as its own file, so the cameras stay independent."""
        registry._settings["processing"] = {"upload_dir": str(tmp_path / "uploads")}
        manager = CameraManager({"video": {}}, registry, RecordingRunner(), store)

        saved = manager.assign_uploads([("one_recording.mp4", b"\x00" * 512)])

        assert set(saved) == {"CAM001", "CAM002", "CAM003", "CAM004"}
        paths = [str(p) for p in saved.values()]
        assert len(set(paths)) == 4, "cameras must not share a video_path"
        assert all(Path(p).is_file() for p in paths)
        # And the registry now points each camera at its own copy.
        for camera_id, path in saved.items():
            assert manager.registry.require(camera_id).video_path == str(path)

    def test_one_file_per_camera_is_assigned_in_queue_order(self, registry, store, tmp_path):
        registry._settings["processing"] = {"upload_dir": str(tmp_path / "uploads")}
        manager = CameraManager({"video": {}}, registry, RecordingRunner(), store)

        saved = manager.assign_uploads([
            (f"clip_{index}.mp4", bytes([index]) * 256) for index in range(4)
        ])

        assert list(saved) == ["CAM001", "CAM002", "CAM003", "CAM004"]
        # Queue order, not upload order: the first file went to the first camera.
        assert Path(saved["CAM001"]).read_bytes()[0] == 0
        assert Path(saved["CAM004"]).read_bytes()[0] == 3

    def test_fewer_files_than_cameras_leaves_the_rest_alone(self, registry, store, tmp_path):
        """A partial re-upload must not wipe the cameras it does not cover."""
        registry._settings["processing"] = {"upload_dir": str(tmp_path / "uploads")}
        manager = CameraManager({"video": {}}, registry, RecordingRunner(), store)
        before = manager.registry.require("CAM004").video_path

        saved = manager.assign_uploads([
            ("a.mp4", b"a" * 128), ("b.mp4", b"b" * 128),
        ])

        assert set(saved) == {"CAM001", "CAM002"}
        assert manager.registry.require("CAM004").video_path == before

    def test_more_files_than_cameras_is_rejected(self, registry, store, tmp_path):
        """Silently dropping footage the operator meant to process would be
        worse than refusing the upload."""
        registry._settings["processing"] = {"upload_dir": str(tmp_path / "uploads")}
        manager = CameraManager({"video": {}}, registry, RecordingRunner(), store)

        with pytest.raises(ValueError, match="only 4 camera"):
            manager.assign_uploads([(f"{i}.mp4", b"x" * 64) for i in range(5)])

    def test_empty_bulk_upload_is_rejected(self, registry, store):
        manager = CameraManager({"video": {}}, registry, RecordingRunner(), store)
        with pytest.raises(ValueError, match="No files"):
            manager.assign_uploads([])

    def test_bulk_upload_respects_disabled_cameras(self, registry, store, tmp_path):
        registry._settings["processing"] = {"upload_dir": str(tmp_path / "uploads")}
        registry.replace(dataclasses.replace(registry.require("CAM002"), enabled=False))
        manager = CameraManager({"video": {}}, registry, RecordingRunner(), store)

        saved = manager.assign_uploads([("one.mp4", b"\x00" * 256)])
        assert "CAM002" not in saved
        assert set(saved) == {"CAM001", "CAM003", "CAM004"}

    def test_broadcast_videos_actually_drive_a_full_sequential_run(
        self, registry, store, tmp_path, video_file
    ):
        """End to end: one bulk upload, then four cameras process it in turn."""
        registry._settings["processing"] = {"upload_dir": str(tmp_path / "uploads")}
        runner = RecordingRunner()
        manager = CameraManager({"video": {}}, registry, runner, store)

        manager.assign_uploads([("demo.mp4", Path(video_file).read_bytes())])
        manager.start()
        assert manager.wait(timeout=30)

        assert runner.order == ["CAM001", "CAM002", "CAM003", "CAM004"]
        assert runner.overlaps == []
        # Each camera processed its OWN file, not a shared one.
        sources = [call["source_override"] for call in runner.calls]
        assert len(set(sources)) == 4

    def test_switching_a_camera_to_rtsp_needs_only_the_url(self, registry, store):
        """The production migration path, end to end at the manager level."""
        manager = CameraManager({"video": {}}, registry, RecordingRunner(), store)
        updated = manager.set_rtsp_url("CAM001", "rtsp://192.168.1.50:554/stream")
        assert updated.source_type.value == "rtsp"
        assert updated.source_uri == "rtsp://192.168.1.50:554/stream"
        # Identity and position are untouched -- it is the same site.
        assert updated.camera_name == "India Gate"
        assert updated.order == 1


class TestSourceSelection:
    """Each camera takes exactly one source: an upload OR a stream."""

    def test_setting_a_stream_clears_the_uploaded_video(self, registry, store, tmp_path, video_file):
        registry._settings["processing"] = {"upload_dir": str(tmp_path / "uploads")}
        manager = CameraManager({"video": {}}, registry, RecordingRunner(), store)

        manager.save_upload("CAM001", "clip.mp4", Path(video_file).read_bytes())
        assert manager.registry.require("CAM001").video_path is not None

        manager.set_rtsp_url("CAM001", "rtsp://192.168.1.50:554/stream1")
        camera = manager.registry.require("CAM001")
        assert camera.video_path is None, "the old file is still attached"
        assert camera.rtsp_url == "rtsp://192.168.1.50:554/stream1"
        assert camera.source_type is SourceType.RTSP

    def test_uploading_a_video_clears_the_stream(self, registry, store, tmp_path, video_file):
        registry._settings["processing"] = {"upload_dir": str(tmp_path / "uploads")}
        manager = CameraManager({"video": {}}, registry, RecordingRunner(), store)

        manager.set_rtsp_url("CAM001", "rtsp://host/stream")
        manager.save_upload("CAM001", "clip.mp4", Path(video_file).read_bytes())

        camera = manager.registry.require("CAM001")
        assert camera.rtsp_url is None, "the old stream is still attached"
        assert camera.source_type is SourceType.UPLOAD

    def test_exactly_one_source_field_is_ever_populated(self, registry, store, tmp_path, video_file):
        registry._settings["processing"] = {"upload_dir": str(tmp_path / "uploads")}
        manager = CameraManager({"video": {}}, registry, RecordingRunner(), store)

        for action in (
            lambda: manager.save_upload("CAM001", "a.mp4", Path(video_file).read_bytes()),
            lambda: manager.set_rtsp_url("CAM001", "rtsp://host/a"),
            lambda: manager.save_upload("CAM001", "b.mp4", Path(video_file).read_bytes()),
            lambda: manager.set_rtsp_url("CAM001", "rtsp://host/b"),
        ):
            action()
            camera = manager.registry.require("CAM001")
            assert bool(camera.video_path) ^ bool(camera.rtsp_url), (
                f"both or neither source set: video={camera.video_path} "
                f"rtsp={camera.rtsp_url}"
            )

    def test_a_bad_stream_url_is_rejected_before_it_reaches_the_registry(
        self, registry, store
    ):
        """Caught at entry, not several minutes into a processing run."""
        manager = CameraManager({"video": {}}, registry, RecordingRunner(), store)
        before = manager.registry.require("CAM002")

        with pytest.raises(VideoSourceError, match="not a recognised stream URL"):
            manager.set_rtsp_url("CAM002", "192.168.1.50/stream")

        assert manager.registry.require("CAM002").source_type is before.source_type
        assert manager.registry.require("CAM002").rtsp_url is None

    def test_clear_source_detaches_without_needing_a_replacement(
        self, registry, store, tmp_path, video_file
    ):
        registry._settings["processing"] = {"upload_dir": str(tmp_path / "uploads")}
        manager = CameraManager({"video": {}}, registry, RecordingRunner(), store)

        manager.save_upload("CAM001", "clip.mp4", Path(video_file).read_bytes())
        manager.clear_source("CAM001")

        camera = manager.registry.require("CAM001")
        assert camera.video_path is None and camera.rtsp_url is None
        assert camera.source_uri is None

    def test_a_mixed_queue_runs_both_kinds_of_camera(self, registry, store, tmp_path, video_file):
        """Upload and RTSP cameras in one sequential run -- the pipeline is
        driven identically for both."""
        registry._settings["processing"] = {
            "upload_dir": str(tmp_path / "uploads"),
            "live_duration_seconds": 5,
        }
        runner = RecordingRunner()
        manager = CameraManager({"video": {}}, registry, runner, store)

        manager.set_rtsp_url("CAM001", "rtsp://host/stream")
        for camera_id in ("CAM002", "CAM003", "CAM004"):
            manager.save_upload(camera_id, "clip.mp4", Path(video_file).read_bytes())

        manager.start()
        assert manager.wait(timeout=30)

        assert runner.order == ["CAM001", "CAM002", "CAM003", "CAM004"]
        assert runner.overlaps == []
        sources = [call["source_override"] for call in runner.calls]
        assert sources[0] == "rtsp://host/stream"
        assert all(s.endswith(".mp4") for s in sources[1:])


class TestPreflightChecks:
    """Refuse or skip work that cannot produce anything, rather than
    starting a run that was always going to fail camera by camera."""

    def test_start_is_refused_when_no_camera_has_a_source(self, registry, store):
        """The dashboard disables its START button for this, but the API and
        any script can call start() directly -- so the guard lives here."""
        manager = CameraManager({"video": {}}, registry, RecordingRunner(), store)
        for camera in list(manager.registry.all):
            manager.clear_source(camera.camera_id)

        with pytest.raises(RuntimeError, match="No camera has a source"):
            manager.start()

    def test_cameras_without_a_source_are_skipped_not_failed(
        self, registry, store, video_file
    ):
        """SKIPPED means the operator left it empty; FAILED means something
        broke. Conflating them turns a normal partial run into an error."""
        manager = CameraManager({"video": {}}, registry, RecordingRunner(), store)
        for camera_id in ("CAM001", "CAM003", "CAM004"):
            manager.clear_source(camera_id)

        manager.start()
        assert manager.wait(timeout=30)

        status = manager.get_status()
        states = {c["camera_id"]: c["state"] for c in status["cameras"]}
        assert states["CAM002"] == CameraState.COMPLETED.value
        for camera_id in ("CAM001", "CAM003", "CAM004"):
            assert states[camera_id] == CameraState.SKIPPED.value
        # A run whose only omissions were deliberate is not a failed run.
        assert status["state"] == SessionState.COMPLETED.value

    def test_has_usable_source_tracks_reality(self, registry, store, tmp_path, video_file):
        registry._settings["processing"] = {"upload_dir": str(tmp_path / "uploads")}
        manager = CameraManager({"video": {}}, registry, RecordingRunner(), store)

        manager.clear_source("CAM001")
        assert manager.has_usable_source(manager.registry.require("CAM001")) is False

        manager.set_rtsp_url("CAM001", "rtsp://host/stream")
        assert manager.has_usable_source(manager.registry.require("CAM001")) is True

        saved = manager.save_upload("CAM002", "clip.mp4", Path(video_file).read_bytes())
        assert manager.has_usable_source(manager.registry.require("CAM002")) is True
        # A file that disappears between upload and START is not usable.
        Path(saved).unlink()
        assert manager.has_usable_source(manager.registry.require("CAM002")) is False


class TestForgetSession:
    """After a session's events are deleted, nothing may keep describing it."""

    def _finished(self, registry, store):
        manager = CameraManager({"video": {}}, registry, RecordingRunner(), store)
        session_id = manager.start(camera_ids=["CAM001"])
        assert manager.wait(timeout=30)
        return manager, session_id

    def test_forgetting_the_last_session_returns_the_page_to_idle(self, registry, store):
        """The reported bug: every session deleted, yet the status panel still
        showed "4/4 complete, 88 detections" for a run that no longer existed."""
        manager, session_id = self._finished(registry, store)
        assert manager.get_status()["state"] == SessionState.COMPLETED.value

        assert manager.forget_session(session_id) is True

        status = manager.get_status()
        assert status["state"] == SessionState.IDLE.value
        assert status["cameras"] == []
        assert store.read() is None, "the status file still describes the deleted run"

    def test_another_process_stops_seeing_the_deleted_session_too(self, registry, store):
        """The API and the dashboard are different processes; both read the
        status file, so clearing only the in-memory copy is not enough."""
        manager, session_id = self._finished(registry, store)
        manager.forget_session(session_id)

        fresh = CameraManager({"video": {}}, registry, RecordingRunner(), store)
        assert fresh.get_status()["state"] == SessionState.IDLE.value

    def test_deleting_an_older_session_keeps_the_latest_status(self, registry, store):
        manager, session_id = self._finished(registry, store)

        assert manager.forget_session("SOME-OLDER-RUN") is False
        assert manager.get_status()["session_id"] == session_id
        assert store.read()["session_id"] == session_id

    def test_a_running_session_cannot_be_forgotten(self, registry, store):
        def slow(**kwargs):
            time.sleep(0.5)
            return {}

        manager = CameraManager({"video": {}}, registry, slow, store)
        session_id = manager.start(camera_ids=["CAM001"])
        try:
            assert manager.forget_session(session_id) is False
            assert manager.get_status()["session_id"] == session_id
        finally:
            manager.stop()
            manager.wait(timeout=30)


class TestLiveDwellTime:
    """A live stream never ends, so the queue must bound how long it samples
    one -- otherwise camera 1 runs forever and camera 2 never starts."""

    def test_a_live_camera_is_given_a_time_limit(self, registry, store):
        registry._settings["processing"] = {"live_duration_seconds": 45}
        runner = RecordingRunner()
        manager = CameraManager({"video": {}}, registry, runner, store)

        manager.set_rtsp_url("CAM001", "rtsp://host/stream")
        manager.start(camera_ids=["CAM001"])
        assert manager.wait(timeout=30)

        assert runner.calls[0]["max_duration_seconds"] == 45

    def test_a_recorded_file_is_never_cut_short(self, registry, store):
        """A file ends on its own; imposing a limit would truncate a long
        clip for no reason."""
        registry._settings["processing"] = {"live_duration_seconds": 45}
        runner = RecordingRunner()
        manager = CameraManager({"video": {}}, registry, runner, store)

        manager.start(camera_ids=["CAM002"])
        assert manager.wait(timeout=30)

        assert runner.calls[0]["max_duration_seconds"] is None

    def test_zero_means_unbounded(self, registry, store):
        registry._settings["processing"] = {"live_duration_seconds": 0}
        runner = RecordingRunner()
        manager = CameraManager({"video": {}}, registry, runner, store)

        manager.set_rtsp_url("CAM001", "rtsp://host/stream")
        manager.start(camera_ids=["CAM001"])
        assert manager.wait(timeout=30)

        assert runner.calls[0]["max_duration_seconds"] is None

    def test_a_malformed_setting_falls_back_to_a_usable_default(self, registry, store):
        """A hand-edited config must not leave the queue unbounded by
        accident -- that is the failure that looks like a hang."""
        registry._settings["processing"] = {"live_duration_seconds": "soon"}
        manager = CameraManager({"video": {}}, registry, RecordingRunner(), store)
        assert manager.live_duration_seconds == 60.0


class TestSingleton:
    def test_the_singleton_survives_repeated_lookups(self):
        """Streamlit re-runs its script constantly; a manager rebuilt each
        time would lose the running thread."""
        reset_camera_manager()
        try:
            first = get_camera_manager({"video": {}}, "config/camera_config.yaml")
            second = get_camera_manager({"video": {}}, "config/camera_config.yaml")
            assert first is second
        finally:
            reset_camera_manager()
