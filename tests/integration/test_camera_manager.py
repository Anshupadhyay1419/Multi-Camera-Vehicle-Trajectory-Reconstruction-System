"""
Integration tests for the CameraManager's two schedules (cameras/manager.py).

Recorded files are processed STRICTLY ONE AT A TIME, in queue order -- the
demo depends on it for monotonic timestamps, and the Jetson depends on it
because the ALPR pipeline already saturates the single GPU. Live streams are
the opposite: they all run TOGETHER and until stopped, because a stream that
has to wait its turn does not delay its footage, it loses it. Everything else
here protects those guarantees' usefulness: a failing camera must not take
the queue down, and a stop must be cooperative so buffered plates are still
flushed.

A fake runner stands in for the real pipeline, so these run in under a
second with no GPU, no models and no video decoding.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from datetime import datetime, timezone
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
def registry(video_file, camera_config_file):
    """This suite's own four-camera registry, each camera on a real file.

    Fixed, and in a temp directory (see conftest.camera_config_file): cameras
    are added, renamed and removed from the dashboard, so the deployment's own
    config/camera_config.yaml is operator state -- reading it would make these
    tests depend on how somebody last configured the Jetson, and writing it
    would change their deployment.
    """
    registry = load_camera_registry(str(camera_config_file))
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
        """Upload and RTSP cameras in one run -- the pipeline is driven
        identically for both, and the files go first."""
        registry._settings["processing"] = {"upload_dir": str(tmp_path / "uploads")}
        runner = RecordingRunner()
        manager = CameraManager({"video": {}}, registry, runner, store)

        manager.set_rtsp_url("CAM001", "rtsp://host/stream")
        for camera_id in ("CAM002", "CAM003", "CAM004"):
            manager.save_upload(camera_id, "clip.mp4", Path(video_file).read_bytes())

        manager.start()
        assert manager.wait(timeout=30)

        # Recorded files in queue order and strictly in turn; the live camera
        # afterwards, since it would otherwise hold the files up forever.
        assert runner.order == ["CAM002", "CAM003", "CAM004", "CAM001"]
        assert runner.overlaps == []
        by_camera = {call["camera_meta"]["camera_id"]: call["source_override"]
                     for call in runner.calls}
        assert by_camera["CAM001"] == "rtsp://host/stream"
        assert all(by_camera[c].endswith(".mp4") for c in ("CAM002", "CAM003", "CAM004"))


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


class LiveRunner:
    """Fake pipeline for live streams: runs until stopped, like a real one.

    Records which cameras were inside it at the same moment, which is the
    property the live schedule exists to provide.
    """

    def __init__(self):
        self.calls: list[dict] = []
        self.started = threading.Event()
        self.peak_concurrency = 0
        self._active: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, config, camera_meta, session_context, progress,
                 source_override, max_duration_seconds=None, models=None):
        camera_id = camera_meta["camera_id"]
        with self._lock:
            self.calls.append({
                "camera_id": camera_id,
                "max_duration_seconds": max_duration_seconds,
                "models": models,
            })
            self._active.append(camera_id)
            self.peak_concurrency = max(self.peak_concurrency, len(self._active))
        self.started.set()

        frames = 0
        while not progress.should_stop():
            frames += 1
            progress.on_frame(frames_processed=frames, fps=25.0)
            time.sleep(0.01)

        with self._lock:
            self._active.remove(camera_id)
        return {"frames_processed": frames}


class TestLiveStreamsRunTogether:
    """A live stream carries what is happening NOW.

    Queueing one behind another does not delay its footage, it discards it:
    every vehicle passing camera 2 during camera 1's turn is never seen. So
    live cameras start together and run until they are stopped, while
    recorded files keep their strict one-at-a-time order.
    """

    def _live_manager(self, registry, store, runner, count=4):
        manager = CameraManager({"video": {}}, registry, runner, store)
        for camera in list(registry.enabled)[:count]:
            manager.set_rtsp_url(camera.camera_id, f"rtsp://host/{camera.camera_id}")
        return manager

    def test_every_live_camera_runs_at_the_same_time(self, registry, store):
        runner = LiveRunner()
        manager = self._live_manager(registry, store, runner)

        manager.start()
        try:
            deadline = time.time() + 20
            while runner.peak_concurrency < 4 and time.time() < deadline:
                time.sleep(0.05)
            assert runner.peak_concurrency == 4, "live cameras did not run together"
            assert {call["camera_id"] for call in runner.calls} == {
                camera.camera_id for camera in list(registry.enabled)[:4]
            }
        finally:
            manager.stop()
            assert manager.wait(timeout=30)

    def test_a_live_stream_is_never_cut_short_by_a_clock(self, registry, store):
        """No dwell time: the stream ends when the operator stops it."""
        runner = LiveRunner()
        manager = self._live_manager(registry, store, runner, count=1)

        manager.start(camera_ids=["CAM001"])
        try:
            assert runner.started.wait(timeout=20)
            assert runner.calls[0]["max_duration_seconds"] is None
            # Still running well after any old 60s sampling window would
            # have mattered -- the only thing that ends it is stop().
            time.sleep(0.3)
            assert manager.is_running()
        finally:
            manager.stop()
            assert manager.wait(timeout=30)

    def test_stopping_completes_a_live_camera_rather_than_skipping_it(
        self, registry, store
    ):
        """Being stopped is how a live camera finishes; it is not a camera
        whose clip was cut short."""
        runner = LiveRunner()
        manager = self._live_manager(registry, store, runner, count=2)

        manager.start(camera_ids=["CAM001", "CAM002"])
        assert runner.started.wait(timeout=20)
        manager.stop()
        assert manager.wait(timeout=30)

        status = manager.get_status()
        assert status["state"] == SessionState.COMPLETED.value
        live = [c for c in status["cameras"] if c["camera_id"] in ("CAM001", "CAM002")]
        assert [c["state"] for c in live] == [CameraState.COMPLETED.value] * 2

    def test_recorded_files_still_run_one_at_a_time_before_the_streams(
        self, registry, store
    ):
        """The upload path is unchanged: files in turn, then the live wall."""
        recorded = RecordingRunner()
        live = LiveRunner()

        def runner(**kwargs):
            camera_id = kwargs["camera_meta"]["camera_id"]
            if camera_id in ("CAM003", "CAM004"):
                return live(**kwargs)
            return recorded(**kwargs)

        manager = CameraManager({"video": {}}, registry, runner, store)
        manager.set_rtsp_url("CAM003", "rtsp://host/three")
        manager.set_rtsp_url("CAM004", "rtsp://host/four")

        manager.start()
        try:
            assert live.started.wait(timeout=30)
            assert recorded.overlaps == [], "uploaded videos overlapped"
            assert recorded.order == ["CAM001", "CAM002"]
            deadline = time.time() + 20
            while live.peak_concurrency < 2 and time.time() < deadline:
                time.sleep(0.05)
            assert live.peak_concurrency == 2
        finally:
            manager.stop()
            assert manager.wait(timeout=30)

    def test_each_live_camera_gets_its_own_guarded_view_of_the_models(self):
        """Concurrent cameras must not share one TensorRT context directly."""
        from src.cameras.pipeline_models import PipelineModels

        class FakeDetector:
            def __init__(self):
                self.concurrent = 0
                self.peak = 0
                self.confidence = 0.5

            def detect(self, _frame):
                self.concurrent += 1
                self.peak = max(self.peak, self.concurrent)
                time.sleep(0.01)
                self.concurrent -= 1
                return []

        detector = FakeDetector()
        models = PipelineModels(detector, detector, detector, detector)
        views = [models.shared() for _ in range(4)]

        threads = [threading.Thread(target=lambda v=v: [v.vehicle_detector.detect(None)
                                                        for _ in range(5)])
                   for v in views]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert detector.peak == 1, "two threads were inside the model at once"
        # Attributes pass through; closing a borrowed view leaves the real
        # models alone (the session, not the camera, owns them).
        assert views[0].vehicle_detector.confidence == 0.5
        views[0].close()
        assert models._closed is False


class TestAddingAndRemovingCameras:
    """Cameras are managed from the dashboard, so the manager owns the
    add/remove rules: a new site is complete from the moment it exists, and
    removing one takes its feed with it but keeps its recorded history."""

    def test_a_new_camera_is_queued_and_configurable_like_any_other(
        self, registry, store, tmp_path, video_file
    ):
        registry._settings["processing"] = {"upload_dir": str(tmp_path / "uploads")}
        runner = RecordingRunner()
        manager = CameraManager({"video": {}}, registry, runner, store)

        camera = manager.add_camera("Rajiv Chowk", 28.6328, 77.2197)

        # Same controls as the shipped cameras: upload, RTSP, clear.
        assert camera.camera_id in [c.camera_id for c in manager.cameras]
        assert camera.order == max(c.order for c in manager.cameras)
        assert camera.has_location and camera.camera_name == "Rajiv Chowk"
        assert manager.has_source_configured(camera) is False

        manager.save_upload(camera.camera_id, "clip.mp4", Path(video_file).read_bytes())
        assert manager.has_usable_source(manager.registry.require(camera.camera_id))
        manager.set_rtsp_url(camera.camera_id, "rtsp://host/new")
        assert manager.is_live_camera(manager.registry.require(camera.camera_id))
        manager.clear_source(camera.camera_id)

        # And it is processed, in its queue position, by the same pipeline.
        manager.save_upload(camera.camera_id, "clip.mp4", Path(video_file).read_bytes())
        manager.start()
        assert manager.wait(timeout=60)
        assert runner.order[-1] == camera.camera_id
        stamped = runner.calls[-1]["camera_meta"]
        assert stamped["camera_name"] == "Rajiv Chowk"
        assert (stamped["latitude"], stamped["longitude"]) == (28.6328, 77.2197)

    def test_coordinates_are_required_and_validated(self, registry, store):
        """A camera with no position silently vanishes from every map."""
        manager = CameraManager({"video": {}}, registry, RecordingRunner(), store)

        for latitude, longitude in [("", 77.0), ("north", 77.0), (91.0, 77.0),
                                    (28.6, 181.0), (28.6, None)]:
            with pytest.raises(ValueError):
                manager.add_camera("Nowhere", latitude, longitude)
        with pytest.raises(ValueError, match="place name"):
            manager.add_camera("   ", 28.6, 77.2)

    def test_a_camera_cannot_be_added_or_removed_mid_session(self, registry, store):
        def slow(**kwargs):
            time.sleep(0.5)
            return {}

        manager = CameraManager({"video": {}}, registry, slow, store)
        manager.start(camera_ids=["CAM001"])
        try:
            with pytest.raises(RuntimeError, match="running"):
                manager.add_camera("Rajiv Chowk", 28.63, 77.21)
            with pytest.raises(RuntimeError, match="running"):
                manager.remove_camera("CAM002")
        finally:
            manager.stop()
            manager.wait(timeout=30)

    def test_renaming_changes_only_the_label(self, registry, store):
        manager = CameraManager({"video": {}}, registry, RecordingRunner(), store)
        before = manager.registry.require("CAM002")

        renamed = manager.rename_camera("CAM002", "  Rajiv Chowk  ")

        assert renamed.camera_name == "Rajiv Chowk", "whitespace is trimmed"
        assert renamed.camera_id == before.camera_id
        assert (renamed.latitude, renamed.longitude) == (before.latitude, before.longitude)
        assert renamed.order == before.order
        assert renamed.source_uri == before.source_uri
        assert manager.registry.require("CAM002").camera_name == "Rajiv Chowk"

    def test_a_renamed_camera_stamps_the_new_name_on_new_events(self, registry, store):
        runner = RecordingRunner()
        manager = CameraManager({"video": {}}, registry, runner, store)
        manager.rename_camera("CAM001", "North Gate")

        manager.start(camera_ids=["CAM001"])
        assert manager.wait(timeout=30)

        assert runner.calls[0]["camera_meta"]["camera_name"] == "North Gate"

    def test_a_blank_name_is_refused(self, registry, store):
        manager = CameraManager({"video": {}}, registry, RecordingRunner(), store)
        with pytest.raises(ValueError, match="place name"):
            manager.rename_camera("CAM001", "   ")
        assert manager.registry.require("CAM001").camera_name == "India Gate"

    def test_renaming_an_unknown_camera_raises(self, registry, store):
        manager = CameraManager({"video": {}}, registry, RecordingRunner(), store)
        with pytest.raises(KeyError):
            manager.rename_camera("CAM999", "Somewhere")

    def test_a_camera_cannot_be_renamed_mid_session(self, registry, store):
        def slow(**kwargs):
            time.sleep(0.5)
            return {}

        manager = CameraManager({"video": {}}, registry, slow, store)
        manager.start(camera_ids=["CAM001"])
        try:
            with pytest.raises(RuntimeError, match="running"):
                manager.rename_camera("CAM002", "Renamed")
        finally:
            manager.stop()
            manager.wait(timeout=30)

    def test_a_rename_survives_a_reload(self, camera_config_file, store):
        config_copy = camera_config_file

        first = CameraManager(
            {"video": {}}, load_camera_registry(str(config_copy)), RecordingRunner(), store
        )
        first.rename_camera("CAM001", "North Gate")

        reloaded = load_camera_registry(str(config_copy))
        assert reloaded.require("CAM001").camera_name == "North Gate"

    def test_removing_a_camera_takes_its_feed_and_upload_with_it(
        self, registry, store, tmp_path, video_file
    ):
        frames = tmp_path / "live_frames"
        frames.mkdir()
        (frames / "CAM002.jpg").write_bytes(b"frame")
        (frames / "CAM003.jpg").write_bytes(b"other camera")
        registry._settings["processing"] = {"upload_dir": str(tmp_path / "uploads")}
        manager = CameraManager(
            {"video": {}, "api": {"live_frames_dir": str(frames)}},
            registry, RecordingRunner(), store,
        )
        upload = manager.save_upload("CAM002", "clip.mp4", Path(video_file).read_bytes())

        removed = manager.remove_camera("CAM002")

        assert removed.camera_id == "CAM002"
        assert "CAM002" not in [c.camera_id for c in manager.cameras]
        # The camera wall is drawn from the registry, so its panel goes with
        # it -- and its last frame must not linger behind.
        assert not (frames / "CAM002.jpg").exists()
        assert (frames / "CAM003.jpg").exists(), "other cameras' frames are untouched"
        assert not upload.exists()

    def test_removing_a_camera_keeps_the_events_it_recorded(self, registry, store, tmp_path):
        """They are history: they carry their own camera id and coordinates,
        so trajectories that passed this site still plot."""
        from src.database import db as database

        database.init_db(str(tmp_path / "events.db"))
        try:
            with database.get_session() as session:
                database.insert_event(session, {
                    "plate_number": "DL8CA1234", "vehicle_type": "Private",
                    "plate_color": "White", "series_type": "normal", "direction": "IN",
                    "image_path": "", "camera_id": "CAM002",
                    "camera_name": "Connaught Place", "latitude": 28.6315,
                    "longitude": 77.2167, "processing_session": "S1",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

            manager = CameraManager({"video": {}}, registry, RecordingRunner(), store)
            manager.remove_camera("CAM002")

            with database.get_session() as session:
                kept = database.get_plate_detections(session, "DL8CA1234")
            assert len(kept) == 1
            assert kept[0]["camera_id"] == "CAM002"
            assert kept[0]["latitude"] == 28.6315
        finally:
            database._engine = database._SessionFactory = database._db_type = None
            database._db_url = None

    def test_changes_persist_for_the_next_process(self, camera_config_file, store):
        """The dashboard restarts; the deployment should not revert."""
        config_copy = camera_config_file

        first = CameraManager(
            {"video": {}}, load_camera_registry(str(config_copy)), RecordingRunner(), store
        )
        added = first.add_camera("Rajiv Chowk", 28.6328, 77.2197)
        first.remove_camera("CAM001")

        second = CameraManager(
            {"video": {}}, load_camera_registry(str(config_copy)), RecordingRunner(), store
        )
        assert [c.camera_id for c in second.cameras] == [c.camera_id for c in first.cameras]
        assert second.registry.require(added.camera_id).camera_name == "Rajiv Chowk"
        assert second.registry.get("CAM001") is None


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
