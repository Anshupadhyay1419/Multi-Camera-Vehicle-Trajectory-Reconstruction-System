"""
CameraManager -- runs the camera queue, strictly one camera at a time.

    load registry -> accept uploads -> build queue -> run each in turn

Sequential by design, not by accident. The Jetson has one GPU and the ALPR
pipeline already saturates it; four concurrent cameras would contend for the
same TensorRT execution context and finish later than four run in turn,
while making every per-camera latency figure meaningless. It is also what
the demo needs: each camera's pass completes before the next opens its
source, so the timestamps ordering a reconstructed trajectory are
monotonically increasing across the queue even though all four cameras may
be replaying the same footage.

The ALPR pipeline itself is untouched. This module calls the existing
`run_pipeline()` once per camera, handing it that camera's metadata, a
session context, and a progress reporter -- all parameters that default to
the single-gate behaviour when nobody passes them.

Threading model: `start()` returns immediately and the queue runs on one
background daemon thread. Progress is published to a JSON file (StatusStore)
rather than kept in memory, because the process watching a run is often not
the process running it.
"""

from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from src.cameras.models import (
    CameraConfig,
    CameraProgress,
    CameraState,
    SessionState,
    SessionStatus,
    SourceType,
)
from src.cameras.registry import CameraRegistry, load_camera_registry
from src.cameras.sources import VideoSourceError, create_video_source
from src.cameras.status_store import StatusStore
from src.utils.logger import get_logger

_logger = get_logger("cameras.manager")

# Most recent log lines kept in the status document. The dashboard shows a
# scrolling log; the whole run's output belongs in logs/alpr.log, not in a
# file that is rewritten several times a second.
_MAX_STATUS_LOGS = 200


class ProgressReporter:
    """Bridges one camera's pipeline run to the shared status document.

    An instance is handed to `run_pipeline(progress=...)`, which calls
    `on_start` / `on_frame` / `on_detection` as it works and checks
    `should_stop()` once per frame. Every method is deliberately cheap and
    non-throwing: the pipeline treats this as fire-and-forget instrumentation
    and must never lose a frame or an event because reporting failed.
    """

    def __init__(
        self,
        progress: CameraProgress,
        publish: Callable[[], None],
        stop_event: threading.Event,
        log: Callable[[str], None],
    ) -> None:
        self._progress = progress
        self._publish = publish
        self._stop_event = stop_event
        self._log = log
        self._plates: set[str] = set()

    def on_start(self, total_frames: int = 0) -> None:
        self._progress.total_frames = max(0, int(total_frames or 0))
        self._publish()

    def on_frame(
        self,
        frames_processed: int,
        fps: float,
        avg_ocr_ms: float = 0.0,
        avg_detection_ms: float = 0.0,
    ) -> None:
        self._progress.frames_processed = int(frames_processed)
        self._progress.fps = round(float(fps), 2)
        self._progress.avg_ocr_ms = round(float(avg_ocr_ms), 2)
        self._progress.avg_detection_ms = round(float(avg_detection_ms), 2)
        self._publish()

    def on_detection(
        self,
        plate_number: str,
        confidence: Optional[float] = None,
        plate_image: Optional[str] = None,
        vehicle_image: Optional[str] = None,
    ) -> None:
        self._progress.detections += 1
        self._plates.add(plate_number)
        self._progress.unique_plates = len(self._plates)
        self._progress.last_plate = plate_number
        self._progress.last_plate_image = plate_image or None
        self._progress.last_vehicle_image = vehicle_image or None
        confidence_text = f" ({confidence:.0%})" if confidence is not None else ""
        self._log(
            f"{self._progress.camera_name}: detected {plate_number}{confidence_text}"
        )
        self._publish()

    def should_stop(self) -> bool:
        """True once a stop has been requested. Polled once per frame."""
        return self._stop_event.is_set()


class CameraManager:
    """Owns the camera registry and runs processing sessions over it.

    Args:
        config:   The full ALPR config dict (load_config()). Passed through
                  to the pipeline unchanged.
        registry: The camera registry. Loaded from camera_config.yaml when
                  not supplied.
        runner:   The function that processes one camera. Defaults to the
                  real `scripts.run_pipeline.run_pipeline`; tests inject a
                  fake so the scheduling logic can be verified without a GPU
                  or a video file.
    """

    def __init__(
        self,
        config: dict,
        registry: Optional[CameraRegistry] = None,
        runner: Optional[Callable[..., dict]] = None,
        status_store: Optional[StatusStore] = None,
    ) -> None:
        self._config = config
        self._registry = registry if registry is not None else load_camera_registry()
        self._runner = runner
        # Whether a runner was INJECTED (tests, or an embedding caller with
        # its own pipeline). Recorded here because _resolve_runner() caches
        # the real run_pipeline into self._runner on first use -- so checking
        # `self._runner is None` later would be False for a genuine pipeline
        # session too, and the second session would skip loading its models.
        self._uses_real_pipeline = runner is None
        processing = self._registry.processing
        self._status_store = status_store or StatusStore(
            processing.get("status_file", "data/processing_status.json")
        )
        self._continue_on_error = bool(processing.get("continue_on_error", True))
        # How long each LIVE camera is sampled before the queue moves on. A
        # recorded file ends by itself and ignores this; an RTSP stream never
        # does, so without a bound the queue would stop dead on its first
        # live camera. 0 or None means "run until stopped", which is only
        # sensible for a single-camera queue.
        self._live_duration_seconds = processing.get("live_duration_seconds", 60)
        self._upload_dir = Path(processing.get("upload_dir", "data/uploads"))

        # Loaded on the first session and KEPT for the life of this
        # process. Two reasons, both learned the hard way:
        #
        #   * Releasing them killed the dashboard. PipelineModels.close()
        #     frees PyCUDA device buffers and drops the TensorRT objects,
        #     and doing that inside a long-lived process aborts it outright
        #     ("PyCUDA ERROR: context stack was not empty") -- so the server
        #     died the moment a run finished and the browser showed
        #     "Connection error".
        #   * Keeping them makes every session after the first start
        #     immediately instead of paying the load again.
        #
        # The memory stays allocated, which is exactly what a second run
        # would have re-allocated anyway. A short-lived CLI run is unaffected:
        # run_pipeline still builds and releases its own models there.
        self._models = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._status: Optional[SessionStatus] = None

    # ── registry access ───────────────────────────────────────────────────

    @property
    def registry(self) -> CameraRegistry:
        return self._registry

    @property
    def cameras(self) -> list[CameraConfig]:
        return self._registry.all

    @property
    def live_duration_seconds(self) -> float:
        """Seconds each live camera is sampled before the queue moves on.

        0.0 means unbounded -- see the `live_duration_seconds` note in
        camera_config.yaml. Exposed so the dashboard can tell the operator
        what a live run will actually do, rather than letting a queue that
        is waiting on an endless stream look stuck.
        """
        try:
            value = float(self._live_duration_seconds or 0)
        except (TypeError, ValueError):
            return 60.0
        return max(0.0, value)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── uploads ───────────────────────────────────────────────────────────

    def save_upload(self, camera_id: str, filename: str, data: bytes) -> Path:
        """Store an uploaded video for *camera_id* and point the camera at it.

        The file is named after the camera, not the upload, so re-uploading
        replaces the previous clip instead of accumulating one file per
        attempt -- and so four uploads of the SAME source video still land in
        four distinct files, one per camera. That matters: the demo feeds one
        recording to four cameras, and each must own its own copy so nothing
        downstream can accidentally treat them as one shared source.

        camera_config.yaml is NOT rewritten. The YAML declares where the
        cameras *are*; which clip they are replaying this run is session
        state, applied to the in-memory registry via `registry.replace()`.

        Args:
            camera_id: Camera the video belongs to.
            filename:  Original filename, used only for its extension.
            data:      The video bytes.

        Returns:
            Path to the saved file.

        Raises:
            KeyError:   No such camera in the registry.
            ValueError: Empty upload.
            OSError:    The file could not be written.
        """
        camera = self._registry.require(camera_id)
        if not data:
            raise ValueError(f"Camera {camera_id}: uploaded file is empty")

        suffix = Path(filename or "").suffix.lower() or ".mp4"
        self._upload_dir.mkdir(parents=True, exist_ok=True)
        destination = self._upload_dir / f"{camera_id}{suffix}"
        destination.write_bytes(data)

        # Reading a video the operator just chose is also the cheapest way to
        # find out it is not a video at all -- better to say so now than to
        # fail four cameras into a run.
        import dataclasses

        # Switching to a file CLEARS any RTSP url on this camera. A camera
        # has exactly one source: source_type already decides which field is
        # read, but leaving the other populated makes the dashboard show a
        # stream the camera will never open, and makes "which source is this
        # camera on?" a two-field question instead of one.
        updated = dataclasses.replace(
            camera,
            video_path=str(destination),
            rtsp_url=None,
            source_type=SourceType.UPLOAD,
        )
        self._registry.replace(updated)

        _logger.info(
            "Camera %s (%s): stored upload %s (%.1f MB)",
            camera_id, camera.camera_name, destination, len(data) / 1e6,
        )
        return destination

    def assign_uploads(
        self,
        uploads: "Sequence[tuple[str, bytes]]",
        camera_ids: Optional[list[str]] = None,
    ) -> dict[str, Path]:
        """Distribute one bulk upload across the camera queue.

        Lets an operator drop in every camera's footage in a single action
        instead of working through one uploader per camera. How the files are
        spread depends on how many arrive:

          1 file           broadcast to EVERY target camera. This is the demo
                           case from the brief -- one recording standing in
                           for four separate sites.
          one per camera   assigned in queue order: 1st file to the 1st
                           camera, and so on.
          fewer (but >1)   fills the first N cameras in queue order; the rest
                           keep whatever source they already had, so a
                           partial re-upload does not wipe the others.
          more than there  rejected, rather than silently dropping footage
          are cameras      the operator clearly meant to process.

        Whatever the split, **each camera gets its own file on disk** --
        `save_upload()` names the copy after the camera, so no two cameras
        ever share a `video_path`. That matters even when the bytes are
        identical: the cameras must stay independent observers, which is what
        makes the demo a faithful stand-in for four real streams and what
        lets any one of them be swapped for an RTSP feed later. The cost is
        one copy of the file per camera on disk.

        Args:
            uploads:    (filename, data) pairs, in the order given.
            camera_ids: Restrict to these cameras. None targets every enabled
                        camera, in queue order.

        Returns:
            {camera_id: saved path} for the cameras that were actually given
            a file.

        Raises:
            ValueError: No files, or more files than cameras.
            KeyError:   An unknown camera_id was requested.
            OSError:    A file could not be written.
        """
        uploads = list(uploads)
        if not uploads:
            raise ValueError("No files were uploaded")

        targets = self.build_queue(camera_ids)
        if not targets:
            raise ValueError(
                "No cameras available to receive the upload -- every camera "
                "is disabled, or none matched the request"
            )

        if len(uploads) > len(targets):
            raise ValueError(
                f"Received {len(uploads)} file(s) but only {len(targets)} "
                f"camera(s) are available "
                f"({', '.join(c.camera_id for c in targets)}). Remove the "
                f"extra file(s), or enable more cameras."
            )

        if len(uploads) == 1:
            # One recording, every camera. Deliberately a broadcast rather
            # than "assign it to camera 1 only": a single file in a bulk
            # upload is the four-virtual-cameras demo, not a partial upload.
            pairs = [(camera, uploads[0]) for camera in targets]
        else:
            pairs = list(zip(targets, uploads))

        saved: dict[str, Path] = {}
        for camera, (filename, data) in pairs:
            saved[camera.camera_id] = self.save_upload(
                camera.camera_id, filename, data
            )

        _logger.info(
            "Assigned %d uploaded file(s) to %d camera(s): %s",
            len(uploads), len(saved),
            ", ".join(f"{cid}<-{path.name}" for cid, path in saved.items()),
        )
        return saved

    def set_rtsp_url(self, camera_id: str, rtsp_url: str) -> CameraConfig:
        """Switch a camera to a live RTSP stream for this session.

        The production path, and the whole point of the VideoSource split:
        nothing but this field changes when a demo site becomes a real
        camera. Persisting it belongs in camera_config.yaml; this only
        affects the running process.
        """
        import dataclasses

        camera = self._registry.require(camera_id)
        # Mirror of save_upload(): one source per camera, so pointing this
        # camera at a stream clears the uploaded file it was using.
        updated = dataclasses.replace(
            camera,
            rtsp_url=(rtsp_url or "").strip(),
            video_path=None,
            source_type=SourceType.RTSP,
        )

        # Validate the URL now rather than at run time. The alternative is
        # that a typo sits quietly in the registry until START PROCESSING,
        # and surfaces several minutes into a queue as a failed camera --
        # long after the operator could have simply corrected it.
        create_video_source(updated, self._config).validate()

        self._registry.replace(updated)
        _logger.info("Camera %s: switched to RTSP source %r", camera_id, updated.rtsp_url)
        return updated

    def clear_source(self, camera_id: str) -> CameraConfig:
        """Detach whatever source a camera is currently using.

        Lets an operator undo a mis-typed stream or a wrong upload without
        having to supply a replacement first. The uploaded FILE is left on
        disk -- this clears the camera's reference to it, not the demo
        footage someone may still want.
        """
        import dataclasses

        camera = self._registry.require(camera_id)
        updated = dataclasses.replace(
            camera, video_path=None, rtsp_url=None, source_type=SourceType.UPLOAD
        )
        self._registry.replace(updated)
        _logger.info("Camera %s: source cleared", camera_id)
        return updated

    # ── session control ───────────────────────────────────────────────────

    @staticmethod
    def has_source_configured(camera: CameraConfig) -> bool:
        """True if anyone has given this camera a source at all.

        Kept distinct from has_usable_source() because the two answer different
        questions and deserve different outcomes:

          nothing configured    -> the operator left this camera out. SKIPPED.
          configured but broken -> a file has gone missing, a URL is wrong.
                                   That is a fault worth surfacing, so the
                                   camera runs, fails validation, and is
                                   reported FAILED with the reason.

        Collapsing them would quietly hide a deleted video behind the same
        neutral "skipped" badge as a camera nobody filled in.
        """
        return bool(camera.source_uri)

    def has_usable_source(self, camera: CameraConfig) -> bool:
        """True if this camera looks ready to process right now.

        Cheap and non-throwing -- it drives the dashboard's "N of M cameras
        have a source" line and its START button, and runs on every rerun.
        The authoritative check is VideoSource.validate() at processing time;
        this only has to be right about the obvious cases.
        """
        if not self.has_source_configured(camera):
            return False
        if camera.source_type is SourceType.RTSP:
            # Reachability is a run-time concern -- a camera that is briefly
            # down should not be dropped from the queue.
            return True
        try:
            return Path(camera.source_uri).is_file()
        except OSError:
            return False

    def build_queue(self, camera_ids: Optional[list[str]] = None) -> list[CameraConfig]:
        """Return the cameras to process, in the order they will run.

        Args:
            camera_ids: Restrict the run to these cameras (still processed in
                        registry order, not in the order given -- the queue
                        order is a property of the deployment, not of the
                        request). None runs every enabled camera.
        """
        queue = self._registry.enabled
        if camera_ids:
            wanted = set(camera_ids)
            queue = [camera for camera in queue if camera.camera_id in wanted]
        return queue

    def start(
        self,
        camera_ids: Optional[list[str]] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Begin a processing session on a background thread.

        Returns immediately -- poll `get_status()` (or the status file) to
        follow it.

        Args:
            camera_ids: Restrict the run to these cameras.
            session_id: Override the generated session id. Every event stored
                        during the run carries it, which is what lets the
                        dashboard scope a trajectory to one run.

        Returns:
            The session id.

        Raises:
            RuntimeError: A session is already running, or the queue is empty.
        """
        with self._lock:
            if self.is_running():
                raise RuntimeError(
                    "A processing session is already running; stop it before starting another"
                )

            queue = self.build_queue(camera_ids)
            if not queue:
                raise RuntimeError(
                    "No cameras to process -- every camera is disabled, or none matched the request"
                )

            # Refuse a run that cannot possibly do anything. Without this the
            # session starts, every camera fails its source check in turn,
            # and the operator watches a "running" queue that was never going
            # to produce a detection. The dashboard disables its START button
            # for this case, but the API and any script can still call start()
            # directly, so the guard belongs here rather than in the UI.
            if not any(self.has_source_configured(camera) for camera in queue):
                raise RuntimeError(
                    "No camera has a source yet -- upload a video or set an "
                    "RTSP URL for at least one camera before starting"
                )

            session_id = session_id or self._new_session_id()
            self._stop_event.clear()
            self._status = SessionStatus(
                session_id=session_id,
                state=SessionState.RUNNING,
                started_at=time.time(),
                cameras=[
                    CameraProgress(
                        camera_id=camera.camera_id,
                        camera_name=camera.camera_name,
                        order=camera.order,
                    )
                    for camera in queue
                ],
            )
            # Drop last session's frames so the camera wall never shows a
            # previous run's video under a camera that has not started yet.
            self._clear_live_frames()
            self._append_log(
                f"Session {session_id} queued {len(queue)} camera(s): "
                + " -> ".join(camera.camera_name for camera in queue)
            )
            self._publish()

            self._thread = threading.Thread(
                target=self._run_session,
                args=(queue, session_id),
                name=f"camera-session-{session_id}",
                # Daemon so a Ctrl+C on the dashboard is not blocked waiting
                # for a long video to finish. The cooperative stop below is
                # the clean way out; this is just the backstop.
                daemon=True,
            )
            self._thread.start()
            return session_id

    def stop(self) -> bool:
        """Request that the running session stop after the current frame.

        Cooperative, not a kill: the pipeline notices at its next frame,
        breaks its loop, and still runs its final fusion flush -- so plates
        that were mid-vote are stored rather than discarded. Returns False
        when nothing was running.
        """
        if not self.is_running():
            return False
        self._stop_event.set()
        self._append_log("Stop requested -- finishing the current camera.")
        self._publish()
        return True

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until the session finishes. Returns True if it did."""
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def get_status(self) -> dict[str, Any]:
        """Current session status.

        Prefers this process's own in-memory status and falls back to the
        status file, so a dashboard that did not start the run still sees it.
        """
        if self._status is not None:
            return self._status.to_dict()
        stored = self._status_store.read()
        if stored is not None:
            return stored
        return SessionStatus(session_id="", state=SessionState.IDLE).to_dict()

    # ── the sequential run ────────────────────────────────────────────────

    def _run_session(self, queue: list[CameraConfig], session_id: str) -> None:
        """Process every queued camera in turn. Runs on the worker thread."""
        status = self._status
        assert status is not None  # set by start() before the thread launched

        # Make the CUDA primary context current on THIS thread, and make sure
        # the stack is empty again before the thread ends -- see
        # cameras.cuda_context. Skipping the drain aborts the whole process
        # when the worker exits, which is what used to take the dashboard
        # down every time a run finished.
        from src.cameras import cuda_context

        cuda_context.push_primary_context()

        failures = 0
        models = None
        try:
            # Load the heavy models ONCE for the whole queue. Building them
            # per camera meant deserialising three TensorRT plans four times
            # (~2m40s each on a Jetson Orin), so most of a four-camera sweep
            # was spent loading rather than processing.
            #
            # Reported as it happens: until the first frame is decoded there
            # is nothing else to show, and a motionless "0% / 0.0 FPS" for
            # minutes is indistinguishable from a hang.
            if self._uses_real_pipeline:
                if self._models is not None:
                    models = self._models
                    self._append_log("Reusing the models already loaded in this process.")
                    self._publish()
                else:
                    from src.cameras.pipeline_models import PipelineModels

                    self._append_log(
                        "Loading detection and OCR models (once per process)…"
                    )
                    self._publish()
                    started = time.monotonic()
                    models = PipelineModels.load(
                        self._config, on_progress=self._append_log
                    )
                    self._models = models
                    self._append_log(
                        f"Models ready in {time.monotonic() - started:.0f}s — "
                        f"shared across all {len(queue)} camera(s)."
                    )
                    self._publish()

            for position, camera in enumerate(queue, start=1):
                if self._stop_event.is_set():
                    self._append_log("Session cancelled before " + camera.camera_name)
                    for remaining in status.cameras:
                        if remaining.state is CameraState.PENDING:
                            remaining.state = CameraState.SKIPPED
                    status.state = SessionState.CANCELLED
                    break

                progress = self._progress_for(camera.camera_id)
                if progress is None:      # cannot happen; queue built the list
                    continue

                status.current_camera_id = camera.camera_id
                if not self._process_camera(camera, position, session_id, progress, models):
                    failures += 1
                    if not self._continue_on_error:
                        status.state = SessionState.FAILED
                        status.error = progress.error
                        self._append_log(
                            "Stopping the session: continue_on_error is false"
                        )
                        break
            else:
                # Loop completed without break -- every camera was attempted.
                status.state = (
                    SessionState.COMPLETED if failures == 0 else SessionState.FAILED
                )
                if failures:
                    status.error = f"{failures} of {len(queue)} camera(s) failed"

        except Exception as exc:
            # A failure in the scheduler itself, as opposed to in one
            # camera's pipeline (which _process_camera already contains).
            _logger.exception("Processing session %s failed", session_id)
            status.state = SessionState.FAILED
            status.error = f"{type(exc).__name__}: {exc}"
            self._append_log(f"Session failed: {exc}")

        finally:
            # The models are deliberately NOT released here. They outlive the
            # session as well as the queue: see self._models above -- closing
            # them mid-process aborts it and takes the dashboard down with it.
            status.current_camera_id = None
            status.finished_at = time.time()
            self._append_log(
                f"Session {session_id} {status.state.value}: "
                f"{status.total_detections} detection(s) in "
                f"{status.elapsed_seconds:.1f}s"
            )
            self._publish()
            _logger.info(
                "Session %s finished with state=%s (%d detections)",
                session_id, status.state.value, status.total_detections,
            )

            # Last thing this thread does. The models stay loaded and stay
            # valid -- the primary context is reference-counted and survives
            # the pop; the next session's worker pushes it again.
            cuda_context.drain_contexts()

    def _process_camera(
        self,
        camera: CameraConfig,
        position: int,
        session_id: str,
        progress: CameraProgress,
        models=None,
    ) -> bool:
        """Run the ALPR pipeline over one camera. Returns True on success.

        Every failure is contained here and recorded on the camera's own
        progress entry, so one unreadable upload marks one camera failed
        rather than taking down the queue.
        """
        if not self.has_source_configured(camera):
            # Not a failure: nobody gave this camera anything to process.
            # Distinguishing the two matters -- FAILED means something broke
            # and is worth investigating, SKIPPED means the operator left it
            # empty, and a session of only-skipped cameras is not an error.
            progress.state = CameraState.SKIPPED
            progress.started_at = progress.finished_at = time.time()
            self._append_log(
                f"[{position}] Skipped {camera.camera_name}: no video or stream set"
            )
            self._publish()
            return True

        progress.state = CameraState.RUNNING
        progress.started_at = time.time()
        self._append_log(
            f"[{position}] Starting {camera.camera_name} ({camera.camera_id})"
        )
        self._publish()

        try:
            # Validate the source before the pipeline touches it: this is
            # where "you never uploaded a video for camera 3" gets caught,
            # with a message naming the camera.
            source = create_video_source(camera, self._config)
            source.validate()
            progress.total_frames = source.total_frames()

            reporter = ProgressReporter(
                progress=progress,
                publish=self._publish,
                stop_event=self._stop_event,
                log=self._append_log,
            )

            # Only a live source gets a time limit; a recorded file runs to
            # its natural end however long that takes.
            duration_limit = None
            if source.is_live:
                try:
                    limit = float(self._live_duration_seconds or 0)
                except (TypeError, ValueError):
                    limit = 60.0
                duration_limit = limit if limit > 0 else None
                if duration_limit is not None:
                    self._append_log(
                        f"{camera.camera_name} is a live stream — sampling for "
                        f"{duration_limit:.0f}s, then moving to the next camera."
                    )

            summary = self._resolve_runner()(
                config=self._config,
                camera_meta=camera.event_metadata(),
                session_context={
                    "processing_session": session_id,
                    # The camera's queue position as it actually applied for
                    # this run, frozen onto every event -- re-ordering
                    # camera_config.yaml later cannot rewrite this history.
                    "trajectory_order": camera.order,
                    "video_source": source.uri,
                },
                progress=reporter,
                source_override=source.uri,
                max_duration_seconds=duration_limit,
                models=models,
            )

            if isinstance(summary, dict):
                progress.frames_processed = summary.get(
                    "frames_processed", progress.frames_processed
                )
                progress.avg_ocr_ms = round(float(summary.get("avg_ocr_ms", 0.0)), 2)
                progress.avg_detection_ms = round(
                    float(summary.get("avg_plate_detection_ms", 0.0)), 2
                )

            progress.state = (
                CameraState.SKIPPED if self._stop_event.is_set() else CameraState.COMPLETED
            )
            progress.finished_at = time.time()
            self._append_log(
                f"[{position}] Finished {camera.camera_name}: "
                f"{progress.detections} detection(s), "
                f"{progress.unique_plates} unique plate(s) in "
                f"{progress.elapsed_seconds:.1f}s"
            )
            self._publish()
            return True

        except VideoSourceError as exc:
            return self._fail_camera(camera, position, progress, str(exc))
        except Exception as exc:
            _logger.exception("Camera %s failed", camera.camera_id)
            return self._fail_camera(
                camera, position, progress, f"{type(exc).__name__}: {exc}"
            )

    def _fail_camera(
        self,
        camera: CameraConfig,
        position: int,
        progress: CameraProgress,
        message: str,
    ) -> bool:
        progress.state = CameraState.FAILED
        progress.error = message
        progress.finished_at = time.time()
        self._append_log(f"[{position}] FAILED {camera.camera_name}: {message}")
        self._publish()
        return False

    def _resolve_runner(self) -> Callable[..., dict]:
        """Return the per-camera runner, importing the real one on first use.

        Deferred so that constructing a CameraManager -- which the API does
        just to list cameras -- does not import the whole vision stack
        (ultralytics, TensorRT, OpenCV) into a process that will never run a
        frame through it.
        """
        if self._runner is None:
            from scripts.run_pipeline import run_pipeline

            self._runner = run_pipeline
        return self._runner

    # ── status plumbing ───────────────────────────────────────────────────

    def _progress_for(self, camera_id: str) -> Optional[CameraProgress]:
        if self._status is None:
            return None
        for progress in self._status.cameras:
            if progress.camera_id == camera_id:
                return progress
        return None

    def _append_log(self, message: str) -> None:
        if self._status is None:
            return
        stamped = f"{datetime.now().strftime('%H:%M:%S')}  {message}"
        self._status.logs.append(stamped)
        # Trim from the front so the newest lines always survive.
        if len(self._status.logs) > _MAX_STATUS_LOGS:
            del self._status.logs[:-_MAX_STATUS_LOGS]
        _logger.info(message)

    def _publish(self) -> None:
        """Snapshot the status to the shared file for other processes."""
        if self._status is not None:
            self._status_store.write(self._status.to_dict())

    def forget_session(self, session_id: str) -> bool:
        """Drop the live status of a session whose events have been deleted.

        The status panel, the header and the camera wall all read the
        manager's status (in memory, and the status file for other
        processes). Deleting a session's events from the database does not
        touch either, so without this the page went on describing a run that
        no longer exists -- "4/4 complete, 88 detections" -- after its data
        was gone.

        Only forgets the status if it describes THIS session: deleting an
        older run must leave the most recent run's status alone. Refuses
        while a session is running, since that status is still live.

        Returns True if anything was cleared.
        """
        if self.is_running():
            return False

        cleared = False
        with self._lock:
            if self._status is not None and self._status.session_id == session_id:
                self._status = None
                cleared = True
            stored = self._status_store.read()
            if stored is not None and stored.get("session_id") == session_id:
                self._status_store.clear()
                cleared = True

        if cleared:
            # The preview frames on the camera wall belong to that run too.
            self._clear_live_frames()
            _logger.info("Forgot the status of deleted session %s", session_id)
        return cleared

    def release_models(self) -> None:
        """Free the shared models. For a process that is about to exit.

        NOT called at the end of a session on purpose. Releasing PyCUDA
        buffers and TensorRT objects inside a long-lived process aborts it
        ("PyCUDA ERROR: context stack was not empty"), which killed the
        dashboard every time a run completed. A caller that is shutting down
        anyway can use this; a server should simply let process exit handle
        it.
        """
        if self._models is not None:
            self._models.close()
            self._models = None

    def _clear_live_frames(self) -> None:
        """Delete the per-camera preview frames from the previous session."""
        directory = Path(
            (self._config.get("api") or {}).get("live_frames_dir", "data/live_frames")
        )
        try:
            if directory.is_dir():
                for frame in directory.glob("*.jpg"):
                    frame.unlink(missing_ok=True)
        except OSError as exc:
            # Cosmetic cleanup only -- never worth failing a run over.
            _logger.debug("Could not clear live frames in %s: %s", directory, exc)

    @staticmethod
    def _new_session_id() -> str:
        """A readable, sortable, collision-free session id.

        Timestamp first so sessions sort chronologically in a picker, plus a
        short random suffix so two runs started in the same second (a retry,
        or two operators) cannot collide.
        """
        return (
            datetime.now().strftime("%Y%m%d-%H%M%S")
            + "-"
            + uuid.uuid4().hex[:6]
        )


# ── process-wide singleton ────────────────────────────────────────────────
#
# Streamlit re-runs its whole script on every interaction, so a manager held
# in a local variable would be rebuilt (losing the running thread) several
# times a second. A module-level singleton survives those reruns because the
# module stays imported. Guarded by a lock so two near-simultaneous reruns
# cannot build two managers and race for the same GPU.

_manager: Optional[CameraManager] = None
_manager_lock = threading.Lock()


def get_camera_manager(
    config: dict,
    camera_config_path: Optional[str] = None,
) -> CameraManager:
    """Return this process's CameraManager, creating it on first call.

    Args:
        config:             The full ALPR config dict.
        camera_config_path: Override the camera registry location. Only read
                            on the call that actually creates the manager.
    """
    global _manager
    with _manager_lock:
        if _manager is None:
            registry = (
                load_camera_registry(camera_config_path)
                if camera_config_path
                else load_camera_registry()
            )
            _manager = CameraManager(config=config, registry=registry)
            _logger.info("Camera manager created with %d camera(s)", len(registry))
        return _manager


def reset_camera_manager() -> None:
    """Drop the singleton. For tests, and for reloading a changed registry."""
    global _manager
    with _manager_lock:
        _manager = None
