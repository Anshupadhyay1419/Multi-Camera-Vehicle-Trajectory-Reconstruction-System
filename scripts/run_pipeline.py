"""
ALPR University Gate — Main Pipeline Entry Point

Pipeline flow per frame:
  Frame Capture → Vehicle Detection → Vehicle Tracking
  → Plate Detection → Preprocessing → OCR → Validation
  → OCR Fusion → Color + Vehicle Type → Duplicate Filter
  → Direction Detection → Database Storage

Super-resolution (sr_enhancer.enhance()) runs on every live OCR attempt, not
just once at fusion time -- detection/OCR are fast enough on this hardware
that this fits inside the per-frame budget, and enhance() already no-ops for
crops at or above sr_threshold_px. On a backend where Real-ESRGAN's actual
neural upscaler is active (not the OpenCV fallback), this is heavier work
running every frame instead of once per vehicle; watch per-frame latency if
you enable it here.

Usage:
  python scripts/run_pipeline.py
  python scripts/run_pipeline.py --source ALPR.mp4
  python scripts/run_pipeline.py --config config/config.yaml --source rtsp://...
"""

from __future__ import annotations

import argparse
import sys
import time
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.benchmark import BenchmarkRecorder
from src.utils.config import get_camera_metadata, load_config
from src.utils.logger import get_logger
from src.ocr.ocr_postprocessor import correct_ocr_text, remove_noise_characters

_logger = None


def _init_logger(config: dict):
    global _logger
    _logger = get_logger("main_pipeline", config=config)
    return _logger


def _ensure_system_site_packages() -> None:
    """Enable system-installed Jetson Python packages inside a virtualenv.

    Delegates to utils.jetson_paths so the multi-camera session, which loads
    its models before any pipeline starts, applies exactly the same fix.
    """
    from src.utils.jetson_paths import ensure_system_site_packages

    ensure_system_site_packages()


def _store_event(
    plate_number: str,
    series_type: str,
    plate_crop: np.ndarray,
    color_classifier,
    vehicle_classifier,
    dup_filter,
    direction_detector,
    track_id: int,
    centroid: tuple,
    image_save_path: str,
    camera_meta: dict,
    database,
    log,
    vehicle_crop: np.ndarray | None = None,
    confidence: float | None = None,
    ocr_text: str | None = None,
    session_context: dict | None = None,
    vehicle_image_save_path: str = "data/vehicle_crops/",
    progress=None,
) -> bool:
    """Classify, deduplicate, and store a confirmed plate event.

    *camera_meta* is the capturing camera's camera_id / camera_name /
    latitude / longitude -- read once at startup by get_camera_metadata()
    for the single-gate path, or supplied per camera by the multi-camera
    manager.

    The trailing arguments are all optional and all default to the
    single-gate behaviour this function has always had:

      vehicle_crop     Whole-vehicle image to archive alongside the plate
                       crop. None (the default) stores no vehicle image and
                       leaves the column NULL, exactly as before.
      confidence /     Recorded on the event for the map popups and the
      ocr_text         search page. None leaves them NULL.
      session_context  processing_session / trajectory_order / video_source
                       for a multi-camera run. None means this event is not
                       part of one.
      progress         Optional reporter notified of each stored event, so a
                       dashboard can show detections as they happen. Any
                       exception it raises is swallowed -- reporting must
                       never cost the pipeline an event.
    """
    if dup_filter.is_duplicate(plate_number, track_id):
        log.info("DUPLICATE skipped: %s (track %d)", plate_number, track_id)
        return False

    color = color_classifier.classify(plate_crop)
    vehicle_type = vehicle_classifier.classify(color)
    direction = direction_detector.update(track_id, centroid) or "IN"
    image_path = database.save_plate_image(plate_crop, plate_number, image_save_path)

    # Only written when a crop was actually passed in, so the single-gate
    # path does no extra disk I/O and the column stays NULL for it.
    vehicle_image_path = ""
    if vehicle_crop is not None and getattr(vehicle_crop, "size", 0):
        vehicle_image_path = database.save_vehicle_image(
            vehicle_crop, plate_number, vehicle_image_save_path
        )

    event_data = {
        "plate_number": plate_number,
        "vehicle_type": vehicle_type,
        "plate_color":  color,
        "series_type":  series_type,
        "direction":    direction,
        "image_path":   image_path,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        # camera_id / camera_name / latitude / longitude
        **camera_meta,
        "vehicle_image_path": vehicle_image_path or None,
        "confidence": confidence,
        "ocr_text":   ocr_text,
        # processing_session / trajectory_order / video_source
        **(session_context or {}),
    }

    try:
        with database.get_session() as session:
            database.insert_event(session, event_data)
        dup_filter.record(plate_number, track_id)
        log.info(
            "✅ STORED: plate=%s type=%s color=%s dir=%s camera=%s conf=%s",
            plate_number, vehicle_type, color, direction,
            camera_meta.get("camera_id"),
            f"{confidence:.2f}" if confidence is not None else "n/a",
        )
        if progress is not None:
            try:
                progress.on_detection(
                    plate_number=plate_number,
                    confidence=confidence,
                    plate_image=image_path,
                    vehicle_image=vehicle_image_path,
                )
            except Exception as exc:
                log.debug("Progress reporter raised on_detection: %s", exc)
        return True
    except Exception as exc:
        log.error("Failed to store event for %s: %s", plate_number, exc)
        return False


def _validate_ocr_text(plate_validator, text: str) -> tuple[str | None, str | None]:
    """Validate OCR text without over-correcting already valid plates."""
    if not text:
        return (None, None)

    plate_number, series_type = plate_validator.validate(text)
    if plate_number:
        return (plate_number, series_type)

    corrected = correct_ocr_text(text)
    return plate_validator.validate(corrected)


# Overlay colors, BGR. Unconfirmed tracks (still accumulating OCR votes)
# draw in amber; a track fusion has actually stored draws in green.
_OVERLAY_COLOR_UNCONFIRMED = (0, 165, 255)
_OVERLAY_COLOR_CONFIRMED = (0, 200, 0)


def _draw_live_overlay(
    frame: np.ndarray,
    tracks: list,
    track_display_labels: dict[int, tuple[str, bool]],
) -> np.ndarray:
    """Return a copy of `frame` with each active track's box and current
    best-known label drawn on it, for the dashboard's live view.
    """
    overlay = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = track.bbox
        label, confirmed = track_display_labels.get(track.track_id, (f"#{track.track_id}", False))
        color = _OVERLAY_COLOR_CONFIRMED if confirmed else _OVERLAY_COLOR_UNCONFIRMED
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        text_y = max(0, y1 - 8)
        cv2.putText(
            overlay, label, (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
        )
    return overlay


# How often the progress reporter is called, in processed frames. Each call
# ends in an atomic file write for the dashboard to poll, which is orders of
# magnitude more expensive than a frame of bookkeeping -- so it happens a few
# times a second, not thirty.
_PROGRESS_EVERY_N_FRAMES = 10


def _probe_total_frames(source) -> int:
    """Frame count of a recorded source, or 0 when there isn't one.

    Only used to size a progress bar. A live stream, an unreadable file, or
    a container with no frame count all return 0, and the dashboard shows an
    indeterminate bar for that rather than a percentage it cannot honour.
    """
    try:
        probe = cv2.VideoCapture(str(source))
        try:
            return max(0, int(probe.get(cv2.CAP_PROP_FRAME_COUNT)))
        finally:
            probe.release()
    except Exception:
        return 0


class VideoSourceUnavailable(RuntimeError):
    """The configured video source could not be opened.

    Raised instead of calling sys.exit() so an embedding caller -- the
    multi-camera manager, which must mark one camera failed and move on to
    the next -- can catch it. main() still turns it into exit status 1, so
    the CLI behaves exactly as it did.
    """


def run_pipeline(
    config: dict,
    benchmark: bool = False,
    camera_meta: dict | None = None,
    session_context: dict | None = None,
    progress=None,
    source_override: str | None = None,
    max_duration_seconds: float | None = None,
    models=None,
) -> dict:
    """Run the ALPR pipeline over one video source, start to finish.

    The first two arguments are the original single-gate interface and are
    unchanged. The rest are optional and exist so the multi-camera manager
    can run this same pipeline once per camera without forking it:

      camera_meta      The capturing camera's camera_id / camera_name /
                       latitude / longitude. None (default) reads this
                       device's own `camera:` block from config.yaml, which
                       is exactly what the single-gate path has always done.
      session_context  processing_session / trajectory_order / video_source
                       stamped onto every event this run stores.
      progress         Optional reporter (see cameras.manager.ProgressReporter)
                       notified of frames and detections as they happen.
                       Never affects pipeline behaviour; exceptions from it
                       are swallowed.
      source_override  Process this source instead of config["video"]["source"].
                       Applied to a COPY of the video config, so the caller's
                       config dict is not mutated between cameras.
      models           A preloaded cameras.pipeline_models.PipelineModels to
                       use instead of building the detectors and OCR engine
                       here. None (the default) builds them, exactly as the
                       single-gate CLI has always done.

                       The multi-camera manager passes one so a four-camera
                       queue deserialises its TensorRT plans ONCE rather than
                       four times -- measured at ~2m40s per load on a Jetson
                       Orin, so this is the difference between a ten-minute
                       and a three-minute sweep. A bundle that was passed in
                       is NOT released here: the next camera still needs it,
                       and whoever built it closes it.
      max_duration_seconds
                       Stop after roughly this long. None (the default) means
                       run until the source is exhausted -- correct for a
                       recorded file, which ends on its own.

                       A LIVE source never ends, so a sequential queue over
                       RTSP cameras would block forever on the first one and
                       never reach the second. The manager therefore gives
                       every live camera a bounded dwell time, and this is
                       how that bound reaches the loop. Enforced
                       cooperatively at the top of the frame loop, so the
                       final OCR-fusion flush still runs and plates that were
                       mid-vote are stored rather than dropped.

    Returns:
        A summary dict (frames, stored events, elapsed, latency averages).
        The CLI ignores it; the manager records it on the camera's progress.

    Raises:
        VideoSourceUnavailable: The source could not be opened.
    """
    log = _init_logger(config)
    log.info("ALPR University Gate pipeline starting...")

    if source_override:
        # Copy rather than mutate: the manager reuses one config dict across
        # four cameras, and rewriting video.source in place would leak each
        # camera's source into the next one's run.
        config = {**config, "video": {**config.get("video", {}), "source": source_override}}

    _ensure_system_site_packages()

    # ── imports ──────────────────────────────────────────────────────────
    from src.capture.frame_capture import FrameCapture
    from src.detection.vehicle_detector import VehicleDetector
    from src.tracking.vehicle_tracker import VehicleTracker
    from src.detection.plate_detector import PlateDetector
    from src.preprocessing.plate_preprocessor import PlatePreprocessor
    from src.enhancement.super_resolution import SuperResolutionEnhancer
    from src.ocr import create_ocr_engine
    from src.validation.plate_validator import PlateValidator
    from src.validation.ocr_fusion import OCRFusion
    from src.classification.color_classifier import ColorClassifier
    from src.classification.vehicle_classifier import VehicleClassifier
    from src.database.duplicate_filter import DuplicateFilter
    from src.utils.direction_detector import DirectionDetector
    from src.utils.motion_filter import MotionFilter
    from src.utils.live_frame import LiveFramePublisher
    from src.database import db as database

    det_cfg = config["detection"]
    enh_cfg = config["enhancement"]
    ocr_cfg = config["ocr"]
    fus_cfg = config["fusion"]
    db_cfg  = config["database"]

    # ── component init ────────────────────────────────────────────────────
    #
    # Split in two on purpose:
    #
    #   SHARED   the detectors, OCR engine and upscaler are stateless with
    #            respect to the video, so a multi-camera session loads them
    #            once and passes them in. `owns_models` records whether this
    #            run built them and must therefore release them.
    #
    #   PER RUN  everything below carries state between frames -- track ids,
    #            fusion votes, duplicate windows, motion history -- and is
    #            rebuilt for every camera so no camera can contaminate the
    #            next. The duplicate filter especially: sharing one across
    #            cameras would suppress the second, third and fourth sighting
    #            of a plate as duplicates, which is exactly what a trajectory
    #            is made of.
    owns_models = models is None
    if owns_models:
        vehicle_detector = VehicleDetector.from_config(config)
        plate_detector   = PlateDetector.from_config(config)
        sr_enhancer      = SuperResolutionEnhancer(
            enh_cfg["realesrgan_model_path"], int(enh_cfg["sr_threshold_px"])
        )
        ocr_engine       = create_ocr_engine(str(ocr_cfg.get("backend", "paddleocr")), config)
    else:
        vehicle_detector = models.vehicle_detector
        plate_detector   = models.plate_detector
        sr_enhancer      = models.sr_enhancer
        ocr_engine       = models.ocr_engine

    frame_capture      = FrameCapture.from_config(config)
    vehicle_tracker    = VehicleTracker(
        int(config["tracking"]["lost_track_timeout"]),
        float(config["tracking"].get("minimum_matching_threshold", 0.5)),
    )
    plate_preprocessor = PlatePreprocessor.from_config(config)
    plate_validator    = PlateValidator()
    ocr_fusion         = OCRFusion(
        window_size=int(fus_cfg.get("window_size", 5)),
        min_exact_votes=int(fus_cfg.get("min_exact_votes", 1)),
    )
    color_classifier   = ColorClassifier.from_config(config)
    vehicle_classifier = VehicleClassifier()
    dup_filter         = DuplicateFilter.from_config(config)
    direction_detector = DirectionDetector.from_config(config)
    motion_filter      = MotionFilter.from_config(config)

    database.init_db(db_cfg["path"])
    image_save_path = db_cfg.get("image_save_path", "data/plate_crops/")
    vehicle_image_save_path = db_cfg.get("vehicle_image_save_path", "data/vehicle_crops/")
    # Static for the life of this run. For the single-gate path that means
    # one Jetson watching one gate (read once from config.yaml); for a
    # multi-camera run the manager supplies the current camera's block.
    if camera_meta is None:
        camera_meta = get_camera_metadata(config)
    log.info(
        "Camera: %s (%s) at lat=%s lon=%s",
        camera_meta["camera_id"], camera_meta["camera_name"],
        camera_meta["latitude"], camera_meta["longitude"],
    )
    fusion_window   = int(fus_cfg.get("window_size", 5))
    min_conf        = float(fus_cfg.get("min_confidence", 0.70))

    # A live source hands read_frame() whatever the newest camera frame is;
    # when per-frame work falls behind the camera's own rate, frame_capture.py
    # drops frames rather than queuing them (see its docstring). That keeps
    # latency bounded, but it means a big gap between two *processed* frames
    # lets a moving vehicle's box jump far enough that ByteTrack's IOU match
    # fails and the track fragments -- resetting OCR fusion to zero votes
    # right as the vehicle is passing. A stationary vehicle never hits this
    # (its box barely moves no matter how many frames get skipped), which is
    # why this only ever showed up on a live camera and never on a recorded
    # file (a file has no real-time pressure, so every frame gets processed
    # in order and tracks never fragment from a skipped frame).
    #
    # The root cause isn't model speed -- benchmarked on-device, plate
    # detection and OCR are both single-digit-to-low-teens ms. It's that the
    # full plate-detect + preprocess + SR + OCR chain runs for *every*
    # active track on *every* processed frame, so cost scales with how many
    # vehicles are simultaneously in view and can blow well past the
    # camera's own frame interval (measured: ~38ms/frame already with just
    # 1 tracked vehicle, ~98ms/frame with 4, against a ~33ms budget at 30fps).
    # Capping how many tracks get that chain per frame -- rotating through
    # the rest across subsequent frames -- bounds the worst case regardless
    # of how many vehicles are in frame at once, while vehicle detection and
    # tracking (cheap, run for every track regardless) stay close to the
    # camera's real frame rate. Only applies to live sources; a recorded
    # file keeps servicing every eligible track every frame, unchanged.
    live_max_heavy_tracks = max(1, int(config["tracking"].get("live_max_heavy_tracks_per_frame", 1)))
    _heavy_rr_offset = 0

    api_cfg = config.get("api", {})
    # Where this run publishes its annotated frames.
    #
    # Single-gate: one shared file, which the API's /stream endpoint serves.
    #
    # Multi-camera: one file PER CAMERA, so the dashboard can show all four
    # feeds side by side. Cameras run one at a time, so a single shared file
    # would only ever show whichever camera is currently active and the other
    # three panels would show that same camera's frames mislabelled as their
    # own. Keyed by camera_id, each panel keeps the last frame that camera
    # actually produced.
    if session_context and camera_meta.get("camera_id"):
        live_frames_dir = Path(api_cfg.get("live_frames_dir", "data/live_frames"))
        live_frame_target = str(live_frames_dir / f"{camera_meta['camera_id']}.jpg")
    else:
        live_frame_target = api_cfg.get("live_frame_path", "data/live_frame.jpg")

    # Multi-camera runs feed the dashboard's continuous preview streams, which
    # play every frame they are given -- so they publish faster than the
    # single-gate snapshot feed, and downscaled, since a preview panel is far
    # smaller than the source video.
    if session_context and camera_meta.get("camera_id"):
        preview_fps = float(api_cfg.get("live_frames_fps", 15.0))
        preview_width = api_cfg.get("live_frames_max_width", 960)
    else:
        preview_fps = float(api_cfg.get("live_frame_fps", 8.0))
        preview_width = None

    live_frame_publisher = (
        LiveFramePublisher(
            path=live_frame_target,
            max_fps=preview_fps,
            max_width=preview_width,
            # Encode previews off the pipeline thread for camera-wall runs,
            # so showing smooth video does not slow ALPR down.
            background=preview_width is not None,
        )
        if api_cfg.get("live_stream_enabled", True) else None
    )
    # track_id -> (label, confirmed) for the live overlay: no entry means
    # "still accumulating reads" (drawn as just the track id); label is the
    # plate text once OCR has a candidate; confirmed=True once fusion has
    # actually stored it (drawn in a different color).
    track_display_labels: dict[int, tuple[str, bool]] = {}

    log.info("OCR backend: %s | fusion window: %d | min_conf: %.2f",
             ocr_cfg.get("backend"), fusion_window, min_conf)

    benchmark_recorder = BenchmarkRecorder() if benchmark else None

    plate_detection_times: list[float] = []
    plate_detection_calls = 0
    plate_detection_calls_per_track: dict[int, int] = defaultdict(int)

    ocr_preprocess_times: list[float] = []
    ocr_inference_times: list[float] = []
    ocr_postprocess_times: list[float] = []
    ocr_calls = 0
    ocr_calls_per_track: dict[int, int] = defaultdict(int)

    # per-track state
    track_plate_crops: dict[int, np.ndarray] = {}   # best raw plate crop
    track_centroids:   dict[int, tuple]      = {}
    stored_tracks:     set[int]              = set()

    # dup_filter._plate_times / _track_times otherwise grow for the life of
    # the process (is_duplicate() only skips expired entries by time, it
    # doesn't remove them) -- periodically prune via the class's own
    # cleanup() so a 24/7 deployment doesn't pay an ever-growing per-event
    # scan cost.
    _DUP_CLEANUP_INTERVAL_S = 60.0
    _last_dup_cleanup = time.monotonic()

    # ── open video ────────────────────────────────────────────────────────
    try:
        frame_capture.open()
    except RuntimeError as exc:
        log.critical("Cannot open video source: %s", exc)
        raise VideoSourceUnavailable(str(exc)) from exc

    log.info("Pipeline running. Press Ctrl+C to stop.")

    # Progress bookkeeping. Cheap by construction: two integer increments per
    # frame, and the reporter is only *called* every _PROGRESS_EVERY_N_FRAMES
    # -- a status write is a file rename, far too expensive to do at 30fps.
    frames_processed = 0
    events_stored = 0
    run_started = time.perf_counter()
    _last_progress_frame = 0
    # Separate clock for max_duration_seconds, started at the FIRST decoded
    # frame rather than here. The detectors and the OCR engine are lazily
    # loaded on their first call, and deserialising two TensorRT plans plus
    # warmup costs tens of seconds on a Jetson -- all of it inside the first
    # iteration of the loop below. Measuring the dwell time from loop entry
    # therefore spends most of a short budget on warmup and samples almost no
    # video (measured: a 20s budget yielded ONE processed frame). Timing from
    # the first frame makes "sample this camera for N seconds" mean N seconds
    # of actual streaming, which is what the setting promises.
    sampling_started: float | None = None

    if progress is not None:
        try:
            progress.on_start(total_frames=_probe_total_frames(config["video"]["source"]))
        except Exception as exc:
            log.debug("Progress reporter raised on_start: %s", exc)

    # ── main loop ─────────────────────────────────────────────────────────
    try:
        while True:
            if sampling_started is None and frames_processed > 0:
                # Frame 1 is complete, so every lazily-loaded model is now
                # warm. Start the sampling clock here, at the top of the
                # second iteration, rather than inside frame 1 -- placing it
                # there would put the warmup inside the window it is meant to
                # exclude. Done here rather than at the end of the loop body
                # because several paths `continue` before reaching the end.
                sampling_started = time.perf_counter()

            if (
                max_duration_seconds is not None
                and sampling_started is not None
                and time.perf_counter() - sampling_started >= max_duration_seconds
            ):
                log.info(
                    "Reached the %.0fs limit for this source — ending this "
                    "camera's run.", max_duration_seconds,
                )
                break

            if progress is not None and progress.should_stop():
                # Cooperative cancellation: the dashboard's STOP button sets
                # this. Breaking here (rather than killing the thread) means
                # the finally-block below still runs, so buffered plates are
                # flushed and stored instead of silently lost.
                log.info("Stop requested — ending this camera's run early.")
                break

            success, frame = frame_capture.read_frame()
            if not success or frame is None:
                if frame_capture.is_live:
                    # Transient stall/reconnect on a live source -- not the
                    # end of the stream. FrameCapture already logged why
                    # and is handling reconnection internally; just keep
                    # polling instead of tearing down the whole pipeline.
                    continue
                log.info("Video ended — stopping pipeline.")
                break

            frame_start = time.perf_counter()
            frames_processed += 1

            if (
                progress is not None
                and frames_processed - _last_progress_frame >= _PROGRESS_EVERY_N_FRAMES
            ):
                _last_progress_frame = frames_processed
                elapsed = time.perf_counter() - run_started
                try:
                    progress.on_frame(
                        frames_processed=frames_processed,
                        fps=frames_processed / elapsed if elapsed > 0 else 0.0,
                        avg_ocr_ms=(
                            sum(ocr_inference_times) / len(ocr_inference_times)
                            if ocr_inference_times else 0.0
                        ),
                        avg_detection_ms=(
                            sum(plate_detection_times) / len(plate_detection_times)
                            if plate_detection_times else 0.0
                        ),
                    )
                except Exception as exc:
                    log.debug("Progress reporter raised on_frame: %s", exc)

            _now_monotonic = time.monotonic()
            if _now_monotonic - _last_dup_cleanup >= _DUP_CLEANUP_INTERVAL_S:
                dup_filter.cleanup()
                _last_dup_cleanup = _now_monotonic

            # Vehicle detection
            vehicle_detection_start = time.perf_counter()
            detections = vehicle_detector.detect(frame)
            if benchmark_recorder is not None:
                benchmark_recorder.record_vehicle_detection(
                    time.perf_counter() - vehicle_detection_start
                )
            if not detections:
                if benchmark_recorder is not None:
                    benchmark_recorder.record_frame(time.perf_counter() - frame_start)
                if live_frame_publisher is not None:
                    live_frame_publisher.publish(frame)
                continue

            # Vehicle tracking
            tracks = vehicle_tracker.update(detections, frame)

            # Cheap per-track bookkeeping runs for every track, every frame,
            # regardless of the live-source throttle below -- motion history
            # must stay current for every vehicle or the next frame's
            # motion-filter decisions would be wrong.
            eligible_tracks: list[tuple] = []
            for track in tracks:
                tid = track.track_id
                x1, y1, x2, y2 = track.bbox

                # Motion filter — skip for first 5 frames to build history
                motion_filter.update(tid, track.centroid)
                track_centroids[tid] = track.centroid

                # Only apply motion filter after enough history (5 frames)
                buf_size = len(motion_filter._history.get(tid, []))
                if buf_size >= motion_filter.history_frames:
                    if not motion_filter.is_moving(tid):
                        continue

                # Already stored this track (strict once-per-track emission)
                if tid in stored_tracks:
                    continue

                # Crop vehicle
                h, w = frame.shape[:2]
                vehicle_crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                if vehicle_crop.size == 0:
                    continue

                eligible_tracks.append((track, vehicle_crop))

            # Live source: only the chosen tracks get the expensive chain
            # this frame; the rest get their turn on a later frame (see the
            # live_max_heavy_tracks_per_frame comment earlier in this
            # function). Recorded file: every eligible track, every frame,
            # exactly as before.
            if frame_capture.is_live and len(eligible_tracks) > live_max_heavy_tracks:
                start = _heavy_rr_offset % len(eligible_tracks)
                heavy_tracks = [
                    eligible_tracks[(start + i) % len(eligible_tracks)]
                    for i in range(live_max_heavy_tracks)
                ]
                _heavy_rr_offset += 1
            else:
                heavy_tracks = eligible_tracks

            for track, vehicle_crop in heavy_tracks:
                tid = track.track_id

                plate_detection_start = time.perf_counter()
                plate_detection_calls += 1
                plate_detection_calls_per_track[tid] += 1
                plate_crops = plate_detector.detect(vehicle_crop)
                plate_detection_elapsed = time.perf_counter() - plate_detection_start
                plate_detection_times.append(plate_detection_elapsed * 1000.0)
                if benchmark_recorder is not None:
                    benchmark_recorder.record_plate_detection(plate_detection_elapsed)
                if not plate_crops:
                    continue

                plate_crop = plate_crops[0]

                # Save best crop for this track (used at fusion time).
                # Store the raw plate_crop so the preprocessor can run fresh
                # at fusion time (potentially with SR enhancement).
                if tid not in track_plate_crops:
                    track_plate_crops[tid] = plate_crop
                    track_plate_crops[f"{tid}_vehicle"] = vehicle_crop

                # Preprocessing, including super-resolution on every attempt
                # (not just once at fusion time) -- for a poor-quality camera
                # source, a single enhance() at the end doesn't help the live
                # OCR reads that feed fusion in the first place. Detection/OCR
                # are now fast enough (single-digit ms) that this fits inside
                # the per-frame budget; sr_enhancer.enhance() already no-ops
                # above the configured sr_threshold_px, so this doesn't touch
                # crops that are already large enough.
                preprocess_start = time.perf_counter()
                preprocessed = plate_preprocessor.process(plate_crop)
                ocr_input = sr_enhancer.enhance(preprocessed)
                preprocess_elapsed = time.perf_counter() - preprocess_start
                ocr_preprocess_times.append(preprocess_elapsed * 1000.0)

                # OCR on upscaled preprocessed crop
                ocr_calls += 1
                ocr_calls_per_track[tid] += 1
                ocr_start = time.perf_counter()
                raw_text, confidence = ocr_engine.recognize(ocr_input)
                inference_elapsed = time.perf_counter() - ocr_start
                ocr_inference_times.append(inference_elapsed * 1000.0)
                if benchmark_recorder is not None:
                    benchmark_recorder.record_ocr(inference_elapsed)
                if not raw_text:
                    continue

                # Post-process OCR text
                postprocess_start = time.perf_counter()
                raw_text = remove_noise_characters(raw_text)

                # Validate
                plate_number, series_type = _validate_ocr_text(plate_validator, raw_text)
                if plate_number is None:
                    postprocess_elapsed = time.perf_counter() - postprocess_start
                    ocr_postprocess_times.append(postprocess_elapsed * 1000.0)
                    continue

                # Confidence filter
                if confidence < min_conf:
                    postprocess_elapsed = time.perf_counter() - postprocess_start
                    ocr_postprocess_times.append(postprocess_elapsed * 1000.0)
                    continue
                postprocess_elapsed = time.perf_counter() - postprocess_start
                ocr_postprocess_times.append(postprocess_elapsed * 1000.0)

                log.info("Track %d: OCR='%s' conf=%.2f", tid, plate_number, confidence)
                if tid not in stored_tracks:
                    track_display_labels[tid] = (plate_number, False)

                # Add to fusion buffer
                ocr_fusion.add_result(tid, plate_number, confidence)

                # Check if fusion window is full → emit result
                buf = ocr_fusion._buffers.get(tid)
                if buf and len(buf) >= fusion_window:
                    result = ocr_fusion.flush(tid)
                    if result:
                        fused_plate, fused_conf = result
                        plate_val, s_type = _validate_ocr_text(plate_validator, fused_plate)
                        if plate_val and fused_conf >= min_conf:
                            # Apply SR on the best saved crop at fusion time
                            best_crop = track_plate_crops.get(tid, plate_crop)
                            color_crop = track_plate_crops.get(f"{tid}_vehicle", best_crop)
                            best_preprocessed = plate_preprocessor.process(best_crop)
                            enhanced_crop = sr_enhancer.enhance(best_preprocessed)

                            # Re-run OCR on SR-enhanced crop for final result
                            sr_text, sr_conf = ocr_engine.recognize(enhanced_crop)
                            if sr_text:
                                sr_text = remove_noise_characters(sr_text)
                                sr_plate, sr_series = _validate_ocr_text(plate_validator, sr_text)
                                if sr_plate and sr_conf >= min_conf:
                                    if sr_conf >= fused_conf:
                                        plate_val = sr_plate
                                        s_type = sr_series
                                        log.info("Track %d: SR improved OCR → '%s' conf=%.2f",
                                                 tid, plate_val, sr_conf)

                            # Save the plate crop (prefer the SR-enhanced crop
                            # for archival clarity) instead of the full vehicle image.
                            stored = _store_event(
                                plate_number=plate_val,
                                series_type=s_type,
                                plate_crop=enhanced_crop,
                                color_classifier=color_classifier,
                                vehicle_classifier=vehicle_classifier,
                                dup_filter=dup_filter,
                                direction_detector=direction_detector,
                                track_id=tid,
                                centroid=track.centroid,
                                image_save_path=image_save_path,
                                camera_meta=camera_meta,
                                database=database,
                                log=log,
                                # color_crop is this track's saved whole-vehicle
                                # image (see track_plate_crops above) -- what the
                                # map popup shows, since a plate crop alone is
                                # not recognisable as a vehicle.
                                vehicle_crop=color_crop,
                                confidence=fused_conf,
                                ocr_text=fused_plate,
                                session_context=session_context,
                                vehicle_image_save_path=vehicle_image_save_path,
                                progress=progress,
                            )
                            if stored:
                                events_stored += 1
                                # Enforce strict once-per-track emission.
                                # After this point, no further flush/store attempts for this tid.
                                stored_tracks.add(tid)
                                track_display_labels[tid] = (plate_val, True)
                                continue

            # Only draw the overlay when the publisher will actually write
            # this frame -- drawing copies the whole frame, and at a capped
            # preview rate most frames would be discarded right after.
            if live_frame_publisher is not None and live_frame_publisher.is_due():
                live_frame_publisher.publish(
                    _draw_live_overlay(frame, tracks, track_display_labels)
                )

            if benchmark_recorder is not None:
                benchmark_recorder.record_frame(time.perf_counter() - frame_start)

    except KeyboardInterrupt:
        log.info("Keyboard interrupt — shutting down...")

    finally:
        # Flush remaining buffers
        log.info("Flushing remaining OCR fusion buffers...")
        pending = ocr_fusion.flush_all()
        for tid, (plate_number, confidence) in pending.items():
            if tid in stored_tracks:
                continue
            if not plate_number or confidence < min_conf * 0.9:
                log.info(
                    "Track %d: dropped at final flush (fused_plate=%r confidence=%.2f "
                    "below threshold %.2f, or fusion had too few matching votes)",
                    tid, plate_number, confidence, min_conf * 0.9,
                )
                continue
            plate_val, s_type = _validate_ocr_text(plate_validator, plate_number)
            if not plate_val:
                log.info(
                    "Track %d: dropped at final flush (fused_plate=%r failed format validation)",
                    tid, plate_number,
                )
                continue
            best_crop = track_plate_crops.get(tid, np.zeros((20, 80, 3), dtype=np.uint8))
            # Use the saved plate crop for storage (not the full vehicle image).
            plate_image_to_save = best_crop
            centroid  = track_centroids.get(tid, (0.0, 0.0))
            if _store_event(
                plate_number=plate_val,
                series_type=s_type,
                plate_crop=plate_image_to_save,
                color_classifier=color_classifier,
                vehicle_classifier=vehicle_classifier,
                dup_filter=dup_filter,
                direction_detector=direction_detector,
                track_id=tid,
                centroid=centroid,
                image_save_path=image_save_path,
                camera_meta=camera_meta,
                database=database,
                log=log,
                vehicle_crop=track_plate_crops.get(f"{tid}_vehicle"),
                confidence=confidence,
                ocr_text=plate_number,
                session_context=session_context,
                vehicle_image_save_path=vehicle_image_save_path,
                progress=progress,
            ):
                events_stored += 1

        frame_capture.release()
        if live_frame_publisher is not None:
            live_frame_publisher.close()

        # Release the OCR backend's GPU resources. Matters because the
        # multi-camera manager runs one pipeline per camera inside a single
        # process: without this, a four-camera queue would build four
        # TensorRT engines and free none of them until the process exited.
        # Duck-typed and swallowed -- close() is optional on the OCREngine
        # interface, and a cleanup failure must never mask a real error or
        # stop the next camera from starting.
        # Release the GPU models only if this run built them. When the
        # multi-camera manager supplied them, the next camera in the queue
        # still needs them and the manager closes them once the queue ends.
        if owns_models:
            for name, component in (
                ("OCR engine", ocr_engine),
                ("vehicle detector", vehicle_detector),
                ("plate detector", plate_detector),
            ):
                try:
                    close = getattr(component, "close", None)
                    if callable(close):
                        close()
                        log.info("%s released.", name.capitalize())
                except Exception as exc:
                    log.warning("Failed to release the %s: %s", name, exc)

        if benchmark_recorder is not None:
            summary = benchmark_recorder.summary()
            log.info(
                "BENCHMARK summary: processed_frames=%d fps=%.2f vehicle_detection_latency_ms(avg=%.2f min=%.2f max=%.2f p95=%.2f) "
                "ocr_latency_ms(avg=%.2f min=%.2f max=%.2f p95=%.2f) plate_detection_latency_ms(avg=%.2f min=%.2f max=%.2f p95=%.2f)",
                summary["processed_frames"],
                summary["fps"],
                summary["vehicle_detection_latency_ms"]["avg_ms"],
                summary["vehicle_detection_latency_ms"]["min_ms"],
                summary["vehicle_detection_latency_ms"]["max_ms"],
                summary["vehicle_detection_latency_ms"]["p95_ms"],
                summary["ocr_latency_ms"]["avg_ms"],
                summary["ocr_latency_ms"]["min_ms"],
                summary["ocr_latency_ms"]["max_ms"],
                summary["ocr_latency_ms"]["p95_ms"],
                summary["plate_detection_latency_ms"]["avg_ms"],
                summary["plate_detection_latency_ms"]["min_ms"],
                summary["plate_detection_latency_ms"]["max_ms"],
                summary["plate_detection_latency_ms"]["p95_ms"],
            )

        def _stats_ms(values: list[float]) -> tuple[float, float, float, float]:
            if not values:
                return 0.0, 0.0, 0.0, 0.0
            v = sorted(values)
            avg = statistics.mean(v)
            p95 = statistics.quantiles(v, n=20)[-1] if len(v) >= 20 else v[-1]
            return avg, v[0], v[-1], p95

        plate_avg, plate_min, plate_max, plate_p95 = _stats_ms(plate_detection_times)
        pre_avg, pre_min, pre_max, pre_p95 = _stats_ms(ocr_preprocess_times)
        inf_avg, inf_min, inf_max, inf_p95 = _stats_ms(ocr_inference_times)
        post_avg, post_min, post_max, post_p95 = _stats_ms(ocr_postprocess_times)

        log.info(
            "PROFILE summary: plate_detection_calls=%d tracks=%d avg_per_track=%.2f max_per_track=%d "
            "plate_detection_ms(avg=%.2f min=%.2f max=%.2f p95=%.2f) "
            "ocr_calls=%d tracks=%d avg_per_track=%.2f "
            "ocr_preprocess_ms(avg=%.2f p95=%.2f) ocr_inference_ms(avg=%.2f p95=%.2f) ocr_postprocess_ms(avg=%.2f p95=%.2f)",
            plate_detection_calls,
            len(plate_detection_calls_per_track),
            plate_detection_calls / len(plate_detection_calls_per_track) if plate_detection_calls_per_track else 0.0,
            max(plate_detection_calls_per_track.values(), default=0),
            plate_avg,
            plate_min,
            plate_max,
            plate_p95,
            ocr_calls,
            len(ocr_calls_per_track),
            ocr_calls / len(ocr_calls_per_track) if ocr_calls_per_track else 0.0,
            pre_avg,
            pre_p95,
            inf_avg,
            inf_p95,
            post_avg,
            post_p95,
        )
        log.info("Pipeline shut down cleanly.")

    # Returned (not just logged) so the multi-camera manager can record what
    # each camera actually achieved on that camera's progress entry. Built
    # after the finally-block so it includes plates stored by the final
    # fusion flush.
    return {
        "frames_processed": frames_processed,
        "events_stored": events_stored,
        "elapsed_seconds": time.perf_counter() - run_started,
        "avg_ocr_ms": inf_avg,
        "avg_plate_detection_ms": plate_avg,
        "camera_id": camera_meta.get("camera_id"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ALPR University Gate — Main Pipeline")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--source", default=None, help="Video source (RTSP URL or file path)")
    parser.add_argument("--benchmark", action="store_true", help="Print FPS and latency summary at shutdown")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.source:
        config["video"]["source"] = args.source

    try:
        run_pipeline(config, benchmark=args.benchmark)
    except VideoSourceUnavailable:
        # Already logged as critical by run_pipeline. Exit status 1 here
        # keeps the CLI contract the same as when this was a sys.exit(1)
        # inside the pipeline itself.
        sys.exit(1)


if __name__ == "__main__":
    main()
