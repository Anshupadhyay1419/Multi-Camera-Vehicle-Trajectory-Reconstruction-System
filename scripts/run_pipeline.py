"""
ALPR University Gate — Main Pipeline Entry Point

Pipeline flow per frame:
  Frame Capture → Vehicle Detection → Vehicle Tracking
  → Plate Detection → Preprocessing → OCR → Validation
  → OCR Fusion → Color + Vehicle Type → Duplicate Filter
  → Direction Detection → Database Storage

Real-ESRGAN is applied ONCE per vehicle at fusion time (not every frame)
to avoid blocking the live processing loop.

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
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.ocr.ocr_postprocessor import correct_ocr_text, remove_noise_characters

_logger = None


def _init_logger(config: dict):
    global _logger
    _logger = get_logger("main_pipeline", config=config)
    return _logger


def _ensure_system_site_packages() -> None:
    """Enable system-installed Jetson Python packages inside a virtualenv."""
    if sys.prefix == sys.base_prefix:
        return

    for path in ("/usr/lib/python3.12/dist-packages", "/usr/lib/python3/dist-packages"):
        if Path(path).is_dir() and str(path) not in sys.path:
            sys.path.append(str(path))


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
    database,
    log,
) -> bool:
    """Classify, deduplicate, and store a confirmed plate event."""
    if dup_filter.is_duplicate(plate_number, track_id):
        log.info("DUPLICATE skipped: %s (track %d)", plate_number, track_id)
        return False

    color = color_classifier.classify(plate_crop)
    vehicle_type = vehicle_classifier.classify(color)
    direction = direction_detector.update(track_id, centroid) or "IN"
    image_path = database.save_plate_image(plate_crop, plate_number, image_save_path)

    event_data = {
        "plate_number": plate_number,
        "vehicle_type": vehicle_type,
        "plate_color":  color,
        "series_type":  series_type,
        "direction":    direction,
        "image_path":   image_path,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
    }

    try:
        with database.get_session() as session:
            database.insert_event(session, event_data)
        dup_filter.record(plate_number, track_id)
        log.info(
            "✅ STORED: plate=%s type=%s color=%s dir=%s",
            plate_number, vehicle_type, color, direction,
        )
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


def run_pipeline(config: dict, benchmark: bool = False) -> None:
    log = _init_logger(config)
    log.info("ALPR University Gate pipeline starting...")

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
    from src.database import db as database

    det_cfg = config["detection"]
    enh_cfg = config["enhancement"]
    ocr_cfg = config["ocr"]
    fus_cfg = config["fusion"]
    db_cfg  = config["database"]

    # ── component init ────────────────────────────────────────────────────
    frame_capture      = FrameCapture.from_config(config)
    vehicle_detector   = VehicleDetector(det_cfg["vehicle_model_path"], float(det_cfg["vehicle_confidence"]))
    vehicle_tracker    = VehicleTracker(int(config["tracking"]["lost_track_timeout"]))
    plate_detector     = PlateDetector(det_cfg["plate_model_path"], float(det_cfg["plate_confidence"]))
    plate_preprocessor = PlatePreprocessor.from_config(config)
    sr_enhancer        = SuperResolutionEnhancer(enh_cfg["realesrgan_model_path"], int(enh_cfg["sr_threshold_px"]))
    ocr_engine         = create_ocr_engine(str(ocr_cfg.get("backend", "paddleocr")), config)
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
    fusion_window   = int(fus_cfg.get("window_size", 5))
    min_conf        = float(fus_cfg.get("min_confidence", 0.70))

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

    # ── open video ────────────────────────────────────────────────────────
    try:
        frame_capture.open()
    except RuntimeError as exc:
        log.critical("Cannot open video source: %s", exc)
        sys.exit(1)

    log.info("Pipeline running. Press Ctrl+C to stop.")

    # ── main loop ─────────────────────────────────────────────────────────
    try:
        while True:
            success, frame = frame_capture.read_frame()
            if not success or frame is None:
                log.info("Video ended — stopping pipeline.")
                break

            frame_start = time.perf_counter()

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
                continue

            # Vehicle tracking
            tracks = vehicle_tracker.update(detections, frame)

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

                # Preprocessing (fast — no SR in live loop)
                preprocess_start = time.perf_counter()
                preprocessed = plate_preprocessor.process(plate_crop)

                # Upscale small plates with fast OpenCV resize before OCR.
                # Use the preprocessed crop (CLAHE + denoise + sharpen applied)
                # rather than the raw plate_crop so OCR gets a cleaner input.
                h_p, w_p = preprocessed.shape[:2]
                if w_p < 200:
                    scale = max(200 / w_p, 2.0)
                    ocr_input = cv2.resize(
                        preprocessed, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_CUBIC
                    )
                else:
                    ocr_input = preprocessed
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
                                database=database,
                                log=log,
                            )
                            if stored:
                                # Enforce strict once-per-track emission.
                                # After this point, no further flush/store attempts for this tid.
                                stored_tracks.add(tid)
                                continue


    except KeyboardInterrupt:
        log.info("Keyboard interrupt — shutting down...")

    finally:
        # Flush remaining buffers
        log.info("Flushing remaining OCR fusion buffers...")
        pending = ocr_fusion.flush_all()
        for tid, (plate_number, confidence) in pending.items():
            if tid in stored_tracks:
                continue
            if confidence < min_conf * 0.9:
                continue
            plate_val, s_type = _validate_ocr_text(plate_validator, plate_number)
            if not plate_val:
                continue
            best_crop = track_plate_crops.get(tid, np.zeros((20, 80, 3), dtype=np.uint8))
            # Use the saved plate crop for storage (not the full vehicle image).
            plate_image_to_save = best_crop
            centroid  = track_centroids.get(tid, (0.0, 0.0))
            _store_event(
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
                database=database,
                log=log,
            )

        frame_capture.release()
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


def main() -> None:
    parser = argparse.ArgumentParser(description="ALPR University Gate — Main Pipeline")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--source", default=None, help="Video source (RTSP URL or file path)")
    parser.add_argument("--benchmark", action="store_true", help="Print FPS and latency summary at shutdown")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.source:
        config["video"]["source"] = args.source

    run_pipeline(config, benchmark=args.benchmark)


if __name__ == "__main__":
    main()
