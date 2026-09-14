"""
Heavy models, loaded once and shared across every camera in a session.

Loading the vehicle detector, the plate detector and the OCR engine means
deserialising three TensorRT plans and warming each one up. On a Jetson Orin
that is around two and a half minutes -- measured from a real run:

    20:37:06  session started, camera 1 begins
    20:39:46  TensorRT + OCR engines loaded
    20:39:47  first frame

`run_pipeline()` builds those itself, which is right for the single-gate CLI
(one process, one run) but wrong for a sequential queue: four cameras meant
four loads, roughly ten minutes of a demo spent doing nothing visible, for
clips a few hundred frames long.

These three are safe to share because they are stateless with respect to the
video: they map an image to a result and keep no per-camera memory. Anything
that DOES carry state between frames -- the tracker (track ids must restart
per camera), OCR fusion, the duplicate filter, the direction detector, the
motion filter -- stays per camera and is still rebuilt for every run, so
cameras cannot contaminate each other's results.

That last point matters most for the duplicate filter: sharing one across
cameras would suppress the second, third and fourth sighting of a plate as
"duplicates", and a trajectory needs exactly those sightings.
"""

from __future__ import annotations

import functools
import threading
from typing import Any, Optional

from src.utils.logger import get_logger

_logger = get_logger("cameras.pipeline_models")

# The calls that reach the GPU. Guarded when several live cameras share one
# bundle; everything else (thresholds, class names, paths) is read-only and
# passes straight through.
_GPU_CALLS = frozenset({"detect", "recognize", "enhance", "predict", "read"})


class _Serialized:
    """One model, callable from several camera threads without overlapping.

    A thin proxy: the calls in `_GPU_CALLS` take the shared lock, and every
    other attribute is the wrapped model's own. Proxying rather than editing
    the detectors keeps the single-camera and CLI paths exactly as they
    were -- they never see this class.
    """

    __slots__ = ("_wrapped", "_lock")

    def __init__(self, wrapped: Any, lock: "threading.RLock") -> None:
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_lock", lock)

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(object.__getattribute__(self, "_wrapped"), name)
        if name not in _GPU_CALLS or not callable(attribute):
            return attribute
        lock = object.__getattribute__(self, "_lock")

        @functools.wraps(attribute)
        def guarded(*args, **kwargs):
            with lock:
                return attribute(*args, **kwargs)

        return guarded

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_wrapped"), name, value)

    def __repr__(self) -> str:
        return f"_Serialized({object.__getattribute__(self, '_wrapped')!r})"


class PipelineModels:
    """The video-stateless models a pipeline run needs.

    Build one per session and hand it to each camera's `run_pipeline()` call.
    Ownership is explicit: a run given a bundle uses it and does NOT release
    it, because the next camera still needs it. Whoever built it calls
    close().
    """

    def __init__(
        self,
        vehicle_detector: Any,
        plate_detector: Any,
        ocr_engine: Any,
        sr_enhancer: Any,
    ) -> None:
        self.vehicle_detector = vehicle_detector
        self.plate_detector = plate_detector
        self.ocr_engine = ocr_engine
        self.sr_enhancer = sr_enhancer
        self._closed = False
        # Held by every view handed out by shared(), so concurrent live
        # cameras enter the GPU one at a time.
        self._lock = threading.RLock()

    @classmethod
    def load(cls, config: dict, on_progress=None) -> "PipelineModels":
        """Build and warm up every shared model.

        Args:
            config:      The full ALPR config dict.
            on_progress: Optional callable taking a short status string, so a
                         dashboard can say which model is loading instead of
                         showing a motionless 0% for minutes.

        Raises:
            Whatever the underlying constructors raise. A session that cannot
            load its models has nothing to run, so this failure is not
            swallowed.
        """
        def report(message: str) -> None:
            _logger.info(message)
            if on_progress is not None:
                try:
                    on_progress(message)
                except Exception as exc:      # reporting must never block work
                    _logger.debug("model-load progress reporter raised: %s", exc)

        # Must happen before ultralytics/tensorrt are imported. On JetPack,
        # tensorrt lives in the system dist-packages; without this a
        # virtualenv cannot see it and ultralytics tries to pip-install
        # `tensorrt-cu13` from the network, which blocks for minutes and then
        # fails. run_pipeline has always done this at startup -- the session
        # now loads models BEFORE the pipeline runs, so it must too.
        from src.utils.jetson_paths import ensure_system_site_packages

        ensure_system_site_packages()

        from src.detection.plate_detector import PlateDetector
        from src.detection.vehicle_detector import VehicleDetector
        from src.enhancement.super_resolution import SuperResolutionEnhancer
        from src.ocr import create_ocr_engine

        enhancement_cfg = config["enhancement"]
        ocr_cfg = config["ocr"]

        # Both detectors load their weights lazily on first detect(), so
        # constructing them costs nothing and the real work would otherwise
        # land inside the first camera's first frame -- exactly where it is
        # invisible and where it made the dashboard look hung. _load_model()
        # is the documented warm-up path (it runs one dummy inference), so
        # calling it here moves the cost to where the UI is reporting it.
        report("Loading vehicle detector…")
        vehicle_detector = VehicleDetector.from_config(config)
        vehicle_detector._load_model()

        report("Loading plate detector…")
        plate_detector = PlateDetector.from_config(config)
        plate_detector._load_model()

        report("Loading super-resolution…")
        sr_enhancer = SuperResolutionEnhancer(
            enhancement_cfg["realesrgan_model_path"],
            int(enhancement_cfg["sr_threshold_px"]),
        )

        # The OCR engine loads eagerly in its constructor, so this one needs
        # no separate warm-up call.
        backend = str(ocr_cfg.get("backend", "paddleocr"))
        report(f"Loading OCR engine ({backend})…")
        ocr_engine = create_ocr_engine(backend, config)

        report("Models ready")
        return cls(vehicle_detector, plate_detector, ocr_engine, sr_enhancer)

    def shared(self) -> "PipelineModels":
        """A view of these models that is safe to use from another thread.

        Live cameras all run at once, but the GPU underneath them does not:
        a TensorRT execution context processes one inference at a time, and
        two threads calling into the same one corrupt each other's bindings.

        Every view returned here wraps the SAME models behind the SAME lock,
        so the four streams take turns inside inference while decoding,
        tracking, OCR fusion and database work carry on in parallel -- which
        is where a live stream actually spends its time. Recorded files do
        not need this: they never run concurrently.

        The view does not own the models, so closing it is a no-op; whoever
        loaded them still owns them.
        """
        view = PipelineModels(
            _Serialized(self.vehicle_detector, self._lock),
            _Serialized(self.plate_detector, self._lock),
            _Serialized(self.ocr_engine, self._lock),
            _Serialized(self.sr_enhancer, self._lock),
        )
        view._closed = True          # borrowed, never released by the borrower
        return view

    def close(self) -> None:
        """Release every model that knows how to release itself.

        Idempotent and non-throwing: called from a session's shutdown path,
        where raising would mask whatever is already unwinding.
        """
        if self._closed:
            return
        self._closed = True
        for name, component in (
            ("OCR engine", self.ocr_engine),
            ("vehicle detector", self.vehicle_detector),
            ("plate detector", self.plate_detector),
        ):
            try:
                close = getattr(component, "close", None)
                if callable(close):
                    close()
                    _logger.info("Released the %s.", name)
            except Exception as exc:
                _logger.warning("Failed to release the %s: %s", name, exc)
