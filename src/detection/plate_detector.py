"""
License plate detector for the ALPR University Gate system.

Uses the custom-trained YOLOv8m model to detect license plate regions
within vehicle crops. Returns cropped plate images as NumPy arrays.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.runtime.device import DeviceSpec, device_from_config
from src.utils.logger import get_logger

_logger = get_logger("detection.plate_detector")


class PlateDetector:
    """Detect license plates within a vehicle crop using custom YOLOv8m.

    Args:
        model_path:           Path to the trained plate detection weights.
        confidence_threshold: Minimum confidence to keep a plate detection.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.4,
        imgsz: int = 320,
        device: str | int = 0,
        half: bool = True,
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        # Plate crops are already localized inside the vehicle box, so a
        # small imgsz is plenty and cuts inference time substantially versus
        # ultralytics' 640 default.
        self.imgsz = imgsz
        self.device = device
        self.half = half
        self._model = None
        # Pre-exported TensorRT/ONNX plans already fix their precision and
        # device at export time; passing `half=`/`device=` again at predict()
        # time only applies to a live .pt/torch model.
        self._predict_kwargs = (
            {"imgsz": self.imgsz}
            if Path(model_path).suffix.lower() in {".engine", ".onnx"}
            else {"imgsz": self.imgsz, "device": self.device, "half": self.half}
        )

    @classmethod
    def from_config(
        cls,
        config: dict,
        device: "DeviceSpec | None" = None,
    ) -> "PlateDetector":
        """Build from config, on an already-resolved device.

        `device` is the DeviceSpec the process resolved. Passing it is what
        keeps every model in a process on the SAME device: this class used
        to read `detection.device` itself, as did the other detector, while
        the OCR engine just assumed CUDA -- so nothing in the system owned
        the answer and nothing noticed when the answers disagreed.

        It stays optional so a one-off script need not build one, and the
        fallback resolves the same config key through the same resolver, so
        a lone script gets a validated device rather than a raw dict value.
        Callers loading several models (see PipelineModels.load) MUST resolve
        once and pass it, because the first model to load decides the
        process's CUDA visibility for every model after it.
        """
        det_cfg = config.get("detection", {})
        if device is None:
            device = device_from_config(config)
        return cls(
            model_path=det_cfg["plate_model_path"],
            confidence_threshold=float(det_cfg.get("plate_confidence", 0.4)),
            imgsz=int(det_cfg.get("plate_imgsz", 320)),
            device=device.ultralytics,
            half=device.use_half,
        )

    def _load_model(self) -> None:
        """Lazy-load the YOLO model on first use and warm it up.

        A model's first inference pays for CUDA context / TensorRT engine
        initialization (can be 1-2s), which otherwise shows up as a huge
        outlier on whichever frame happens to trigger it mid-pipeline.
        Running one dummy inference here moves that cost to startup.
        """
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self._model(dummy, verbose=False, **self._predict_kwargs)
            _logger.info(
                "Plate detector loaded from '%s' (imgsz=%d device=%s half=%s)",
                self.model_path, self.imgsz, self.device, self.half,
            )
        except Exception as exc:
            _logger.error(
                "Failed to load plate detector from '%s': %s", self.model_path, exc
            )
            raise

    def close(self) -> None:
        """Release the loaded model and its GPU memory.

        Needed because the multi-camera manager runs one pipeline per camera
        inside a single process: a four-camera queue otherwise loads four
        copies of this detector and frees none of them until the process
        exits. With a TensorRT (.engine) model that also leaves CUDA state
        alive per copy, which surfaces as a context-stack error at
        interpreter teardown.

        Idempotent and non-throwing -- called from a finally-block during
        shutdown, where raising would mask whatever is already unwinding.
        """
        model, self._model = self._model, None
        if model is None:
            return
        try:
            # Ultralytics keeps the torch module (or the TensorRT wrapper) on
            # .model; dropping both references lets the allocator reclaim the
            # device memory at the next collection.
            inner = getattr(model, "model", None)
            if inner is not None and hasattr(inner, "cpu"):
                inner.cpu()
            del inner, model
            import gc

            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                # torch is optional at runtime for an ONNX/TensorRT-only
                # deployment; nothing to reclaim through it in that case.
                pass
        except Exception as exc:
            _logger.warning("Failed to release %s: %s", type(self).__name__, exc)

    def detect(self, vehicle_crop: np.ndarray) -> list[np.ndarray]:
        """Detect license plates within a vehicle crop.

        Args:
            vehicle_crop: BGR image crop of a detected vehicle (H, W, 3).

        Returns:
            List of plate crop images (NumPy arrays). Empty list if no plate
            detected above the confidence threshold.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return []

        self._load_model()

        try:
            results = self._model(vehicle_crop, verbose=False, **self._predict_kwargs)
        except Exception as exc:
            _logger.warning("Plate detection inference failed: %s", exc)
            return []

        plate_crops: list[np.ndarray] = []
        h, w = vehicle_crop.shape[:2]

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                if conf < self.confidence_threshold:
                    continue

                xyxy = boxes.xyxy[i].cpu().numpy()
                x1 = max(0, int(xyxy[0]))
                y1 = max(0, int(xyxy[1]))
                x2 = min(w, int(xyxy[2]))
                y2 = min(h, int(xyxy[3]))

                if x2 <= x1 or y2 <= y1:
                    continue

                plate_crop = vehicle_crop[y1:y2, x1:x2]
                if plate_crop.size > 0:
                    plate_crops.append(plate_crop)

        return plate_crops
