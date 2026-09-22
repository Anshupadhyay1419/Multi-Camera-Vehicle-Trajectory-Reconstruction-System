"""
Vehicle detector for the ALPR University Gate system.

Uses a YOLOv8 model pretrained on COCO to detect vehicles (car, truck,
bus, motorcycle) in each frame. Returns Detection dataclasses filtered
by confidence threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.runtime.device import DeviceSpec, device_from_config
from src.utils.logger import get_logger

_logger = get_logger("detection.vehicle_detector")

# COCO class names for vehicle types we care about
_VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}


@dataclass
class Detection:
    """A single object detection result."""
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2 (pixel coords)
    class_label: str
    confidence: float


class VehicleDetector:
    """Detect vehicles in a frame using YOLOv8 COCO pretrained weights.

    Args:
        model_path:           Path to YOLOv8 weights (.pt file).
        confidence_threshold: Minimum confidence score to keep a detection.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        imgsz: int = 480,
        device: str | int = 0,
        half: bool = True,
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.imgsz = imgsz
        self.device = device
        self.half = half
        self._model = None
        # A pre-exported TensorRT/ONNX plan already fixes its precision and
        # device at export time; only a live .pt/torch model needs these
        # passed again at predict() time.
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
    ) -> "VehicleDetector":
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
            model_path=det_cfg["vehicle_model_path"],
            confidence_threshold=float(det_cfg.get("vehicle_confidence", 0.5)),
            imgsz=int(det_cfg.get("vehicle_imgsz", 480)),
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
                "Vehicle detector loaded from '%s' (imgsz=%d device=%s half=%s)",
                self.model_path, self.imgsz, self.device, self.half,
            )
        except Exception as exc:
            _logger.error("Failed to load vehicle detector from '%s': %s", self.model_path, exc)
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

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run vehicle detection on a single frame.

        Args:
            frame: BGR image as NumPy array (H, W, 3).

        Returns:
            List of Detection objects for detected vehicles above the
            confidence threshold. Returns empty list if none found.
        """
        self._load_model()

        try:
            results = self._model(frame, verbose=False, **self._predict_kwargs)
        except Exception as exc:
            _logger.warning("Vehicle detection inference failed: %s", exc)
            return []

        detections: list[Detection] = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                if conf < self.confidence_threshold:
                    continue

                cls_id = int(boxes.cls[i])
                class_name = result.names.get(cls_id, "").lower()

                if class_name not in _VEHICLE_CLASSES:
                    continue

                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    class_label=class_name,
                    confidence=conf,
                ))

        return self.non_max_suppression(detections, iou_threshold=0.5)

    @staticmethod
    def non_max_suppression(
        detections: list[Detection],
        iou_threshold: float = 0.5,
    ) -> list[Detection]:
        """Remove overlapping duplicate detections using per-class NMS.

        This keeps the highest-confidence box for the same vehicle when the
        detector returns multiple highly overlapping boxes in a single frame.
        """
        if len(detections) <= 1:
            return detections

        kept: list[Detection] = []

        for class_label in {d.class_label for d in detections}:
            class_detections = [d for d in detections if d.class_label == class_label]
            class_detections.sort(key=lambda d: d.confidence, reverse=True)

            while class_detections:
                best = class_detections.pop(0)
                kept.append(best)
                class_detections = [
                    det
                    for det in class_detections
                    if VehicleDetector._iou(best.bbox, det.bbox) < iou_threshold
                ]

        return kept

    @staticmethod
    def _iou(
        box_a: tuple[int, int, int, int],
        box_b: tuple[int, int, int, int],
    ) -> float:
        """Compute intersection-over-union for two boxes."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0

        return inter_area / union

    @staticmethod
    def filter_by_confidence(
        detections: list[Detection], threshold: float
    ) -> list[Detection]:
        """Filter a list of detections to those at or above threshold.

        This static method is used by property-based tests to verify the
        filtering invariant independently of model inference.
        """
        return [d for d in detections if d.confidence >= threshold]
