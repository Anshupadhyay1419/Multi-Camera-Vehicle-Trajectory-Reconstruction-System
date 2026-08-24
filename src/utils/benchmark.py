from __future__ import annotations

import statistics
import time
from collections import deque
from typing import Deque, Dict, List


class BenchmarkRecorder:
    """Collect lightweight performance metrics for the ALPR pipeline."""

    def __init__(self, max_window: int = 200) -> None:
        self.max_window = max_window
        self.frame_times: Deque[float] = deque(maxlen=max_window)
        self.vehicle_detection_latencies_ms: Deque[float] = deque(maxlen=max_window)
        self.ocr_latencies_ms: Deque[float] = deque(maxlen=max_window)
        self.plate_detection_latencies_ms: Deque[float] = deque(maxlen=max_window)
        self._started_at = time.perf_counter()

    def record_frame(self, elapsed_s: float) -> None:
        self.frame_times.append(elapsed_s)

    def record_vehicle_detection(self, elapsed_s: float) -> None:
        self.vehicle_detection_latencies_ms.append(elapsed_s * 1000.0)

    def record_ocr(self, elapsed_s: float) -> None:
        self.ocr_latencies_ms.append(elapsed_s * 1000.0)

    def record_plate_detection(self, elapsed_s: float) -> None:
        self.plate_detection_latencies_ms.append(elapsed_s * 1000.0)

    def summary(self) -> Dict[str, object]:
        processed = len(self.frame_times)
        elapsed_s = max(sum(self.frame_times), 1e-6) if self.frame_times else max(time.perf_counter() - self._started_at, 1e-6)
        fps = processed / elapsed_s if processed else 0.0

        def _stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {"avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "p95_ms": 0.0}
            return {
                "avg_ms": statistics.mean(values),
                "min_ms": min(values),
                "max_ms": max(values),
                "p95_ms": statistics.quantiles(values, n=20)[-1] if len(values) >= 20 else max(values),
            }

        return {
            "processed_frames": processed,
            "fps": fps,
            "elapsed_s": elapsed_s,
            "vehicle_detection_latency_ms": _stats(list(self.vehicle_detection_latencies_ms)),
            "ocr_latency_ms": _stats(list(self.ocr_latencies_ms)),
            "plate_detection_latency_ms": _stats(list(self.plate_detection_latencies_ms)),
        }
