"""Standalone GPU validation and benchmark for the PARSeq TensorRT engine."""

from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from src.ocr.parseq_tensorrt_engine import PARSeqTensorRTOCREngine


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the PARSeq TensorRT OCR engine without the ALPR pipeline")
    parser.add_argument("--engine", default="models/ocr/parseq.engine")
    parser.add_argument("--image", default="OCR_data/images/127.jpg")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    return parser.parse_args()


def _jetson_utilization() -> tuple[str, str]:
    """Read Jetson's native GR3D (GPU) and CPU counters when tegrastats exists."""
    try:
        text = subprocess.check_output(["tegrastats", "--interval", "100", "--count", "1"], text=True)
        gpu = re.search(r"GR3D_FREQ\s+(\d+)%", text)
        cpu_match = re.search(r"CPU\s*\[([^\]]+)\]", text)
        cpus = [int(value) for value in re.findall(r"(\d+)%@", cpu_match.group(1))] if cpu_match else []
        return (f"{gpu.group(1)}%" if gpu else "unavailable", f"{statistics.mean(cpus):.1f}%" if cpus else "unavailable")
    except Exception:
        return "unavailable", "unavailable"


def _gpu_memory() -> str:
    try:
        import pynvml
        pynvml.nvmlInit()
        info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0))
        return f"{info.used / 2**20:.1f} / {info.total / 2**20:.1f} MiB"
    except Exception:
        return "unavailable (NVML is normally absent on Jetson)"


def main() -> None:
    args = _parse_args()
    image = cv2.imread(str(Path(args.image)))
    if image is None:
        raise FileNotFoundError(f"Could not load OCR image: {args.image}")
    engine = PARSeqTensorRTOCREngine(args.engine)
    if engine._init_failed:
        raise RuntimeError("PARSeq TensorRT engine failed to initialize")

    for _ in range(args.warmup):
        engine.recognize(image)
    samples, gpu_samples = [], []
    text, confidence = "", 0.0
    for _ in range(args.runs):
        started = time.perf_counter()
        text, confidence = engine.recognize(image)
        samples.append((time.perf_counter() - started) * 1000.0)
        if engine.last_inference_ms is not None:
            gpu_samples.append(engine.last_inference_ms)

    gpu_util, cpu_util = _jetson_utilization()
    print(f"Predicted text: {text}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Inference latency (ms): {statistics.mean(gpu_samples) if gpu_samples else 0.0:.3f}")
    print(f"GPU memory usage: {_gpu_memory()}")
    print(f"Average OCR latency (ms): {statistics.mean(samples):.3f}")
    print(f"P95 OCR latency (ms): {np.percentile(samples, 95):.3f}")
    print(f"Average FPS: {1000.0 / statistics.mean(samples):.2f}")
    print(f"GPU utilization: {gpu_util}")
    print(f"CPU utilization: {cpu_util}")
    print(f"Average inference time (ms): {statistics.mean(gpu_samples) if gpu_samples else 0.0:.3f}")
    print(f"Maximum inference time (ms): {max(gpu_samples) if gpu_samples else 0.0:.3f}")


if __name__ == "__main__":
    main()
