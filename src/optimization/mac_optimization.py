"""
Mac Mini optimization module for Apple Silicon (M1/M2/M3) acceleration.

Features:
  - Metal Performance Shaders (MPS) for GPU acceleration
  - Async/await for concurrent processing  
  - Bounded frame queue with intelligent dropping under load
  - Memory profiling and optimization
  - Model compilation for inference speedup
  - CPU/GPU/memory monitoring

Usage:
  python -m src.optimization.mac_optimization --profile
  python -m src.optimization.mac_optimization --convert-to-mlpackage
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import os
import platform
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

_logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """Current system resource usage."""
    cpu_percent: float
    memory_percent: float
    gpu_memory_used_mb: float
    fps: float
    frame_queue_size: int
    dropped_frames: int


class MacOptimizer:
    """Optimize ALPR pipeline for Mac Mini Apple Silicon."""

    def __init__(self, max_queue_size: int = 30, target_fps: int = 30):
        """
        Args:
            max_queue_size: Maximum frames in buffer before dropping
            target_fps: Target inference FPS (controls frame skipping)
        """
        self.max_queue_size = max_queue_size
        self.target_fps = target_fps
        self.platform_is_mac = platform.system() == "Darwin"
        self.has_mps = self._check_mps_availability()
        
        self.frame_queue: deque = deque(maxlen=max_queue_size)
        self.dropped_frames = 0
        self.processed_frames = 0
        self.start_time = time.time()

        _logger.info(
            "Mac Optimizer initialized — platform=%s, MPS=%s, queue_size=%d",
            platform.system(), self.has_mps, max_queue_size
        )

    def _check_mps_availability(self) -> bool:
        """Check if Metal Performance Shaders are available."""
        if not self.platform_is_mac:
            return False

        try:
            import torch
            available = torch.backends.mps.is_available()
            if available:
                _logger.info("✓ MPS (Metal Performance Shaders) available")
            return available
        except Exception as e:
            _logger.warning("MPS not available: %s", e)
            return False

    def get_device(self) -> str:
        """Return optimal PyTorch device: 'mps', 'cuda', or 'cpu'."""
        if self.has_mps:
            return "mps"
        
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except:
            pass
        
        return "cpu"

    def configure_torch_for_mac(self) -> None:
        """Apply Mac-specific PyTorch optimizations."""
        try:
            import torch
            
            if self.has_mps:
                # Enable MPS
                torch.backends.mps.enabled = True
                
                # Disable gradient computation for inference
                torch.set_grad_enabled(False)
                
                # Use float16 for memory efficiency
                torch.set_float32_matmul_precision('high')
                
                _logger.info("✓ PyTorch configured for Mac optimization")
        except Exception as e:
            _logger.error("Failed to configure PyTorch for Mac: %s", e)

    def add_frame(self, frame: np.ndarray) -> bool:
        """Add frame to queue with intelligent dropping under load.

        Returns:
            True if frame was queued, False if dropped due to queue overflow.
        """
        queue_usage = len(self.frame_queue) / self.max_queue_size
        
        if queue_usage > 0.9:
            # Queue 90%+ full — start dropping frames
            self.dropped_frames += 1
            _logger.warning(
                "Frame dropped (queue %.1f%% full). Dropped: %d",
                queue_usage * 100, self.dropped_frames
            )
            return False

        self.frame_queue.append(frame)
        return True

    def get_frame(self) -> Optional[np.ndarray]:
        """Get next frame from queue (FIFO)."""
        try:
            return self.frame_queue.popleft()
        except IndexError:
            return None

    def should_process_frame(self, frame_number: int, skip_interval: int = 2) -> bool:
        """Determine if frame should be processed based on FPS target.

        Args:
            frame_number: Current frame number
            skip_interval: Skip this many frames between processing

        Returns:
            True if frame should be processed
        """
        # Calculate skip interval to achieve target FPS
        expected_skip = max(1, (30 // self.target_fps))  # 30 FPS input source
        return frame_number % expected_skip == 0

    def get_metrics(self) -> SystemMetrics:
        """Get current system resource metrics."""
        try:
            import psutil
            
            proc = psutil.Process(os.getpid())
            cpu_percent = proc.cpu_percent(interval=0.1)
            memory_info = proc.memory_info()
            memory_percent = proc.memory_percent()
            
            # Estimate GPU memory (rough approximation)
            gpu_memory = 0
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            except:
                pass
            
            elapsed_time = time.time() - self.start_time
            fps = self.processed_frames / elapsed_time if elapsed_time > 0 else 0
            
            return SystemMetrics(
                cpu_percent=float(cpu_percent),
                memory_percent=float(memory_percent),
                gpu_memory_used_mb=float(gpu_memory),
                fps=float(fps),
                frame_queue_size=len(self.frame_queue),
                dropped_frames=self.dropped_frames,
            )
        except ImportError:
            _logger.warning("psutil not installed, returning dummy metrics")
            return SystemMetrics(0, 0, 0, 0, len(self.frame_queue), self.dropped_frames)

    def convert_model_to_mlpackage(self, model_path: str, output_dir: str = "models/mlpackage/") -> str:
        """Convert PyTorch model to Core ML format for Mac deployment.

        Args:
            model_path: Path to .pt model file
            output_dir: Output directory for .mlpackage

        Returns:
            Path to converted model
        """
        try:
            import coremltools as ct
            import torch
            from ultralytics import YOLO

            _logger.info("Converting model to Core ML format...")

            # Load YOLO model
            model = YOLO(model_path)

            # Convert to TorchScript first
            scripted_model = torch.jit.script(model.model)

            # Convert to Core ML
            example_input = torch.randn(1, 3, 640, 640)
            traced_model = torch.jit.trace(scripted_model, example_input)

            ml_model = ct.convert(
                traced_model,
                inputs=[ct.ImageType(name="image", shape=(1, 3, 640, 640))],
                outputs=[ct.TensorType(name="output", shape=(1, 25200, 85))],
                compute_units=ct.ComputeUnit.CPU_AND_NE,  # Neural Engine
            )

            output_path = f"{output_dir}/detector.mlpackage"
            ml_model.save(output_path)

            _logger.info(f"✓ Model converted to Core ML: {output_path}")
            return output_path

        except ImportError as e:
            _logger.error("coremltools not available: %s", e)
            raise
        except Exception as e:
            _logger.error("Failed to convert model: %s", e)
            raise

    def profile_inference(
        self,
        model_path: str,
        test_image: np.ndarray,
        num_runs: int = 100,
    ) -> dict:
        """Profile inference performance on test image.

        Args:
            model_path: Path to model (.pt file)
            test_image: Sample image for profiling
            num_runs: Number of inference runs

        Returns:
            Profiling results (latency, throughput, memory)
        """
        from ultralytics import YOLO
        import statistics

        _logger.info(f"Profiling inference on {num_runs} runs...")

        model = YOLO(model_path)
        model.to(self.get_device())

        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(test_image, verbose=False)
            latencies.append((time.perf_counter() - start) * 1000)  # ms

        results = {
            "mean_latency_ms": statistics.mean(latencies),
            "median_latency_ms": statistics.median(latencies),
            "stdev_latency_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "throughput_fps": 1000 / statistics.mean(latencies),
            "device": self.get_device(),
            "num_runs": num_runs,
        }

        return results

    async def async_inference(
        self,
        model,
        frame: np.ndarray,
        semaphore: asyncio.Semaphore,
    ):
        """Run inference asynchronously to avoid blocking main loop.

        Args:
            model: YOLO model
            frame: Image frame
            semaphore: Limit concurrent inference tasks

        Returns:
            Detection results
        """
        async with semaphore:
            # Run in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: model(frame, verbose=False)
            )
            return results


class MemoryOptimizer:
    """Reduce memory footprint for continuous operation."""

    @staticmethod
    def enable_memory_efficient_inference(model) -> None:
        """Configure model for lower memory usage."""
        try:
            import torch
            
            # Enable gradient checkpointing (reduces activation memory)
            if hasattr(model, 'gradient_checkpointing'):
                model.gradient_checkpointing = True

            # Use mixed precision
            model.half()

            _logger.info("✓ Memory-efficient inference enabled")
        except Exception as e:
            _logger.error("Failed to enable memory optimization: %s", e)

    @staticmethod
    def profile_memory(model, test_frame: np.ndarray) -> dict:
        """Profile memory usage of model inference."""
        try:
            import torch
            import tracemalloc

            tracemalloc.start()
            
            start_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024  # MB
            _ = model(test_frame, verbose=False)
            end_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024  # MB

            tracemalloc.stop()

            return {
                "peak_memory_mb": end_memory,
                "inference_memory_delta_mb": end_memory - start_memory,
            }
        except Exception as e:
            _logger.error("Failed to profile memory: %s", e)
            return {}


def main():
    """CLI interface for Mac optimization."""
    import argparse

    parser = argparse.ArgumentParser(description="Mac Mini optimization tools")
    parser.add_argument("--profile", action="store_true", help="Profile inference performance")
    parser.add_argument("--model", default="models/plate_detector/best.pt")
    parser.add_argument("--convert-mlpackage", action="store_true", help="Convert to Core ML")
    parser.add_argument("--target-fps", type=int, default=30)

    args = parser.parse_args()

    optimizer = MacOptimizer(target_fps=args.target_fps)
    optimizer.configure_torch_for_mac()

    if args.profile:
        import cv2
        test_image = cv2.imread("data/test_image.jpg")
        if test_image is None:
            print("Error: test_image.jpg not found")
            return

        results = optimizer.profile_inference(args.model, test_image)
        print("Profiling Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")

    if args.convert_mlpackage:
        try:
            output = optimizer.convert_model_to_mlpackage(args.model)
            print(f"✓ Converted: {output}")
        except Exception as e:
            print(f"✗ Conversion failed: {e}")


if __name__ == "__main__":
    main()
