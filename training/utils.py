"""
Utility functions for YOLOv8 training pipeline.

Includes logging, GPU monitoring, result tracking, and report generation.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import time

import psutil
import torch
import numpy as np
from pathlib import Path

# Configure logging
def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with file and console handlers.

    Args:
        name: Logger name
        log_file: Optional file to log to

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    simple_formatter = logging.Formatter("%(levelname)s - %(message)s")

    # Console handler (INFO level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    # File handler (DEBUG level, if log file provided)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

    return logger


class GPUMonitor:
    """Monitor GPU memory and utilization during training."""

    def __init__(self, device: int = 0):
        """
        Initialize GPU monitor.

        Args:
            device: GPU device ID
        """
        self.device = device
        self.metrics = []

    def get_status(self) -> Dict[str, float]:
        """
        Get current GPU status.

        Returns:
            Dict with gpu_util and gpu_mem
        """
        if not torch.cuda.is_available():
            return {"gpu_util": 0.0, "gpu_mem": 0.0}

        try:
            # Get GPU memory
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3

            gpu_mem = (allocated / total) * 100 if total > 0 else 0

            return {
                "gpu_util": 0.0,  # Would need nvidia-ml-py for actual util
                "gpu_mem": gpu_mem,
                "gpu_mem_gb": allocated,
                "gpu_total_gb": total,
            }
        except Exception as e:
            logging.warning(f"Could not get GPU status: {e}")
            return {"gpu_util": 0.0, "gpu_mem": 0.0}

    def log_status(self, epoch: int, logger: logging.Logger) -> None:
        """Log current GPU status."""
        status = self.get_status()
        logger.debug(
            f"Epoch {epoch}: GPU Memory: {status['gpu_mem']:.1f}% "
            f"({status.get('gpu_mem_gb', 0):.1f}GB / {status.get('gpu_total_gb', 0):.1f}GB)"
        )


class SystemMonitor:
    """Monitor system resources during training."""

    def __init__(self):
        """Initialize system monitor."""
        self.start_time = time.time()
        self.metrics = []

    def get_status(self) -> Dict[str, Any]:
        """
        Get current system status.

        Returns:
            Dict with cpu, memory, and training stats
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            elapsed = time.time() - self.start_time

            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "elapsed_time": elapsed,
            }
        except Exception as e:
            logging.warning(f"Could not get system status: {e}")
            return {}


class TrainingTracker:
    """Track training metrics and results."""

    def __init__(self, save_dir: str):
        """
        Initialize training tracker.

        Args:
            save_dir: Directory to save tracking files
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = []
        self.best_metrics = {}

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Log metrics for an epoch.

        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
        """
        record = {"epoch": epoch, "timestamp": datetime.now().isoformat(), **metrics}
        self.metrics.append(record)

    def update_best(self, metrics: Dict[str, float]) -> bool:
        """
        Update best metrics if current is better.

        Args:
            metrics: Current metrics

        Returns:
            True if metrics improved
        """
        improved = False

        # Check if mAP@0.5 improved
        if "mAP@0.5" in metrics:
            if "mAP@0.5" not in self.best_metrics or metrics["mAP@0.5"] > self.best_metrics["mAP@0.5"]:
                self.best_metrics.update(metrics)
                improved = True

        return improved

    def save_metrics(self, filename: str = "metrics.json") -> None:
        """
        Save metrics to JSON file.

        Args:
            filename: Output filename
        """
        output_file = self.save_dir / filename
        with open(output_file, "w") as f:
            json.dump(self.metrics, f, indent=2)

    def get_best_epoch(self, metric: str = "mAP@0.5") -> Optional[int]:
        """
        Get epoch with best metric.

        Args:
            metric: Metric name to check

        Returns:
            Best epoch number or None
        """
        if not self.metrics:
            return None

        best_epoch = 0
        best_value = -1

        for record in self.metrics:
            if metric in record and record[metric] > best_value:
                best_value = record[metric]
                best_epoch = record["epoch"]

        return best_epoch if best_value >= 0 else None


def format_time(seconds: float) -> str:
    """
    Format seconds to readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (HH:MM:SS)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def estimate_remaining_time(
    epochs_completed: int, total_epochs: int, time_per_epoch: float
) -> str:
    """
    Estimate remaining training time.

    Args:
        epochs_completed: Number of completed epochs
        total_epochs: Total epochs for training
        time_per_epoch: Average time per epoch in seconds

    Returns:
        Formatted estimated time remaining
    """
    if epochs_completed == 0:
        return "Estimating..."

    remaining_epochs = total_epochs - epochs_completed
    remaining_seconds = remaining_epochs * time_per_epoch
    return format_time(remaining_seconds)


def log_training_start(
    logger: logging.Logger,
    model_name: str,
    dataset_size: int,
    batch_size: int,
    epochs: int,
) -> None:
    """
    Log training start information.

    Args:
        logger: Logger instance
        model_name: Model name
        dataset_size: Training dataset size
        batch_size: Batch size
        epochs: Number of epochs
    """
    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Dataset size: {dataset_size:,} images")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Steps per epoch: {dataset_size // batch_size}")
    logger.info(f"Total epochs: {epochs}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    logger.info("=" * 80)


def log_epoch_summary(
    logger: logging.Logger,
    epoch: int,
    total_epochs: int,
    metrics: Dict[str, float],
    time_elapsed: float,
) -> None:
    """
    Log summary for an epoch.

    Args:
        logger: Logger instance
        epoch: Current epoch
        total_epochs: Total epochs
        metrics: Epoch metrics
        time_elapsed: Time elapsed for epoch
    """
    logger.info(f"\nEpoch {epoch + 1}/{total_epochs} - {format_time(time_elapsed)}")

    # Log key metrics
    if "train_loss" in metrics:
        logger.info(f"  Train Loss: {metrics['train_loss']:.4f}")
    if "val_loss" in metrics:
        logger.info(f"  Val Loss: {metrics['val_loss']:.4f}")
    if "mAP@0.5" in metrics:
        logger.info(f"  mAP@0.5: {metrics['mAP@0.5']:.4f}")
    if "precision" in metrics:
        logger.info(f"  Precision: {metrics['precision']:.4f}")
    if "recall" in metrics:
        logger.info(f"  Recall: {metrics['recall']:.4f}")


def log_training_end(
    logger: logging.Logger, best_epoch: int, best_metrics: Dict[str, float], total_time: float
) -> None:
    """
    Log training completion information.

    Args:
        logger: Logger instance
        best_epoch: Best epoch number
        best_metrics: Best metrics achieved
        total_time: Total training time in seconds
    """
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Total time: {format_time(total_time)}")

    if best_metrics:
        logger.info("\nBest metrics:")
        for metric, value in best_metrics.items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")

    logger.info("=" * 80)


def save_config_snapshot(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration snapshot for reproducibility.

    Args:
        config: Configuration dictionary
        save_path: Path to save config
    """
    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        json.dump(config, f, indent=2, default=str)


def load_results_csv(csv_path: str) -> List[Dict[str, Any]]:
    """
    Load training results from CSV.

    Args:
        csv_path: Path to results.csv from YOLOv8

    Returns:
        List of result dictionaries
    """
    results = []

    try:
        with open(csv_path, "r") as f:
            lines = f.readlines()

            if len(lines) < 2:
                return results

            # Parse header
            header = [col.strip() for col in lines[0].split(",")]

            # Parse data rows
            for line in lines[1:]:
                values = [val.strip() for val in line.split(",")]
                if len(values) == len(header):
                    result = {header[i]: values[i] for i in range(len(header))}
                    results.append(result)

    except Exception as e:
        logging.warning(f"Could not load results CSV: {e}")

    return results


if __name__ == "__main__":
    # Test utilities
    logger = setup_logger("test", log_file="/tmp/test.log")
    logger.info("Logger working")

    gpu_monitor = GPUMonitor()
    print("GPU Status:", gpu_monitor.get_status())

    tracker = TrainingTracker("/tmp/tracker")
    tracker.log_epoch(1, {"loss": 0.5, "mAP@0.5": 0.6})
    print("Remaining time:", estimate_remaining_time(1, 100, 60))
