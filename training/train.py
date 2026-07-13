"""
Training script for YOLOv8 License Plate Detector.

Trains YOLOv8s model on custom license plate dataset with optimized
hyperparameters for RTX 2000 Ada GPU and Jetson Orin Nano deployment.
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional
import json
import shutil
import time

import torch
from ultralytics import YOLO
import yaml

# Add training directory to path
training_dir = Path(__file__).parent
sys.path.insert(0, str(training_dir))

from config import (
    MODEL_NAME,
    MODEL_PRETRAINED,
    TRAINING_CONFIG,
    AUGMENTATION_CONFIG,
    VALIDATION_CONFIG,
    DATASET_STATS,
    PATHS,
    HARDWARE,
    get_config_summary,
)
from utils import (
    setup_logger,
    GPUMonitor,
    SystemMonitor,
    TrainingTracker,
    log_training_start,
    log_epoch_summary,
    log_training_end,
    save_config_snapshot,
    format_time,
)

logger = setup_logger(
    __name__, log_file=PATHS["training_dir"] + "/train.log"
)


def _parse_results_csv(csv_path: Path) -> Dict:
    """Parse YOLOv8 results.csv and return best metrics."""
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        # Get last row which has best metrics
        if len(df) > 0:
            last_row = df.iloc[-1]
            metrics = {}
            
            # Map column names to readable names
            column_mapping = {
                'epoch': 'Epoch',
                'train/box_loss': 'Train Box Loss',
                'train/cls_loss': 'Train Class Loss',
                'train/dfl_loss': 'Train DFL Loss',
                'val/box_loss': 'Val Box Loss',
                'val/cls_loss': 'Val Class Loss',
                'val/dfl_loss': 'Val DFL Loss',
                'metrics/precision(B)': 'Precision',
                'metrics/recall(B)': 'Recall',
                'metrics/mAP50(B)': 'mAP@0.5',
                'metrics/mAP50-95(B)': 'mAP@0.5:0.95',
                'fitness': 'Fitness'
            }
            
            for col, readable_name in column_mapping.items():
                if col in df.columns:
                    value = last_row[col]
                    if pd.notna(value):
                        metrics[readable_name] = float(value)
            
            return metrics
    except Exception as e:
        logger.debug(f"Could not parse results CSV: {e}")
    
    return {}


def check_environment() -> bool:
    """
    Check if environment is properly set up.

    Returns:
        True if environment is valid
    """
    logger.info("Checking environment...")

    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("❌ CUDA is not available. GPU training not possible.")
        return False

    logger.info(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Check dataset
    data_yaml = Path(PATHS["data_yaml"])
    if not data_yaml.exists():
        logger.error(f"❌ Dataset config not found: {data_yaml}")
        return False

    logger.info(f"✓ Dataset: {data_yaml}")

    # Check model availability
    logger.info(f"✓ Model: YOLOv8s (will download pretrained weights if needed)")

    return True


def prepare_training_dir() -> None:
    """Prepare training output directory."""
    train_dir = Path(PATHS["plate_detector_dir"])
    train_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration snapshot
    config_to_save = {
        "model": MODEL_NAME,
        "training_config": TRAINING_CONFIG,
        "augmentation_config": AUGMENTATION_CONFIG,
        "dataset_stats": DATASET_STATS,
        "hardware": HARDWARE,
    }
    save_config_snapshot(config_to_save, str(train_dir / "config.json"))
    logger.info(f"✓ Configuration saved to {train_dir / 'config.json'}")


def create_yolo_training_args() -> Dict:
    """
    Create YOLO training arguments from config.

    Returns:
        Dictionary of training arguments for YOLO
    """
    args = {
        # Basic settings
        "model": f"{MODEL_NAME}.pt",
        "data": PATHS["data_yaml"],
        "epochs": TRAINING_CONFIG["epochs"],
        "imgsz": TRAINING_CONFIG["imgsz"],
        "batch": TRAINING_CONFIG["batch_size"],
        "device": TRAINING_CONFIG["device"],
        
        # Optimization
        "optimizer": TRAINING_CONFIG["optimizer"],
        "lr0": TRAINING_CONFIG["lr0"],
        "lrf": TRAINING_CONFIG["lrf"],
        "momentum": TRAINING_CONFIG["momentum"],
        "weight_decay": TRAINING_CONFIG["weight_decay"],
        "warmup_epochs": TRAINING_CONFIG["warmup_epochs"],
        "warmup_momentum": TRAINING_CONFIG["warmup_momentum"],
        "warmup_bias_lr": TRAINING_CONFIG["warmup_bias_lr"],
        "cos_lr": TRAINING_CONFIG["cos_lr"],
        
        # Early stopping
        "patience": TRAINING_CONFIG["patience"],
        
        # Mixed precision
        "amp": TRAINING_CONFIG["amp"],
        
        # Data loading
        "workers": TRAINING_CONFIG["workers"],
        "cache": TRAINING_CONFIG["cache"],
        "rect": TRAINING_CONFIG["rect"],
        
        # Augmentation (YOLO uses hsv_h, hsv_s, hsv_v directly)
        "degrees": AUGMENTATION_CONFIG["degrees"],
        "translate": AUGMENTATION_CONFIG["translate"],
        "scale": AUGMENTATION_CONFIG["scale"],
        "flipud": AUGMENTATION_CONFIG["flipud"],
        "fliplr": AUGMENTATION_CONFIG["fliplr"],
        "perspective": AUGMENTATION_CONFIG["perspective"],
        "shear": AUGMENTATION_CONFIG["shear"],
        "hsv_h": AUGMENTATION_CONFIG["hsv_h"],
        "hsv_s": AUGMENTATION_CONFIG["hsv_s"],
        "hsv_v": AUGMENTATION_CONFIG["hsv_v"],
        "mixup": AUGMENTATION_CONFIG["mixup"],
        
        # Validation
        "val": TRAINING_CONFIG["val"],
        "save": TRAINING_CONFIG["save"],
        "save_period": TRAINING_CONFIG["save_period"],
        
        # Output
        "project": PATHS["plate_detector_dir"],
        "name": "train",
        "plots": TRAINING_CONFIG["plots"],
        "verbose": TRAINING_CONFIG["verbose"],
        "seed": TRAINING_CONFIG["seed"],
    }
    
    return args


def train() -> bool:
    """
    Run training pipeline.

    Returns:
        True if training completed successfully
    """
    try:
        # Print configuration
        print(get_config_summary())

        # Check environment
        logger.info("\n" + "=" * 80)
        if not check_environment():
            return False
        logger.info("=" * 80)

        # Prepare directories
        prepare_training_dir()

        # Initialize monitors
        gpu_monitor = GPUMonitor(device=TRAINING_CONFIG["device"])
        system_monitor = SystemMonitor()
        tracker = TrainingTracker(PATHS["plate_detector_dir"])

        # Log training start
        log_training_start(
            logger,
            MODEL_NAME,
            DATASET_STATS["train_images"],
            TRAINING_CONFIG["batch_size"],
            TRAINING_CONFIG["epochs"],
        )

        # Create YOLO model
        logger.info(f"\nLoading model: YOLOv8s...")
        model = YOLO(f"{MODEL_NAME}.pt")  # Auto-downloads pretrained weights

        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  Parameters: {sum(p.numel() for p in model.model.parameters()):,}")

        # Create training arguments
        args = create_yolo_training_args()

        logger.info("\nStarting training...")
        logger.info("-" * 80)
        logger.info("Training will display metrics for each epoch...")
        logger.info("Watch GPU usage and loss metrics below:")
        logger.info("-" * 80)

        # Train model with verbose output
        epoch_times = []
        start_time = time.time()

        # Run training (verbose is already set in args)
        results = model.train(**args)

        total_time = time.time() - start_time

        logger.info("-" * 80)

        # Find best model checkpoint
        train_dir = Path(PATHS["plate_detector_dir"]) / "train"
        best_pt = train_dir / "weights" / "best.pt"
        last_pt = train_dir / "weights" / "last.pt"
        results_csv = train_dir / "results.csv"

        if best_pt.exists():
            # Copy best model to standard location
            shutil.copy(str(best_pt), PATHS["best_model"])
            logger.info(f"✓ Best model: {PATHS['best_model']}")

        if last_pt.exists():
            # Copy last model to standard location
            shutil.copy(str(last_pt), PATHS["last_model"])
            logger.info(f"✓ Last model: {PATHS['last_model']}")

        # Display final metrics
        logger.info("\n" + "=" * 80)
        logger.info("FINAL TRAINING METRICS")
        logger.info("=" * 80)

        if results_csv.exists():
            final_metrics = _parse_results_csv(results_csv)
            if final_metrics:
                logger.info("\nBest Epoch Metrics:")
                logger.info("-" * 80)
                for key, value in sorted(final_metrics.items()):
                    if isinstance(value, float):
                        if "loss" in key.lower():
                            logger.info(f"  {key:.<40} {value:.6f}")
                        else:
                            logger.info(f"  {key:.<40} {value:.4f}")
                    else:
                        logger.info(f"  {key:.<40} {value}")
                logger.info("-" * 80)

        logger.info(f"\n✓ Training completed successfully!")
        logger.info(f"  Total time: {format_time(total_time)}")
        logger.info(f"  Output: {PATHS['plate_detector_dir']}")

        return True

    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 License Plate Detector"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )

    args = parser.parse_args()

    # Override config if specified
    if args.epochs:
        TRAINING_CONFIG["epochs"] = args.epochs
    if args.batch_size:
        TRAINING_CONFIG["batch_size"] = args.batch_size

    success = train()
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
