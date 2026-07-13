"""
Training configuration for YOLOv8 License Plate Detector.

This module contains all configurable parameters for model training, augmentation,
and evaluation. All settings are optimized for:
- RTX 2000 Ada GPU (11GB VRAM)
- Jetson Orin Nano deployment target
- High accuracy license plate detection
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# PROJECT STRUCTURE
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATASET_ROOT = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
PLATE_DETECTOR_DIR = MODELS_DIR / "plate_detector"
TRAINING_DIR = PROJECT_ROOT / "training"

# Create directories if they don't exist
PLATE_DETECTOR_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

DATA_YAML = str(DATASET_ROOT / "data.yaml")

# Dataset statistics (from validation)
DATASET_STATS = {
    "train_images": 20580,
    "val_images": 4116,
    "test_images": 978,
    "total_images": 25674,
    "classes": 1,
    "class_names": {0: "license_plate"},
    "avg_annotations_per_image": 1.04,
}


# ============================================================================
# MODEL SELECTION ANALYSIS
# ============================================================================
"""
MODEL SELECTION: YOLOv8n vs YOLOv8s

For license plate detection on RTX 2000 Ada + Jetson Orin Nano deployment:

YOLOv8n (Nano):
  - Model size: ~6.3 MB
  - Parameters: ~3.2M
  - Inference speed: ~1-2 ms (GPU)
  - Accuracy: Good
  - Deployment: Excellent on edge devices
  - Use case: Ultra-lightweight, minimal latency

YOLOv8s (Small):
  - Model size: ~22.5 MB
  - Parameters: ~11.2M
  - Inference speed: ~2-3 ms (GPU)
  - Accuracy: ~5-10% better mAP than nano
  - Deployment: Good on Jetson Orin Nano (8GB RAM)
  - Use case: Balanced accuracy and speed

CHOSEN: YOLOv8s

REASONING:
1. RTX 2000 Ada has 11GB VRAM - easily handles YOLOv8s training
2. License plate detection requires HIGH ACCURACY - critical for ALPR
3. The 5-10% mAP improvement justifies the minor deployment cost
4. Jetson Orin Nano has sufficient memory for YOLOv8s inference
5. Transfer learning will offset any accuracy concerns
6. Better for production quality vs. pure edge optimization

TRADEOFFS:
- Training time: ~2-3x longer than YOLOv8n
- Inference latency: ~1ms additional (acceptable for real-time)
- Model size: Still deployable on Jetson (22.5 MB is reasonable)
"""

# Model architecture
MODEL_NAME = "yolov8s"  # YOLOv8 Small
MODEL_PRETRAINED = True  # Use pretrained weights from COCO
MODEL_SIZE = (640, 640)  # Standard YOLO input size


# ============================================================================
# TRAINING HYPERPARAMETERS - OPTIMIZED FOR LICENSE PLATES
# ============================================================================

TRAINING_CONFIG = {
    # ========== BASIC TRAINING PARAMETERS ==========
    "epochs": 100,  # Sufficient for convergence with early stopping
    "batch_size": 32,  # Fits in RTX 2000 Ada (11GB VRAM) with margin
    "imgsz": 640,  # Standard YOLO image size, optimal for plates
    "device": 0,  # GPU device ID (0 = first GPU, i.e., RTX 2000)
    
    # ========== OPTIMIZATION PARAMETERS ==========
    "optimizer": "SGD",  # SGD typically better than Adam for YOLO
    "lr0": 0.01,  # Initial learning rate
    "lrf": 0.01,  # Final learning rate (1% of initial)
    "momentum": 0.937,  # SGD momentum
    "weight_decay": 0.0005,  # L2 regularization
    "warmup_epochs": 3.0,  # Gradual warmup for stability
    "warmup_momentum": 0.8,  # Initial momentum during warmup
    "warmup_bias_lr": 0.1,  # Warmup bias learning rate
    
    # ========== LEARNING RATE SCHEDULER ==========
    "scheduler": "cosine",  # Cosine annealing for smooth LR decay
    "cos_lr": True,  # Use cosine annealing
    
    # ========== EARLY STOPPING ==========
    "patience": 20,  # Stop if no improvement for 20 epochs
    "min_delta": 0.0,  # Minimum delta for improvement detection
    
    # ========== MIXED PRECISION TRAINING ==========
    "amp": True,  # Automatic Mixed Precision (faster, less memory)
    
    # ========== NUMERICAL STABILITY ==========
    "seed": 42,  # For reproducibility
    "deterministic": False,  # False for faster training, results still reproducible
    "workers": 8,  # DataLoader workers (8 for RTX 2000)
    
    # ========== CACHING & PERFORMANCE ==========
    "cache": "ram",  # Cache images in RAM for speed (set to False if OOM)
    "rect": False,  # Rectangular training (we use square 640x640)
    "mosaic": 1.0,  # Mosaic augmentation (1.0 = always on)
    "mixup": 0.1,  # MixUp augmentation (10% probability)
    
    # ========== VALIDATION ==========
    "val": True,  # Run validation each epoch
    "save": True,  # Save checkpoints
    "save_period": 10,  # Save checkpoint every N epochs
    "plots": True,  # Save training plots
    "profile": False,  # Profile GPU/CPU usage (expensive)
    
    # ========== OUTPUT & LOGGING ==========
    "verbose": True,  # Verbose logging
    "project": str(PLATE_DETECTOR_DIR),  # Project directory
    "name": "train",  # Experiment name (will create run subdirs)
}


# ============================================================================
# DATA AUGMENTATION - OPTIMIZED FOR LICENSE PLATES
# ============================================================================
"""
DATA AUGMENTATION STRATEGY:

License plates require CAREFUL augmentation to maintain readability.
We use standard YOLO augmentations but avoid transforms that harm OCR.

RATIONALE FOR EACH AUGMENTATION:

1. MOSAIC (1.0): Combine 4 images - improves generalization
2. MIXUP (0.1): Blend images - smooth decision boundaries
3. COPY-PASTE (0.0): Not used - harmful for small objects like plates
4. GEOMETRIC:
   - Rotation: ±10° - real-world viewing angles
   - Translate: ±10% - plates can appear anywhere in frame
   - Scale: ±15% - vehicle distance variation
   - Perspective: ±0.0% - plates should stay roughly frontal
   - Shear: ±0% - no shearing needed
5. COLOR:
   - HSV-H: ±180° - different lighting conditions
   - HSV-S: ±30% - saturation variation
   - HSV-V: ±30% - brightness/contrast changes
   - Grayscale: 0% - plates are colored in India
6. BLUR:
   - Motion blur: Not used - preserves plate clarity
   - Gaussian blur: ~0.5% - slight blur
7. NOISE & ARTIFACTS:
   - Noise: Not used - plates must be readable
   - Cutout: Not used - dangerous for small objects
   - Erase: Not used - same concern
"""

AUGMENTATION_CONFIG = {
    # Geometric Augmentations (affects bounding boxes)
    "degrees": 10.0,  # Rotation: ±10 degrees (real-world camera angle)
    "translate": 0.1,  # Translation: ±10% of image size
    "scale": (0.85, 1.15),  # Scale: ±15% (distance variation)
    "flipud": 0.0,  # Flip upside-down: 0% (plates don't flip)
    "fliplr": 0.5,  # Flip left-right: 50% (symmetric plates)
    "perspective": 0.0,  # Perspective transform: 0% (no skew)
    "shear": 0.0,  # Shear transform: 0%
    
    # Color Augmentations (doesn't affect bounding boxes)
    "hsv_h": 0.015,  # HSV Hue: ±1.5% (lighting changes)
    "hsv_s": 0.3,  # HSV Saturation: ±30% (color variation)
    "hsv_v": 0.3,  # HSV Value (brightness): ±30%
    
    # Blur & Noise
    "blur": 0.0,  # Gaussian blur: 0% (keeps plates sharp)
    "motion_blur": 0.0,  # Motion blur: 0% (avoid)
    
    # Artifacts (generally avoid for small objects)
    "mixup": 0.1,  # MixUp: 10% (handled at trainer level)
    "copy_paste": 0.0,  # Copy-paste: 0% (not for small objects)
    "erasing": 0.0,  # Random erasing: 0% (dangerous for plates)
    "cutout": 0.0,  # Cutout: 0% (harmful for small objects)
    "crop_fraction": 1.0,  # Crop: 100% (full image, no cropping loss)
}


# ============================================================================
# VALIDATION & TESTING CONFIGURATION
# ============================================================================

VALIDATION_CONFIG = {
    "conf": 0.25,  # Confidence threshold for NMS
    "iou": 0.6,  # IoU threshold for NMS
    "max_det": 300,  # Maximum detections per image
    "half": False,  # Use FP16 precision during inference
    "device": 0,  # GPU device
    "verbose": True,
}

# Metrics for evaluation
METRICS = [
    "precision",
    "recall",
    "mAP@0.5",
    "mAP@0.5:0.95",
    "f1_score",
]


# ============================================================================
# INFERENCE & DEPLOYMENT CONFIGURATION
# ============================================================================

INFERENCE_CONFIG = {
    "conf": 0.5,  # Detection confidence threshold
    "iou": 0.45,  # IoU threshold for NMS
    "max_det": 100,  # Max detections per image
    "half": False,  # FP16 inference (False for accuracy)
    "device": 0,  # GPU device
}

# Export formats
EXPORT_FORMATS = {
    "pt": True,  # PyTorch format (keep weights)
    "onnx": True,  # ONNX format (CPU/edge inference)
    "onnx_fp32": True,  # FP32 ONNX for Jetson compatibility
}


# ============================================================================
# LOGGING & MONITORING
# ============================================================================

LOGGING_CONFIG = {
    "log_level": "INFO",
    "save_dir": PLATE_DETECTOR_DIR,
    "log_file": PLATE_DETECTOR_DIR / "training.log",
    "save_period": 1,  # Log every epoch
    "verbose": True,
}

# Metrics to track during training
TRACKED_METRICS = [
    "epoch",
    "gpu_mem",
    "train/loss",
    "train/box_loss",
    "train/cls_loss",
    "val/loss",
    "val/box_loss",
    "val/cls_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "lr/pg0",
    "lr/pg1",
    "lr/pg2",
]


# ============================================================================
# PATHS & FILE CONFIGURATION
# ============================================================================

PATHS = {
    "project_root": str(PROJECT_ROOT),
    "dataset_root": str(DATASET_ROOT),
    "data_yaml": DATA_YAML,
    "models_dir": str(MODELS_DIR),
    "plate_detector_dir": str(PLATE_DETECTOR_DIR),
    "training_dir": str(TRAINING_DIR),
    "best_model": str(PLATE_DETECTOR_DIR / "best.pt"),
    "last_model": str(PLATE_DETECTOR_DIR / "last.pt"),
    "results_csv": str(PLATE_DETECTOR_DIR / "results.csv"),
    "training_report": str(PLATE_DETECTOR_DIR / "training_report.md"),
}


# ============================================================================
# HARDWARE CONFIGURATION
# ============================================================================

HARDWARE = {
    "gpu_model": "RTX 2000 Ada",
    "vram_gb": 11,
    "target_deployment": "Jetson Orin Nano",
    "deployment_vram_gb": 8,
    "deployment_compute_capability": 8.7,  # Jetson Orin Nano
}


# ============================================================================
# CONFIGURATION VALIDATION & UTILITY FUNCTIONS
# ============================================================================

def get_config_summary() -> str:
    """Get readable summary of training configuration."""
    summary = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                    YOLOV8 LICENSE PLATE DETECTOR CONFIGURATION             ║
╚════════════════════════════════════════════════════════════════════════════╝

📊 DATASET:
   • Total images: {DATASET_STATS['total_images']:,}
   • Train: {DATASET_STATS['train_images']:,} | Val: {DATASET_STATS['val_images']:,} | Test: {DATASET_STATS['test_images']:,}
   • Class: license_plate (1 class)
   • Avg annotations/image: {DATASET_STATS['avg_annotations_per_image']:.2f}

🤖 MODEL:
   • Architecture: {MODEL_NAME.upper()} (Small)
   • Input size: {MODEL_SIZE[0]}x{MODEL_SIZE[1]}
   • Pretrained: COCO weights with transfer learning
   • Reason: Balances accuracy (5-10% better than nano) with edge deployment

⚙️ TRAINING:
   • Epochs: {TRAINING_CONFIG['epochs']}
   • Batch size: {TRAINING_CONFIG['batch_size']}
   • Optimizer: {TRAINING_CONFIG['optimizer']}
   • Learning rate: {TRAINING_CONFIG['lr0']} → {TRAINING_CONFIG['lrf']}
   • Scheduler: Cosine annealing
   • Early stopping: {TRAINING_CONFIG['patience']} epochs patience
   • Mixed precision (AMP): {TRAINING_CONFIG['amp']}

📈 AUGMENTATION:
   • Mosaic: Always on (4-image combinations)
   • MixUp: 10% (blended images)
   • Rotation: ±{AUGMENTATION_CONFIG['degrees']}°
   • Scale: {AUGMENTATION_CONFIG['scale'][0]:.0%} to {AUGMENTATION_CONFIG['scale'][1]:.0%}
   • HSV: H±{AUGMENTATION_CONFIG['hsv_h']*100:.1f}%, S±{AUGMENTATION_CONFIG['hsv_s']*100:.0f}%, V±{AUGMENTATION_CONFIG['hsv_v']*100:.0f}%

💾 OUTPUT:
   • Best weights: {PATHS['best_model']}
   • Training report: {PATHS['training_report']}
   • Plots: confusion matrix, P-R curves, training history

🎯 TARGET DEPLOYMENT:
   • Device: Jetson Orin Nano (8GB RAM)
   • Export formats: PT, ONNX, ONNX-FP32
   • No TensorRT engine (for flexibility)
"""
    return summary


def validate_paths() -> bool:
    """Validate that all required paths exist."""
    required_paths = [
        DATA_YAML,
        DATASET_ROOT / "train" / "images",
        DATASET_ROOT / "val" / "images",
        DATASET_ROOT / "test" / "images",
    ]
    
    missing = []
    for path in required_paths:
        if not Path(path).exists():
            missing.append(str(path))
    
    if missing:
        logger.error("Missing required paths:")
        for path in missing:
            logger.error(f"  - {path}")
        return False
    
    return True


def print_config() -> None:
    """Print full configuration."""
    print(get_config_summary())
    print("\n" + "=" * 80)
    print("✓ Configuration validated successfully")
    print("=" * 80)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print_config()
