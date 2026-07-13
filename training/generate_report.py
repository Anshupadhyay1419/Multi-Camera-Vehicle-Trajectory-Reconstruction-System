"""
Generate comprehensive training report for YOLOv8 License Plate Detector.

Creates markdown report with training results, metrics, visualizations, and recommendations.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Optional, List, Any
import json
from datetime import datetime
import csv

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add training directory to path
training_dir = Path(__file__).parent
sys.path.insert(0, str(training_dir))

from config import DATASET_STATS, TRAINING_CONFIG, AUGMENTATION_CONFIG, HARDWARE, PATHS
from utils import setup_logger

logger = setup_logger(__name__, log_file=str(Path(PATHS["plate_detector_dir"]) / "report.log"))


class TrainingReportGenerator:
    """Generate comprehensive training report."""

    def __init__(self, training_dir: str, model_name: str = "YOLOv8s"):
        """
        Initialize report generator.

        Args:
            training_dir: Directory containing training results
            model_name: Model name for report
        """
        self.training_dir = Path(training_dir)
        self.model_name = model_name
        self.report_path = self.training_dir / "training_report.md"

        self.metrics = {}
        self.config = {}

    def load_results(self) -> bool:
        """
        Load training results from directory.

        Returns:
            True if results loaded successfully
        """
        logger.info(f"Loading results from {self.training_dir}")

        # Load config
        config_file = self.training_dir / "config.json"
        if config_file.exists():
            with open(config_file, "r") as f:
                self.config = json.load(f)
                logger.info("✓ Config loaded")

        # Load evaluation report
        eval_file = self.training_dir / "evaluation_report.json"
        if eval_file.exists():
            with open(eval_file, "r") as f:
                eval_results = json.load(f)
                if "metrics" in eval_results:
                    self.metrics.update(eval_results["metrics"])
                logger.info("✓ Evaluation results loaded")

        return True

    def generate_report(self) -> str:
        """
        Generate comprehensive markdown report.

        Returns:
            Markdown report content
        """
        report_lines = []

        # Title
        report_lines.append("# YOLOv8 License Plate Detector - Training Report")
        report_lines.append("")

        # Metadata
        report_lines.append("## Report Information")
        report_lines.append(f"- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"- **Model**: {self.model_name}")
        report_lines.append(f"- **Training Directory**: {self.training_dir}")
        report_lines.append("")

        # Executive Summary
        report_lines.extend(self._generate_executive_summary())

        # Dataset Information
        report_lines.extend(self._generate_dataset_section())

        # Model Architecture
        report_lines.extend(self._generate_model_section())

        # Training Configuration
        report_lines.extend(self._generate_training_config_section())

        # Augmentation Configuration
        report_lines.extend(self._generate_augmentation_section())

        # Training Results
        report_lines.extend(self._generate_results_section())

        # Hardware & Performance
        report_lines.extend(self._generate_hardware_section())

        # Recommendations
        report_lines.extend(self._generate_recommendations())

        # Appendix
        report_lines.extend(self._generate_appendix())

        return "\n".join(report_lines)

    def _generate_executive_summary(self) -> List[str]:
        """Generate executive summary section."""
        lines = [
            "## Executive Summary",
            "",
            "This report documents the training of a YOLOv8s-based license plate detector",
            "optimized for Indian license plates. The model is trained on a custom dataset",
            "and optimized for deployment on NVIDIA Jetson Orin Nano edge devices.",
            "",
        ]

        if self.metrics:
            lines.append("### Key Performance Metrics")
            lines.append("")

            metrics_to_show = {
                "precision": ("Precision", "{:.4f}"),
                "recall": ("Recall", "{:.4f}"),
                "mAP50": ("mAP@0.5", "{:.4f}"),
                "mAP": ("mAP@0.5:0.95", "{:.4f}"),
            }

            for key, (display_name, fmt) in metrics_to_show.items():
                if key in self.metrics:
                    value = self.metrics[key]
                    lines.append(f"- **{display_name}**: {fmt.format(float(value))}")

            lines.append("")

        return lines

    def _generate_dataset_section(self) -> List[str]:
        """Generate dataset information section."""
        lines = [
            "## Dataset Information",
            "",
            "### Dataset Structure",
            f"- **Total Images**: {DATASET_STATS['total_images']:,}",
            f"- **Training Images**: {DATASET_STATS['train_images']:,}",
            f"- **Validation Images**: {DATASET_STATS['val_images']:,}",
            f"- **Test Images**: {DATASET_STATS['test_images']:,}",
            "",
            "### Dataset Statistics",
            f"- **Classes**: {DATASET_STATS['classes']}",
            f"- **Class Name**: {', '.join(DATASET_STATS['class_names'].values())}",
            f"- **Average Annotations/Image**: {DATASET_STATS['avg_annotations_per_image']:.2f}",
            f"- **Image Format**: YOLO (.txt labels)",
            "",
        ]

        return lines

    def _generate_model_section(self) -> List[str]:
        """Generate model architecture section."""
        lines = [
            "## Model Architecture",
            "",
            f"- **Base Model**: {self.model_name}",
            "- **Framework**: PyTorch with Ultralytics YOLOv8",
            "- **Input Size**: 640x640 pixels",
            "- **Output**: Bounding boxes with class confidence",
            "",
            "### Model Selection Rationale",
            "",
            "**YOLOv8s (Small) was selected over YOLOv8n (Nano) based on:**",
            "",
            "1. **Accuracy Priority**: License plate detection requires high accuracy",
            "   - YOLOv8s provides 5-10% better mAP than nano",
            "   - Critical for downstream OCR reliability",
            "",
            "2. **GPU Training Capacity**: RTX 2000 Ada (11GB VRAM)",
            "   - Easily handles batch size 32 with YOLOv8s",
            "   - Mixed precision training further reduces memory",
            "",
            "3. **Edge Deployment Feasibility**: Jetson Orin Nano (8GB RAM)",
            "   - Model size: 22.5 MB (well within limits)",
            "   - Inference: 10-15ms per image (acceptable)",
            "   - Throughput: 65-100 images/sec",
            "",
            "4. **Transfer Learning**: COCO pretrained weights",
            "   - Faster convergence",
            "   - Better generalization to new scenarios",
            "",
        ]

        return lines

    def _generate_training_config_section(self) -> List[str]:
        """Generate training configuration section."""
        lines = [
            "## Training Configuration",
            "",
            "### Optimization Parameters",
            f"- **Epochs**: {TRAINING_CONFIG['epochs']}",
            f"- **Batch Size**: {TRAINING_CONFIG['batch_size']}",
            f"- **Image Size**: {TRAINING_CONFIG['imgsz']}x{TRAINING_CONFIG['imgsz']}",
            f"- **Optimizer**: {TRAINING_CONFIG['optimizer']}",
            f"- **Learning Rate**: {TRAINING_CONFIG['lr0']} → {TRAINING_CONFIG['lrf']}",
            f"- **Momentum**: {TRAINING_CONFIG['momentum']}",
            f"- **Weight Decay**: {TRAINING_CONFIG['weight_decay']}",
            "",
            "### Learning Rate Schedule",
            f"- **Scheduler**: Cosine Annealing",
            f"- **Warmup Epochs**: {TRAINING_CONFIG['warmup_epochs']}",
            f"- **Warmup Momentum**: {TRAINING_CONFIG['warmup_momentum']}",
            "",
            "### Training Parameters",
            f"- **Early Stopping Patience**: {TRAINING_CONFIG['patience']} epochs",
            f"- **Mixed Precision (AMP)**: {TRAINING_CONFIG['amp']}",
            f"- **Cache Mode**: {TRAINING_CONFIG['cache']}",
            f"- **Data Workers**: {TRAINING_CONFIG['workers']}",
            "",
        ]

        return lines

    def _generate_augmentation_section(self) -> List[str]:
        """Generate augmentation configuration section."""
        lines = [
            "## Data Augmentation Strategy",
            "",
            "### Rationale",
            "Augmentations are carefully selected to enhance generalization while",
            "maintaining license plate readability (critical for downstream OCR).",
            "",
            "### Augmentation Techniques",
            "",
            "#### Geometric Transforms",
            f"- **Rotation**: ±{AUGMENTATION_CONFIG['degrees']:.1f}° (realistic viewing angles)",
            f"- **Translation**: ±{AUGMENTATION_CONFIG['translate']*100:.0f}% (positioning variation)",
            f"- **Scale**: {AUGMENTATION_CONFIG['scale'][0]:.0%} - {AUGMENTATION_CONFIG['scale'][1]:.0%} (distance changes)",
            f"- **Flip LR**: {AUGMENTATION_CONFIG['fliplr']*100:.0f}% (symmetry in plates)",
            f"- **Perspective**: {AUGMENTATION_CONFIG['perspective']*100:.0f}% (frontal plates don't skew)",
            "",
            "#### Color Space",
            f"- **HSV Hue**: ±{AUGMENTATION_CONFIG['hsv_h']*100:.1f}% (lighting conditions)",
            f"- **HSV Saturation**: ±{AUGMENTATION_CONFIG['hsv_s']*100:.0f}% (color variation)",
            f"- **HSV Value**: ±{AUGMENTATION_CONFIG['hsv_v']*100:.0f}% (brightness/contrast)",
            "",
            "#### Image Composition",
            f"- **Mosaic**: Always enabled (4-image combinations for diversity)",
            f"- **MixUp**: {AUGMENTATION_CONFIG['mixup']*100:.0f}% (blended images)",
            "",
            "#### Avoided Augmentations",
            "- **Motion Blur**: Keeps license plates sharp",
            "- **Gaussian Blur**: Maintains text clarity",
            "- **Cutout/Erasing**: Dangerous for small objects",
            "- **Copy-Paste**: Not suitable for single-class detection",
            "",
        ]

        return lines

    def _generate_results_section(self) -> List[str]:
        """Generate training results section."""
        lines = [
            "## Training Results",
            "",
        ]

        if self.metrics:
            lines.append("### Final Metrics")
            lines.append("")

            metric_categories = {
                "Detection Performance": ["precision", "recall", "mAP50", "mAP"],
                "Model Fitness": ["fitness"],
            }

            for category, keys in metric_categories.items():
                category_metrics = {k: v for k, v in self.metrics.items() if k in keys and v is not None}
                if category_metrics:
                    lines.append(f"#### {category}")
                    lines.append("")
                    for key, value in category_metrics.items():
                        if isinstance(value, (int, float)):
                            lines.append(f"- **{key.replace('_', ' ').title()}**: {float(value):.4f}")
                    lines.append("")

        # Visualizations
        lines.append("### Training Visualizations")
        lines.append("")

        visualization_files = [
            ("results.png", "Training Results Plot"),
            ("confusion_matrix.png", "Confusion Matrix"),
            ("confusion_matrix_normalized.png", "Normalized Confusion Matrix"),
            ("P_curve.png", "Precision Curve"),
            ("R_curve.png", "Recall Curve"),
            ("F1_curve.png", "F1 Score Curve"),
            ("PR_curve.png", "Precision-Recall Curve"),
            ("labels.jpg", "Dataset Labels Distribution"),
        ]

        train_dir = Path(PATHS["plate_detector_dir"]) / "train"
        for filename, description in visualization_files:
            if (train_dir / "weights" / filename).exists():
                rel_path = f"train/weights/{filename}"
                lines.append(f"#### {description}")
                lines.append(f"![{description}]({rel_path})")
                lines.append("")

        return lines

    def _generate_hardware_section(self) -> List[str]:
        """Generate hardware and performance section."""
        lines = [
            "## Hardware & Performance",
            "",
            "### Training Environment",
            f"- **GPU**: {HARDWARE['gpu_model']}",
            f"- **GPU VRAM**: {HARDWARE['vram_gb']} GB",
            f"- **Batch Size**: {TRAINING_CONFIG['batch_size']}",
            f"- **Mixed Precision**: Enabled",
            "",
            "### Deployment Target",
            f"- **Device**: {HARDWARE['target_deployment']}",
            f"- **RAM**: {HARDWARE['deployment_vram_gb']} GB",
            f"- **Compute Capability**: {HARDWARE['deployment_compute_capability']}",
            "",
            "### Expected Performance on Jetson Orin Nano",
            "- **Inference Latency**: 10-15 ms per image",
            "- **Throughput**: 65-100 images/sec",
            "- **Memory Usage**: 500-800 MB",
            "- **Format**: ONNX with ONNX Runtime (recommended)",
            "",
        ]

        return lines

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations section."""
        lines = [
            "## Recommendations",
            "",
            "### For Production Deployment",
            "",
            "1. **Model Export**",
            "   - Export to ONNX format for Jetson Orin Nano deployment",
            "   - Use FP32 precision for maximum accuracy",
            "   - Run `export.py` with `--format all --include-guide` flags",
            "",
            "2. **Performance Optimization**",
            "   - Compile ONNX model with TensorRT for additional speed (optional)",
            "   - Use batch processing for higher throughput",
            "   - Set Jetson to max performance mode: `sudo jetson_clocks`",
            "",
            "3. **Integration with OCR**",
            "   - Ensure detected plates are at least 50x150 pixels for OCR",
            "   - Apply NMS with iou_threshold=0.45 to remove duplicates",
            "   - Use PaddleOCR or TrOCR for character recognition",
            "",
            "### For Model Improvement",
            "",
            "1. **Data Augmentation**",
            "   - Collect more challenging scenarios (nighttime, rain, extreme angles)",
            "   - Add hard negative samples (non-plates)",
            "   - Increase rotation diversity",
            "",
            "2. **Hyperparameter Tuning**",
            "   - If underfitting: Increase epochs, reduce early stopping patience",
            "   - If overfitting: Add more augmentation, increase weight decay",
            "   - Experiment with learning rate schedules",
            "",
            "3. **Model Architecture**",
            "   - Consider YOLOv8m for higher accuracy (requires more resources)",
            "   - Implement multi-scale detection for varying plate sizes",
            "   - Add attention mechanisms if accuracy plateaus",
            "",
            "### Monitoring & Maintenance",
            "",
            "1. **Continuous Evaluation**",
            "   - Monitor real-world detection performance",
            "   - Collect failure cases for retraining",
            "   - Track metric drift over time",
            "",
            "2. **Retraining Schedule**",
            "   - Retrain quarterly with new data",
            "   - Implement active learning for hard examples",
            "   - Maintain version control of models",
            "",
        ]

        return lines

    def _generate_appendix(self) -> List[str]:
        """Generate appendix section."""
        lines = [
            "## Appendix",
            "",
            "### File Locations",
            f"- **Best Model**: `{PATHS['best_model']}`",
            f"- **Last Model**: `{PATHS['last_model']}`",
            f"- **Training Results**: `{PATHS['results_csv']}`",
            f"- **Configuration**: `{Path(PATHS['plate_detector_dir']) / 'config.json'}`",
            "",
            "### Usage Examples",
            "",
            "#### Prediction on Image",
            "```bash",
            "python training/predict.py --source image.jpg --model models/plate_detector/best.pt",
            "```",
            "",
            "#### Evaluation on Test Set",
            "```bash",
            "python training/evaluate.py --model models/plate_detector/best.pt",
            "```",
            "",
            "#### Export Model",
            "```bash",
            "python training/export.py --model models/plate_detector/best.pt --format all",
            "```",
            "",
            "### References",
            "- [YOLOv8 Documentation](https://docs.ultralytics.com/)",
            "- [Jetson Orin Nano Setup](https://docs.nvidia.com/jetson/jetson-orin-nano-devkit/)",
            "- [ONNX Runtime](https://onnxruntime.ai/)",
            "",
            "---",
            f"*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        ]

        return lines

    def save_report(self) -> bool:
        """
        Save report to markdown file.

        Returns:
            True if saved successfully
        """
        try:
            report_content = self.generate_report()

            with open(self.report_path, "w") as f:
                f.write(report_content)

            logger.info(f"✓ Report saved to {self.report_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to save report: {e}", exc_info=True)
            return False


def main():
    """Main report generation pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate training report")
    parser.add_argument(
        "--training-dir",
        type=str,
        default=PATHS["plate_detector_dir"],
        help="Training results directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output report path",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("TRAINING REPORT GENERATOR")
    logger.info("=" * 80)

    try:
        generator = TrainingReportGenerator(args.training_dir)
        generator.load_results()

        if generator.save_report():
            logger.info("✓ Report generated successfully")
            exit(0)
        else:
            exit(1)

    except Exception as e:
        logger.error(f"❌ Report generation failed: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
