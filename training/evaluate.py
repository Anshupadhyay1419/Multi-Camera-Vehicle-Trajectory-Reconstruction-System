"""
Evaluation script for trained YOLOv8 License Plate Detector.

Evaluates model on test dataset and generates detailed metrics and visualizations.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
import time

import torch
import numpy as np
from ultralytics import YOLO
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add training directory to path
training_dir = Path(__file__).parent
sys.path.insert(0, str(training_dir))

from config import PATHS, VALIDATION_CONFIG, MODEL_NAME
from utils import setup_logger, format_time

logger = setup_logger(__name__, log_file=str(Path(PATHS["plate_detector_dir"]) / "evaluate.log"))


class ModelEvaluator:
    """Evaluate trained YOLOv8 model on test dataset."""

    def __init__(self, model_path: str, data_yaml: str, device: int = 0):
        """
        Initialize evaluator.

        Args:
            model_path: Path to trained model (.pt file)
            data_yaml: Path to data.yaml
            device: GPU device ID
        """
        self.model_path = Path(model_path)
        self.data_yaml = Path(data_yaml)
        self.device = device
        self.results = {}

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not self.data_yaml.exists():
            raise FileNotFoundError(f"Dataset config not found: {data_yaml}")

        # Load model
        logger.info(f"Loading model: {model_path}")
        self.model = YOLO(str(model_path))
        logger.info("✓ Model loaded")

    def evaluate_on_test_set(self) -> Dict:
        """
        Evaluate model on test dataset.

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATING ON TEST DATASET")
        logger.info("=" * 80)

        start_time = time.time()

        try:
            # Run validation on test set
            results = self.model.val(
                data=str(self.data_yaml),
                device=self.device,
                split="test",
                conf=VALIDATION_CONFIG["conf"],
                iou=VALIDATION_CONFIG["iou"],
                verbose=VALIDATION_CONFIG["verbose"],
            )

            elapsed = time.time() - start_time

            logger.info(f"\nEvaluation completed in {format_time(elapsed)}")

            # Extract metrics
            metrics = self._extract_metrics(results)
            self.results = metrics

            self._log_metrics(metrics)

            return metrics

        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}", exc_info=True)
            return {}

    def _extract_metrics(self, results) -> Dict:
        """
        Extract metrics from YOLO validation results.

        Args:
            results: Results object from YOLO validation

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Extract from results object
        if hasattr(results, "box"):
            box_results = results.box
            if hasattr(box_results, "conf"):
                metrics["confidence"] = float(box_results.conf.mean()) if len(box_results.conf) > 0 else 0.0

        # Standard metrics
        metrics_names = [
            "precision",
            "recall",
            "mAP50",
            "mAP",
            "fitness"
        ]

        for attr in dir(results):
            if not attr.startswith("_"):
                try:
                    value = getattr(results, attr)
                    if isinstance(value, (int, float, np.number)):
                        if not np.isnan(value) and not np.isinf(value):
                            metrics[attr] = float(value)
                except:
                    pass

        # Ensure key metrics exist
        if "metrics" in dir(results) and hasattr(results.metrics, "keys"):
            for key in results.metrics.keys():
                value = results.metrics[key]
                if isinstance(value, (int, float, np.number)):
                    if not np.isnan(value) and not np.isinf(value):
                        metrics[key] = float(value)

        return metrics

    def _log_metrics(self, metrics: Dict) -> None:
        """
        Log evaluation metrics.

        Args:
            metrics: Dictionary of metrics
        """
        logger.info("\n" + "=" * 80)
        logger.info("TEST SET METRICS")
        logger.info("=" * 80)

        if not metrics:
            logger.warning("No metrics available")
            return

        # Log metrics in organized groups
        logger.info("\nDetection Metrics:")
        for key in ["precision", "recall", "mAP50", "mAP", "fitness"]:
            if key in metrics:
                logger.info(f"  {key}: {metrics[key]:.4f}")

        # Log all other metrics
        other_metrics = {k: v for k, v in metrics.items() 
                        if k not in ["precision", "recall", "mAP50", "mAP", "fitness"]}
        
        if other_metrics:
            logger.info("\nAdditional Metrics:")
            for key, value in sorted(other_metrics.items()):
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}")
                else:
                    logger.info(f"  {key}: {value}")

        logger.info("=" * 80)

    def infer_on_images(self, image_dir: str, output_dir: Optional[str] = None) -> None:
        """
        Run inference on test images and save results.

        Args:
            image_dir: Directory containing test images
            output_dir: Directory to save predictions (default: {plate_detector_dir}/predictions)
        """
        image_dir = Path(image_dir)
        
        if output_dir is None:
            output_dir = Path(PATHS["plate_detector_dir"]) / "predictions"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        if not image_dir.exists():
            logger.error(f"Image directory not found: {image_dir}")
            return

        logger.info(f"\nRunning inference on images in {image_dir}...")

        # Get all image files
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        image_files += list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.bmp"))

        if not image_files:
            logger.warning(f"No images found in {image_dir}")
            return

        logger.info(f"Found {len(image_files)} images")

        # Run inference
        successful = 0
        for img_path in tqdm(image_files, desc="Inferencing"):
            try:
                results = self.model.predict(
                    str(img_path),
                    conf=VALIDATION_CONFIG["conf"],
                    iou=VALIDATION_CONFIG["iou"],
                    device=self.device,
                    verbose=False,
                )

                if results and len(results) > 0:
                    # Get annotated image
                    result = results[0]
                    annotated_img = result.plot()

                    # Save
                    output_path = output_dir / f"pred_{img_path.stem}.jpg"
                    cv2.imwrite(str(output_path), annotated_img)
                    successful += 1

            except Exception as e:
                logger.debug(f"Error processing {img_path}: {e}")

        logger.info(f"✓ Saved {successful}/{len(image_files)} predictions to {output_dir}")

    def save_evaluation_report(self, output_path: str) -> None:
        """
        Save evaluation results to JSON file.

        Args:
            output_path: Path to save JSON report
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "model": str(self.model_path),
            "metrics": self.results,
        }

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"✓ Evaluation report saved to {output_file}")


def main():
    """Main evaluation pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 License Plate Detector")
    parser.add_argument(
        "--model",
        type=str,
        default=PATHS["best_model"],
        help="Path to trained model",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=PATHS["data_yaml"],
        help="Path to data.yaml",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU device ID",
    )
    parser.add_argument(
        "--infer-images",
        type=str,
        default=None,
        help="Run inference on images in this directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for predictions",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("YOLOV8 LICENSE PLATE DETECTOR EVALUATION")
    logger.info("=" * 80)

    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(args.model, args.data, device=args.device)

        # Evaluate on test set
        metrics = evaluator.evaluate_on_test_set()

        # Save report
        report_path = Path(PATHS["plate_detector_dir"]) / "evaluation_report.json"
        evaluator.save_evaluation_report(str(report_path))

        # Run inference if specified
        if args.infer_images:
            evaluator.infer_on_images(args.infer_images, args.output_dir)

        logger.info("\n✓ Evaluation completed successfully")
        exit(0)

    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
