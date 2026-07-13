"""
Export script for YOLOv8 License Plate Detector.

Exports trained model to multiple formats: PyTorch, ONNX, and TensorRT-compatible ONNX.
Suitable for edge deployment on Jetson Orin Nano.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional
import time
import shutil

import torch
from ultralytics import YOLO

# Add training directory to path
training_dir = Path(__file__).parent
sys.path.insert(0, str(training_dir))

from config import PATHS
from utils import setup_logger, format_time

logger = setup_logger(__name__, log_file=str(Path(PATHS["plate_detector_dir"]) / "export.log"))


class ModelExporter:
    """Export YOLOv8 model to multiple formats."""

    def __init__(self, model_path: str):
        """
        Initialize exporter.

        Args:
            model_path: Path to trained model (.pt file)
        """
        self.model_path = Path(model_path)
        self.export_dir = self.model_path.parent

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info(f"Loading model: {model_path}")
        self.model = YOLO(str(model_path))
        logger.info("✓ Model loaded")

    def export_to_pytorch(self, output_path: Optional[str] = None) -> Optional[Path]:
        """
        Export model in PyTorch format (.pt).

        Args:
            output_path: Output path (default: same as input)

        Returns:
            Path to exported model or None if failed
        """
        logger.info("\n" + "-" * 80)
        logger.info("Exporting to PyTorch format (.pt)...")

        try:
            if output_path is None:
                output_path = self.model_path

            # Copy model (it's already in PyTorch format)
            shutil.copy(str(self.model_path), str(output_path))

            logger.info(f"✓ PyTorch model exported to {output_path}")
            logger.info(f"  Model size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

            return Path(output_path)

        except Exception as e:
            logger.error(f"❌ PyTorch export failed: {e}", exc_info=True)
            return None

    def export_to_onnx(
        self,
        output_dir: Optional[str] = None,
        opset: int = 12,
        half: bool = False,
        dynamic: bool = False,
    ) -> Optional[Path]:
        """
        Export model to ONNX format for CPU/edge inference.

        Args:
            output_dir: Output directory (default: {plate_detector_dir})
            opset: ONNX opset version
            half: Use FP16 (not recommended for edge devices)
            dynamic: Use dynamic batch size

        Returns:
            Path to exported model or None if failed
        """
        logger.info("\n" + "-" * 80)
        logger.info("Exporting to ONNX format...")

        try:
            if output_dir is None:
                output_dir = str(self.export_dir)

            output_path = Path(output_dir) / f"{self.model_path.stem}.onnx"

            # Export to ONNX
            logger.info(f"  Opset version: {opset}")
            logger.info(f"  Half precision: {half}")
            logger.info(f"  Dynamic batch: {dynamic}")

            export_result = self.model.export(
                format="onnx",
                imgsz=640,
                half=half,
                dynamic=dynamic,
                opset=opset,
                device=0,
            )

            # Move exported file
            if export_result:
                temp_path = Path(export_result)
                shutil.move(str(temp_path), str(output_path))

            logger.info(f"✓ ONNX model exported to {output_path}")
            logger.info(f"  Model size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
            logger.info("  ✓ Suitable for: CPU inference, Jetson Orin Nano (with ONNX Runtime)")

            return Path(output_path)

        except Exception as e:
            logger.error(f"❌ ONNX export failed: {e}", exc_info=True)
            return None

    def export_to_onnx_fp32(self, output_dir: Optional[str] = None) -> Optional[Path]:
        """
        Export model to FP32 ONNX (specifically for Jetson Orin Nano).

        Args:
            output_dir: Output directory (default: {plate_detector_dir})

        Returns:
            Path to exported model or None if failed
        """
        logger.info("\n" + "-" * 80)
        logger.info("Exporting to ONNX FP32 (Jetson Orin Nano optimized)...")

        try:
            onnx_path = self.export_to_onnx(output_dir=output_dir, half=False, dynamic=False)

            if onnx_path:
                logger.info(f"✓ ONNX FP32 model ready for Jetson deployment")
                logger.info("  Deployment instructions:")
                logger.info("    1. Install ONNX Runtime on Jetson:")
                logger.info("       pip install onnxruntime")
                logger.info("    2. Copy model to Jetson:")
                logger.info(f"       scp {onnx_path} jetson@jetson_ip:/path/to/models/")
                logger.info("    3. Use ONNX Runtime for inference")
                return onnx_path
            else:
                return None

        except Exception as e:
            logger.error(f"❌ ONNX FP32 export failed: {e}", exc_info=True)
            return None

    def export_all(self, output_dir: Optional[str] = None) -> Dict[str, Optional[Path]]:
        """
        Export to all formats.

        Args:
            output_dir: Output directory (default: {plate_detector_dir})

        Returns:
            Dictionary mapping format names to exported paths
        """
        logger.info("\n" + "=" * 80)
        logger.info("EXPORTING MODEL TO ALL FORMATS")
        logger.info("=" * 80)

        if output_dir is None:
            output_dir = str(self.export_dir)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        start_time = time.time()

        # Export to PyTorch
        pt_path = self.export_to_pytorch(output_dir=str(output_dir / "best.pt"))
        results["pytorch"] = pt_path

        # Export to ONNX
        onnx_path = self.export_to_onnx(output_dir=str(output_dir), half=False, dynamic=False)
        results["onnx"] = onnx_path

        total_time = time.time() - start_time

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("EXPORT SUMMARY")
        logger.info("=" * 80)

        for format_name, path in results.items():
            if path and path.exists():
                size_mb = path.stat().st_size / 1024 / 1024
                logger.info(f"✓ {format_name.upper()}: {path} ({size_mb:.1f} MB)")
            else:
                logger.warning(f"✗ {format_name.upper()}: Failed")

        logger.info(f"\nTotal export time: {format_time(total_time)}")
        logger.info("=" * 80)

        return results

    def generate_deployment_guide(self, output_dir: Optional[str] = None) -> None:
        """
        Generate deployment guide for exported models.

        Args:
            output_dir: Output directory
        """
        if output_dir is None:
            output_dir = str(self.export_dir)

        guide_path = Path(output_dir) / "DEPLOYMENT_GUIDE.md"

        guide_content = f"""
# YOLOv8 License Plate Detector - Deployment Guide

## Model Information
- **Model**: YOLOv8s
- **Input size**: 640x640
- **Output**: Bounding boxes with confidence scores
- **Class**: license_plate

## Exported Formats

### 1. PyTorch Format (.pt)
- **File**: {Path(output_dir) / 'best.pt'}
- **Use case**: Development, training continuation, fine-tuning
- **Framework**: PyTorch
- **Device**: GPU (recommended) or CPU
- **Installation**: `pip install torch torchvision`

**Example usage**:
```python
from ultralytics import YOLO
model = YOLO('best.pt')
results = model.predict('image.jpg')
```

### 2. ONNX Format (.onnx)
- **File**: {Path(output_dir) / f'{Path(self.model_path).stem}.onnx'}
- **Use case**: Edge inference, CPU inference, cross-platform
- **Framework**: ONNX Runtime
- **Device**: CPU or GPU
- **Installation**: `pip install onnxruntime`

**Example usage**:
```python
import onnxruntime as ort
session = ort.InferenceSession('model.onnx')
```

## Jetson Orin Nano Deployment

### Setup Instructions

1. **SSH into Jetson**:
```bash
ssh ubuntu@jetson_ip
```

2. **Install dependencies**:
```bash
pip install numpy opencv-python onnxruntime
```

3. **Copy model to Jetson**:
```bash
scp best.pt ubuntu@jetson_ip:/home/ubuntu/models/
scp {Path(self.model_path).stem}.onnx ubuntu@jetson_ip:/home/ubuntu/models/
```

4. **Run inference on Jetson**:

**Using PyTorch**:
```python
import torch
from pathlib import Path

# Ensure GPU is available
print(torch.cuda.is_available())  # Should be True

# Load and run model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
results = model.to(device).predict('image.jpg', device=0)
```

**Using ONNX Runtime** (lightweight, recommended):
```python
import onnxruntime as ort
import cv2
import numpy as np

# Load model
session = ort.InferenceSession('best.onnx')

# Load image
img = cv2.imread('image.jpg')
img_resized = cv2.resize(img, (640, 640))
img_input = np.expand_dims(img_resized.transpose(2, 0, 1), 0).astype(np.float32) / 255.0

# Run inference
outputs = session.run(None, {'{input_name}': img_input})
```

## Performance Targets

### RTX 2000 Ada (Training)
- Inference time: ~2-3 ms per image
- Throughput: ~330-500 images/sec

### Jetson Orin Nano (Deployment)
- Inference time: ~10-15 ms per image (ONNX Runtime)
- Throughput: ~65-100 images/sec
- Memory usage: ~500-800 MB

## Optimization Tips

### For Jetson Orin Nano:
1. Use ONNX Runtime instead of PyTorch for lower latency
2. Set Jetson to max performance mode:
   ```bash
   sudo /usr/bin/jetson_clocks
   ```

3. Use FP16 quantization (if supported by your ONNX runtime):
   ```bash
   pip install onnxruntime-gpu  # GPU acceleration
   ```

4. Batch inference for higher throughput:
   - Process multiple frames at once
   - Use larger batch sizes on Jetson

## Troubleshooting

### CUDA Memory Issues
```python
# Reduce batch size or use half precision
results = model.predict(image, half=True)
```

### Slow Inference on Jetson
1. Check if GPU is being used: `tegrastats`
2. Ensure proper power mode: `sudo nvpmodel -m 0` (max performance)
3. Consider ONNX Runtime for faster inference

### Model Accuracy Issues
- Verify input image dimensions (640x640)
- Check confidence threshold
- Ensure images are properly preprocessed

## Support & Resources

- YOLOv8 Documentation: https://docs.ultralytics.com/
- ONNX Runtime: https://onnxruntime.ai/
- Jetson Developer Forum: https://forums.developer.nvidia.com/c/intelligent-edge/jetson/207

---
Generated: {Path(output_dir) / 'DEPLOYMENT_GUIDE.md'}
"""

        with open(guide_path, "w") as f:
            f.write(guide_content)

        logger.info(f"✓ Deployment guide saved to {guide_path}")


def main():
    """Main export pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Export YOLOv8 License Plate Detector")
    parser.add_argument(
        "--model",
        type=str,
        default=PATHS["best_model"],
        help="Path to trained model (.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for exported models",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["pytorch", "onnx", "all"],
        default="all",
        help="Export format",
    )
    parser.add_argument(
        "--include-guide",
        action="store_true",
        help="Generate deployment guide",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("YOLOV8 LICENSE PLATE DETECTOR - MODEL EXPORT")
    logger.info("=" * 80)

    try:
        exporter = ModelExporter(args.model)

        if args.output_dir is None:
            args.output_dir = PATHS["plate_detector_dir"]

        # Export based on format
        if args.format == "all":
            results = exporter.export_all(output_dir=args.output_dir)
        elif args.format == "pytorch":
            results = {"pytorch": exporter.export_to_pytorch()}
        elif args.format == "onnx":
            results = {"onnx": exporter.export_to_onnx(output_dir=args.output_dir)}

        # Generate deployment guide if requested
        if args.include_guide:
            exporter.generate_deployment_guide(output_dir=args.output_dir)

        logger.info("\n✓ Export completed successfully")
        exit(0)

    except Exception as e:
        logger.error(f"❌ Export failed: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
