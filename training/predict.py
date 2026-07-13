"""
Prediction script for YOLOv8 License Plate Detector.

Supports inference on single images, image folders, videos, and webcam streams.
Saves results with annotated bounding boxes.
"""

import sys
import logging
from pathlib import Path
from typing import Optional, List
import time

import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# Add training directory to path
training_dir = Path(__file__).parent
sys.path.insert(0, str(training_dir))

from config import PATHS, INFERENCE_CONFIG
from utils import setup_logger, format_time

logger = setup_logger(__name__, log_file=str(Path(PATHS["plate_detector_dir"]) / "predict.log"))


class PlateDetector:
    """License plate detector using trained YOLOv8 model."""

    def __init__(self, model_path: str, device: int = 0, conf: float = 0.5):
        """
        Initialize detector.

        Args:
            model_path: Path to trained model (.pt file)
            device: GPU device ID
            conf: Confidence threshold
        """
        self.model_path = Path(model_path)
        self.device = device
        self.conf = conf

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info(f"Loading model: {model_path}")
        self.model = YOLO(str(model_path))
        logger.info("✓ Model loaded")

    def predict_image(self, image_path: str) -> dict:
        """
        Run inference on single image.

        Args:
            image_path: Path to image

        Returns:
            Dictionary with results and annotations
        """
        image_path = Path(image_path)

        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return {"success": False, "error": "Image not found"}

        try:
            # Run inference
            results = self.model.predict(
                str(image_path),
                conf=self.conf,
                iou=INFERENCE_CONFIG["iou"],
                device=self.device,
                verbose=False,
            )

            if not results or len(results) == 0:
                return {"success": False, "error": "No detections"}

            result = results[0]

            # Extract detections
            detections = []
            for box in result.boxes:
                detection = {
                    "class": int(box.cls[0]),
                    "class_name": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": {
                        "x1": float(box.xyxy[0][0]),
                        "y1": float(box.xyxy[0][1]),
                        "x2": float(box.xyxy[0][2]),
                        "y2": float(box.xyxy[0][3]),
                    },
                }
                detections.append(detection)

            return {
                "success": True,
                "image_path": str(image_path),
                "detections": detections,
                "num_detections": len(detections),
                "annotated_image": result.plot(),
            }

        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return {"success": False, "error": str(e)}

    def predict_folder(
        self, folder_path: str, output_dir: Optional[str] = None, extensions: List[str] = None
    ) -> None:
        """
        Run inference on folder of images.

        Args:
            folder_path: Path to folder containing images
            output_dir: Output directory for annotated images (default: {plate_detector_dir}/predictions)
            extensions: Image file extensions to process (default: jpg, png, jpeg, bmp)
        """
        folder_path = Path(folder_path)

        if not folder_path.exists():
            logger.error(f"Folder not found: {folder_path}")
            return

        if extensions is None:
            extensions = [".jpg", ".png", ".jpeg", ".bmp"]

        if output_dir is None:
            output_dir = Path(PATHS["plate_detector_dir"]) / "predictions"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Get all images
        image_files = []
        for ext in extensions:
            image_files.extend(folder_path.glob(f"*{ext}"))
            image_files.extend(folder_path.glob(f"*{ext.upper()}"))

        image_files = sorted(set(image_files))  # Remove duplicates

        if not image_files:
            logger.warning(f"No images found in {folder_path}")
            return

        logger.info(f"Found {len(image_files)} images in {folder_path}")

        # Process each image
        successful = 0
        for img_path in tqdm(image_files, desc="Processing"):
            result = self.predict_image(str(img_path))

            if result["success"]:
                # Save annotated image
                output_path = output_dir / f"pred_{img_path.stem}.jpg"
                cv2.imwrite(str(output_path), result["annotated_image"])
                successful += 1

                if result["num_detections"] > 0:
                    logger.debug(f"Found {result['num_detections']} plate(s) in {img_path.name}")

        logger.info(f"✓ Processed {successful}/{len(image_files)} images")
        logger.info(f"✓ Saved to {output_dir}")

    def predict_video(
        self, video_path: str, output_path: Optional[str] = None, skip_frames: int = 1
    ) -> None:
        """
        Run inference on video file.

        Args:
            video_path: Path to video file
            output_path: Output path for annotated video (default: predictions/{video_stem}_annotated.mp4)
            skip_frames: Process every Nth frame for speed
        """
        video_path = Path(video_path)

        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            return

        if output_path is None:
            output_dir = Path(PATHS["plate_detector_dir"]) / "predictions"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"{video_path.stem}_annotated.mp4")
        else:
            output_path = str(output_path)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing video: {video_path}")

        try:
            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return

            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            logger.info(f"  Resolution: {width}x{height}")
            logger.info(f"  FPS: {fps:.1f}")
            logger.info(f"  Total frames: {total_frames}")

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_idx = 0
            processed_frames = 0
            inference_times = []

            with tqdm(total=total_frames, desc="Processing frames") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_idx % skip_frames == 0:
                        # Run inference
                        start = time.time()
                        results = self.model.predict(
                            frame,
                            conf=self.conf,
                            iou=INFERENCE_CONFIG["iou"],
                            device=self.device,
                            verbose=False,
                        )
                        inference_time = time.time() - start
                        inference_times.append(inference_time)

                        if results and len(results) > 0:
                            result = results[0]
                            frame = result.plot()
                            processed_frames += 1

                    out.write(frame)
                    frame_idx += 1
                    pbar.update(1)

            cap.release()
            out.release()

            logger.info(f"✓ Video processing completed")
            logger.info(f"  Frames processed: {processed_frames}/{total_frames}")

            if inference_times:
                avg_time = np.mean(inference_times)
                fps_inference = 1.0 / avg_time
                logger.info(f"  Average inference time: {avg_time*1000:.2f} ms")
                logger.info(f"  Average inference FPS: {fps_inference:.1f}")

            logger.info(f"✓ Saved to {output_path}")

        except Exception as e:
            logger.error(f"Error processing video: {e}", exc_info=True)
        finally:
            cap.release()

    def predict_webcam(self, duration_seconds: int = 30) -> None:
        """
        Run inference on webcam stream.

        Args:
            duration_seconds: Duration to run webcam inference
        """
        logger.info("Starting webcam inference...")
        logger.info(f"Duration: {duration_seconds} seconds")
        logger.info("Press 'q' to quit")

        try:
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                logger.error("Could not open webcam")
                return

            # Set resolution for better speed
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            start_time = time.time()
            frame_count = 0
            detections_count = 0
            inference_times = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Run inference
                start = time.time()
                results = self.model.predict(
                    frame,
                    conf=self.conf,
                    iou=INFERENCE_CONFIG["iou"],
                    device=self.device,
                    verbose=False,
                )
                inference_time = time.time() - start
                inference_times.append(inference_time)

                if results and len(results) > 0:
                    result = results[0]
                    frame = result.plot()
                    detections_count += len(result.boxes)

                # Display FPS
                fps = 1.0 / inference_time if inference_time > 0 else 0
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("License Plate Detection", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                frame_count += 1
                elapsed = time.time() - start_time

                if elapsed > duration_seconds:
                    break

            cap.release()
            cv2.destroyAllWindows()

            # Print statistics
            logger.info(f"\n✓ Webcam inference completed")
            logger.info(f"  Frames: {frame_count}")
            logger.info(f"  Duration: {format_time(elapsed)}")
            logger.info(f"  Average FPS: {frame_count / elapsed:.1f}")
            logger.info(f"  Total detections: {detections_count}")

            if inference_times:
                avg_time = np.mean(inference_times)
                logger.info(f"  Average inference time: {avg_time*1000:.2f} ms")

        except Exception as e:
            logger.error(f"Error in webcam inference: {e}", exc_info=True)
        finally:
            cap.release()
            cv2.destroyAllWindows()


def main():
    """Main prediction pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Predict with YOLOv8 License Plate Detector")
    parser.add_argument(
        "--model",
        type=str,
        default=PATHS["best_model"],
        help="Path to trained model",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Image, folder, video, or 'webcam'",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory or file path",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=INFERENCE_CONFIG["conf"],
        help="Confidence threshold",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU device ID",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=1,
        help="Skip N frames in video (for speed)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Webcam duration in seconds",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("YOLOV8 LICENSE PLATE DETECTOR - PREDICTION")
    logger.info("=" * 80)

    try:
        detector = PlateDetector(args.model, device=args.device, conf=args.conf)

        source = Path(args.source)

        # Determine input type and run prediction
        if args.source.lower() == "webcam" or args.source == "0":
            detector.predict_webcam(duration_seconds=args.duration)

        elif source.is_file():
            # Single image or video
            if source.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                logger.info(f"Processing image: {source}")
                result = detector.predict_image(str(source))
                if result["success"]:
                    output_dir = Path(args.output) if args.output else Path(PATHS["plate_detector_dir"]) / "predictions"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_file = output_dir / f"pred_{source.stem}.jpg"
                    cv2.imwrite(str(output_file), result["annotated_image"])
                    logger.info(f"✓ Saved to {output_file}")

            elif source.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
                logger.info(f"Processing video: {source}")
                detector.predict_video(str(source), output_path=args.output, skip_frames=args.skip_frames)

        elif source.is_dir():
            logger.info(f"Processing folder: {source}")
            detector.predict_folder(str(source), output_dir=args.output)

        else:
            logger.error(f"Invalid source: {args.source}")
            exit(1)

        logger.info("\n✓ Prediction completed successfully")
        exit(0)

    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
