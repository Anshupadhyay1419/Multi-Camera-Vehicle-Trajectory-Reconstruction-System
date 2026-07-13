"""
TensorRT OCR engine for the ALPR University Gate system.

This engine runs a TensorRT .engine file built from a fine-tuned PARSeq
model for Indian license plate recognition. It is the GPU inference path
on JetPack 7.2, where onnxruntime-gpu is unavailable but TensorRT 10.16
is pre-installed as part of JetPack itself.

─── Workflow ────────────────────────────────────────────────────────────────

  Phase 1 (current):  config['ocr']['backend'] = "rapidocr"
                      RapidOCR runs on CPU via onnxruntime. Works out of
                      the box, ~80–120ms/crop on Jetson Orin Nano.

  Phase 2 (after fine-tuning PARSeq):
    1. Fine-tune PARSeq on your Indian plate crop dataset (dev machine):
         cd /path/to/parseq
         ./train.py +experiment=parseq charset=36_lowercase \\
             data.root_dir=<your_plate_lmdb> trainer.accelerator=gpu

    2. Export to ONNX (dev machine):
         python export_onnx.py pretrained=<your_checkpoint>.ckpt \\
             --output models/ocr/parseq_plate.onnx

    3. Compile to TensorRT engine ON the Jetson (required — TensorRT
       engines are device-specific):
         trtexec \\
           --onnx=models/ocr/parseq_plate.onnx \\
           --saveEngine=models/ocr/parseq_plate.engine \\
           --fp16 \\
           --minShapes=input:1x3x32x128 \\
           --optShapes=input:1x3x32x128 \\
           --maxShapes=input:8x3x32x128

    4. Switch backend in config.yaml:
         ocr:
           backend: "tensorrt"
           tensorrt:
             engine_path: "models/ocr/parseq_plate.engine"
             input_size: [32, 128]       # H x W PARSeq expects
             charset: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

─── TensorRT Python bindings ────────────────────────────────────────────────

  TensorRT is pre-installed on JetPack. Expose Python bindings with:
    sudo apt-get install python3-libnvinfer python3-libnvinfer-dev
  Verify: python3 -c "import tensorrt; print(tensorrt.__version__)"

─── PARSeq character set ────────────────────────────────────────────────────

  Indian plates use: 0-9 + A-Z (36 chars). Train with charset=36_lowercase
  (PARSeq encodes lowercase + digits, decodes to uppercase for our validator).
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from src.ocr.base import OCREngine
from src.utils.logger import get_logger

_logger = get_logger("ocr.tensorrt_engine")

# PARSeq default character set for Indian plates (0–9 + A–Z, 36 chars)
_DEFAULT_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class TensorRTOCREngine(OCREngine):
    """OCR engine backed by a TensorRT .engine file (PARSeq fine-tuned).

    Requires TensorRT Python bindings to be installed:
        sudo apt-get install python3-libnvinfer python3-libnvinfer-dev

    Args:
        engine_path:  Path to the compiled .engine file on disk.
        input_size:   (H, W) that the engine was compiled for. Default
                      matches PARSeq's standard plate input: (32, 128).
        charset:      Character vocabulary the model was trained on.
                      Must match the charset used during PARSeq training.
    """

    def __init__(
        self,
        engine_path: str,
        input_size: tuple[int, int] = (32, 128),
        charset: str = _DEFAULT_CHARSET,
    ) -> None:
        self.engine_path = engine_path
        self.input_h, self.input_w = input_size
        self.charset = charset
        self._context = None
        self._engine = None
        self._init_failed = False
        self._initialize()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        """Load and deserialize the TensorRT engine."""
        engine_file = Path(self.engine_path)

        if not engine_file.exists():
            _logger.error(
                "TensorRT engine not found at '%s'. "
                "Build it with trtexec — see docstring for full command.",
                self.engine_path,
            )
            self._init_failed = True
            return

        try:
            import tensorrt as trt

            trt_logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(trt_logger)

            with open(engine_file, "rb") as f:
                engine_data = f.read()

            self._engine = runtime.deserialize_cuda_engine(engine_data)
            if self._engine is None:
                raise RuntimeError("deserialize_cuda_engine returned None")

            self._context = self._engine.create_execution_context()
            _logger.info(
                "TensorRT engine loaded: '%s' | input %dx%d | charset %d chars",
                self.engine_path, self.input_h, self.input_w, len(self.charset),
            )

        except ImportError:
            _logger.error(
                "TensorRT Python bindings not found. Install with:\n"
                "  sudo apt-get install python3-libnvinfer python3-libnvinfer-dev\n"
                "Then verify: python3 -c \"import tensorrt; print(tensorrt.__version__)\""
            )
            self._init_failed = True
        except Exception as exc:
            _logger.error("TensorRT engine load failed: %s", exc)
            self._init_failed = True

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize and normalize a plate crop for PARSeq inference.

        PARSeq expects:  float32 tensor (1, 3, H, W), values in [-1, 1]
        Input image:     BGR or grayscale uint8 NumPy array.
        """
        # Convert to RGB
        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to PARSeq input size
        resized = cv2.resize(
            rgb, (self.input_w, self.input_h),
            interpolation=cv2.INTER_BICUBIC,
        )

        # Normalize to [-1, 1] (mean=0.5, std=0.5 per channel)
        tensor = resized.astype(np.float32) / 255.0
        tensor = (tensor - 0.5) / 0.5

        # HWC → CHW → NCHW
        tensor = tensor.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)
        return np.ascontiguousarray(tensor, dtype=np.float32)

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def _decode_greedy(self, logits: np.ndarray) -> tuple[str, float]:
        """Greedy decode PARSeq logits to a plate string + confidence.

        Args:
            logits: (1, T, vocab_size) float32 array from engine output.
                    T = max_label_length + 1 (includes EOS position).
                    vocab_size = len(charset) + 2  ([EOS] + [PAD]).

        Returns:
            (decoded_text, mean_character_confidence)
        """
        # EOS token is the last index in PARSeq's vocabulary
        eos_idx = len(self.charset)

        probs = self._softmax(logits[0])   # (T, vocab_size)
        chars = []
        confidences = []

        for t in range(probs.shape[0]):
            idx = int(np.argmax(probs[t]))
            if idx == eos_idx:
                break
            if idx < len(self.charset):
                chars.append(self.charset[idx])
                confidences.append(float(probs[t, idx]))

        text = "".join(chars).upper()
        # Strip non-alphanumeric in case of unexpected tokens
        text = re.sub(r"[^A-Z0-9]", "", text)
        conf = float(np.mean(confidences)) if confidences else 0.0
        return (text, conf)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax along last axis."""
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def recognize(self, image: np.ndarray) -> tuple[str, float]:
        """Run TensorRT PARSeq inference on a plate crop.

        Args:
            image: Grayscale (H, W) or BGR (H, W, 3) NumPy array.
                   Should be the cropped plate region from PlateDetector.

        Returns:
            (text, confidence) — ("", 0.0) on any failure.
        """
        if self._init_failed or self._context is None:
            return ("", 0.0)

        if image is None or image.size == 0:
            return ("", 0.0)

        try:
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401  — initialises CUDA context

            input_tensor = self._preprocess(image)

            # Allocate I/O buffers
            input_binding = self._engine.get_binding_index("input")
            output_binding = self._engine.get_binding_index("output")

            input_mem = cuda.mem_alloc(input_tensor.nbytes)
            # Determine output shape from engine binding
            output_shape = tuple(self._engine.get_binding_shape(output_binding))
            output_tensor = np.empty(output_shape, dtype=np.float32)
            output_mem = cuda.mem_alloc(output_tensor.nbytes)

            stream = cuda.Stream()

            # Host → device
            cuda.memcpy_htod_async(input_mem, input_tensor, stream)

            # Run inference
            self._context.execute_async_v2(
                bindings=[int(input_mem), int(output_mem)],
                stream_handle=stream.handle,
            )

            # Device → host
            cuda.memcpy_dtoh_async(output_tensor, output_mem, stream)
            stream.synchronize()

            # Free device memory
            input_mem.free()
            output_mem.free()

            text, conf = self._decode_greedy(output_tensor)
            return (text, conf)

        except ImportError:
            _logger.error(
                "pycuda not installed. Install with:\n"
                "  pip install pycuda\n"
                "or via apt: sudo apt-get install python3-pycuda"
            )
            self._init_failed = True
            return ("", 0.0)
        except Exception as exc:
            _logger.warning("TensorRT inference failed: %s", exc)
            return ("", 0.0)
