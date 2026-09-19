"""Version-compatible TensorRT implementation for the PARSeq OCR model.

TensorRT 10 uses named I/O tensors and ``execute_async_v3``.  TensorRT 8
uses indexed bindings and ``execute_async_v2``.  Keeping those paths separate
is important: a TensorRT 10 context deliberately does not expose v2.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.ocr.base import OCREngine
from src.utils.logger import get_logger

_logger = get_logger("ocr.parseq_tensorrt_engine")
_DEFAULT_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_OFFICIAL_PARSEQ_CHARSET = (
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
)


class _IncompatibleEngineError(RuntimeError):
    """Raised only for TensorRT's explicit cross-device plan warning."""


class PARSeqTensorRTOCREngine(OCREngine):
    """Run the existing PARSeq TensorRT engine entirely on the GPU."""

    def __init__(
        self,
        engine_path: str,
        input_size: tuple[int, int] = (32, 128),
        charset: str = _DEFAULT_CHARSET,
        onnx_path: str = "models/ocr/parseq.onnx",
        decoder_mode: str = "legacy",
    ) -> None:
        self.engine_path = engine_path
        self.onnx_path = Path(onnx_path)
        self.input_h, self.input_w = input_size
        self.charset = charset
        self.decoder_mode = decoder_mode
        self._init_failed = False
        self._engine: Any = None
        self._context: Any = None
        self._runtime: Any = None  # Keep Python-owned TRT objects alive.
        self._trt_logger: Any = None
        self._api = ""
        self._rebuild_attempted = False
        self._execution_rebuild_attempted = False
        self._input_name = ""
        self._output_name = ""
        self._input_index = -1
        self._output_index = -1
        self._input_dtype = np.dtype(np.float32)
        self._output_dtype = np.dtype(np.float32)
        self.last_inference_ms: float | None = None
        # Persistent GPU buffers + stream, allocated once (input size is fixed
        # at batch=1) instead of cudaMalloc/cudaFree on every recognize() call.
        self._stream: Any = None
        self._input_mem: Any = None
        self._output_mem: Any = None
        self._host_output: np.ndarray | None = None
        self._output_shape: tuple[int, ...] | None = None
        self._initialize()

    # ---- Startup and engine compatibility ---------------------------------

    def _initialize(self) -> None:
        engine_file = Path(self.engine_path)
        if not engine_file.exists():
            _logger.error("TensorRT engine not found at '%s'", engine_file)
            self._init_failed = True
            return
        if importlib.util.find_spec("tensorrt") is None:
            _logger.error("TensorRT Python bindings missing; install python3-libnvinfer")
            self._init_failed = True
            return

        try:
            # TensorRT, PyCUDA and PyTorch must use one primary CUDA context.
            # ``autoinit`` creates a separate context and causes execution
            # failures once the YOLO/PyTorch detector has touched the GPU.
            import pycuda.autoprimaryctx  # noqa: F401
            import tensorrt as trt

            info = self.runtime_info()
            _logger.info(
                "TensorRT startup: version=%s cuda=%s jetpack=%s gpu=%s "
                "execute_async_v2=%s execute_async_v3=%s",
                info["tensorrt_version"], info["cuda_version"], info["jetpack_version"],
                info["gpu_name"], info["execute_async_v2"], info["execute_async_v3"],
            )
            if self._metadata_device_mismatch(engine_file, str(info["gpu_name"])):
                raise _IncompatibleEngineError("engine metadata identifies a different GPU model")

            self._trt_logger = self._make_trt_logger(trt)
            self._runtime = trt.Runtime(self._trt_logger)
            self._engine = self._runtime.deserialize_cuda_engine(engine_file.read_bytes())

            # TensorRT emits this warning while deserializing an incompatible
            # plan.  Never continue with that plan: rebuild from the existing ONNX.
            device_warning = any(
                "across different models of devices" in message.lower()
                or "not compatible with the current device" in message.lower()
                for message in getattr(self._trt_logger, "messages", [])
            )
            if device_warning:
                raise _IncompatibleEngineError("TensorRT engine was built for a different GPU model")
            if self._engine is None:
                raise RuntimeError("TensorRT failed to deserialize CUDA engine")

            self._context = self._engine.create_execution_context()
            if self._context is None:
                raise RuntimeError("TensorRT failed to create an execution context")
            self._configure_io(trt)
            self._allocate_io_buffers()
            self._write_engine_metadata(engine_file, info)
            _logger.info(
                "PARSeq TensorRT engine loaded from %s using %s (wrapper=%s)",
                engine_file, self._api, Path(__file__).resolve(),
            )
        except _IncompatibleEngineError as exc:
            if not self._rebuild_attempted and self._rebuild_engine(engine_file, str(exc)):
                self._rebuild_attempted = True
                self._initialize()
                return
            _logger.error("Failed to load PARSeq TensorRT engine: %s", exc)
            self._init_failed = True
        except Exception as exc:
            _logger.error("Failed to load PARSeq TensorRT engine: %s", exc)
            self._init_failed = True

    @staticmethod
    def _make_trt_logger(trt: Any) -> Any:
        class CapturingLogger(trt.ILogger):
            def __init__(self) -> None:
                trt.ILogger.__init__(self)
                self.messages: list[str] = []

            def log(self, severity: Any, message: str) -> None:
                self.messages.append(str(message))
                if severity <= trt.ILogger.Severity.WARNING:
                    _logger.warning("TensorRT: %s", message)

        return CapturingLogger()

    @classmethod
    def runtime_info(cls) -> dict[str, str | bool]:
        """Return the runtime diagnostics printed whenever the backend starts."""
        try:
            import tensorrt as trt
            context_cls = trt.IExecutionContext
            trt_version = trt.__version__
            has_v2 = hasattr(context_cls, "execute_async_v2")
            has_v3 = hasattr(context_cls, "execute_async_v3")
        except Exception:
            trt_version, has_v2, has_v3 = "unavailable", False, False
        return {
            "tensorrt_version": trt_version,
            "cuda_version": cls._command_version(["nvcc", "--version"], "release"),
            "jetpack_version": cls._jetpack_version(),
            "gpu_name": cls._gpu_name(),
            "execute_async_v2": has_v2,
            "execute_async_v3": has_v3,
        }

    @staticmethod
    def _command_version(command: list[str], needle: str) -> str:
        try:
            output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
            return next((line.strip() for line in output.splitlines() if needle in line.lower()), "unknown")
        except Exception:
            return "unknown"

    @staticmethod
    def _jetpack_version() -> str:
        for path in (Path("/etc/nv_tegra_release"), Path("/etc/nv_boot_control.conf")):
            try:
                return path.read_text(encoding="utf-8").strip()
            except OSError:
                pass
        return "unknown"

    @staticmethod
    def _gpu_name() -> str:
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
            )
            if output.strip():
                return output.strip().splitlines()[0]
        except Exception:
            pass
        try:
            import pycuda.driver as cuda
            cuda.init()
            return cuda.Device(0).name()
        except Exception:
            return "unknown"

    def _rebuild_engine(self, engine_file: Path, reason: str) -> bool:
        if not self.onnx_path.exists():
            _logger.error("Cannot rebuild incompatible engine: ONNX model is missing at %s", self.onnx_path)
            return False
        _logger.warning("Rebuilding PARSeq engine on this Jetson (%s): %s", self._gpu_name(), reason)
        command = ["trtexec", f"--onnx={self.onnx_path}", f"--saveEngine={engine_file}", "--fp16"]
        if self.decoder_mode != "official_parseq_94":
            # scripts/export_official_parseq.py exports a fixed batch=1
            # static-shape graph (no dynamic_axes declared) -- trtexec
            # rejects *any* --*Shapes flags against a static graph
            # ("Static model does not take explicit shapes"). The older
            # custom-trained model (training/export_onnx.py) does declare a
            # dynamic batch axis and needs an explicit shape profile to
            # build at all, so only add these for that model.
            command += [
                "--minShapes=input:1x3x%dx%d" % (self.input_h, self.input_w),
                "--optShapes=input:1x3x%dx%d" % (self.input_h, self.input_w),
                "--maxShapes=input:1x3x%dx%d" % (self.input_h, self.input_w),
            ]
        try:
            subprocess.run(command, cwd=Path(__file__).resolve().parents[2], check=True)
            return True
        except FileNotFoundError:
            return self._rebuild_engine_with_python(engine_file)
        except subprocess.CalledProcessError as exc:
            _logger.error("TensorRT engine rebuild failed: %s", exc)
            return False

    def _rebuild_engine_with_python(self, engine_file: Path) -> bool:
        """Build an engine when the TensorRT Python package has no trtexec CLI."""
        try:
            import tensorrt as trt

            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            creation_flag = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
            network = builder.create_network(
                1 << int(creation_flag) if creation_flag is not None else 0
            )
            parser = trt.OnnxParser(network, logger)
            if not parser.parse(self.onnx_path.read_bytes()):
                errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
                raise RuntimeError(f"ONNX parse failed: {errors}")

            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
            fp16_flag = getattr(trt.BuilderFlag, "FP16", None)
            if fp16_flag is not None and getattr(builder, "platform_has_fast_fp16", True):
                config.set_flag(fp16_flag)

            serialized = builder.build_serialized_network(network, config)
            if serialized is None:
                raise RuntimeError("TensorRT returned no serialized engine")
            engine_file.write_bytes(bytes(serialized))
            self._write_engine_metadata(engine_file, self.runtime_info())
            _logger.info("PARSeq TensorRT engine rebuilt with the TensorRT Python API: %s", engine_file)
            return True
        except Exception as exc:
            _logger.error("TensorRT Python engine rebuild failed: %s", exc)
            return False

    @staticmethod
    def _write_engine_metadata(engine_file: Path, info: dict[str, str | bool]) -> None:
        """Record where a successfully loaded engine was validated/built."""
        try:
            engine_file.with_suffix(engine_file.suffix + ".metadata.json").write_text(
                json.dumps(info, indent=2) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            _logger.warning("Could not write TensorRT engine metadata: %s", exc)

    @staticmethod
    def _metadata_device_mismatch(engine_file: Path, current_gpu: str) -> bool:
        """Detect a moved engine before deserialization when our sidecar exists."""
        if current_gpu == "unknown":
            return False
        try:
            recorded = json.loads(
                engine_file.with_suffix(engine_file.suffix + ".metadata.json").read_text(encoding="utf-8")
            ).get("gpu_name", "unknown")
            return recorded != "unknown" and recorded != current_gpu
        except (OSError, ValueError, TypeError):
            return False

    # ---- TensorRT API split -------------------------------------------------

    def _configure_io(self, trt: Any) -> None:
        if hasattr(self._context, "execute_async_v3") and hasattr(self._engine, "num_io_tensors"):
            self._api = "TensorRT 10 named-tensor / execute_async_v3"
            inputs, outputs = [], []
            for index in range(self._engine.num_io_tensors):
                name = self._engine.get_tensor_name(index)
                if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    inputs.append(name)
                else:
                    outputs.append(name)
            if len(inputs) != 1 or len(outputs) != 1:
                raise RuntimeError("PARSeq engine must have exactly one input and one output")
            self._input_name, self._output_name = inputs[0], outputs[0]
            self._input_dtype = np.dtype(trt.nptype(self._engine.get_tensor_dtype(self._input_name)))
            self._output_dtype = np.dtype(trt.nptype(self._engine.get_tensor_dtype(self._output_name)))
            _logger.info(
                "TensorRT 10 I/O: input=%s shape=%s dtype=%s; output=%s shape=%s dtype=%s",
                self._input_name, tuple(self._engine.get_tensor_shape(self._input_name)), self._input_dtype,
                self._output_name, tuple(self._engine.get_tensor_shape(self._output_name)), self._output_dtype,
            )
            return

        if hasattr(self._context, "execute_async_v2") and hasattr(self._engine, "num_bindings"):
            self._api = "TensorRT 8 binding / execute_async_v2"
            for index in range(self._engine.num_bindings):
                if self._engine.binding_is_input(index):
                    self._input_index = index
                else:
                    self._output_index = index
            if self._input_index < 0 or self._output_index < 0:
                raise RuntimeError("PARSeq TensorRT 8 engine needs one input and one output binding")
            self._input_dtype = np.dtype(trt.nptype(self._engine.get_binding_dtype(self._input_index)))
            self._output_dtype = np.dtype(trt.nptype(self._engine.get_binding_dtype(self._output_index)))
            return
        raise RuntimeError("Installed TensorRT exposes neither supported execution API")

    def _free_io_buffers(self) -> None:
        for attr in ("_input_mem", "_output_mem"):
            mem = getattr(self, attr)
            if mem is not None:
                try:
                    mem.free()
                except Exception as exc:
                    # A handle can already be invalid if the CUDA context was
                    # disrupted (e.g. by another CUDA-using library). Freeing
                    # is best-effort cleanup, never a reason to crash.
                    _logger.warning("Failed to free TensorRT buffer %s: %s", attr, exc)
            setattr(self, attr, None)
        self._stream = None
        self._host_output = None
        self._output_shape = None

    def close(self) -> None:
        """Free the GPU buffers and drop the TensorRT objects.

        Called once per camera by the pipeline's shutdown path. Without it,
        a four-camera queue builds four engines in one process and frees
        none of them until the process exits -- which on a Jetson's unified
        memory is a real ceiling, and which also leaves the CUDA context
        stack non-empty at interpreter teardown.

        Idempotent and non-throwing: safe to call twice, and safe to call on
        an engine whose initialisation failed.
        """
        self._free_io_buffers()
        # Dropped newest-first: the execution context references the engine,
        # which references the runtime. Releasing in the other order leaves
        # TensorRT holding a pointer to something already collected.
        self._context = None
        self._engine = None
        self._runtime = None
        self._trt_logger = None
        self._init_failed = True   # any further recognize() no-ops instead of
                                   # touching freed device memory

    def _allocate_io_buffers(self) -> None:
        """Allocate the fixed-shape GPU input/output buffers and CUDA stream
        once at load time. Input size is always batch=1 x 3 x H x W, so these
        never need to be resized between calls — reusing them avoids a
        cudaMalloc/cudaFree pair (and its implicit device sync) per frame,
        which otherwise dominates OCR latency variance under load.
        """
        import pycuda.driver as cuda

        self._free_io_buffers()
        input_shape = (1, 3, self.input_h, self.input_w)
        host_input = np.empty(input_shape, dtype=self._input_dtype)

        if self._api.startswith("TensorRT 10"):
            if not self._context.set_input_shape(self._input_name, input_shape):
                raise RuntimeError("TensorRT rejected PARSeq input shape %s" % (input_shape,))
            output_shape = self._shape(self._context.get_tensor_shape(self._output_name))
        else:
            if not self._context.set_binding_shape(self._input_index, input_shape):
                raise RuntimeError("TensorRT rejected PARSeq input shape %s" % (input_shape,))
            output_shape = self._shape(self._context.get_binding_shape(self._output_index))

        self._output_shape = output_shape
        self._host_output = np.empty(output_shape, dtype=self._output_dtype)
        try:
            self._input_mem = cuda.mem_alloc(host_input.nbytes)
            self._output_mem = cuda.mem_alloc(self._host_output.nbytes)
            self._stream = cuda.Stream()
        except Exception:
            # Free whichever allocation(s) succeeded before the failure
            # (e.g. _input_mem allocated, then _output_mem raised on a
            # near-OOM GPU) instead of leaking them for the process lifetime.
            self._free_io_buffers()
            raise

    # ---- Preprocessing, allocation and inference --------------------------

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB if image.ndim == 2 else cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_CUBIC)
        tensor = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        return np.ascontiguousarray(tensor.transpose(2, 0, 1)[np.newaxis], dtype=self._input_dtype)

    @staticmethod
    def _shape(shape: Any) -> tuple[int, ...]:
        result = tuple(int(dim) for dim in shape)
        if any(dim < 0 for dim in result):
            raise RuntimeError("TensorRT returned an unresolved dynamic output shape: %s" % (result,))
        return result

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        values = values - np.max(values, axis=-1, keepdims=True)
        exp = np.exp(values)
        return exp / exp.sum(axis=-1, keepdims=True)

    def _decode(self, logits: np.ndarray) -> tuple[str, float]:
        if self.decoder_mode == "official_parseq_94":
            return self._decode_official_parseq(logits)
        probs, eos = self._softmax(logits[0]), len(self.charset)
        characters, confidence = [], []
        for row in probs:
            index = int(np.argmax(row))
            if index == eos:
                break
            if index < len(self.charset):
                characters.append(self.charset[index])
                confidence.append(float(row[index]))
        return re.sub(r"[^A-Z0-9]", "", "".join(characters).upper()), float(np.mean(confidence)) if confidence else 0.0

    def _decode_official_parseq(self, logits: np.ndarray) -> tuple[str, float]:
        """Decode baudm/parseq logits: output 0 is EOS, chars start at 1."""
        probs = self._softmax(logits[0])
        characters, confidence = [], []
        for row in probs:
            index = int(np.argmax(row))
            if index == 0:  # Official PARSeq EOS is head output index 0.
                break
            char_index = index - 1
            if 0 <= char_index < len(_OFFICIAL_PARSEQ_CHARSET):
                char = _OFFICIAL_PARSEQ_CHARSET[char_index]
                if char.isalnum():
                    characters.append(char.upper())
                    confidence.append(float(row[index]))
        return "".join(characters), float(np.mean(confidence)) if confidence else 0.0

    def recognize(self, image: np.ndarray) -> tuple[str, float]:
        if self._init_failed or self._context is None or image is None or image.size == 0:
            return "", 0.0
        if self._input_mem is None or self._output_mem is None or self._stream is None:
            return "", 0.0
        try:
            import pycuda.driver as cuda
            import pycuda.autoprimaryctx  # noqa: F401: retains the CUDA primary context.

            host_input = self._preprocess(image)
            stream = self._stream
            input_mem, output_mem = self._input_mem, self._output_mem
            host_output = self._host_output
            start_event, end_event = cuda.Event(), cuda.Event()
            start_event.record(stream)
            cuda.memcpy_htod_async(input_mem, host_input, stream)

            if self._api.startswith("TensorRT 10"):
                if not self._context.set_tensor_address(self._input_name, int(input_mem)):
                    raise RuntimeError("Failed to set TensorRT input address")
                if not self._context.set_tensor_address(self._output_name, int(output_mem)):
                    raise RuntimeError("Failed to set TensorRT output address")
                if not self._context.execute_async_v3(stream.handle):
                    raise RuntimeError("TensorRT execute_async_v3 failed")
            else:
                # A 10.x runtime must never enter this branch.  If it does,
                # Python imported a different/stale wrapper or TRT binding.
                if self._trt_major_version() >= 10:
                    raise RuntimeError(
                        "TensorRT 10 selected the legacy binding path; verify the wrapper path printed at startup"
                    )
                bindings = [0] * self._engine.num_bindings
                bindings[self._input_index], bindings[self._output_index] = int(input_mem), int(output_mem)
                if not self._context.execute_async_v2(bindings, stream.handle):
                    raise RuntimeError("TensorRT execute_async_v2 failed")
            cuda.memcpy_dtoh_async(host_output, output_mem, stream)
            end_event.record(stream)
            stream.synchronize()  # Required before host_output is decoded.
            self.last_inference_ms = float(start_event.time_till(end_event))
            return self._decode(host_output)
        except ImportError:
            _logger.error("pycuda is required for PARSeq TensorRT inference")
            self._init_failed = True
            return "", 0.0
        except Exception as exc:
            _logger.warning("PARSeq TensorRT inference failed: %s", exc)
            try:
                if self._should_rebuild_after_execution_error():
                    # Release the persistent buffers before trtexec writes a
                    # replacement plan; _initialize() reallocates them on success.
                    self._free_io_buffers()
                    self._execution_rebuild_attempted = True
                    if self._rebuild_engine(Path(self.engine_path), "TensorRT Cask execution failure"):
                        self._init_failed = False
                        self._initialize()
                        if not self._init_failed:
                            _logger.info("Retrying PARSeq inference with engine rebuilt on this Jetson")
                            return self.recognize(image)
                    else:
                        # _free_io_buffers() above already cleared
                        # _input_mem/_output_mem/_stream, so every future
                        # recognize() call would otherwise hit the top-of-
                        # method buffer guard and return ("", 0.0) silently
                        # forever. Mark the engine failed and log once here
                        # so this doesn't read as "OCR has no errors" when
                        # it has in fact stopped working.
                        _logger.error(
                            "PARSeq TensorRT engine rebuild failed after a Cask "
                            "execution error; OCR is disabled for the rest of "
                            "this process."
                        )
                        self._init_failed = True
            except Exception as recovery_exc:
                # Never let a failure during error recovery (e.g. freeing an
                # already-invalid CUDA handle) escape and take down the
                # pipeline — recognize() must always degrade to ("", 0.0).
                _logger.error("PARSeq TensorRT recovery failed: %s", recovery_exc)
                self._init_failed = True
            return "", 0.0

    def _trt_major_version(self) -> int:
        try:
            import tensorrt as trt
            return int(str(trt.__version__).split(".", maxsplit=1)[0])
        except Exception:
            return 0

    def _should_rebuild_after_execution_error(self) -> bool:
        """A Cask execution error means the serialized tactic is unusable here."""
        if self._execution_rebuild_attempted:
            return False
        messages = getattr(self._trt_logger, "messages", [])
        return any("cask convolution execution" in str(message).lower() for message in messages[-16:])
