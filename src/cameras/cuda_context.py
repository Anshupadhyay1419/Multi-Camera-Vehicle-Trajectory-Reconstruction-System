"""
CUDA context lifecycle for the worker thread that runs a processing session.

The problem this solves kills the dashboard outright, so it is worth stating
plainly. `pycuda.autoprimaryctx` -- which the PARSeq TensorRT engine imports,
and must, since TensorRT, PyCUDA and PyTorch have to share one primary
context -- retains the device's primary context and PUSHES it onto whichever
thread imports it. In the multi-camera system that thread is the session
worker, not the main thread.

When a thread exits with a non-empty CUDA context stack, PyCUDA aborts the
process:

    PyCUDA ERROR: The context stack was not empty upon module cleanup.
    ... The program will be aborted now.

For the single-gate CLI that was invisible: the process was exiting anyway.
For a long-lived Streamlit server it meant the whole site died the moment a
run finished -- the browser just showed "Connection error", with no Python
traceback anywhere, because abort() is not an exception.

So the worker thread has to leave its context stack as it found it: push the
primary context when it starts, drain the stack before it ends. Every
function here is best-effort and never raises -- a machine with no CUDA at
all (CI, a laptop) must be completely unaffected.
"""

from __future__ import annotations

from typing import Any, Optional

from src.utils.logger import get_logger

_logger = get_logger("cameras.cuda_context")


def _driver() -> Optional[Any]:
    """Return pycuda.driver, or None when CUDA is not available here."""
    try:
        import pycuda.driver as cuda

        return cuda
    except Exception:
        # No pycuda, no CUDA, or no device -- nothing to manage.
        return None


def push_primary_context() -> bool:
    """Make the device's primary context current on the calling thread.

    Needed for every session worker AFTER the first: the first one inherits
    the context `autoprimaryctx` pushed at import time, but drain_contexts()
    empties the stack when that thread ends, so later threads start with
    nothing current and every CUDA call would fail.

    Returns True if a context is current when this returns.
    """
    cuda = _driver()
    if cuda is None:
        return False
    try:
        cuda.init()
        if cuda.Context.get_current() is not None:
            return True
        # retain_primary_context() hands back the SAME context TensorRT,
        # PyCUDA and PyTorch already share -- creating a fresh one instead
        # would put this thread on a context the engines know nothing about.
        context = cuda.Device(0).retain_primary_context()
        context.push()
        _logger.debug("Pushed the CUDA primary context onto the session worker")
        return True
    except Exception as exc:
        _logger.debug("Could not push a CUDA context: %s", exc)
        return False


def drain_contexts() -> int:
    """Pop every CUDA context current on the calling thread.

    Call this before a worker thread exits. Leaving even one context on the
    stack is what makes PyCUDA abort the process.

    Returns how many contexts were popped.
    """
    cuda = _driver()
    if cuda is None:
        return 0

    popped = 0
    try:
        # Bounded rather than `while True`: a genuinely stuck stack should
        # not spin here forever during shutdown.
        for _ in range(16):
            if cuda.Context.get_current() is None:
                break
            cuda.Context.pop()
            popped += 1
    except Exception as exc:
        _logger.debug("Stopped draining CUDA contexts after %d: %s", popped, exc)

    if popped:
        _logger.debug("Popped %d CUDA context(s) before the worker exited", popped)
    return popped
