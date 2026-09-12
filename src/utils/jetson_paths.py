"""
Make Jetson's system-installed Python packages visible inside a virtualenv.

On JetPack, `tensorrt` (and some CUDA bindings) are installed as Debian
packages into /usr/lib/python3.*/dist-packages, not via pip. A virtualenv
created without --system-site-packages cannot see them, and the symptom is
not a clean ImportError: ultralytics decides TensorRT is missing and tries
to pip-install `tensorrt-cu13` from the network, which blocks for a long
time and then fails.

This has to run BEFORE anything imports ultralytics or tensorrt. The main
pipeline has always done it at startup; the multi-camera session now loads
its models before the pipeline starts, so it has to do the same -- hence
this shared helper rather than a second copy of the logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DIST_PACKAGES = (
    "/usr/lib/python3.12/dist-packages",
    "/usr/lib/python3/dist-packages",
)


def ensure_system_site_packages() -> None:
    """Append Jetson's dist-packages to sys.path when inside a virtualenv.

    A no-op outside a virtualenv (where those paths are already present) and
    idempotent, so it is safe to call from several entry points.
    """
    if sys.prefix == sys.base_prefix:
        return

    for path in _DIST_PACKAGES:
        if Path(path).is_dir() and path not in sys.path:
            sys.path.append(path)
