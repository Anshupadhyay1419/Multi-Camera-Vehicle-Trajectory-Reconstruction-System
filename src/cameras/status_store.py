"""
Cross-process handoff for live processing status.

The worker that runs the camera queue and the screens that watch it are not
in the same process: the Streamlit dashboard, the FastAPI server and the CLI
can each be watching a run that a different one of them started. They share
a single small JSON file -- one writer, many readers -- rather than a queue
or a broker, because the payload is a few hundred bytes and adding Redis to
a Jetson deployment to move it would be out of proportion.

Durability rules that matter here:

  * Writes are atomic (write to a temp file in the same directory, then
    os.replace). A reader polling every second WILL eventually read while a
    write is in flight; without the rename it reads a half-written file and
    the dashboard throws a JSON error mid-run.
  * Reads never raise. A missing file just means nothing has run yet, and a
    corrupt one (killed mid-write on a filesystem without atomic rename)
    must degrade to "unknown" rather than take the dashboard down with it.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional

from src.utils.logger import get_logger

_logger = get_logger("cameras.status_store")

DEFAULT_STATUS_PATH = "data/processing_status.json"


class StatusStore:
    """Atomic reader/writer for the processing status document.

    Args:
        path: Status file location. Its parent directory is created on first
              write, not in __init__, so constructing a store is side-effect
              free (the API builds one at import time just to read).
    """

    def __init__(self, path: str = DEFAULT_STATUS_PATH) -> None:
        self.path = Path(path)
        # Serialises writers inside one process. Cross-process safety comes
        # from os.replace being atomic, not from this lock -- there is only
        # ever one writer, so the lock is just guarding against a second
        # thread in the same process (a cancel racing an update).
        self._lock = threading.Lock()

    def write(self, payload: dict[str, Any]) -> bool:
        """Persist *payload*, replacing whatever was there.

        Returns:
            True on success. False (logged, never raised) on failure -- a
            status update that cannot be written must not abort the
            processing run it is merely reporting on.
        """
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                # Temp file in the SAME directory: os.replace is only
                # guaranteed atomic within one filesystem, and /tmp is
                # frequently a different one.
                handle, temp_path = tempfile.mkstemp(
                    dir=str(self.path.parent),
                    prefix=f".{self.path.name}.",
                    suffix=".tmp",
                )
                try:
                    with os.fdopen(handle, "w", encoding="utf-8") as stream:
                        json.dump(payload, stream, default=str)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temp_path, self.path)
                except Exception:
                    # Never leave the temp file behind on a failed write --
                    # a long run updating twice a second would otherwise
                    # litter the data directory.
                    Path(temp_path).unlink(missing_ok=True)
                    raise
                return True
            except Exception as exc:
                _logger.warning("Could not write status to %s: %s", self.path, exc)
                return False

    def read(self) -> Optional[dict[str, Any]]:
        """Return the current status document, or None if unavailable.

        None covers both "no run has ever started" and "the file is
        unreadable"; callers treat either as "nothing to show", so they are
        not worth distinguishing at this layer.
        """
        try:
            if not self.path.is_file():
                return None
            raw = self.path.read_text(encoding="utf-8")
            if not raw.strip():
                return None
            return json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            _logger.debug("Could not read status from %s: %s", self.path, exc)
            return None

    def clear(self) -> None:
        """Remove the status file. Safe when it does not exist."""
        with self._lock:
            try:
                self.path.unlink(missing_ok=True)
            except OSError as exc:
                _logger.warning("Could not clear status file %s: %s", self.path, exc)
