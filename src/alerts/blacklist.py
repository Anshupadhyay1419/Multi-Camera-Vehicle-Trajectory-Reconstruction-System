"""
Vehicle blacklist: plates that raise an alert whenever they are detected.

Stored in config/blacklist.yaml so it survives restarts and can be edited by
hand or from the dashboard. The file is re-read whenever it changes on disk,
so a plate added in the dashboard is picked up by a processing run already
in progress (the pipeline worker and the dashboard share this process, and
the API is a separate one).

Matching is exact after normalisation (upper case, no spaces or dashes). OCR
can misread a character, so a blacklisted vehicle read incorrectly will not
match -- the alert is only as reliable as the plate read.
"""

from __future__ import annotations

import re
import threading
from datetime import date
from pathlib import Path
from typing import Optional

import yaml

from src.utils.logger import get_logger

_logger = get_logger("alerts.blacklist")

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "blacklist.yaml"


def normalize_plate(plate: Optional[str]) -> str:
    return re.sub(r"[\s\-]", "", (plate or "")).upper()


class Blacklist:
    def __init__(self, path: Path = DEFAULT_PATH) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._mtime: Optional[float] = None
        self._entries: dict[str, dict] = {}

    # ── reading ──────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            self._entries, self._mtime = {}, None
            return
        if mtime == self._mtime:
            return
        try:
            data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            _logger.warning("Could not read blacklist %s: %s", self.path, exc)
            return
        entries = {}
        for item in data.get("plates") or []:
            if isinstance(item, str):
                item = {"plate": item}
            plate = normalize_plate(item.get("plate") if isinstance(item, dict) else None)
            if plate:
                entries[plate] = {"plate": plate, "reason": item.get("reason") or "",
                                  "added": str(item.get("added") or "")}
        self._entries, self._mtime = entries, mtime

    def entries(self) -> list[dict]:
        with self._lock:
            self._refresh()
            return sorted(self._entries.values(), key=lambda e: e["plate"])

    def plates(self) -> set[str]:
        return {e["plate"] for e in self.entries()}

    def get(self, plate: Optional[str]) -> Optional[dict]:
        with self._lock:
            self._refresh()
            return self._entries.get(normalize_plate(plate))

    def contains(self, plate: Optional[str]) -> bool:
        return self.get(plate) is not None

    # ── writing ──────────────────────────────────────────────────────────

    def _save(self) -> None:
        body = {"plates": [
            {"plate": e["plate"], "reason": e["reason"], "added": e["added"]}
            for e in sorted(self._entries.values(), key=lambda e: e["plate"])
        ]}
        header = "# Blacklisted vehicles -- detections of these plates raise an alert.\n"
        tmp = self.path.with_suffix(".yaml.tmp")
        tmp.write_text(header + yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
        tmp.replace(self.path)
        self._mtime = None

    def add(self, plate: str, reason: str = "") -> str:
        normalized = normalize_plate(plate)
        if not re.fullmatch(r"[A-Z0-9]{4,12}", normalized):
            raise ValueError(f"{plate!r} is not a valid plate number")
        with self._lock:
            self._refresh()
            self._entries[normalized] = {"plate": normalized, "reason": reason.strip(),
                                         "added": date.today().isoformat()}
            self._save()
        _logger.info("Blacklisted %s (%s)", normalized, reason)
        return normalized

    def remove(self, plate: str) -> bool:
        normalized = normalize_plate(plate)
        with self._lock:
            self._refresh()
            if normalized not in self._entries:
                return False
            del self._entries[normalized]
            self._save()
        _logger.info("Removed %s from the blacklist", normalized)
        return True


_default: Optional[Blacklist] = None


def get_blacklist() -> Blacklist:
    global _default
    if _default is None:
        _default = Blacklist()
    return _default
