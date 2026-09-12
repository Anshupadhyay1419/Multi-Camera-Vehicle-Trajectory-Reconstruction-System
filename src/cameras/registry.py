"""
Loader and lookup for config/camera_config.yaml.

Parses the camera registry into `CameraConfig` objects and exposes them in
processing order. Validation is deliberately split in two:

  * Structural problems that make the file meaningless -- not a mapping, no
    `cameras:` list, a duplicate camera_id -- raise CameraConfigError. There
    is no sane way to continue, and silently running a subset of the
    deployment would corrupt trajectories in a way nobody would notice.

  * Per-field problems that cost one camera something but not everything --
    an unsurveyed or swapped coordinate, an unknown source_type -- are
    logged and degraded. A camera with no coordinates still records events
    and still appears in the timeline; it just cannot be plotted. Losing a
    site's whole feed because someone mistyped a latitude would be worse.

This mirrors utils/config.py's existing posture (a hand-edited config should
never stop the gate recording vehicles) and reuses its coordinate parser
rather than growing a second one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import yaml

from src.cameras.models import CameraConfig, SourceType
from src.utils.config import _coerce_coordinate
from src.utils.logger import get_logger

_logger = get_logger("cameras.registry")

DEFAULT_CAMERA_CONFIG_PATH = "config/camera_config.yaml"

# Fallbacks for the whole `processing:` / `trajectory:` blocks, so a config
# written before either existed still loads with sensible behaviour.
_DEFAULT_PROCESSING = {
    "upload_dir": "data/uploads",
    "status_file": "data/processing_status.json",
    "continue_on_error": True,
}
_DEFAULT_TRAJECTORY = {
    "order_by": "auto",
    "collapse_per_camera": True,
    "revisit_gap_seconds": 300,
}


class CameraConfigError(Exception):
    """Raised when camera_config.yaml is missing, unparseable, or structurally invalid."""


def _parse_source_type(raw: Any, camera_id: str) -> SourceType:
    """Resolve `source_type`, falling back to UPLOAD for anything unknown.

    UPLOAD is the safe fallback: it points at a local file that either
    exists or fails loudly at open time. Guessing RTSP instead would make a
    typo look like a network outage.
    """
    if isinstance(raw, SourceType):
        return raw
    text = str(raw or "").strip().lower()
    try:
        return SourceType(text)
    except ValueError:
        _logger.warning(
            "Camera %s: unknown source_type %r -- falling back to %r. "
            "Valid values: %s",
            camera_id, raw, SourceType.UPLOAD.value,
            ", ".join(member.value for member in SourceType),
        )
        return SourceType.UPLOAD


def _as_optional_str(value: Any) -> Optional[str]:
    """Normalise a YAML scalar to a non-empty string, or None.

    YAML gives `null` for an omitted value but an empty string for a key
    left as `""`, and both mean "not set" here.
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_camera(entry: Any, index: int, seen_ids: set[str], defaults: dict) -> CameraConfig:
    """Build one CameraConfig from a raw YAML mapping."""
    if not isinstance(entry, dict):
        raise CameraConfigError(
            f"cameras[{index}] must be a mapping, got {type(entry).__name__}"
        )

    camera_id = _as_optional_str(entry.get("camera_id"))
    if not camera_id:
        raise CameraConfigError(f"cameras[{index}] is missing a camera_id")
    if camera_id in seen_ids:
        # Two sites sharing an id silently merge every trajectory that
        # passes either of them, and no amount of downstream care can
        # separate them again -- so this is fatal, not a warning.
        raise CameraConfigError(
            f"Duplicate camera_id {camera_id!r} at cameras[{index}]; "
            "camera ids must be unique across the deployment"
        )
    seen_ids.add(camera_id)

    # A missing name is cosmetic, not fatal -- fall back to the id so the
    # dashboard has something to render instead of a blank label.
    camera_name = _as_optional_str(entry.get("camera_name")) or camera_id

    def _setting(key: str, fallback: Any = None) -> Any:
        if key in entry:
            return entry[key]
        return defaults.get(key, fallback)

    try:
        order = int(_setting("order", index + 1))
    except (TypeError, ValueError):
        _logger.warning(
            "Camera %s: order %r is not an integer -- using file position %d",
            camera_id, entry.get("order"), index + 1,
        )
        order = index + 1

    return CameraConfig(
        camera_id=camera_id,
        camera_name=camera_name,
        latitude=_coerce_coordinate(entry.get("latitude"), f"{camera_id}.latitude", 90.0),
        longitude=_coerce_coordinate(entry.get("longitude"), f"{camera_id}.longitude", 180.0),
        order=order,
        source_type=_parse_source_type(_setting("source_type", "upload"), camera_id),
        video_path=_as_optional_str(_setting("video_path")),
        rtsp_url=_as_optional_str(_setting("rtsp_url")),
        enabled=bool(_setting("enabled", True)),
    )


class CameraRegistry:
    """The deployment's cameras, in processing order.

    Ordering is settled once here rather than at every call site: `order`
    ascending, ties broken by the position the entry appears in the file, so
    two cameras that both say `order: 2` still get a stable, reproducible
    sequence instead of one that depends on dict iteration.
    """

    def __init__(self, cameras: Iterable[CameraConfig], settings: Optional[dict] = None) -> None:
        self._cameras: list[CameraConfig] = sorted(
            cameras, key=lambda camera: (camera.order, camera.camera_id)
        )
        self._by_id: dict[str, CameraConfig] = {c.camera_id: c for c in self._cameras}
        self._settings: dict = settings or {}

    # ── access ────────────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[CameraConfig]:
        return iter(self._cameras)

    def __len__(self) -> int:
        return len(self._cameras)

    @property
    def all(self) -> list[CameraConfig]:
        """Every configured camera, enabled or not, in processing order."""
        return list(self._cameras)

    @property
    def enabled(self) -> list[CameraConfig]:
        """Only the cameras that should actually be queued."""
        return [camera for camera in self._cameras if camera.enabled]

    def get(self, camera_id: str) -> Optional[CameraConfig]:
        return self._by_id.get(camera_id)

    def require(self, camera_id: str) -> CameraConfig:
        camera = self._by_id.get(camera_id)
        if camera is None:
            raise KeyError(
                f"No camera {camera_id!r} in the registry "
                f"(known: {', '.join(self._by_id) or 'none'})"
            )
        return camera

    def replace(self, camera: CameraConfig) -> None:
        """Swap in a modified copy of an already-registered camera.

        The dashboard uses this after an upload to point a camera at the
        file the operator just provided, without rewriting camera_config.yaml
        -- the YAML stays the declaration of *where the cameras are*, while
        the per-run source is session state. Re-sorts because `order` is
        part of what may have changed.
        """
        if camera.camera_id not in self._by_id:
            raise KeyError(f"Cannot replace unknown camera {camera.camera_id!r}")
        self._by_id[camera.camera_id] = camera
        self._cameras = sorted(
            self._by_id.values(), key=lambda c: (c.order, c.camera_id)
        )

    # ── settings blocks ───────────────────────────────────────────────────

    @property
    def processing(self) -> dict:
        return {**_DEFAULT_PROCESSING, **(self._settings.get("processing") or {})}

    @property
    def trajectory(self) -> dict:
        return {**_DEFAULT_TRAJECTORY, **(self._settings.get("trajectory") or {})}

    def to_dict(self) -> dict:
        return {
            "cameras": [camera.to_dict() for camera in self._cameras],
            "processing": self.processing,
            "trajectory": self.trajectory,
        }


def load_camera_registry(path: str = DEFAULT_CAMERA_CONFIG_PATH) -> CameraRegistry:
    """Read and validate the camera registry at *path*.

    Args:
        path: Filesystem path to camera_config.yaml.

    Returns:
        A CameraRegistry holding every configured camera in processing order.

    Raises:
        CameraConfigError: File missing, unparseable, not a mapping, has no
            usable `cameras:` list, or contains a duplicate camera_id.
    """
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        msg = f"Camera configuration file not found: '{path}'"
        _logger.critical(msg)
        raise CameraConfigError(msg) from None
    except yaml.YAMLError as exc:
        msg = f"Failed to parse camera configuration '{path}': {exc}"
        _logger.critical(msg)
        raise CameraConfigError(msg) from exc

    if not isinstance(raw, dict):
        msg = (
            f"Camera configuration '{path}' must be a YAML mapping at the top "
            f"level, got {type(raw).__name__}"
        )
        _logger.critical(msg)
        raise CameraConfigError(msg)

    entries = raw.get("cameras")
    if not isinstance(entries, list) or not entries:
        msg = f"Camera configuration '{path}' must define a non-empty 'cameras' list"
        _logger.critical(msg)
        raise CameraConfigError(msg)

    defaults = raw.get("defaults") or {}
    if not isinstance(defaults, dict):
        _logger.warning(
            "'defaults' in %s must be a mapping, got %s -- ignoring it",
            path, type(defaults).__name__,
        )
        defaults = {}

    seen_ids: set[str] = set()
    cameras = [
        _parse_camera(entry, index, seen_ids, defaults)
        for index, entry in enumerate(entries)
    ]

    registry = CameraRegistry(cameras, settings=raw)
    _logger.info(
        "Loaded %d camera(s) from %s (%d enabled): %s",
        len(registry), path, len(registry.enabled),
        ", ".join(f"{c.camera_id}/{c.camera_name}" for c in registry),
    )
    return registry
