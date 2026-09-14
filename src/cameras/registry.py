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

# Where cameras added or removed from the dashboard are kept.
#
# camera_config.yaml is hand-written and heavily commented -- rewriting it
# from a YAML dump would throw every one of those comments away. So the
# shipped file stays the declaration an engineer edits, and operator changes
# live in this small file beside it, written and read only by code. When it
# holds a camera list, that list REPLACES the shipped one (the operator's
# view of the deployment is the current one); everything else -- defaults,
# processing, trajectory -- still comes from camera_config.yaml. Delete the
# file to go back to the shipped cameras.
RUNTIME_CAMERAS_FILENAME = "cameras_runtime.yaml"

_RUNTIME_HEADER = """\
# Cameras as managed from the dashboard (Add camera / Remove).
#
# Written by src/cameras/registry.py -- edit camera_config.yaml instead if you
# want comments to survive. This list REPLACES the `cameras:` list in
# camera_config.yaml; delete this file to go back to the shipped cameras.
"""

# The camera fields worth persisting. `video_path` and `rtsp_url` are left
# out on purpose: which clip or stream a camera is replaying is session
# state, chosen per run, exactly as save_upload()/set_rtsp_url() already
# treat it.
_PERSISTED_FIELDS = ("camera_id", "camera_name", "latitude", "longitude",
                     "order", "enabled")


def runtime_cameras_path(base_path: str = DEFAULT_CAMERA_CONFIG_PATH) -> Path:
    """The operator-managed camera file that sits beside *base_path*."""
    return Path(base_path).resolve().parent / RUNTIME_CAMERAS_FILENAME

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

    def __init__(
        self,
        cameras: Iterable[CameraConfig],
        settings: Optional[dict] = None,
        source_path: Optional[str] = None,
    ) -> None:
        self._cameras: list[CameraConfig] = sorted(
            cameras, key=lambda camera: (camera.order, camera.camera_id)
        )
        self._by_id: dict[str, CameraConfig] = {c.camera_id: c for c in self._cameras}
        self._settings: dict = settings or {}
        # Where this registry was loaded from, so add/remove know which
        # runtime file to write. None for a registry built in memory (tests),
        # which then simply does not persist.
        self._source_path = source_path

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

    def add(self, camera: CameraConfig) -> CameraConfig:
        """Register a new camera site.

        Raises:
            ValueError: A camera with that id is already registered. Two
                sites sharing an id merge every trajectory that passes
                either of them, which nothing downstream can undo.
        """
        if camera.camera_id in self._by_id:
            raise ValueError(f"Camera {camera.camera_id!r} already exists")
        self._by_id[camera.camera_id] = camera
        self._cameras = sorted(
            self._by_id.values(), key=lambda c: (c.order, c.camera_id)
        )
        return camera

    def remove(self, camera_id: str) -> CameraConfig:
        """Unregister a camera site and return what was removed.

        Only the site's *configuration* goes. Events it already recorded stay
        in the database: they are history, and a trajectory that passed this
        camera yesterday still happened. They keep the camera_id and the
        coordinates stamped on them at capture time, so past trajectories
        still plot correctly.

        Raises:
            KeyError:   No such camera.
            ValueError: It is the last one -- a deployment with no cameras
                        cannot be loaded back (camera_config.yaml requires a
                        non-empty list), so refusing here is kinder than
                        writing a file that fails at the next start.
        """
        camera = self.require(camera_id)
        if len(self._cameras) == 1:
            raise ValueError(
                "Cannot remove the last camera -- a deployment needs at "
                "least one. Add its replacement first."
            )
        del self._by_id[camera_id]
        self._cameras = [c for c in self._cameras if c.camera_id != camera_id]
        return camera

    def next_camera_id(self, prefix: str = "CAM") -> str:
        """An unused id in the deployment's CAMnnn style."""
        used = set(self._by_id)
        number = 1
        while f"{prefix}{number:03d}" in used:
            number += 1
        return f"{prefix}{number:03d}"

    def next_order(self) -> int:
        """The queue position a newly added camera should take: last."""
        return max((c.order for c in self._cameras), default=0) + 1

    def save(self) -> Optional[Path]:
        """Persist the current camera list so it survives a restart.

        Writes the runtime file beside the config this registry was loaded
        from. Returns the path written, or None for an in-memory registry
        with nowhere to write.
        """
        if not self._source_path:
            return None
        path = runtime_cameras_path(self._source_path)
        payload = yaml.safe_dump(
            {"cameras": [
                {field: value for field, value in camera.to_dict().items()
                 if field in _PERSISTED_FIELDS}
                for camera in self._cameras
            ]},
            sort_keys=False,
            allow_unicode=True,
        )
        # Write-and-rename: the dashboard may be reading this file in another
        # process, and a half-written camera list is a registry that refuses
        # to load.
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(_RUNTIME_HEADER + payload, encoding="utf-8")
        temporary.replace(path)
        _logger.info("Wrote %d camera(s) to %s", len(self._cameras), path)
        return path

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

    # Cameras the operator has added or removed from the dashboard replace
    # the shipped list. A missing, empty or unreadable runtime file is not an
    # error: it just means nobody has changed the deployment yet.
    runtime_path = runtime_cameras_path(path)
    if runtime_path.is_file():
        try:
            runtime_raw = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            msg = f"Failed to parse '{runtime_path}': {exc}"
            _logger.critical(msg)
            raise CameraConfigError(msg) from exc
        runtime_entries = (runtime_raw or {}).get("cameras") \
            if isinstance(runtime_raw, dict) else None
        if isinstance(runtime_entries, list) and runtime_entries:
            _logger.info(
                "Using the %d camera(s) managed from the dashboard (%s)",
                len(runtime_entries), runtime_path,
            )
            entries = runtime_entries
        elif runtime_entries is not None:
            _logger.warning(
                "%s has no usable 'cameras' list -- using the cameras in %s",
                runtime_path, path,
            )

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

    registry = CameraRegistry(cameras, settings=raw, source_path=str(config_path))
    _logger.info(
        "Loaded %d camera(s) from %s (%d enabled): %s",
        len(registry), path, len(registry.enabled),
        ", ".join(f"{c.camera_id}/{c.camera_name}" for c in registry),
    )
    return registry
