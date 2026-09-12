"""
Config loader utility for the ALPR University Gate system.

Provides:
- ConfigError: custom exception raised on any configuration problem
- load_config(path): reads and validates config.yaml, returning the full dict
- get_camera_metadata(config): this device's camera identity/location fields

Secrets (camera URLs with embedded credentials, etc.) should never be
written into config.yaml -- it's tracked in git. Put them in a local .env
file instead (already gitignored); load_config() applies a small set of
recognized environment-variable overrides on top of the YAML.
"""

import os
from pathlib import Path

import yaml

from src.utils.logger import get_logger

# Logger for this module (uses defaults since config may not be loaded yet)
_logger = get_logger("utils.config")

# All top-level keys that must be present and must map to a dict
REQUIRED_KEYS = [
    "video",
    "detection",
    "tracking",
    "preprocessing",
    "enhancement",
    "ocr",
    "fusion",
    "deduplication",
    "color_classifier",
    "direction",
    "database",
    "api",
    "dashboard",
    "logging",
    "training",
]


class ConfigError(Exception):
    """Raised when the configuration file is missing, unparseable, or invalid."""


# config key (dot path) -> environment variable name. Extend this as more
# values need to move out of the tracked config.yaml.
_ENV_OVERRIDES = {
    ("video", "source"): "ALPR_VIDEO_SOURCE",
    # Point the whole system at a different database without editing the
    # tracked config.yaml -- a copy of production for a demo, a scratch file
    # for a test run, or a PostgreSQL URL in production. Everything that
    # reads config (pipeline, API, both dashboards) honours it, so they all
    # stay pointed at the same place.
    ("database", "path"): "ALPR_DB_PATH",
}


def _load_dotenv(path: str = ".env") -> None:
    """Load KEY=value lines from a local .env file into os.environ.

    Never overwrites a variable already set in the real environment (so an
    explicit `export FOO=bar` before running always wins). No external
    dependency -- the file format needed here (plain KEY=value lines,
    '#' comments) doesn't warrant one.
    """
    env_path = Path(path)
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _apply_env_overrides(config: dict) -> None:
    _load_dotenv()
    for (top_key, sub_key), env_var in _ENV_OVERRIDES.items():
        value = os.environ.get(env_var)
        if value:
            config[top_key][sub_key] = value
            _logger.info("Config override from $%s: %s.%s", env_var, top_key, sub_key)


def load_config(path: str = "config/config.yaml") -> dict:
    """
    Read and validate the YAML configuration file at *path*.

    Validation rules:
    - The file must exist and be readable.
    - The file must be valid YAML that parses to a dict.
    - Every key in REQUIRED_KEYS must be present at the top level.
    - Each required key must map to a dict (not None, not a scalar, not a list).

    Args:
        path: Filesystem path to the YAML config file.
              Defaults to "config/config.yaml".

    Returns:
        The full configuration as a plain Python dict.

    Raises:
        ConfigError: If the file is missing, cannot be parsed, or fails validation.
    """
    # --- 1. Read the file ---
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except FileNotFoundError:
        msg = f"Configuration file not found: '{path}'"
        _logger.critical(msg)
        raise ConfigError(msg) from None
    except yaml.YAMLError as exc:
        msg = f"Failed to parse configuration file '{path}': {exc}"
        _logger.critical(msg)
        raise ConfigError(msg) from exc

    # --- 2. Top-level must be a dict ---
    if not isinstance(raw, dict):
        msg = (
            f"Configuration file '{path}' must contain a YAML mapping at the top "
            f"level, got {type(raw).__name__}."
        )
        _logger.critical(msg)
        raise ConfigError(msg)

    # --- 3. Validate required keys ---
    for key in REQUIRED_KEYS:
        if key not in raw:
            msg = (
                f"Required configuration key '{key}' is missing from '{path}'."
            )
            _logger.critical(msg)
            raise ConfigError(msg)

        value = raw[key]
        if value is None:
            msg = (
                f"Required configuration key '{key}' in '{path}' must be a "
                f"mapping (dict), but its value is None."
            )
            _logger.critical(msg)
            raise ConfigError(msg)

        if not isinstance(value, dict):
            msg = (
                f"Required configuration key '{key}' in '{path}' must be a "
                f"mapping (dict), but got {type(value).__name__}."
            )
            _logger.critical(msg)
            raise ConfigError(msg)

    _apply_env_overrides(raw)
    return raw


# ---------------------------------------------------------------------------
# Camera identity / siting
# ---------------------------------------------------------------------------

# Used when config.yaml has no `camera:` block, or leaves a field blank.
# Deliberately an obviously-wrong placeholder: an operator seeing "UNKNOWN"
# on the dashboard knows the block needs filling in, where a plausible
# default like "GATE-01" would silently mislabel every event instead.
#
# `camera` is intentionally NOT in REQUIRED_KEYS -- a config predating the
# block must keep loading, just without camera attribution.
_CAMERA_PLACEHOLDER_ID = "UNKNOWN"
_CAMERA_PLACEHOLDER_NAME = "Unconfigured camera"


def _coerce_coordinate(value, field: str, limit: float) -> float | None:
    """Parse one decimal-degrees coordinate, or return None if unusable.

    Returns None (rather than raising) for every bad input: a missing,
    blank, non-numeric, or out-of-range coordinate should cost this event
    its map position, not stop the gate from recording vehicles. Each case
    is logged so it's still visible in the pipeline log.

    *limit* is the valid magnitude for the field: 90 for latitude, 180 for
    longitude. Out-of-range almost always means the two were swapped in
    config.yaml (a longitude of 77.2 dropped into `latitude` stays silently
    in range, but a latitude of 28.6 in `longitude` does not), so it's
    rejected rather than stored -- a wrong coordinate plots the gate
    somewhere it isn't, which is worse than having none.
    """
    if value is None or value == "":
        return None
    try:
        coordinate = float(value)
    except (TypeError, ValueError):
        _logger.warning(
            "camera.%s is not a number (%r); storing events without coordinates",
            field, value,
        )
        return None
    if not -limit <= coordinate <= limit:
        _logger.warning(
            "camera.%s = %s is outside the valid range [-%g, %g] -- are "
            "latitude and longitude swapped? Storing events without "
            "coordinates.",
            field, coordinate, limit, limit,
        )
        return None
    return coordinate


def get_camera_metadata(config: dict) -> dict:
    """Return this device's camera fields, ready to merge into an event row.

    One Jetson watches one gate, so the camera's id, name and coordinates
    are static configuration rather than anything the pipeline can detect.
    They're read once at startup and stamped onto every event the device
    stores, which keeps a stored event correct even if the camera is later
    renamed or re-sited (a join against a live cameras table would instead
    rewrite history).

    Args:
        config: The full config dict from load_config().

    Returns:
        Dict with exactly the keys "camera_id", "camera_name", "latitude"
        and "longitude" -- the same names as the VehicleEvent columns, so
        callers can splat it straight into an event_data dict. The two ids
        are always non-empty strings; the coordinates are floats or None.
    """
    camera_cfg = config.get("camera") or {}
    if not isinstance(camera_cfg, dict):
        _logger.warning(
            "Config key 'camera' must be a mapping, got %s -- using placeholders",
            type(camera_cfg).__name__,
        )
        camera_cfg = {}

    camera_id = str(camera_cfg.get("camera_id") or "").strip()
    camera_name = str(camera_cfg.get("camera_name") or "").strip()

    if not camera_id:
        _logger.warning(
            "camera.camera_id is not set in config.yaml -- events will be "
            "stored as '%s'", _CAMERA_PLACEHOLDER_ID,
        )
        camera_id = _CAMERA_PLACEHOLDER_ID
    if not camera_name:
        camera_name = _CAMERA_PLACEHOLDER_NAME

    return {
        "camera_id":   camera_id,
        "camera_name": camera_name,
        "latitude":    _coerce_coordinate(camera_cfg.get("latitude"), "latitude", 90.0),
        "longitude":   _coerce_coordinate(camera_cfg.get("longitude"), "longitude", 180.0),
    }
