"""
Config loader utility for the ALPR University Gate system.

Provides:
- ConfigError: custom exception raised on any configuration problem
- load_config(path): reads and validates config.yaml, returning the full dict

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
