"""Runtime settings, read from the environment.

Deliberately plain stdlib rather than pydantic-settings. This module lives
inside an existing deployment whose virtualenv is pinned around the ALPR
pipeline (torch, ultralytics, OpenCV all built against numpy 1.26), and the
registry needs exactly one thing from a settings library -- read a handful
of environment variables with defaults. Adding a dependency, and the
pydantic version floor it drags with it, is not worth that.

PostgreSQL is the deployment target. The default below is a local SQLite
file so a fresh checkout can run the tests and the README examples without
a database server; every model here is written to the PostgreSQL feature
set regardless.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

_ENV_PREFIX = "SENTINEL_"


def _env(name: str, default: str) -> str:
    return os.environ.get(f"{_ENV_PREFIX}{name}", default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(f"{_ENV_PREFIX}{name}")
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(f"{_ENV_PREFIX}{name}")
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{_ENV_PREFIX}{name} must be an integer, got {raw!r}"
        ) from exc
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{_ENV_PREFIX}{name} must be between {minimum} and {maximum}, got {value}"
        )
    return value


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.environ.get(f"{_ENV_PREFIX}{name}")
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"{_ENV_PREFIX}{name} must be a number, got {raw!r}"
        ) from exc
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{_ENV_PREFIX}{name} must be between {minimum} and {maximum}, got {value}"
        )
    return value


@dataclass(frozen=True)
class Settings:
    """Environment-driven configuration.

    Frozen: settings are read once at process start and a component that
    mutates them at runtime would leave other components holding the old
    values, which is a bug that only shows up under load.
    """

    database_url: str = field(
        default_factory=lambda: _env("DATABASE_URL", "sqlite+pysqlite:///./sentinel.db")
    )
    sql_echo: bool = field(default_factory=lambda: _env_bool("SQL_ECHO", False))
    db_pool_size: int = field(
        default_factory=lambda: _env_int("DB_POOL_SIZE", 5, minimum=1, maximum=100)
    )
    db_max_overflow: int = field(
        default_factory=lambda: _env_int("DB_MAX_OVERFLOW", 10, minimum=0, maximum=100)
    )
    #: Whether DELETE may destroy a row rather than retire it. Off by
    #: default: a camera that has been in service is referenced by incident
    #: reports, so the destructive path has to be switched on deliberately
    #: rather than reached by adding a query parameter.
    allow_hard_delete: bool = field(
        default_factory=lambda: _env_bool("ALLOW_HARD_DELETE", False)
    )

    # ── camera verification (M1.3) ────────────────────────────────────────
    #: End-to-end ceiling on one verification, in seconds. An operator is
    #: waiting on this, so it is bounded well below OpenCV's own ~30s per
    #: connection attempt.
    verification_timeout: float = field(
        default_factory=lambda: _env_float(
            "VERIFICATION_TIMEOUT", 12.0, minimum=1.0, maximum=120.0
        )
    )
    #: Frames to read before deciding. More than one because a single frame
    #: gives no interval and therefore no achieved frame rate; few enough
    #: that a 25fps camera is still sampled in well under a second.
    frames_to_sample: int = field(
        default_factory=lambda: _env_int(
            "FRAMES_TO_SAMPLE", 5, minimum=2, maximum=120
        )
    )
    #: JPEG quality for the stored thumbnail.
    thumbnail_quality: int = field(
        default_factory=lambda: _env_int(
            "THUMBNAIL_QUALITY", 80, minimum=1, maximum=100
        )
    )
    #: Where thumbnails are written. Under the ALPR API's existing
    #: `data/thumbnails` mount, so a saved thumbnail is served at
    #: /thumbnails/cameras/<file> with no new static route.
    thumbnail_directory: str = field(
        default_factory=lambda: _env("THUMBNAIL_DIRECTORY", "data/thumbnails/cameras")
    )

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith("postgresql")

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


_settings: Settings | None = None


def get_settings() -> Settings:
    """The process-wide settings object, built once."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Drop the cached settings so the next call re-reads the environment."""
    global _settings
    _settings = None
