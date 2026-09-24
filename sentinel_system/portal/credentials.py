"""Portal credentials: from the environment or a Git-ignored local file.

Never from source code, never from the committed config, never printed. The
two sources, in order of precedence:

  1. Environment variables  SENTINEL_PORTAL_URL, SENTINEL_USERNAME,
                            SENTINEL_PASSWORD
  2. config/sentinel_portal.env   KEY=VALUE lines, listed in .gitignore.
                            A committed template lives beside it as
                            sentinel_portal.env.example, with no real values.

Environment variables win so a deployment can override a developer's local
file without editing it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from sentinel_system.portal.errors import PortalNotConfiguredError

_REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_FILE = _REPO_ROOT / "config" / "sentinel_portal.env"

_KEYS = ("SENTINEL_PORTAL_URL", "SENTINEL_USERNAME", "SENTINEL_PASSWORD")


def mask_identity(value: str | None) -> str:
    """'ankit.s@example.com' -> 'an***@example.com'; anything else -> 'an***'.

    Enough of the account left to tell two apart in a log, not enough to be
    the account.
    """
    if not value:
        return "<unset>"
    if "@" in value:
        local, _, domain = value.partition("@")
        return f"{local[:2]}***@{domain}"
    return f"{value[:2]}***"


@dataclass(frozen=True)
class PortalCredentials:
    """Everything needed to authenticate. The password never renders."""

    base_url: str
    username: str
    password: str = field(repr=False)

    def __repr__(self) -> str:
        return (
            f"PortalCredentials(base_url={self.base_url!r}, "
            f"username={mask_identity(self.username)!r}, password='***')"
        )

    __str__ = __repr__


def _read_local_file(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines; ignore blanks and comments. Never raises."""
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, OSError):
        return values
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_credentials(local_file: Path | None = None) -> PortalCredentials:
    """Resolve credentials, or say exactly which ones are missing.

    Raises:
        PortalNotConfiguredError: listing every missing setting by name, and
            never echoing any value that WAS found.
    """
    local = _read_local_file(local_file or LOCAL_FILE)
    resolved = {key: os.environ.get(key) or local.get(key) or "" for key in _KEYS}
    missing = [key for key, value in resolved.items() if not value]
    if missing:
        raise PortalNotConfiguredError(
            [f"{key} (environment variable, or config/sentinel_portal.env)"
             for key in missing]
        )
    return PortalCredentials(
        base_url=resolved["SENTINEL_PORTAL_URL"].rstrip("/"),
        username=resolved["SENTINEL_USERNAME"],
        password=resolved["SENTINEL_PASSWORD"],
    )


__all__ = ["LOCAL_FILE", "PortalCredentials", "load_credentials", "mask_identity"]
