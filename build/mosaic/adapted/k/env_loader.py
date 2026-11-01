"""Local environment loading for MOSAIC (adapted from K/tools/env_loader.py).

This intentionally avoids printing or logging secrets. Values already present
in the process environment win over values in files.

Adaptation (docs/REUSE_LEDGER.md): the search path is fixed to this
repository's git-ignored .env only. The original also read ``~/.env`` and
the KB root's .env; HANDOFF §3 item 5 forbids reading K's .env, and a home
.env could supply an unrelated key. ``require_secret`` returns a value
without ever echoing it and raises with the variable NAME only.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path

from mosaic import REPO_ROOT

MOSAIC_ENV_FILE = REPO_ROOT / ".env"


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    if key.startswith("export "):
        key = key.removeprefix("export ").strip()
    if not key:
        return None
    value = value.strip()
    if value:
        try:
            parsed = shlex.split(value, posix=True)
            if len(parsed) == 1:
                value = parsed[0]
        except ValueError:
            value = value.strip("'\"")
    return key, value


def load_mosaic_env(path: Path | None = None) -> list[str]:
    """Load this repository's .env and return the NAMES that were populated."""

    env_path = Path(path) if path is not None else MOSAIC_ENV_FILE
    loaded: list[str] = []
    if not env_path.exists():
        return loaded
    for line in env_path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        parsed = _parse_env_line(line)
        if not parsed:
            continue
        key, value = parsed
        if key not in os.environ and value:
            os.environ[key] = value
            loaded.append(key)
    return loaded


class MissingSecret(RuntimeError):
    pass


def require_secret(name: str) -> str:
    """Return the secret named ``name`` from the environment or raise.

    The value is never included in any message.
    """
    load_mosaic_env()
    value = os.environ.get(name, "")
    if not value.strip():
        raise MissingSecret(f"environment variable {name} is not set (see .env; value never logged)")
    return value
