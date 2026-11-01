"""The single write boundary for pipeline code.

Every filesystem write made by the pipeline goes through a ``Workspace``. It
accepts paths beneath this repository or beneath the OS temp directory and
raises ``WorkspaceViolation`` for anything else. It also refuses the legacy
read-only roots (E, K, X, Q, DL), the immutable handoff inputs under
docs/inputs, and the repository's .git directory, even if one of them were
somehow nested under an allowed root.

This is MOSAIC's local equivalent of EVOKE's Command Center guard
(audit C-02): it does not intercept builtins, so tests/test_workspace_guard.py
statically checks that pipeline modules do not write around it.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import IO, Iterable

from mosaic import LEGACY_ROOTS, REPO_ROOT
from mosaic.canonical import canonical_bytes


class WorkspaceViolation(PermissionError):
    """A write was attempted outside the permitted roots."""


class ImmutableArtifactError(PermissionError):
    """A write would change an existing immutable artifact."""


def _norm(path: Path) -> str:
    # realpath resolves symlinks/junctions; normcase folds case on Windows.
    return os.path.normcase(os.path.realpath(str(path)))


def _is_within(child: str, parent: str) -> bool:
    parent = parent.rstrip("\\/")
    return child == parent or child.startswith(parent + os.sep)


class Workspace:
    """Guarded writer rooted at the repository (plus OS temp for scratch)."""

    def __init__(
        self,
        repo_root: Path | None = None,
        temp_root: Path | None = None,
        extra_forbidden: Iterable[Path] = (),
    ) -> None:
        self.repo_root = Path(repo_root or REPO_ROOT)
        self.temp_root = Path(temp_root or tempfile.gettempdir())
        self._allowed = [_norm(self.repo_root), _norm(self.temp_root)]
        forbidden = [Path(p) for p in LEGACY_ROOTS.values()]
        forbidden += [
            self.repo_root / "docs" / "inputs",
            self.repo_root / "docs" / "INPUT_MANIFEST.json",
            self.repo_root / ".git",
            self.repo_root / ".env",
        ]
        forbidden += [Path(p) for p in extra_forbidden]
        self._forbidden = [_norm(p) for p in forbidden]

    # ------------------------------------------------------------------ checks
    def check(self, path: Path | str) -> Path:
        """Return the resolved path if a write there is permitted, else raise."""
        raw = str(path)
        if raw.startswith("\\\\") or raw.lower().startswith("//wsl") or "wsl$" in raw.lower():
            raise WorkspaceViolation(f"UNC/WSL paths are never written: {raw}")
        p = Path(path)
        if not p.is_absolute():
            p = self.repo_root / p
        resolved = _norm(p)
        for bad in self._forbidden:
            if _is_within(resolved, bad):
                raise WorkspaceViolation(f"write refused inside protected root: {p}")
        if not any(_is_within(resolved, ok) for ok in self._allowed):
            raise WorkspaceViolation(f"write refused outside workspace roots: {p}")
        return Path(os.path.realpath(str(p)))

    def is_writable_path(self, path: Path | str) -> bool:
        try:
            self.check(path)
            return True
        except WorkspaceViolation:
            return False

    # ------------------------------------------------------------------ writes
    def mkdir(self, path: Path | str) -> Path:
        target = self.check(path)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def write_bytes(self, path: Path | str, data: bytes, *, overwrite: bool = False) -> Path:
        target = self.check(path)
        if target.exists() and not overwrite:
            raise ImmutableArtifactError(f"refusing to overwrite existing file: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + f".tmp-{os.getpid()}")
        self.check(tmp)
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
        return target

    def write_text(self, path: Path | str, text: str, *, overwrite: bool = False) -> Path:
        # UTF-8 without BOM, LF line endings (HANDOFF §3 item 8).
        return self.write_bytes(path, text.replace("\r\n", "\n").encode("utf-8"), overwrite=overwrite)

    def write_once(self, path: Path | str, data: bytes) -> Path:
        """Write-once semantics for immutable artifacts.

        Identical bytes already present: no-op. Different bytes: raise.
        """
        target = self.check(path)
        if target.exists():
            if target.read_bytes() == data:
                return target
            raise ImmutableArtifactError(f"immutable artifact differs from new content: {target}")
        return self.write_bytes(target, data)

    def write_canonical_json(self, path: Path | str, obj: object, *, once: bool = True) -> Path:
        data = canonical_bytes(obj)
        return self.write_once(path, data) if once else self.write_bytes(path, data, overwrite=True)

    def write_json_pretty(self, path: Path | str, obj: object, *, overwrite: bool = False) -> Path:
        text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        return self.write_text(path, text, overwrite=overwrite)

    def append_line(self, path: Path | str, line: str) -> Path:
        """Append one line to an append-only ledger; never truncates."""
        if "\n" in line or "\r" in line:
            raise ValueError("ledger lines must not contain newlines")
        target = self.check(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        before = target.stat().st_size if target.exists() else 0
        with open(target, "ab") as fh:
            fh.write(line.encode("utf-8") + b"\n")
            fh.flush()
            os.fsync(fh.fileno())
        after = target.stat().st_size
        if after < before:
            raise ImmutableArtifactError(f"append-only ledger shrank: {target}")
        return target

    def copy_file(self, source: Path | str, dest: Path | str, *, expected_sha256: str | None = None) -> Path:
        data = Path(source).read_bytes()
        if expected_sha256 is not None:
            actual = hashlib.sha256(data).hexdigest()
            if actual != expected_sha256:
                raise ImmutableArtifactError(f"source hash mismatch for {source}: {actual}")
        return self.write_once(dest, data)

    def sqlite_connect(self, path: Path | str) -> sqlite3.Connection:
        """Open a writable SQLite database inside the workspace."""
        target = self.check(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(str(target))

    def open_lock_file(self, path: Path | str) -> IO[str]:
        """Open (creating it if needed) a lock file inside the workspace for K's byte-range writer lock
        (`mosaic/adapted/k/kb_writer_lock.py`). The returned handle stays open while the lock is held."""
        target = self.check(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        return open(target, "a+", encoding="utf-8")


def readonly_sqlite(path: Path | str, *, immutable: bool = False,
                    check_same_thread: bool = True) -> sqlite3.Connection:
    """Open an external SQLite database strictly read-only (HANDOFF §2)."""
    uri = "file:" + Path(path).as_posix() + "?mode=ro"
    if immutable:
        uri += "&immutable=1"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=check_same_thread)
    conn.execute("PRAGMA query_only = ON")
    return conn


def default_workspace() -> Workspace:
    return Workspace()
