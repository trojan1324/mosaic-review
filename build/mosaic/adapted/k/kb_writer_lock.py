"""K's single-writer lock for SQLite databases (adapted from K/tools/kb_writer_lock.py and portable_file_lock.py).

Kept from K (REUSE_LEDGER R002, R003): the lock file sits beside the database;
byte 0 is the locked byte and the holder record (one JSON line) starts at
byte 1, so a refused writer can name the holder; acquisition never blocks;
a held lock surfaces as `KBWriterLockHeld`, a SystemExit with code 3 whose
message names the lock path and the holder.

Changed: the lock guards whichever database the caller names (the Package->KB
adapter's disposable copy), never K's own `library/kb.db`; its file is opened
through the Workspace, so it cannot be created inside K; the child-process
environment flag is dropped (the adapter spawns no writers); and `release`
is exposed so the dry-run fixtures can hold and free the lock (FINAL v2 §11.2,
audit M-48).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

from mosaic.workspace import Workspace

EXIT_CODE = 3
_HOLDER_OFFSET = 1


class KBWriterLockHeld(SystemExit):
    def __init__(self, lock_path: Path, holder: str):
        super().__init__(EXIT_CODE)
        self.lock_path = lock_path
        self.holder = holder

    def __str__(self) -> str:
        return (
            "REFUSING TO START: another kb.db writer holds the lock.\n"
            f"  lock:   {self.lock_path}\n"
            f"  holder: {self.holder or 'unknown (record unreadable or not yet written)'}"
        )


def lock_path_for(db_path: Path) -> Path:
    return Path(db_path).with_name(Path(db_path).name + ".writer.lock")


def _lock_nonblocking(handle: IO[str]) -> None:
    if os.name == "nt":
        import msvcrt

        handle.flush()
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write("\0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise BlockingIOError(str(exc)) from exc
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(handle: IO[str]) -> None:
    if os.name == "nt":
        import msvcrt

        handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_holder(lock_path: Path) -> str:
    try:
        with open(lock_path, "rb", buffering=0) as raw:
            raw.seek(_HOLDER_OFFSET)
            return raw.read().decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def acquire(db_path: Path, workspace: Workspace, label: str) -> IO[str]:
    """Take the writer lock beside `db_path` or raise KBWriterLockHeld. Keep the handle open while writing."""
    lock_path = lock_path_for(db_path)
    handle = workspace.open_lock_file(lock_path)
    try:
        _lock_nonblocking(handle)
    except BlockingIOError:
        handle.close()
        raise KBWriterLockHeld(lock_path, _read_holder(lock_path)) from None
    handle.seek(0)
    handle.truncate()
    handle.write(" " + json.dumps({"pid": os.getpid(), "label": label,
                                   "acquired_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}) + "\n")
    handle.flush()
    return handle


def release(handle: IO[str]) -> None:
    _unlock(handle)
    handle.close()
