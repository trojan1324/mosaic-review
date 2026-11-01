"""Minimal authorization interface shared by private and public Jev callers.

This module deliberately carries no work roster, sealed-set reader, or legacy
root.  ``JevClient`` needs only this fail-closed check; the private
``DocumentScope`` and the public explicit scope each implement it.
"""

from __future__ import annotations

from typing import Protocol


class JevScope(Protocol):
    """Authorize one source-bound Jev request or raise a hard stop."""

    def check(self, work_id: str, source_sha256s: list[str], text_provenance: str) -> None: ...
