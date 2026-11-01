"""Minimal request-authorization interface shared by vLLM callers.

The public local-runtime scope and the private screening scope both implement
this contract.  Keeping it free of roster and sealed-set imports lets an
explicit public scope use the common client without loading private policy.
"""

from __future__ import annotations

from typing import Protocol


class VLLMScopeContract(Protocol):
    """Authorize one source-bound vLLM request or raise a hard stop."""

    def check(self, work_id: str, source_sha256s: list[str], text_provenance: str, *, remote: bool,
              rights: list[str] | None = None) -> None: ...
