"""Append-only Jev call ledger and cumulative caps (hard stops 2 and 3).

Every call writes an `intent` row before anything is sent and a `result` row
after the response. "Requests so far" is the number of intent rows ever
written; "spent so far" is the sum of `gateway_cost_usd` over result rows,
plus the pre-send estimate for any intent whose result is missing or whose
result is `unknown_outcome` (billing status unconfirmed). Both are
recomputed from the file on every check, never kept in a process counter.
A malformed row stops everything rather than being skipped.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from mosaic import REPO_ROOT
from mosaic.canonical import canonical_bytes
from mosaic.package import utc_now
from mosaic.validation import registry
from mosaic.workspace import Workspace

LEDGER_PATH = REPO_ROOT / "ledgers" / "jev_calls.jsonl"
BUDGET_PATH = REPO_ROOT / "config" / "budget.json"


def _validation_fingerprint() -> str:
    """Refuse hot validation edits rather than trusting stale registry caches."""
    digest = hashlib.sha256(registry.schemas_digest().encode("ascii"))
    for name in ("registry.py", "objects.py"):
        digest.update((REPO_ROOT / "mosaic" / "validation" / name).read_bytes())
    return digest.hexdigest()


_PROCESS_VALIDATION_FINGERPRINT = _validation_fingerprint()


class HardStop(RuntimeError):
    def __init__(self, stop: int, code: str, message: str):
        super().__init__(f"hard stop {stop} ({code}): {message}")
        self.stop, self.code = stop, code


@dataclass
class Totals:
    requests: int
    spent_usd: Decimal
    unresolved: list[str] = field(default_factory=list)
    first_nonzero_cost: dict[str, Any] | None = None


@dataclass
class Budget:
    usd_cap: Decimal
    request_cap: int

    @classmethod
    def load(cls, path: Path = BUDGET_PATH) -> "Budget":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(Decimal(str(data["usd_cap"])), int(data["request_cap"]))


class JevLedger:
    def __init__(self, path: Path = LEDGER_PATH, workspace: Workspace | None = None):
        self.path = Path(path)
        self.workspace = workspace or Workspace()
        # Cache only successful validation of exact line text. Parsed rows and
        # accounting totals are always fresh; changed bytes cannot hit this set.
        self._validated_lines: set[str] = set()

    def _check_validation_environment(self) -> None:
        if _validation_fingerprint() != _PROCESS_VALIDATION_FINGERPRINT:
            raise HardStop(3, "ledger_validation_environment_changed",
                           "schemas or validator code changed; start a fresh process before sending")

    def rows(self) -> list[dict[str, Any]]:
        self._check_validation_environment()
        if not self.path.exists():
            return []
        out = []
        try:
            text = self.path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise HardStop(3, "ledger_unreadable", "ledger is not UTF-8; nothing is sent") from None
        for n, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HardStop(3, "ledger_unreadable", f"line {n} is not JSON ({exc.msg}); nothing is sent") from None
            if line not in self._validated_lines:
                errors = registry.validate("jev-call", row)
                if errors:
                    raise HardStop(3, "ledger_row_invalid", f"line {n}: {errors[:2]}")
                self._validated_lines.add(line)
            out.append(row)
        return out

    def totals(self) -> Totals:
        rows = self.rows()
        intents = [r for r in rows if r["row_kind"] == "intent"]
        results = {(r["idempotency_key"], r.get("attempt", 1)): r for r in rows if r["row_kind"] == "result"}
        spent = Decimal("0")
        unresolved = []
        first = None
        for r in rows:
            if r["row_kind"] == "result" and r.get("gateway_cost_usd"):
                cost = Decimal(r["gateway_cost_usd"])
                spent += cost
                if cost > 0 and first is None:
                    first = r
        for i in intents:
            result = results.get((i["idempotency_key"], i.get("attempt", 1)))
            # No result at all, or a result whose billing status is itself unknown: its cost is exactly
            # as uncertain as an unresolved intent, so the pre-send estimate counts toward spend until reconciled.
            if result is None or result.get("outcome") == "unknown_outcome":
                unresolved.append(i["idempotency_key"])
                spent += Decimal(i.get("estimated_cost_usd", "0"))
        return Totals(len(intents), spent, unresolved, first)

    def attempts(self, key: str) -> list[dict[str, Any]]:
        return [r for r in self.rows() if r["idempotency_key"] == key]

    def append(self, row: dict[str, Any]) -> None:
        self._check_validation_environment()
        row = {**row, "timestamp_utc": row.get("timestamp_utc") or utc_now()}
        errors = registry.validate("jev-call", row)
        if errors:
            raise HardStop(2, "ledger_row_invalid", f"refusing to write an invalid row: {errors[:2]}")
        try:
            self.workspace.append_line(self.path, canonical_bytes(row).decode("utf-8"))
        except Exception as exc:  # any failure to persist stops the call
            raise HardStop(2, "ledger_unwritable", f"{type(exc).__name__}: {exc}") from None

    def reconcile(self) -> list[str]:
        """Close intents left without a result (a crash between send and receipt) as unknown_outcome.
        Assumes one process uses the ledger at a time."""
        rows = self.rows()
        done = {(r["idempotency_key"], r.get("attempt", 1)) for r in rows if r["row_kind"] == "result"}
        closed = []
        for i in rows:
            k = (i["idempotency_key"], i.get("attempt", 1))
            if i["row_kind"] == "intent" and k not in done:
                self.append({k2: i[k2] for k2 in ("idempotency_key", "route", "request_sha256", "prompt_state_sha256",
                                                   "target_work_id", "attempt") if k2 in i}
                            | {"row_kind": "result", "outcome": "unknown_outcome", "zero_data_retention": True,
                               "detail": "no result row at startup; outcome unknown; never retried"})
                closed.append(i["idempotency_key"])
        return closed
