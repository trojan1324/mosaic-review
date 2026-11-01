"""Append-only vLLM call ledger and cumulative caps, one file per candidate (VS2, VS3, VS8).

Mirrors `mosaic/jev/ledger.py`: an `intent` row is written before anything is
sent and a `result` row after the response; totals are recomputed from the
file on every check, never kept in a process counter; a malformed row stops
everything rather than being skipped.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mosaic import REPO_ROOT
from mosaic.canonical import canonical_bytes
from mosaic.package import utc_now
from mosaic.validation import registry
from mosaic.workspace import Workspace

LEDGER_ROOT = REPO_ROOT / "ledgers" / "vllm_calls"
BUDGET_PATH = REPO_ROOT / "config" / "vllm_budget.json"
DEVELOPMENT_ALLOWANCE_PATH = REPO_ROOT / "config" / "protocols" / "development_iteration_allowance.json"
_SAFE_NAME = re.compile(r"[^a-zA-Z0-9._\-]+")


class VLLMHardStop(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"vLLM hard stop ({code}): {message}")
        self.code = code


def ledger_path(candidate: str, root: Path = LEDGER_ROOT) -> Path:
    return root / f"{_SAFE_NAME.sub('_', candidate)}.jsonl"


@dataclass
class Totals:
    requests: int
    output_tokens: int
    wall_clock_seconds: float
    unresolved: list[str] = field(default_factory=list)


@dataclass
class ArticleTotals:
    requests: int


@dataclass
class VLLMBudget:
    request_cap: int
    output_token_cap_total: int
    wall_clock_cap_seconds: float
    max_requests_per_article: int
    development_iteration_allowance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = BUDGET_PATH,
             allowance_path: Path = DEVELOPMENT_ALLOWANCE_PATH) -> "VLLMBudget":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(int(data["request_cap"]), int(data["output_token_cap_total"]),
                   float(data["wall_clock_cap_seconds"]), int(data["max_requests_per_article"]),
                   _load_development_allowance(Path(allowance_path)))

    def article_limit(self, candidate: str, work_id: str) -> int:
        """Return the narrowly authorized per-article limit, or the base limit.

        This is deliberately a two-key exception.  Historical ledger intents
        are still counted by ``totals_for_article``; this method only chooses
        the ceiling against which that count is checked.
        """
        allowance = self.development_iteration_allowance
        if (allowance.get("status") == "authorized"
                and candidate in allowance.get("candidates", [])
                and work_id in allowance.get("work_ids", [])):
            return int(allowance["max_requests_per_article"])
        return self.max_requests_per_article


def _load_development_allowance(path: Path) -> dict[str, Any]:
    """Load the authorized exception without silently widening a malformed file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {"status", "max_requests_per_article", "work_ids", "candidates",
                "unchanged_per_candidate_request_cap", "unchanged_output_token_cap_total",
                "unchanged_wall_clock_cap_seconds", "cumulative_rental_ceiling_usd"}
    if not required.issubset(data) or data.get("status") != "authorized":
        raise ValueError("development_iteration_allowance_invalid")
    if (not isinstance(data["max_requests_per_article"], int)
            or data["max_requests_per_article"] < 1
            or not all(isinstance(value, str) and value for value in data["work_ids"])
            or not all(isinstance(value, str) and value for value in data["candidates"])):
        raise ValueError("development_iteration_allowance_invalid")
    return data


class VLLMLedger:
    def __init__(self, candidate: str, root: Path = LEDGER_ROOT, workspace: Workspace | None = None):
        self.candidate = candidate
        self.path = ledger_path(candidate, root)
        self.workspace = workspace or Workspace()

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        for n, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise VLLMHardStop("ledger_unreadable", f"line {n} is not JSON ({exc.msg}); nothing is sent") from None
            errors = registry.validate("vllm-call", row)
            if errors:
                raise VLLMHardStop("ledger_row_invalid", f"line {n}: {errors[:2]}")
            out.append(row)
        return out

    def totals(self) -> Totals:
        rows = self.rows()
        intents = [r for r in rows if r["row_kind"] == "intent"]
        results = {(r["idempotency_key"], r.get("attempt", 1)): r for r in rows if r["row_kind"] == "result"}
        output_tokens = 0.0
        seconds = 0.0
        for r in rows:
            if r["row_kind"] == "result":
                output_tokens += r.get("output_tokens") or 0
                seconds += (r.get("latency_ms") or 0) / 1000.0
        unresolved = [i["idempotency_key"] for i in intents if (i["idempotency_key"], i.get("attempt", 1)) not in results]
        return Totals(len(intents), int(output_tokens), seconds, unresolved)

    def totals_for_article(self, work_id: str) -> ArticleTotals:
        return ArticleTotals(sum(1 for r in self.rows() if r["row_kind"] == "intent" and r["target_work_id"] == work_id))

    def attempts(self, key: str) -> list[dict[str, Any]]:
        return [r for r in self.rows() if r["idempotency_key"] == key]

    def append(self, row: dict[str, Any]) -> None:
        row = {**row, "timestamp_utc": row.get("timestamp_utc") or utc_now()}
        errors = registry.validate("vllm-call", row)
        if errors:
            raise VLLMHardStop("ledger_row_invalid", f"refusing to write an invalid row: {errors[:2]}")
        try:
            self.workspace.append_line(self.path, canonical_bytes(row).decode("utf-8"))
        except Exception as exc:  # any failure to persist stops the call
            raise VLLMHardStop("ledger_unwritable", f"{type(exc).__name__}: {exc}") from None

    def reconcile(self) -> list[str]:
        """Close intents left without a result (a crash between send and receipt) as unknown_outcome.
        Assumes one process uses this candidate's ledger at a time."""
        rows = self.rows()
        done = {(r["idempotency_key"], r.get("attempt", 1)) for r in rows if r["row_kind"] == "result"}
        closed = []
        for i in rows:
            k = (i["idempotency_key"], i.get("attempt", 1))
            if i["row_kind"] == "intent" and k not in done:
                self.append({k2: i[k2] for k2 in ("idempotency_key", "route", "request_sha256", "packet_sha256",
                                                   "target_work_id", "attempt", "batch_id") if k2 in i}
                            | {"row_kind": "result", "outcome": "unknown_outcome",
                               "detail": "no result row at startup; outcome unknown; never retried"})
                closed.append(i["idempotency_key"])
        return closed
