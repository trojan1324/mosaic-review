"""Stage-state machine for the orchestrator (FINAL v2 §11.2; audit M-48, M-49).

States: planned -> admitted -> running -> completed, with alternatives
retryable_failed, blocked, quarantined, canceled, unknown_outcome and
writer_lock_busy. Policy: one primary attempt and at most one retry, and only
from retryable_failed. unknown_outcome is never retried until reconciled
against a receipt; writer_lock_busy is never eligible for the transport retry
(it waits for the lock holder and is re-admitted as a fresh plan).
"""

from __future__ import annotations

STATES = ("planned", "admitted", "running", "completed", "retryable_failed", "blocked", "quarantined",
          "canceled", "unknown_outcome", "writer_lock_busy")
TERMINAL = {"completed", "blocked", "quarantined", "canceled"}
TRANSITIONS = {
    "planned": {"admitted", "canceled", "blocked"},
    "admitted": {"running", "canceled", "blocked", "writer_lock_busy"},
    "running": {"completed", "retryable_failed", "blocked", "quarantined", "canceled", "unknown_outcome", "writer_lock_busy"},
    "retryable_failed": {"admitted", "blocked"},
    "unknown_outcome": {"completed", "retryable_failed", "blocked"},  # only via reconciliation
    "writer_lock_busy": {"planned", "canceled"},
    "completed": set(), "blocked": set(), "quarantined": set(), "canceled": set(),
}
MAX_ATTEMPTS = 2


class IllegalTransition(ValueError):
    pass


def retry_eligible(state: str | None, *, reconciled: bool = False, attempt_number: int = 1) -> bool:
    if state == "retryable_failed":
        return attempt_number < MAX_ATTEMPTS
    if state == "unknown_outcome":
        return False  # reconciliation moves it to completed or retryable_failed first
    return False


def transition(current: str, new: str, *, reconciled: bool = False) -> str:
    if new not in TRANSITIONS.get(current, set()):
        raise IllegalTransition(f"{current} -> {new} is not allowed")
    if current == "unknown_outcome" and not reconciled:
        raise IllegalTransition("unknown_outcome may only leave its state after reconciliation against a receipt")
    return new
