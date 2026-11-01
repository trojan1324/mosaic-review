"""Candidate generation with a recorded backend (HANDOFF §6.5; FINAL v2 §6.3).

Every candidate carries the backend that produced it and an evidence label:
`fixture` replays recorded real model outputs from Q ("replayed historical
output", never a fresh run); `gateway_jev` is the §6.6 Jev client in its
advisory role ("jev advisory signal"); `real_provider` is an interface that
refuses to run (AUTH-2: no other paid provider is authorized).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mosaic.package import actor

EVIDENCE_LABELS = {"fixture": "replayed historical output", "gateway_jev": "jev advisory signal",
                   "real_provider": "fresh provider output", "deterministic": "deterministic extraction",
                   "vllm": "live vllm candidate output"}


class BackendRefused(RuntimeError):
    """A backend declined to run; the message names the missing authorization."""


@dataclass
class CandidateSet:
    backend: str
    evidence_label: str
    work_id: str
    source_file: str | None
    source_sha256: str | None
    payload: dict[str, Any]
    created_by: dict[str, str] = field(default_factory=dict)
    batch_id: str | None = None

    def origin(self, pointer: str, claim_text: str | None = None) -> dict[str, Any]:
        out = {"backend": self.backend, "evidence_label": self.evidence_label}
        if self.source_file:
            out["source_file"] = self.source_file
        if self.source_sha256:
            out["source_sha256"] = self.source_sha256
        out["source_pointer"] = pointer
        if claim_text:
            out["claim_text"] = claim_text
        return out


def load_pinned_json(path: Path, expected_sha256: str) -> tuple[bytes, dict[str, Any]]:
    import json

    data = Path(path).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise BackendRefused(f"{path}: sha256 {actual} differs from the pinned {expected_sha256}; refusing to replay")
    return data, json.loads(data.decode("utf-8"))


def fixture_actor(module: str) -> dict[str, str]:
    return actor("llm_fixture_replay", module)
