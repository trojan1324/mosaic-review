"""Explicit, post-extraction Jev advice for a public candidate snapshot.

This is an optional sidecar boundary.  It cannot discover sources, alter the
snapshot, qualify science, or decide reader visibility.  Literal context is
rebuilt from hash-pinned source units in the completed snapshot; callers never
pass arbitrary state through to the gateway.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

from mosaic.canonical import canonical_bytes
from mosaic.jev.client import JevClient, JevHTTPError, UnknownOutcome, state_sha256
from mosaic.jev.ledger import Budget, HardStop, JevLedger
from mosaic.workspace import Workspace

TASK_VERSION = "mosaic-public-jev-task/v1"
SIDECAR_VERSION = "mosaic-public-jev-advisory/v1"
# Keep the canonical Jev ledger schema's established literal-source category.
# The public scope below adds the stricter snapshot and exact-witness binding.
PUBLIC_PROVENANCE = "native_article_text"
_HEX = set("0123456789abcdef")
_TASK_ID = re.compile(r"^public-[A-Za-z0-9_-]+$")
_PROTECTED_ALIAS = re.compile(r"(?:^|[-_/:\s])(sealed|holdback|calibration|transcript|student|private|e1|p4)(?:$|[-_/:\s])", re.I)
_UNSAFE_PARTS = frozenset({".env", ".git", "sealed", "references", "calibration", "private"})


class PublicJevRefusal(ValueError):
    """The request is not a safely bound public advisory task."""


def _is_hash(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value.lower()) <= _HEX


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _protected(value: object) -> bool:
    """Reject declared source aliases/categories, never hash-like values."""
    text = str(value).strip()
    return text.upper().startswith("KS-") or bool(_PROTECTED_ALIAS.search(text))


def _safe_input(path: Path | str) -> Path:
    candidate = Path(path)
    linked = lambda item: item.is_symlink() or getattr(item, "is_junction", lambda: False)()
    if linked(candidate) or any(linked(parent) for parent in candidate.parents) or any(part.casefold() in _UNSAFE_PARTS or part.casefold().startswith(".env.") for part in candidate.parts):
        raise PublicJevRefusal("public_jev_unsafe_input_path")
    if candidate.name.casefold().endswith(".private.json"):
        raise PublicJevRefusal("public_jev_unsafe_input_path")
    return candidate


def _json(path: Path | str) -> dict[str, Any]:
    path = _safe_input(path)
    try:
        data = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicJevRefusal(f"public_jev_json_unreadable:{path.name}:{type(exc).__name__}") from None
    if not isinstance(data, dict):
        raise PublicJevRefusal("public_jev_json_object_required")
    return data


@dataclass(frozen=True)
class PublicTask:
    task_id: str
    snapshot_path: Path
    snapshot_sha256: str
    source_sha256: str
    pdf_sha256: str
    aliases: tuple[str, ...]
    declared_protected_hashes: tuple[str, ...]
    caller_permission: Mapping[str, Any]
    selections: tuple[Mapping[str, Any], ...]
    definitions: Mapping[str, Any]
    questions: Mapping[str, Any]
    pattern: str
    task_sha256: str


def load_public_task(path: Path | str) -> PublicTask:
    """Validate explicit configuration only; this reads no registry or private state."""
    task_path = _safe_input(path)
    raw = _json(task_path)
    if raw.get("version") != TASK_VERSION:
        raise PublicJevRefusal("public_jev_task_version")
    task_id, snapshot, source = raw.get("task_id"), raw.get("snapshot"), raw.get("source")
    permission = raw.get("caller_permission")
    if not isinstance(task_id, str) or len(task_id) > 120 or not _TASK_ID.fullmatch(task_id):
        raise PublicJevRefusal("public_jev_task_id")
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("path"), str) or not _is_hash(snapshot.get("sha256")):
        raise PublicJevRefusal("public_jev_snapshot_binding")
    if not isinstance(source, dict) or not _is_hash(source.get("source_sha256")) or not _is_hash(source.get("pdf_sha256")):
        raise PublicJevRefusal("public_jev_exact_source_hashes_required")
    aliases, declared = source.get("aliases"), source.get("declared_protected_hashes")
    if not isinstance(aliases, list) or not aliases or not all(isinstance(item, str) and item for item in aliases):
        raise PublicJevRefusal("public_jev_source_aliases_required")
    if not isinstance(declared, list) or not all(_is_hash(item) for item in declared):
        raise PublicJevRefusal("public_jev_declared_protected_hashes_required")
    selected = {str(source["source_sha256"]).lower(), str(source["pdf_sha256"]).lower()}
    if any(_protected(value) for value in aliases) or selected & {str(item).lower() for item in declared}:
        raise PublicJevRefusal("public_jev_declared_protected_source_refused")
    if not isinstance(permission, dict) or permission.get("authorized") is not True or permission.get("scope") != "post_extraction_advisory" or not isinstance(permission.get("declared_by"), str):
        raise PublicJevRefusal("public_jev_explicit_caller_permission_required")
    selections, definitions, questions, pattern = raw.get("source_selections"), raw.get("definitions"), raw.get("questions"), raw.get("pattern")
    if not isinstance(selections, list) or not selections or not all(isinstance(item, dict) for item in selections):
        raise PublicJevRefusal("public_jev_source_selections_required")
    if not isinstance(definitions, dict) or not definitions or not isinstance(questions, dict) or not questions or not isinstance(pattern, str) or not pattern:
        raise PublicJevRefusal("public_jev_definitions_questions_pattern_required")
    snapshot_path = Path(snapshot["path"])
    if not snapshot_path.is_absolute():
        snapshot_path = task_path.parent / snapshot_path
    return PublicTask(task_id, _safe_input(snapshot_path), str(snapshot["sha256"]).lower(),
                      str(source["source_sha256"]).lower(), str(source["pdf_sha256"]).lower(), tuple(aliases),
                      tuple(sorted({str(item).lower() for item in declared})), permission, tuple(selections),
                      definitions, questions, pattern, _digest(raw))


def _snapshot_and_context(task: PublicTask) -> tuple[dict[str, Any], dict[str, Any]]:
    snapshot_path = _safe_input(task.snapshot_path)
    if not snapshot_path.is_file():
        raise PublicJevRefusal("public_jev_completed_snapshot_missing")
    raw = snapshot_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != task.snapshot_sha256:
        raise PublicJevRefusal("public_jev_snapshot_hash_mismatch")
    snapshot = _json(snapshot_path)
    if snapshot.get("validation_status") not in {"candidate_only", "validated_candidate"} or snapshot.get("scientifically_accepted") is not False:
        raise PublicJevRefusal("public_jev_completed_candidate_required")
    source = snapshot.get("source")
    paper = snapshot.get("paper")
    if not isinstance(source, dict) or not isinstance(paper, dict) or source.get("source_sha256") != task.source_sha256 or paper.get("source_pdf_sha256") != task.pdf_sha256:
        raise PublicJevRefusal("public_jev_snapshot_source_binding_mismatch")
    if _digest(source.get("units")) != source.get("units_sha256"):
        raise PublicJevRefusal("public_jev_snapshot_packet_or_units_hash_mismatch")
    packet = snapshot.get("source_packet_candidate")
    if not isinstance(packet, dict) or _digest(packet) != snapshot.get("source_packet_sha256") or _digest(packet) != task.source_sha256:
        raise PublicJevRefusal("public_jev_snapshot_packet_consistency_mismatch")
    if packet.get("units") != source.get("units") or packet.get("source_policy_sha256") != source.get("source_policy_sha256"):
        raise PublicJevRefusal("public_jev_snapshot_packet_consistency_mismatch")
    units = source.get("units")
    if not isinstance(units, list):
        raise PublicJevRefusal("public_jev_snapshot_units_missing")
    if not all(isinstance(unit, dict) and isinstance(unit.get("unit_id"), str) and isinstance(unit.get("text"), str) for unit in units):
        raise PublicJevRefusal("public_jev_snapshot_unit_invalid")
    by_id = {unit["unit_id"]: unit for unit in units}
    if len(by_id) != len(units):
        raise PublicJevRefusal("public_jev_snapshot_unit_duplicate")
    literals: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for selection in task.selections:
        unit_id, start, end = selection.get("unit_id"), selection.get("start"), selection.get("end")
        if not isinstance(unit_id, str) or type(start) is not int or type(end) is not int or unit_id not in by_id:
            raise PublicJevRefusal("public_jev_selection_invalid")
        unit = by_id[unit_id]; text = unit["text"]
        if not 0 <= start < end <= len(text) or (unit_id, start, end) in seen:
            raise PublicJevRefusal("public_jev_selection_span_invalid")
        seen.add((unit_id, start, end)); literal = text[start:end]
        witness = {"unit_id": unit_id, "page": unit.get("page"), "source_locator": unit.get("source_locator"),
                   "start": start, "end": end, "text": literal}
        witness["literal_sha256"] = _digest(witness)
        literals.append(witness)
    context = {"snapshot_sha256": task.snapshot_sha256, "source_sha256": task.source_sha256,
               "pdf_sha256": task.pdf_sha256, "literal_context": literals, "definitions": dict(task.definitions)}
    return snapshot, context


class ExplicitPublicScope:
    """Closed scope for exactly one task; no protected-set lookup is implied."""
    def __init__(self, task: PublicTask): self.task = task
    def check(self, work_id: str, source_sha256s: list[str], text_provenance: str) -> None:
        if work_id != self.task.task_id or text_provenance != PUBLIC_PROVENANCE:
            raise PublicJevRefusal("public_jev_scope_binding_mismatch")
        if tuple(sorted(set(source_sha256s))) != tuple(sorted({self.task.source_sha256, self.task.pdf_sha256})):
            raise PublicJevRefusal("public_jev_source_hash_scope_mismatch")


def make_public_client(task: PublicTask, *, ledger_path: Path | str, budget: Budget,
                       workspace: Workspace | None = None, **kwargs: Any) -> JevClient:
    """Opt-in construction; callers can instead inject the canonical account client."""
    return JevClient(ledger=JevLedger(Path(ledger_path), workspace=workspace), scope=ExplicitPublicScope(task),
                     budget=budget, **kwargs)


def _sidecar(task: PublicTask, context: Mapping[str, Any], response: Mapping[str, Any]) -> dict[str, Any]:
    return {"version": SIDECAR_VERSION, "kind": "post_extraction_advisory_only", "task_id": task.task_id,
            "task_sha256": task.task_sha256, "snapshot_sha256": task.snapshot_sha256,
            "source_sha256": task.source_sha256, "pdf_sha256": task.pdf_sha256,
            "literal_context_sha256": _digest(context["literal_context"]), "prompt_state_sha256": state_sha256(context, dict(task.questions)),
            "caller_permission": dict(task.caller_permission), "caller_permission_is_not_independent_clearance": True,
            "advice": dict(response.get("answers", {})), "recorded": bool(response.get("recorded", False)),
            "generation_id": response.get("generation_id"), "scientifically_accepted": False, "qualified": False,
            "may_not_gate_visibility": True, "may_not_modify_extraction": True}


def assess_public_task(task_path: Path | str, output_dir: Path | str, *, client: JevClient,
                       workspace: Workspace | None = None, reuse_recorded: bool = False) -> dict[str, Any]:
    """The sole explicit live-call entrypoint; it never modifies the snapshot."""
    task = load_public_task(task_path)
    _, context = _snapshot_and_context(task)
    scoped = workspace or Workspace()
    destination = _safe_input(Path(output_dir).absolute())
    scoped.check(destination)
    if destination.exists() and not destination.is_dir():
        raise PublicJevRefusal("public_jev_output_directory_required")
    path = destination / f"public-jev-advisory-{task.task_id}.json"
    if path.exists():
        if not reuse_recorded:
            raise PublicJevRefusal("public_jev_existing_sidecar_requires_replay")
        sidecar_hash = hashlib.sha256(_safe_input(path).read_bytes()).hexdigest()
        sidecar = replay_public_advisory(task_path, path, expected_sidecar_sha256=sidecar_hash)
        return {"sidecar_path": str(path), "sidecar_sha256": sidecar_hash, "sidecar": sidecar}
    expected = state_sha256(context, dict(task.questions))
    response = client.system_one(work_id=task.task_id, state=context, questions=dict(task.questions),
                                 expected_state_sha256=expected, source_sha256s=[task.source_sha256, task.pdf_sha256],
                                 text_provenance=PUBLIC_PROVENANCE, pattern=task.pattern, purpose="advisory", reuse_recorded=reuse_recorded)
    sidecar = _sidecar(task, context, response)
    path = scoped.write_canonical_json(path, sidecar)
    return {"sidecar_path": str(path), "sidecar_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "sidecar": sidecar}


def replay_public_advisory(task_path: Path | str, sidecar_path: Path | str, *, expected_sidecar_sha256: str) -> dict[str, Any]:
    """Credential-free, ledger-free, network-free replay."""
    task = load_public_task(task_path); _, context = _snapshot_and_context(task); path = _safe_input(sidecar_path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
    if not _is_hash(expected_sidecar_sha256) or actual != expected_sidecar_sha256.lower():
        raise PublicJevRefusal("public_jev_sidecar_hash_mismatch")
    sidecar = _json(path)
    if sidecar.get("version") != SIDECAR_VERSION or sidecar.get("task_sha256") != task.task_sha256 or sidecar.get("snapshot_sha256") != task.snapshot_sha256:
        raise PublicJevRefusal("public_jev_sidecar_task_mismatch")
    if sidecar.get("source_sha256") != task.source_sha256 or sidecar.get("pdf_sha256") != task.pdf_sha256 or sidecar.get("literal_context_sha256") != _digest(context["literal_context"]):
        raise PublicJevRefusal("public_jev_sidecar_binding_mismatch")
    if (sidecar.get("scientifically_accepted") is not False or sidecar.get("qualified") is not False
            or sidecar.get("may_not_gate_visibility") is not True or sidecar.get("may_not_modify_extraction") is not True
            or sidecar.get("caller_permission_is_not_independent_clearance") is not True
            or sidecar.get("prompt_state_sha256") != state_sha256(context, dict(task.questions))):
        raise PublicJevRefusal("public_jev_sidecar_invariant_mismatch")
    return sidecar


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explicit public post-extraction Jev advice")
    subs = parser.add_subparsers(dest="command", required=True)
    assess = subs.add_parser("assess"); assess.add_argument("--task", required=True); assess.add_argument("--out", required=True); assess.add_argument("--ledger", required=True); assess.add_argument("--usd-cap"); assess.add_argument("--request-cap", type=int); assess.add_argument("--send", action="store_true")
    replay = subs.add_parser("replay"); replay.add_argument("--task", required=True); replay.add_argument("--sidecar", required=True); replay.add_argument("--sidecar-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "replay":
            print(json.dumps(replay_public_advisory(args.task, args.sidecar, expected_sidecar_sha256=args.sidecar_sha256), sort_keys=True)); return 0
        if not args.send: raise PublicJevRefusal("public_jev_send_flag_required")
        try:
            usd_cap = Decimal(args.usd_cap) if args.usd_cap is not None else None
        except (InvalidOperation, ValueError):
            usd_cap = None
        if args.request_cap is None or usd_cap is None or not usd_cap.is_finite() or usd_cap < 0 or args.request_cap < 1:
            raise PublicJevRefusal("public_jev_explicit_budget_required")
        task = load_public_task(args.task)
        result = assess_public_task(args.task, args.out, client=make_public_client(task, ledger_path=args.ledger, budget=Budget(usd_cap, args.request_cap)))
        print(json.dumps({key: result[key] for key in ("sidecar_path", "sidecar_sha256")}, sort_keys=True)); return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        code = getattr(exc, "code", None)
        text = str(exc)
        safe = text if re.fullmatch(r"public_jev_[a-z0-9_]+", text) else None
        print(f"public_jev_refused:{code or safe or type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
