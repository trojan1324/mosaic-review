"""Mock-only contract tests for the public post-extraction Jev sidecar."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from mosaic.canonical import canonical_bytes
from mosaic.jev.client import TransportError
from mosaic.jev.ledger import Budget, HardStop
from mosaic.public_capture import capture_public
from mosaic.public_demo import demo_pdf_bytes
from mosaic.public_jev import (PublicJevRefusal, assess_public_task, load_public_task,
                               main, make_public_client, replay_public_advisory)
from mosaic.public_runtime import fixture_replay_for_demo, run_extraction
from mosaic.workspace import Workspace

pytestmark = pytest.mark.evidence_mock


def test_shared_client_import_does_not_require_private_roots():
    command = "import mosaic; mosaic.LEGACY_ROOTS = {}; import mosaic.public_corpus; import mosaic.public_jev; print('public-only')"
    process = subprocess.run([sys.executable, "-c", command], cwd=Path(__file__).parents[1],
                             capture_output=True, text=True, check=False)
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == "public-only"

class Transport:
    def __init__(self, *, unknown: bool = False): self.calls = []; self.unknown = unknown
    def request(self, method, url, headers, body, timeout):
        self.calls.append((method, url, headers, json.loads(body)))
        if self.unknown: raise TransportError("sent then disconnected", sent=True)
        return 200, {}, canonical_bytes({"model": "fixture", "answers": {"classification": "advisory"}, "usage": {}, "provider_metadata": {"gateway": {}}})


def fixture(tmp_path: Path) -> tuple[Workspace, Path, dict, str]:
    ws = Workspace(repo_root=tmp_path, temp_root=tmp_path)
    pdf_hash = "2" * 64
    units = [{"unit_id": "u1", "page": 1, "source_locator": "article:1", "text": "Observed fixture result is positive."}]
    packet = {"units": units, "source_policy_sha256": "3" * 64}
    source_hash = hashlib.sha256(canonical_bytes(packet)).hexdigest()
    snap = {"version": "mosaic-public-snapshot/v1", "validation_status": "candidate_only", "scientifically_accepted": False,
            "paper": {"source_pdf_sha256": pdf_hash}, "source": {"source_sha256": source_hash, "units": units,
            "units_sha256": hashlib.sha256(canonical_bytes(units)).hexdigest(), "source_policy_sha256": "3" * 64},
            "source_packet_candidate": packet, "source_packet_sha256": source_hash}
    snapshot = ws.write_canonical_json(tmp_path / "snapshot.json", snap)
    task = {"version": "mosaic-public-jev-task/v1", "task_id": "public-fixture-1",
            "snapshot": {"path": str(snapshot), "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest()},
            "source": {"source_sha256": source_hash, "pdf_sha256": pdf_hash, "aliases": ["public-fixture"], "declared_protected_hashes": []},
            "caller_permission": {"authorized": True, "scope": "post_extraction_advisory", "declared_by": "fixture owner"},
            "source_selections": [{"unit_id": "u1", "start": 0, "end": 23}],
            "definitions": {"classification": "fixed public advisory label"},
            "questions": {"classification": {"type": "single_label", "labels": ["advisory"]}}, "pattern": "public-fixture-v1"}
    task_path = ws.write_canonical_json(tmp_path / "task.json", task)
    return ws, task_path, task, source_hash


def client(task_path: Path, ws: Workspace, tmp_path: Path, transport: Transport, *, budget: Budget | None = None):
    return make_public_client(load_public_task(task_path), ledger_path=tmp_path / "installation-ledger.jsonl", workspace=ws,
                              budget=budget or Budget(Decimal("10"), 10), transport=transport, api_key="test-key", base_url="https://fixture.invalid")


def test_assessment_rebuilds_snapshot_literal_and_zdr_sidecar(tmp_path: Path):
    ws, task_path, _, _ = fixture(tmp_path); transport = Transport()
    result = assess_public_task(task_path, tmp_path / "sidecars", client=client(task_path, ws, tmp_path, transport), workspace=ws)
    assert result["sidecar"]["scientifically_accepted"] is False
    assert result["sidecar"]["caller_permission_is_not_independent_clearance"] is True
    sent = transport.calls[0][3]
    assert sent["providerOptions"]["gateway"]["zeroDataRetention"] is True
    witness = sent["state"]["literal_context"][0]
    assert {key: witness[key] for key in ("unit_id", "page", "source_locator", "start", "end", "text")} == {"unit_id": "u1", "page": 1, "source_locator": "article:1", "start": 0, "end": 23, "text": "Observed fixture result"}
    assert len(witness["literal_sha256"]) == 64
    replay = replay_public_advisory(task_path, result["sidecar_path"], expected_sidecar_sha256=result["sidecar_sha256"])
    assert replay["advice"] == {"classification": "advisory"}


def test_refuses_unbound_selection_and_declared_protected_alias_or_hash(tmp_path: Path):
    ws, task_path, task, source_hash = fixture(tmp_path)
    bad = dict(task); bad["source_selections"] = [{"unit_id": "not-present", "start": 0, "end": 1}]
    bad_path = ws.write_canonical_json(tmp_path / "bad-selection.json", bad)
    with pytest.raises(PublicJevRefusal, match="selection_invalid"):
        assess_public_task(bad_path, tmp_path / "out", client=client(bad_path, ws, tmp_path, Transport()), workspace=ws)
    protected = dict(task); protected["source"] = {**task["source"], "aliases": ["calibration-paper"]}
    protected_path = ws.write_canonical_json(tmp_path / "protected.json", protected)
    with pytest.raises(PublicJevRefusal, match="declared_protected"):
        load_public_task(protected_path)
    hash_declared = dict(task); hash_declared["source"] = {**task["source"], "declared_protected_hashes": [source_hash]}
    hash_path = ws.write_canonical_json(tmp_path / "protected-hash.json", hash_declared)
    with pytest.raises(PublicJevRefusal, match="declared_protected"):
        load_public_task(hash_path)


def test_budget_duplicate_and_unknown_outcome_preserve_client_stops(tmp_path: Path):
    ws, task_path, _, _ = fixture(tmp_path)
    budget_client = client(task_path, ws, tmp_path, Transport(), budget=Budget(Decimal("10"), 0))
    with pytest.raises(HardStop, match="request_cap"):
        assess_public_task(task_path, tmp_path / "budget", client=budget_client, workspace=ws)
    transport = Transport(); active = client(task_path, ws, tmp_path, transport)
    assess_public_task(task_path, tmp_path / "first", client=active, workspace=ws)
    with pytest.raises(HardStop, match="idempotency_key_already_sent"):
        assess_public_task(task_path, tmp_path / "second", client=active, workspace=ws)
    ws2, task2, _, _ = fixture(tmp_path / "unknown")
    with pytest.raises(Exception, match="not retried"):
        assess_public_task(task2, tmp_path / "unknown/out", client=client(task2, ws2, tmp_path / "unknown", Transport(unknown=True)), workspace=ws2)


def test_replay_detects_tamper_without_client_or_network(tmp_path: Path):
    ws, task_path, _, _ = fixture(tmp_path); result = assess_public_task(task_path, tmp_path / "sidecars", client=client(task_path, ws, tmp_path, Transport()), workspace=ws)
    path = Path(result["sidecar_path"]); tampered = json.loads(path.read_bytes()); tampered["advice"] = {"classification": "tampered"}
    path.write_bytes(canonical_bytes(tampered))
    with pytest.raises(PublicJevRefusal, match="sidecar_hash_mismatch"):
        replay_public_advisory(task_path, path, expected_sidecar_sha256=result["sidecar_sha256"])
    path.write_bytes(canonical_bytes({**tampered, "qualified": True}))
    with pytest.raises(PublicJevRefusal, match="sidecar_invariant_mismatch"):
        replay_public_advisory(task_path, path, expected_sidecar_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_cli_assess_requires_explicit_send_before_client_creation(tmp_path: Path):
    _, task_path, _, _ = fixture(tmp_path)
    assert main(["assess", "--task", str(task_path), "--out", str(tmp_path / "out"), "--ledger", str(tmp_path / "ledger.jsonl")]) == 2


def test_invalid_output_refuses_before_send_and_verified_sidecar_reuses_without_send(tmp_path):
    ws, task_path, _, _ = fixture(tmp_path)
    transport = Transport()
    active = client(task_path, ws, tmp_path, transport)
    occupied = tmp_path / "occupied"
    occupied.write_bytes(b"not a directory")
    with pytest.raises(PublicJevRefusal, match="output_directory_required"):
        assess_public_task(task_path, occupied, client=active, workspace=ws)
    assert transport.calls == []
    first = assess_public_task(task_path, tmp_path / "out", client=active, workspace=ws)
    assert len(transport.calls) == 1
    with pytest.raises(PublicJevRefusal, match="requires_replay"):
        assess_public_task(task_path, tmp_path / "out", client=active, workspace=ws)
    assert assess_public_task(task_path, tmp_path / "out", client=active, workspace=ws, reuse_recorded=True) == first
    assert len(transport.calls) == 1


@pytest.mark.parametrize("cap", ["NaN", "Infinity", "-1"])
def test_cli_rejects_nonfinite_or_negative_budget_without_key(tmp_path: Path, cap: str):
    _, task_path, _, _ = fixture(tmp_path)
    assert main(["assess", "--task", str(task_path), "--out", str(tmp_path / "out"), "--ledger", str(tmp_path / "ledger.jsonl"),
                 "--usd-cap", cap, "--request-cap", "1", "--send"]) == 2


def test_module_cli_propagates_refusal_exit_code_without_traceback(tmp_path: Path):
    _, task_path, _, _ = fixture(tmp_path)
    process = subprocess.run([sys.executable, "-m", "mosaic.public_jev", "assess", "--task", str(task_path),
                              "--out", str(tmp_path / "out"), "--ledger", str(tmp_path / "ledger.jsonl")],
                             cwd=Path(__file__).parents[1], capture_output=True, text=True, check=False)
    assert process.returncode == 2
    assert process.stderr.strip() == "public_jev_refused:public_jev_send_flag_required"
    assert "Traceback" not in process.stderr


def test_safe_paths_and_safe_identifier_do_not_use_substring_matching(tmp_path: Path):
    _, task_path, task, _ = fixture(tmp_path)
    changed = {**task, "task_id": "public-e1a"}
    changed_path = task_path.parent / "safe-id.json"; changed_path.write_bytes(canonical_bytes(changed))
    assert load_public_task(changed_path).task_id == "public-e1a"
    with pytest.raises(PublicJevRefusal, match="unsafe_input_path"):
        load_public_task(tmp_path / ".env")
    with pytest.raises(PublicJevRefusal, match="unsafe_input_path"):
        load_public_task(tmp_path / ".env.local")
    traversal = {**task, "task_id": "public-../../escape"}
    traversal_path = task_path.parent / "traversal.json"; traversal_path.write_bytes(canonical_bytes(traversal))
    with pytest.raises(PublicJevRefusal, match="task_id"):
        load_public_task(traversal_path)


def test_actual_capture_report_snapshot_is_required_and_rechecked(tmp_path: Path):
    """Exercise capture -> real fixture driver -> public snapshot -> advisory."""
    ws = Workspace(repo_root=tmp_path, temp_root=tmp_path)
    sources = ws.mkdir(tmp_path / "sources"); pdf = sources / "fictional-build.pdf"; ws.write_once(pdf, demo_pdf_bytes())
    raw = pdf.read_bytes(); pdf_hash = hashlib.sha256(raw).hexdigest()
    row = {"paper_id": "public-" + pdf_hash, "source_pdf_sha256": pdf_hash, "byte_size": len(raw), "locators": [pdf.name],
           "permission": "fictional_fixture", "source_status": "public_unreviewed_source", "scientifically_accepted": False}
    root = tmp_path / "capture"; capture_public(row, pdf.name, sources, root, title="Fictional fixture", workspace=ws)
    read = lambda name: json.loads((root / name).read_bytes())
    packet = {"capture_receipt": read("receipt.json"), "source": read("source.json"), "source_binding": read("source-binding.json"),
              "prepare_upstream_input": read("prepare-upstream-input.json"), "runtime_authorization": {"kind": "fictional_fixture_local_only", "source_pdf_sha256": pdf_hash}}
    profile = json.loads((Path(__file__).parents[1] / "config" / "public_runtime_reference_v1.json").read_bytes())
    runtime = run_extraction(packet, tmp_path / "runtime", workspace=ws, profile=profile, replay=fixture_replay_for_demo(packet))
    snapshot = json.loads(Path(runtime["snapshot_path"]).read_bytes()); source = snapshot["source"]
    assert hashlib.sha256(canonical_bytes(snapshot["source_packet_candidate"])).hexdigest() == snapshot["source_packet_sha256"] == source["source_sha256"]
    assert hashlib.sha256(canonical_bytes(source["units"])).hexdigest() == source["units_sha256"]
    unit = source["units"][0]
    task = {"version": "mosaic-public-jev-task/v1", "task_id": "public-runtime-fixture",
            "snapshot": {"path": runtime["snapshot_path"], "sha256": hashlib.sha256(Path(runtime["snapshot_path"]).read_bytes()).hexdigest()},
            "source": {"source_sha256": source["source_sha256"], "pdf_sha256": pdf_hash, "aliases": ["fictional-fixture"], "declared_protected_hashes": []},
            "caller_permission": {"authorized": True, "scope": "post_extraction_advisory", "declared_by": "fixture owner"},
            "source_selections": [{"unit_id": unit["unit_id"], "start": 0, "end": min(5, len(unit["text"]))}],
            "definitions": {"classification": "fixed"}, "questions": {"classification": {"type": "single_label"}}, "pattern": "runtime-fixture"}
    task_path = ws.write_canonical_json(tmp_path / "runtime-task.json", task)
    result = assess_public_task(task_path, tmp_path / "advice", client=client(task_path, ws, tmp_path, Transport()), workspace=ws)
    assert result["sidecar"]["source_sha256"] == source["source_sha256"]
