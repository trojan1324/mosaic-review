"""Tamper replay coverage for fictional retained v4 orphan evidence only."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from mosaic.canonical import canonical_bytes
from mosaic.public_linux_orphan_dependencies import OrphanRepairRefusal, load_plan, verify_retained_orphan_dependency_gate
from test_public_linux_orphan_dependencies import create_retained_orphan_fixture


def _rewrite(root: Path, label: str, stdout: bytes) -> None:
    """Update a receipt and outcome command list, so semantic replay is exercised."""
    output = root / "run" / "gpu-smoke"; (output / f"{label}.stdout.log").write_bytes(stdout)
    receipt_path = output / f"{label}.json"; old = json.loads(receipt_path.read_bytes()); receipt = {**old, "stdout_sha256": sha256(stdout).hexdigest()}; receipt_path.write_bytes(canonical_bytes(receipt))
    outcome_path = output / "outcome.json"; outcome = json.loads(outcome_path.read_bytes())
    replaced = False; commands = []
    for row in outcome["commands"]:
        if not replaced and row == old:
            commands.append(receipt); replaced = True
        else: commands.append(row)
    assert replaced; outcome["commands"] = commands
    if label in {"baseline-metadata", "post-repair-freeze"}:
        gate_path = output / "dependency-gate.json"; gate = json.loads(gate_path.read_bytes())
        gate["baseline_metadata_sha256" if label == "baseline-metadata" else "post_repair_freeze_sha256"] = receipt["stdout_sha256"]
        gate_path.write_bytes(canonical_bytes(gate)); outcome["dependency_gate_sha256"] = sha256(gate_path.read_bytes()).hexdigest()
    if label == "post-pdf-metadata":
        pdf_path = output / "dependency-pdf-stage.json"; pdf = json.loads(pdf_path.read_bytes()); pdf["post_pdf_metadata_sha256"] = receipt["stdout_sha256"]
        pdf_path.write_bytes(canonical_bytes(pdf)); outcome["dependency_pdf_stage_sha256"] = sha256(pdf_path.read_bytes()).hexdigest()
    outcome_path.write_bytes(canonical_bytes(outcome))


@pytest.mark.parametrize("case", ["wrong_uninstall", "false_absence", "shared_telemetry_drift", "unexpected_remaining_version", "post_pdf_conflict", "invalid_pdf_lock", "wrong_import", "malformed_direct_url"])
def test_retained_orphan_replay_refuses_semantic_tampering(tmp_path: Path, case: str) -> None:
    root = tmp_path / "fictional-v4"; create_retained_orphan_fixture(root); output = root / "run" / "gpu-smoke"
    if case == "wrong_uninstall":
        receipt = json.loads((output / "orphan-uninstall.json").read_bytes()); receipt["command"][-1] = "other-package"; (output / "orphan-uninstall.json").write_bytes(canonical_bytes(receipt))
        outcome = json.loads((output / "outcome.json").read_bytes()); outcome["commands"] = [receipt if row.get("command") == ["python", "-m", "pip", "uninstall", "--yes", "opentelemetry-exporter-otlp-common"] else row for row in outcome["commands"]]; (output / "outcome.json").write_bytes(canonical_bytes(outcome)); pattern = "command_semantics"
    elif case == "false_absence": _rewrite(root, "orphan-post-remove-absence", b'{"absent_paths":[]}'); pattern = "absence"
    elif case == "shared_telemetry_drift":
        data = json.loads((output / "orphan-after-files.stdout.log").read_bytes()); data["shared_opentelemetry_files"] = {"opentelemetry-sdk": {"shared.py": "0" * 64}}; _rewrite(root, "orphan-after-files", canonical_bytes(data)); pattern = "shared"
    elif case == "unexpected_remaining_version":
        text = (output / "post-repair-freeze.stdout.log").read_bytes().replace(b"opentelemetry-sdk==1.40.0", b"opentelemetry-sdk==9.99.9"); _rewrite(root, "post-repair-freeze", text)
        data = json.loads((output / "post-repair-metadata.stdout.log").read_bytes()); data["installed"] = [{**row, "metadata": {**row["metadata"], "version": "9.99.9"}} if row["metadata"]["name"] == "opentelemetry-sdk" else row for row in data["installed"]]; _rewrite(root, "post-repair-metadata", canonical_bytes(data)); pattern = "delta|conflict|only_nccl"
    elif case == "post_pdf_conflict":
        data = json.loads((output / "post-pdf-metadata.stdout.log").read_bytes()); data["installed"][0]["metadata"]["requires_dist"] = ["missing-package==1"]; _rewrite(root, "post-pdf-metadata", canonical_bytes(data)); pattern = "graph_not_nccl"
    elif case == "invalid_pdf_lock": (output / "requirements-linux-smoke.lock").write_text("bad lock\n", encoding="utf-8"); pattern = "pdf.*lock|lock.*pdf"
    elif case == "wrong_import": _rewrite(root, "orphan-post-imports", b"wrong-import\n"); pattern = "import"
    else:
        data = json.loads((output / "baseline-metadata.stdout.log").read_bytes()); data["installed"][0]["metadata"]["requires_dist"] = ["bad @ https://example.invalid/unpinned.whl"]; _rewrite(root, "baseline-metadata", canonical_bytes(data)); pattern = "baseline_conflict|not_orphan"
    with pytest.raises(OrphanRepairRefusal, match=pattern): verify_retained_orphan_dependency_gate(root, expected_config_sha256=load_plan().config_sha256)
