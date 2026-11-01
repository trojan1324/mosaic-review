"""CPU-only contracts for the fail-closed Linux dependency gate."""
from pathlib import Path
import json

import pytest

from mosaic.public_linux_dependencies import (BASE, NCCL, OTEL, DependencyRefusal,
    load_plan, parse_freeze, run_dependency_gate, validate_pdf_stage)
from mosaic.workspace import Workspace


def _freeze(values):
    return ("\n".join(f"{k}=={v}" for k, v in sorted(values.items())) + "\n").encode()


def _runner(output, baseline, post, checks, calls):
    def run(command, label, timeout, require_success=True, **_):
        calls.append((label, command))
        if label == "baseline-freeze": out = _freeze(baseline)
        elif label == "post-repair-freeze": out = _freeze(post)
        elif label == "post-pdf-freeze": out = _freeze(post)
        elif label.endswith("metadata"):
            values = baseline if label.startswith("baseline") else post
            rows = [{"metadata": {"name": name, "version": version,
                    "requires_dist": ['nvidia-nccl-cu13==2.29.7; platform_system == "Linux"'] if name == "torch" else []}}
                    for name, version in values.items()]
            out = json.dumps({"installed": rows}).encode()
        elif label.endswith("check"): out = checks.pop(0)
        else: out = b"otel-imports-ok\n"
        (output / f"{label}.stdout.log").write_bytes(out)
        (output / f"{label}.stderr.log").write_bytes(b"")
        return {"returncode": 1 if label.endswith("check") and out else 0}
    return run


def _baseline():
    return {**BASE, "torch": "2.13.0+cu130", "nvidia-nccl-cu13": "2.30.7"}


def create_retained_dependency_fixture(root: Path) -> Path:
    """Build an archive-replay-shaped, fictional dependency receipt tree.

    The helper calls the real gate functions; only subprocess output is mocked.
    ``root`` is the public-root and the returned path is ``run/gpu-smoke``.
    """
    from hashlib import sha256
    import mosaic.public_linux_smoke as smoke

    output = root / "run" / "gpu-smoke"; output.mkdir(parents=True)
    workspace = Workspace(repo_root=root, temp_root=root)
    plan, baseline = load_plan(), _baseline()
    repaired = {**baseline, **plan.versions}
    post_pdf = {**repaired, "pymupdf": smoke.PDF_WHEEL_URL + "#sha256=" + smoke.PDF_WHEEL_SHA}
    commands = []
    def metadata(values):
        return json.dumps({"installed": [{"metadata": {"name": name, "version": version,
            "requires_dist": ['nvidia-nccl-cu13==2.29.7; platform_system == "Linux"'] if name == "torch" else []}}
            for name, version in values.items()]}).encode()
    def run(command, label, timeout, require_success=True, **_):
        if label == "baseline-freeze": stdout = _freeze(baseline)
        elif label == "post-repair-freeze": stdout = _freeze(repaired)
        elif label in {"post-pdf-freeze", "installed"}: stdout = _freeze(post_pdf)
        elif label == "baseline-metadata": stdout = metadata(baseline)
        elif label == "post-repair-metadata": stdout = metadata(repaired)
        elif label == "baseline-check": stdout = f"{OTEL}\n{NCCL}\n".encode()
        elif label.endswith("check"): stdout = f"{NCCL}\n".encode()
        elif label == "post-repair-imports": stdout = b"otel-imports-ok\n"
        else: stdout = b""
        code = 1 if label.endswith("check") else 0
        (output / f"{label}.stdout.log").write_bytes(stdout)
        (output / f"{label}.stderr.log").write_bytes(b"")
        receipt = {"command": list(command), "returncode": code, "timed_out": False,
            "stdout_sha256": sha256(stdout).hexdigest(), "stderr_sha256": sha256(b"").hexdigest()}
        workspace.write_canonical_json(output / f"{label}.json", receipt); commands.append(receipt)
        return receipt
    gate = run_dependency_gate(run=run, python="python", output=output, workspace=workspace, image=plan.image, plan=plan)
    lock = f"pymupdf @ {smoke.PDF_WHEEL_URL} --hash=sha256:{smoke.PDF_WHEEL_SHA}\n".encode()
    workspace.write_once(output / "requirements-linux-smoke.lock", lock)
    run(["python", "-m", "pip", "install", "--no-deps", "--require-hashes", "-r", str(output / "requirements-linux-smoke.lock")], "install", 1)
    run(["python", "-m", "pip", "freeze", "--all"], "installed", 1)
    pdf = validate_pdf_stage(run=run, python="python", output=output, workspace=workspace, before=repaired, plan=plan)
    workspace.write_canonical_json(output / "outcome.json", {"commands": commands,
        "dependency_gate_sha256": sha256((output / "dependency-gate.json").read_bytes()).hexdigest(),
        "dependency_config_sha256": plan.config_sha256,
        "dependency_pdf_stage_sha256": sha256((output / "dependency-pdf-stage.json").read_bytes()).hexdigest(),
        "dependency_gate": gate, "dependency_pdf": pdf})
    return output


def test_exact_repair_records_baseline_before_mutation(tmp_path):
    output = tmp_path / "out"; output.mkdir(); plan = load_plan(); baseline = _baseline()
    post = {**baseline, **plan.versions}; calls = []
    run = _runner(output, baseline, post, [f"{OTEL}\n{NCCL}\n".encode(), f"{NCCL}\n".encode()], calls)
    result = run_dependency_gate(run=run, python="python", output=output, workspace=Workspace(repo_root=tmp_path, temp_root=tmp_path), image=plan.image, plan=plan)
    labels = [x[0] for x in calls]
    assert result["baseline_mode"] == "repair_required"
    assert labels.index("baseline-freeze") < labels.index("otel-repair-install") < labels.index("post-repair-check")
    assert (output / "otel-repair.lock").read_bytes() == plan.lock_bytes()


def test_unknown_conflict_refuses_before_install(tmp_path):
    output = tmp_path / "out"; output.mkdir(); plan = load_plan(); calls = []
    run = _runner(output, _baseline(), _baseline(), [b"unexplained package conflict\n"], calls)
    with pytest.raises(DependencyRefusal, match="unknown_conflict"):
        run_dependency_gate(run=run, python="python", output=output, workspace=Workspace(repo_root=tmp_path, temp_root=tmp_path), image=plan.image, plan=plan)
    assert "otel-repair-install" not in [x[0] for x in calls]


def test_post_pdf_rejects_unrelated_drift(tmp_path):
    output = tmp_path / "out"; output.mkdir(); plan = load_plan(); calls = []
    before = {**_baseline(), **plan.versions}
    after = {**before, "certifi": "different"}
    run = _runner(output, before, after, [f"{NCCL}\n".encode()], calls)
    with pytest.raises(DependencyRefusal, match="pdf_drift"):
        validate_pdf_stage(run=run, python="python", output=output, workspace=Workspace(repo_root=tmp_path, temp_root=tmp_path), before=before, plan=plan)


def test_freeze_rejects_malformed_rows():
    with pytest.raises(DependencyRefusal, match="freeze_invalid"):
        parse_freeze(b"not-a-distribution\n")


def test_retained_fixture_uses_real_gate_and_pdf_stage(tmp_path):
    output = create_retained_dependency_fixture(tmp_path)
    import mosaic.public_linux_smoke as smoke
    gate = json.loads((output / "dependency-gate.json").read_text(encoding="utf-8"))
    outcome = json.loads((output / "outcome.json").read_text(encoding="utf-8"))
    verified = smoke.verify_retained_dependency_gate(tmp_path, expected_config_sha256=load_plan().config_sha256)
    assert gate["baseline_mode"] == "repair_required"
    assert (output / "dependency-pdf-stage.json").is_file()
    assert len(outcome["commands"]) == 12
    assert verified["scientifically_accepted"] is False
