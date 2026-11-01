"""Tests for the Linux-only, CPU-only runtime preflight."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import time

import pytest

from mosaic.public_linux_environment import (
    CORE_RUNTIME_IMPORT_CODE,
    PublicLinuxEnvironmentRefusal,
    PDF_IMPORT_CODE,
    PDF_IMPORT_STDOUT,
    VERSION,
    run_linux_environment_preflight,
    verify_retained_linux_environment_preflight,
)
from mosaic.public_linux_orphan_dependencies import load_plan
from mosaic.workspace import Workspace


def test_preflight_refuses_non_linux_before_creating_output(tmp_path, monkeypatch):
    import mosaic.public_linux_environment as environment

    monkeypatch.setattr(environment.platform, "system", lambda: "Windows")
    with pytest.raises(PublicLinuxEnvironmentRefusal, match="linux_required"):
        run_linux_environment_preflight(tmp_path / "preflight", deadline_epoch=time.time() + 60,
                                        workspace=Workspace(repo_root=tmp_path, temp_root=tmp_path))
    assert not (tmp_path / "preflight").exists()


def test_preflight_runs_only_environment_steps_with_fake_linux_process(tmp_path, monkeypatch):
    import mosaic.public_linux_environment as environment

    monkeypatch.setattr(environment.platform, "system", lambda: "Linux")
    labels: list[str] = []

    def fake_gate(*, run, output, workspace, **_):
        for label in ("baseline-freeze", "baseline-metadata", "baseline-check", "orphan-target-record",
                      "orphan-before-files", "orphan-uninstall", "orphan-post-remove-absence",
                      "post-repair-freeze", "post-repair-metadata", "post-repair-check",
                      "orphan-after-files", "orphan-post-imports"):
            run(["python3", "-c", label], label, 1, require_success=label.endswith("check") is False)
        workspace.write_canonical_json(output / "dependency-gate.json", {"fixture": True})
        return {"fixture": True}

    def fake_pdf(*, run, output, workspace, **_):
        for label in ("post-pdf-freeze", "post-pdf-metadata", "post-pdf-check"):
            run(["python3", "-c", label], label, 1, require_success=label.endswith("check") is False)
        workspace.write_canonical_json(output / "dependency-pdf-stage.json", {"fixture": True})
        return {"fixture": True}

    def fake_process(command, **_):
        labels.append(command[-1])
        if command[-1] == CORE_RUNTIME_IMPORT_CODE:
            observed = {"metadata_versions": {"torch": "2.13.0", "tokenizers": "0.23.2", "xgrammar": "0.2.8"},
                        "module_versions": {"torch": "2.13.0", "tokenizers": "0.23.2"}, "torch_cuda": "13.0"}
            return subprocess.CompletedProcess(command, 0, json.dumps(observed, sort_keys=True).encode(), b"")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(environment, "run_orphan_dependency_gate", fake_gate)
    monkeypatch.setattr(environment, "validate_orphan_pdf_stage", fake_pdf)
    monkeypatch.setattr(environment, "parse_freeze", lambda _: {"opentelemetry-api": "1.40.0"})
    result = run_linux_environment_preflight(
        tmp_path / "preflight", deadline_epoch=time.time() + 60,
        workspace=Workspace(repo_root=tmp_path, temp_root=tmp_path), run_process=fake_process)
    assert result["status"] == "environment_preflight_completed"
    assert result["gpu_observed"] is False
    assert result["model_downloaded"] is False
    assert result["model_inference_run"] is False
    assert not any("vllm" in command for command in labels)
    assert not any("hf" in command for command in labels)
    assert CORE_RUNTIME_IMPORT_CODE in labels
    assert (tmp_path / "preflight/preflight-outcome.json").is_file()


def _retained_environment_fixture(tmp_path: Path, *, pdf_stdout: bytes = b"pymupdf-import-ok\n") -> Path:
    # Reuse the exact fictional v4 retained-gate fixture rather than reimplement
    # its dependency graph or ownership evidence in this test.
    from test_public_linux_orphan_dependencies import create_retained_orphan_fixture

    old = create_retained_orphan_fixture(tmp_path)
    output = tmp_path / "preflight"
    old.rename(output)
    old_outcome = json.loads((output / "outcome.json").read_bytes())
    plan = load_plan()
    workspace = Workspace(repo_root=tmp_path, temp_root=tmp_path)
    old_lock = output / "requirements-linux-smoke.lock"
    new_lock = output / "requirements-linux-preflight.lock"
    workspace.write_once(new_lock, old_lock.read_bytes())
    install = json.loads((output / "install.json").read_bytes())
    install["command"][-1] = str(new_lock)
    workspace.write_canonical_json(output / "install.json", install, once=False)
    commands = [install if row["command"] == old_outcome["commands"][12]["command"] else row
                for row in old_outcome["commands"]]
    import_stdout = pdf_stdout
    workspace.write_once(output / "post-pdf-imports.stdout.log", import_stdout)
    workspace.write_once(output / "post-pdf-imports.stderr.log", b"")
    import_receipt = {"command": ["python", "-c", PDF_IMPORT_CODE], "returncode": 0, "timed_out": False,
                      "stdout_sha256": sha256(import_stdout).hexdigest(),
                      "stderr_sha256": sha256(b"").hexdigest()}
    workspace.write_canonical_json(output / "post-pdf-imports.json", import_receipt)
    commands.append(import_receipt)
    runtime_observation = {"metadata_versions": {"torch": "2.13.0", "tokenizers": "0.23.2", "xgrammar": "0.2.8"},
                           "module_versions": {"torch": "2.13.0", "tokenizers": "0.23.2"}, "torch_cuda": "13.0"}
    runtime_stdout = json.dumps(runtime_observation, sort_keys=True).encode() + b"\n"
    workspace.write_once(output / "core-runtime-imports.stdout.log", runtime_stdout)
    workspace.write_once(output / "core-runtime-imports.stderr.log", b"")
    runtime_receipt = {"command": ["python", "-c", CORE_RUNTIME_IMPORT_CODE], "returncode": 0, "timed_out": False,
                       "stdout_sha256": sha256(runtime_stdout).hexdigest(),
                       "stderr_sha256": sha256(b"").hexdigest()}
    workspace.write_canonical_json(output / "core-runtime-imports.json", runtime_receipt)
    commands.append(runtime_receipt)
    outcome = {
        "version": VERSION,
        "status": "environment_preflight_completed",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "declared_image": plan.image,
        "image_provenance": "declared_not_observed; bind an external container receipt before treating this as exact-image evidence",
        "platform": {"system": "Linux", "python": "fixture"},
        "plan_sha256": plan.config_sha256,
        "dependency_gate_sha256": old_outcome["dependency_gate_sha256"],
        "dependency_pdf_stage_sha256": old_outcome["dependency_pdf_stage_sha256"],
        "core_runtime_import_sha256": sha256((output / "core-runtime-imports.json").read_bytes()).hexdigest(),
        "core_runtime": runtime_observation,
        "commands": commands,
        "gpu_observed": False,
        "model_downloaded": False,
        "model_inference_run": False,
        "scientifically_accepted": False,
        "limitations": ["fixture"],
    }
    workspace.write_canonical_json(output / "preflight-outcome.json", outcome)
    return output


def test_retained_preflight_replays_complete_dependency_evidence(tmp_path):
    output = _retained_environment_fixture(tmp_path)
    result = verify_retained_linux_environment_preflight(output, expected_config_sha256=load_plan().config_sha256)
    assert result == {"version": VERSION, "environment_preflight_observed": True,
                      "exact_image_observed": False, "scientifically_accepted": False}


@pytest.mark.parametrize('stdout,accepted', [(PDF_IMPORT_STDOUT[1] + b'\n', True),
    (b'unexpected import output\npymupdf-import-ok\n', False)])
def test_retained_pdf_import_allows_only_the_observed_alias_warning(tmp_path, stdout, accepted):
    output = _retained_environment_fixture(tmp_path, pdf_stdout=stdout)
    if accepted:
        assert verify_retained_linux_environment_preflight(output,
            expected_config_sha256=load_plan().config_sha256)['environment_preflight_observed']
    else:
        with pytest.raises(PublicLinuxEnvironmentRefusal, match='pdf_import_invalid'):
            verify_retained_linux_environment_preflight(output, expected_config_sha256=load_plan().config_sha256)


def test_retained_cpu_preflight_rejects_rehashed_shared_file_drift(tmp_path):
    output = _retained_environment_fixture(tmp_path)
    workspace = Workspace(repo_root=tmp_path, temp_root=tmp_path)
    altered = json.loads((output / 'orphan-after-files.stdout.log').read_bytes())
    altered['shared_opentelemetry_files'] = {'opentelemetry-sdk': {'changed.py': '0' * 64}}
    workspace.write_canonical_json(output / 'orphan-after-files.stdout.log', altered, once=False)
    receipt = json.loads((output / 'orphan-after-files.json').read_bytes())
    original_receipt = dict(receipt)
    receipt['stdout_sha256'] = sha256((output / 'orphan-after-files.stdout.log').read_bytes()).hexdigest()
    workspace.write_canonical_json(output / 'orphan-after-files.json', receipt, once=False)
    outcome = json.loads((output / 'preflight-outcome.json').read_bytes())
    outcome['commands'] = [receipt if row == original_receipt else row for row in outcome['commands']]
    workspace.write_canonical_json(output / 'preflight-outcome.json', outcome, once=False)
    with pytest.raises(PublicLinuxEnvironmentRefusal, match='shared_namespace'):
        verify_retained_linux_environment_preflight(output, expected_config_sha256=load_plan().config_sha256)


def test_retained_preflight_rejects_rehashed_command_semantic_tampering(tmp_path):
    output = _retained_environment_fixture(tmp_path)
    workspace = Workspace(repo_root=tmp_path, temp_root=tmp_path)
    receipt = json.loads((output / "orphan-uninstall.json").read_bytes())
    receipt["command"][-1] = "different-package"
    workspace.write_canonical_json(output / "orphan-uninstall.json", receipt, once=False)
    outcome = json.loads((output / "preflight-outcome.json").read_bytes())
    outcome["commands"] = [receipt if row["command"][1:] == ["-m", "pip", "uninstall", "--yes", "opentelemetry-exporter-otlp-common"] else row
                           for row in outcome["commands"]]
    workspace.write_canonical_json(output / "preflight-outcome.json", outcome, once=False)
    with pytest.raises(PublicLinuxEnvironmentRefusal, match="command_semantics"):
        verify_retained_linux_environment_preflight(output, expected_config_sha256=load_plan().config_sha256)


def test_retained_preflight_rejects_missing_evidence(tmp_path):
    output = _retained_environment_fixture(tmp_path)
    (output / "post-pdf-metadata.stdout.log").unlink()
    with pytest.raises(FileNotFoundError):
        verify_retained_linux_environment_preflight(output, expected_config_sha256=load_plan().config_sha256)


def test_retained_preflight_rejects_core_runtime_version_mismatch(tmp_path):
    output = _retained_environment_fixture(tmp_path)
    workspace = Workspace(repo_root=tmp_path, temp_root=tmp_path)
    observed = {"metadata_versions": {"torch": "2.13.0", "tokenizers": "0.23.2", "xgrammar": "0.2.8"},
                "module_versions": {"torch": "2.13.1", "tokenizers": "0.23.2"}, "torch_cuda": "13.0"}
    raw = json.dumps(observed, sort_keys=True).encode() + b"\n"
    workspace.write_once(output / "tampered-core-runtime.stdout.log", raw)
    receipt = json.loads((output / "core-runtime-imports.json").read_bytes())
    receipt["stdout_sha256"] = sha256(raw).hexdigest()
    workspace.write_canonical_json(output / "core-runtime-imports.json", receipt, once=False)
    workspace.write_once(output / "core-runtime-imports.stdout.log.replacement", raw)
    (output / "core-runtime-imports.stdout.log").unlink()
    (output / "core-runtime-imports.stdout.log.replacement").rename(output / "core-runtime-imports.stdout.log")
    outcome = json.loads((output / "preflight-outcome.json").read_bytes())
    outcome["commands"] = [receipt if row["command"] == receipt["command"] else row for row in outcome["commands"]]
    workspace.write_canonical_json(output / "preflight-outcome.json", outcome, once=False)
    with pytest.raises(PublicLinuxEnvironmentRefusal, match="core_runtime_import_invalid"):
        verify_retained_linux_environment_preflight(output, expected_config_sha256=load_plan().config_sha256)
