"""Regression coverage for the orphan-repair worker's canonical package names."""
from __future__ import annotations

import json
import time
from types import SimpleNamespace


def _freeze(values: dict[str, str]) -> bytes:
    return ("\n".join(f"{name}=={version}" for name, version in sorted(values.items())) + "\n").encode()


def _metadata(values: dict[str, str], *, pdf_url: str, pdf_sha: str) -> bytes:
    installed = []
    for name, version in values.items():
        requires = ['nvidia-nccl-cu13==2.29.7; platform_system == "Linux"'] if name == "torch" else []
        row = {"metadata": {"name": name, "version": "1.28.2" if name == "pymupdf" else version, "requires_dist": requires}}
        if name == "pymupdf":
            row["direct_url"] = {"url": pdf_url, "archive_info": {"hash": "sha256=" + pdf_sha}}
        installed.append(row)
    environment = {"implementation_name": "cpython", "implementation_version": "3.12.0", "os_name": "posix",
                   "platform_machine": "x86_64", "platform_release": "test", "platform_system": "Linux",
                   "platform_version": "test", "python_full_version": "3.12.0", "platform_python_implementation": "CPython",
                   "python_version": "3.12", "sys_platform": "linux"}
    return json.dumps({"installed": installed, "environment": environment}).encode()


def test_worker_entry_uses_orphan_parser_through_pdf_validation_with_dotted_names(tmp_path, monkeypatch):
    """The real worker path reaches the real PDF validator with canonical keys."""
    import platform
    import mosaic.public_export as export
    import mosaic.public_linux_smoke as smoke
    from mosaic.public_linux_orphan_dependencies import PDF_WHEEL_SHA, PDF_WHEEL_URL

    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(export, "verify_export", lambda _: {"version": "mosaic-public-export/v1"})
    monkeypatch.setattr(smoke.shutil, "disk_usage", lambda _: SimpleNamespace(free=smoke.MIN_FREE_BYTES))
    output, model = tmp_path / "run/out", tmp_path / "model"
    before = {"lazr.uri": "1.0", "lazr.restfulclient": "0.14", "torch": "2.13.0+cu130", "nvidia-nccl-cu13": "2.30.7"}
    after = {**before, "pymupdf": PDF_WHEEL_URL + "#sha256=" + PDF_WHEEL_SHA}

    def gate(**kwargs):
        kwargs["workspace"].write_once(output / "post-repair-freeze.stdout.log", _freeze(before))
        kwargs["workspace"].write_canonical_json(output / "dependency-gate.json", {"mock": "only-pre-pdf-stage"})
        return {"mock": "only-pre-pdf-stage"}
    monkeypatch.setattr(smoke, "run_orphan_dependency_gate", gate)
    actual_pdf_validator = smoke.validate_orphan_pdf_stage
    captured = {}
    def pdf_validator(**kwargs):
        captured["before"] = dict(kwargs["before"])
        return actual_pdf_validator(**kwargs)
    monkeypatch.setattr(smoke, "validate_orphan_pdf_stage", pdf_validator)

    def completed(command, **_):
        if command[:2] == ["hf", "--version"]:
            return SimpleNamespace(returncode=0, stdout=b"hf", stderr=b"")
        if command[:4] == [smoke.sys.executable, "-m", "pip", "freeze"]:
            return SimpleNamespace(returncode=0, stdout=_freeze(after), stderr=b"")
        if command[:4] == [smoke.sys.executable, "-m", "pip", "inspect"]:
            return SimpleNamespace(returncode=0, stdout=_metadata(after, pdf_url=PDF_WHEEL_URL, pdf_sha=PDF_WHEEL_SHA), stderr=b"")
        if command[:4] == [smoke.sys.executable, "-m", "pip", "check"]:
            from mosaic.public_linux_orphan_dependencies import NCCL
            return SimpleNamespace(returncode=1, stdout=(NCCL + "\n").encode(), stderr=b"")
        # Stop immediately after the successful PDF stage without a model download.
        if command[:3] == [smoke.sys.executable, "-c", "import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in ('vllm','torch','tokenizers','xgrammar','huggingface-hub','pymupdf')}))"]:
            return SimpleNamespace(returncode=1, stdout=b"", stderr=b"synthetic stop")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
    monkeypatch.setattr(smoke.subprocess, "run", completed)

    result = smoke.execute_linux_smoke(tmp_path, model, output, time.time() + 1200)
    assert captured["before"]["lazr-uri"] == "1.0"
    assert captured["before"]["lazr-restfulclient"] == "0.14"
    assert result["reason"] == "public_linux_smoke_required-runtime_failed"
    assert json.loads((output / "dependency-pdf-stage.json").read_text())["changed_distributions"] == ["pymupdf"]


def test_worker_preserves_safe_orphan_refusal_code_in_outcome(tmp_path, monkeypatch):
    import platform
    import mosaic.public_export as export
    import mosaic.public_linux_smoke as smoke
    from mosaic.public_linux_orphan_dependencies import OrphanRepairRefusal

    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(export, "verify_export", lambda _: {"version": "mosaic-public-export/v1"})
    monkeypatch.setattr(smoke.shutil, "disk_usage", lambda _: SimpleNamespace(free=smoke.MIN_FREE_BYTES))
    monkeypatch.setattr(smoke.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=b"hf", stderr=b""))
    monkeypatch.setattr(smoke, "run_orphan_dependency_gate", lambda **_: (_ for _ in ()).throw(OrphanRepairRefusal("orphan_repair_pdf_delta_invalid")))
    result = smoke.execute_linux_smoke(tmp_path, tmp_path / "model", tmp_path / "run/out", time.time() + 1200)
    assert result["reason"] == "orphan_repair_pdf_delta_invalid"
    assert result["real_model_inference_run"] is False


def test_safe_reason_does_not_expose_arbitrary_exception_payloads():
    from mosaic.public_linux_smoke import _safe_failure_reason
    from mosaic.public_linux_orphan_dependencies import OrphanRepairRefusal

    assert _safe_failure_reason(OrphanRepairRefusal("orphan_repair_pdf_delta_invalid")) == "orphan_repair_pdf_delta_invalid"
    assert _safe_failure_reason(OrphanRepairRefusal("orphan_repair_bad: private diagnostic")) == "OrphanRepairRefusal"
    assert _safe_failure_reason(ValueError("orphan_repair_spoofed")) == "ValueError"
