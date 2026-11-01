"""Execute the actual public fixture route and verify immutable repeat/resume."""
from hashlib import sha256
import json
from pathlib import Path
import socket

import pytest

from mosaic.public_build import PublicBuildRefusal
from mosaic.public_pipeline import build_demo
from mosaic.workspace import Workspace

pytestmark = pytest.mark.evidence_mock


def test_complete_fictional_route_is_repeatable_and_tamper_evident(tmp_path, monkeypatch):
    original_connect = socket.socket.connect
    def local_only(sock, address):
        if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "::1"):
            raise AssertionError("public build attempted a nonlocal connection")
        return original_connect(sock, address)
    monkeypatch.setattr(socket.socket, "connect", local_only)
    project = tmp_path / "candidate"
    workspace = Workspace(repo_root=project, temp_root=tmp_path)
    first = build_demo(project, workspace=workspace)
    report = json.loads(Path(first["build_report_path"]).read_bytes())
    assert report["demo_software_route_ready"]
    assert not report["real_model_inference_run"] and not report["scientifically_accepted"]
    assert report["optional_jev"] == "not_run_separate_advisory_sidecar"
    assert "fictional_replay_not_real_model_inference" in report["current_route_limits"]
    assert "generic_bibliography_reference_measure_materializers" not in report["remaining_gates"]
    assert "public_live_Jev_adapter" not in report["remaining_gates"]
    assert set(report["stages"]) == {"intake", "capture", "inventory", "extract", "verify", "bibliography", "materialize", "serve"}
    runtime_receipts = list((project / "run/demo/stages/extract").rglob("public-runtime-receipt.json"))
    assert len(runtime_receipts) == 1
    runtime = json.loads(runtime_receipts[0].read_bytes())
    assert runtime["network_calls"] == runtime["model_calls"] == 0
    native = next((project / "run/demo/stages/capture").rglob("native-reader.json"))
    assert len(json.loads(native.read_bytes())["units"]) == 14
    saved_hash = sha256(native.read_bytes()).hexdigest()
    assert build_demo(project, workspace=workspace) == first
    assert sha256(native.read_bytes()).hexdigest() == saved_hash
    workspace.write_text(native, "tampered reader", overwrite=True)
    with pytest.raises(PublicBuildRefusal, match="artifact_changed"):
        build_demo(project, workspace=workspace)
