"""Integrate actual capture/native preparation and immutable public handoffs."""
from copy import deepcopy
import builtins
from hashlib import sha256
import json
from pathlib import Path
import socket
from http.server import ThreadingHTTPServer
from threading import Thread

import pytest

from mosaic.public_build import PublicBuildRefusal
from mosaic.public_runtime_manifest import validate_manifest
from mosaic.public_workflow import load_corpus_manifest, materialization_destination, run_public_corpus
from test_public_execution import _prepared, _Loopback

pytestmark = pytest.mark.evidence_mock


def _inputs(tmp_path):
    workspace, capture, prepared = _prepared(tmp_path)
    manifest = json.loads((prepared / "local-model-manifest.json").read_bytes())
    profile = json.loads((prepared / "driver-profile.json").read_bytes())
    manifest["model"]["revision"] = "a" * 40
    profile["model"]["revision"] = "a" * 40
    profile["driver"]["wall_clock_cap_seconds"] = 180
    weights = tmp_path / "weights"
    checked = validate_manifest(manifest, weights_root=weights)
    config = {"version": "mosaic-public-build/v1", "profile": "corpus",
        "sources": {"root": "sources", "files": ["fictional.pdf"], "permission": "user_declared_permitted"},
        "run": {"root": "run/corpus"},
        "runtime": {"kind": "local_open_weights", "base_url": "http://127.0.0.1:8000/v1",
                    "model": "fictional/local", "revision": "a" * 40, "tokenizer_revision": "a" * 40,
                    "weights_sha256": checked["model_files_sha256"], "license": "fictional-test-only"},
        "view": {"columns": ["title", "authors", "journal", "year"], "default_screen": "papers",
                 "sort": "distinct_article_count_desc"},
        "serve": {"port": 8798}, "jev": {"enabled": False, "replay_only": True}}
    kwargs = {"project_root": tmp_path, "workspace": workspace, "model_manifest": manifest,
              "weights_root": weights, "driver_profile": profile}
    return config, kwargs


def test_actual_corpus_preparation_never_connects_and_refuses_identity_mismatch(tmp_path, monkeypatch):
    config, kwargs = _inputs(tmp_path)
    def no_network(*args, **kw):
        raise AssertionError("preparation attempted a network call")
    monkeypatch.setattr(socket.socket, "connect", no_network)
    result = run_public_corpus(config, **kwargs)
    assert result["status"] == "prepared_only"
    assert result["prepared_source_count"] == 1 and result["candidate_source_count"] == 0
    assert not result["real_model_inference_run"] and not result["local_model_requests_completed"] and not result["own_corpus_pipeline_ready"]
    assert Path(result["build_report_path"]).is_file()
    altered = deepcopy(config)
    altered["runtime"]["weights_sha256"] = "0" * 64
    with pytest.raises(PublicBuildRefusal, match="runtime_binding_mismatch"):
        run_public_corpus(altered, **kwargs)


def test_execution_failure_records_stop_without_retrying_or_claiming_completion(tmp_path, monkeypatch):
    config, kwargs = _inputs(tmp_path)
    import mosaic.public_execution
    calls = []
    def fail(*args, **kw):
        calls.append(args)
        raise RuntimeError("untrusted remote text must not enter the report")
    monkeypatch.setattr(mosaic.public_execution, "run_local_program", fail)
    with pytest.raises(PublicBuildRefusal, match="execution_stopped"):
        run_public_corpus(config, **kwargs, live=True)
    reports = [json.loads(p.read_bytes()) for p in (tmp_path / "run/corpus/reports").glob("*.json")]
    stopped = [r for r in reports if r.get("status") == "execution_stopped"]
    assert len(calls) == 1 and len(stopped) == 1
    assert stopped[0]["dispositions"][0]["reason"] == "RuntimeError"
    assert not stopped[0]["own_corpus_pipeline_ready"]
    assert "untrusted remote text" not in json.dumps(stopped)


@pytest.mark.parametrize("name", ["sealed/input.json", ".env", "references/input.json"])
def test_manifest_rejects_protected_paths_before_reading(tmp_path, name):
    from mosaic.workspace import Workspace
    path = tmp_path / name  # absent is intentional: no read may be attempted
    with pytest.raises(PublicBuildRefusal, match="private_path_refused"):
        load_corpus_manifest(path, project_root=tmp_path, workspace=Workspace(repo_root=tmp_path))


def test_manifest_refuses_traversal_and_changed_capture_before_materialization(tmp_path):
    from mosaic.workspace import Workspace
    workspace = Workspace(repo_root=tmp_path)
    manifest = {"version": "mosaic-public-corpus-input/v1", "sources": [{
        "capture_root": "../escape", "capture_receipt_sha256": "0" * 64,
        "report": {"path": "report.json", "sha256": "0" * 64},
        "snapshot": {"path": "snapshot.json", "sha256": "0" * 64}}]}
    path = tmp_path / "input.json"
    workspace.write_canonical_json(path, manifest)
    with pytest.raises(PublicBuildRefusal):
        load_corpus_manifest(path, project_root=tmp_path, workspace=workspace)
    manifest["sources"][0]["capture_root"] = "capture"
    workspace.write_text(tmp_path / "capture/receipt.json", "{}")
    workspace.write_canonical_json(path, manifest, once=False)
    with pytest.raises(PublicBuildRefusal, match="capture_changed"):
        load_corpus_manifest(path, project_root=tmp_path, workspace=workspace)


def test_actual_native_corpus_route_to_http_and_verified_resume(tmp_path):
    """Fictional protocol evidence through actual native/corpus/HTTP code."""
    config, kwargs = _inputs(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Loopback)
    thread = Thread(target=server.serve_forever)
    thread.start()
    config["runtime"]["base_url"] = f"http://127.0.0.1:{server.server_address[1]}/v1"
    try:
        result = run_public_corpus(config, **kwargs, live=True)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    assert result["status"] == "candidate_corpus_served"
    assert result["candidate_source_count"] == 1 and result["local_candidate_workflow_completed"]
    assert result["local_model_requests_completed"]
    assert not result["real_model_inference_run"] and not result["scientifically_accepted"]
    bundle_path = Path(result["materialization"]["bundle_path"])
    assert sha256(bundle_path.read_bytes()).hexdigest() == result["materialization"]["bundle_sha256"]
    readiness = json.loads((bundle_path.parent / "readiness.json").read_bytes())
    assert readiness["http_status"] == 200
    # No inference server is running during reuse. Stored outcomes must verify.
    repeated = run_public_corpus(config, **kwargs, live=True)
    assert repeated == result
    snapshot = next((tmp_path / "run/corpus/native").rglob("public-snapshot.json"))
    snapshot.write_bytes(b"tampered derivative")
    with pytest.raises(PublicBuildRefusal, match="execution_stopped"):
        run_public_corpus(config, **kwargs, live=True)


def test_interrupted_materialization_uses_a_new_bounded_attempt_directory(tmp_path):
    run_root = tmp_path / "run"
    base, reused = materialization_destination(run_root, "view-key")
    assert base == run_root / "display" / "view-key" and not reused
    base.mkdir(parents=True)
    (base / "partial-artifact.json").write_text("{}", encoding="utf-8")
    recovery, reused = materialization_destination(run_root, "view-key")
    assert recovery == base / "attempt-01" and not reused
    recovery.mkdir()
    (recovery / "partial-artifact.json").write_text("{}", encoding="utf-8")
    next_attempt, reused = materialization_destination(run_root, "view-key")
    assert next_attempt == base / "attempt-02" and not reused


@pytest.mark.parametrize("completed,exit_code", [(True, 0), (False, 2)])
def test_run_local_cli_does_not_call_an_incomplete_terminal_receipt_success(tmp_path, monkeypatch, completed, exit_code):
    import bootstrap
    import mosaic.public_execution
    monkeypatch.setattr(bootstrap, "_input_json", lambda p: {"driver": {"wall_clock_cap_seconds": 120}})
    monkeypatch.setattr(mosaic.public_execution, "run_local_program", lambda *a, **kw: {
        "receipt": {"status": "terminal", "native_execution_completed": completed}})
    assert bootstrap.main(["run-local", "--live", "--prepared-root", str(tmp_path / "prepared"),
                           "--capture-root", str(tmp_path / "capture"), "--output", str(tmp_path / "out")]) == exit_code


def test_bootstrap_refusal_reports_only_safe_dependency_and_missing_key_context(tmp_path, monkeypatch, capsys):
    import bootstrap
    original_import = builtins.__import__
    def refuse_pipeline(name, *args, **kwargs):
        if name == "mosaic.public_pipeline":
            raise ImportError("sensitive local detail", name=name)
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", refuse_pipeline)
    assert bootstrap.main(["demo"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"status": "refused", "reason": "public_dependency_unavailable",
                       "own_corpus_pipeline_ready": False, "module": "mosaic.public_pipeline"}
    monkeypatch.setattr(builtins, "__import__", original_import)
    monkeypatch.setattr(bootstrap, "load_config", lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyError("driver_profile")))
    assert bootstrap.main(["plan", "--config", str(tmp_path / "config.json")]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"status": "refused", "reason": "KeyError", "own_corpus_pipeline_ready": False,
                       "missing_key": "driver_profile"}
    monkeypatch.setattr(bootstrap, "load_config", lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyError("secret_value")))
    assert bootstrap.main(["plan", "--config", str(tmp_path / "config.json")]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"status": "refused", "reason": "KeyError", "own_corpus_pipeline_ready": False}
    def refuse_arbitrary_module(name, *args, **kwargs):
        if name == "mosaic.public_pipeline":
            raise ImportError("secret provider detail", name="sk-live-token-ish")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", refuse_arbitrary_module)
    assert bootstrap.main(["demo"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"status": "refused", "reason": "public_dependency_unavailable",
                       "own_corpus_pipeline_ready": False}
