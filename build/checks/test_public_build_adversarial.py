"""Adversarial boundary checks for the isolated public-build scaffold."""
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from mosaic.public_build import (PublicBuildRefusal, StageExecutor, digest,
                                 register_sources, validate_config)
from mosaic.workspace import Workspace


pytestmark = pytest.mark.evidence_mock


@pytest.fixture
def public(tmp_path):
    project = tmp_path / "project"
    ws = Workspace(repo_root=project, temp_root=tmp_path)
    ws.mkdir(project / "sources")
    ws.write_once(project / "sources" / "one.pdf", b"%PDF-1.4\nfictional")
    config = {"version": "mosaic-public-build/v1", "profile": "demo",
        "sources": {"root": "sources", "files": ["one.pdf"], "permission": "fictional_fixture"},
        "run": {"root": "run"}, "runtime": {"kind": "recorded_replay"},
        "view": {"columns": ["title"], "default_screen": "papers", "sort": "title"},
        "serve": {"port": 8991}, "jev": {"enabled": False, "replay_only": True}}
    return project, ws, config


def _action(scoped, attempt):
    return {"output": scoped.write_once(attempt / "output.json", b"{}")}


def test_symlink_source_and_config_are_refused_before_any_source_read(public, tmp_path):
    project, ws, config = public
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"%PDF-outside")
    link = project / "sources" / "linked.pdf"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation unavailable")
    config["sources"]["files"] = ["linked.pdf"]
    manifest = register_sources(config, project_root=project, workspace=ws)
    assert manifest["sources"] == []
    assert manifest["refusals"][0]["locator"] == "linked.pdf"
    assert manifest["refusals"][0]["reason"] in {"public_link_refused", "public_path_escape_refused"}


def test_dependency_requires_the_stored_receipt_not_a_self_consistent_forgery(public):
    project, ws, _ = public
    executor = StageExecutor(project / "run", workspace=ws, implementation_sha256="a" * 64)
    artifact = project / "run" / "unrelated.json"
    ws.write_once(artifact, b"{}")
    identity = {"version": "mosaic-public-build/v1", "stage": "intake", "inputs_sha256": digest({}),
                "implementation_sha256": "a" * 64, "dependencies": {}}
    forged = {**identity, "fingerprint": digest(identity), "status": "succeeded", "attempt": 1,
              "artifacts": {"unrelated": {"path": "unrelated.json", "sha256": sha256(b"{}").hexdigest()}},
              "scientifically_accepted": False}
    with pytest.raises(PublicBuildRefusal, match="public_stage_dependency_receipt_changed"):
        executor.execute("capture", {}, dependencies={"intake": forged}, action=_action)


def test_changed_result_receipt_cannot_resume_or_satisfy_a_child(public):
    project, ws, _ = public
    executor = StageExecutor(project / "run", workspace=ws, implementation_sha256="a" * 64)
    intake = executor.execute("intake", {}, dependencies={}, action=_action)
    result = project / "run" / "stages" / "intake" / intake["fingerprint"] / "result.json"
    tampered = json.loads(result.read_bytes())
    tampered["artifacts"]["output"]["sha256"] = "0" * 64
    ws.write_bytes(result, json.dumps(tampered).encode(), overwrite=True)
    with pytest.raises(PublicBuildRefusal, match="public_stage_artifact_changed"):
        executor.execute("intake", {}, dependencies={}, action=_action)
    with pytest.raises(PublicBuildRefusal):
        executor.execute("capture", {}, dependencies={"intake": intake}, action=_action)


def test_missing_or_changed_input_creates_new_manifest_without_overwrite(public):
    project, ws, config = public
    first = register_sources(config, project_root=project, workspace=ws)
    old = project / "run" / "inputs" / digest(first) / "input-manifest.json"
    ws.write_bytes(project / "sources" / "one.pdf", b"%PDF-new", overwrite=True)
    second = register_sources(config, project_root=project, workspace=ws)
    assert digest(first) != digest(second) and old.is_file()
    config = deepcopy(config); config["sources"]["files"] = ["missing.pdf"]
    missing = register_sources(config, project_root=project, workspace=ws)
    assert missing["sources"] == [] and missing["refusals"][0]["reason"] == "public_source_missing"


def test_runtime_declaration_never_claims_ready_and_view_is_nonsemantic(public):
    project, _, config = public
    corpus = deepcopy(config)
    corpus["profile"] = "corpus"; corpus["sources"]["permission"] = "user_declared_permitted"
    corpus["runtime"] = {"kind": "local_open_weights", "base_url": "http://127.0.0.1:7777/v1", "model": "fixture", "revision": "a" * 40, "tokenizer_revision": "b" * 40, "weights_sha256": "c" * 64, "license": "fixture"}
    assert validate_config(corpus, project_root=project)["runtime"]["model"] == "fixture"
    # Validation is only a declaration check; no completed-runtime claim exists here.
    assert "ready" not in validate_config(corpus, project_root=project)
