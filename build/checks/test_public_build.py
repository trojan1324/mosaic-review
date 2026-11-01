"""Public scaffolding safety and lifecycle, using deliberately fictional bytes.

These tests do not run an extractor or establish support for a PDF/model.
"""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import socket

import pytest

from mosaic.public_build import (EVIDENCE_FIELDS, PublicBuildRefusal, RunWorkspace,
    StageExecutor, build_plan, digest, preflight, register_sources, validate_config, validate_view)
from mosaic.workspace import Workspace, WorkspaceViolation

pytestmark = pytest.mark.evidence_mock


@pytest.fixture
def setup(tmp_path):
    project = tmp_path / "public-project"
    workspace = Workspace(repo_root=project, temp_root=tmp_path)
    workspace.mkdir(project / "example-data")
    workspace.write_once(project / "example-data" / "one.pdf", b"%PDF-1.4\nfictional fixture; not a parseable paper")
    config = {"version": "mosaic-public-build/v1", "profile": "demo",
        "sources": {"root": "example-data", "files": ["one.pdf"], "permission": "fictional_fixture"},
        "run": {"root": "run/demo"}, "runtime": {"kind": "recorded_replay"},
        "view": {"columns": ["title", "authors"], "default_screen": "papers", "sort": "distinct_article_count_desc"},
        "serve": {"port": 8798}, "jev": {"enabled": False, "replay_only": True}}
    return project, workspace, config


def test_explicit_intake_versions_bytes_preserves_old_manifest_and_avoids_network(setup, monkeypatch):
    project, workspace, config = setup
    def no_network(*args, **kwargs):
        raise AssertionError("unexpected network")
    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    before = register_sources(config, project_root=project, workspace=workspace)
    assert before == register_sources(config, project_root=project, workspace=workspace)
    assert before["sources"][0]["source_status"] == "public_unreviewed_source"
    assert before["registry_consulted"] is False
    first = project / "run/demo/inputs" / digest(before) / "input-manifest.json"
    frozen = first.read_bytes()
    # An input-file change gets a new manifest; old input evidence survives.
    workspace.write_bytes(project / "example-data/one.pdf", b"%PDF-1.7\nsecond fictional source version", overwrite=True)
    later = register_sources(config, project_root=project, workspace=workspace)
    assert later["sources"][0]["paper_id"] != before["sources"][0]["paper_id"]
    assert first.read_bytes() == frozen
    assert preflight(config)["network_calls"] == 0


def test_aliases_do_not_multiply_articles_and_bad_inputs_remain_explicit(setup):
    project, workspace, config = setup
    workspace.copy_file(project / "example-data/one.pdf", project / "example-data/alias.pdf")
    workspace.write_once(project / "example-data/bad.pdf", b"HTML error page")
    config["sources"]["files"] = ["one.pdf", "alias.pdf", "bad.pdf", "missing.pdf"]
    result = register_sources(config, project_root=project, workspace=workspace)
    assert len(result["sources"]) == 1
    assert result["sources"][0]["locators"] == ["alias.pdf", "one.pdf"]
    assert {r["reason"] for r in result["refusals"]} == {"public_pdf_signature_invalid", "public_source_missing"}


@pytest.mark.parametrize("bad", ["../one.pdf", "/absolute.pdf", "C:/private.pdf", "one.pdf:stream", "a\\b.pdf", "a/./one.pdf"])
def test_untrusted_member_path_is_refused_before_read(setup, bad):
    project, _, config = setup
    config["sources"]["files"] = [bad]
    with pytest.raises(PublicBuildRefusal):
        validate_config(config, project_root=project)


def test_view_changes_do_not_change_extraction_identity_or_hide_provenance(setup):
    project, _, config = setup
    first = build_plan(validate_config(config, project_root=project))
    revised = deepcopy(config)
    revised["view"]["columns"] = ["measure", "design"]
    revised["view"]["default_screen"] = "measures"
    later = build_plan(validate_config(revised, project_root=project))
    assert first["extraction_config_sha256"] == later["extraction_config_sha256"]
    assert first["view_sha256"] != later["view_sha256"]
    assert set(validate_view(revised["view"])["mandatory_evidence_fields"]) == EVIDENCE_FIELDS
    revised["view"]["scientifically_accepted"] = True
    with pytest.raises(PublicBuildRefusal, match="public_view_keys_invalid"):
        validate_view(revised["view"])


def test_private_inputs_and_credentials_are_not_configurable(setup):
    project, _, config = setup
    for root in ("scratch/private", "references/keys", "sealed", "diagnostic"):
        changed = deepcopy(config)
        changed["sources"]["root"] = root
        with pytest.raises(PublicBuildRefusal, match="public_private_input_root_refused"):
            validate_config(changed, project_root=project)
    config["jev"]["api_key"] = "not-a-secret-test-value"
    with pytest.raises(PublicBuildRefusal, match="public_jev_config_invalid"):
        validate_config(config, project_root=project)


def test_hosted_runtime_and_live_advisor_cannot_be_enabled_by_accident(setup):
    project, _, config = setup
    config["profile"] = "corpus"
    config["sources"]["permission"] = "user_declared_permitted"
    config["runtime"] = {"kind": "local_open_weights", "base_url": "https://external.invalid/v1",
        "model": "fictional-model", "revision": "a" * 40, "tokenizer_revision": "b" * 40,
        "weights_sha256": "a" * 64, "license": "not-selected-test-only"}
    with pytest.raises(PublicBuildRefusal, match="public_loopback_runtime_required"):
        validate_config(config, project_root=project)
    config["runtime"]["base_url"] = "http://127.0.0.1:8000/v1"
    validate_config(config, project_root=project)
    assert preflight(config)["model_runtime"] == "not_tested"
    assert build_plan(config)["own_corpus_pipeline_ready"] is False
    config["jev"] = {"enabled": True, "replay_only": False}
    with pytest.raises(PublicBuildRefusal, match="public_live_advisory_workflow_refused"):
        validate_config(config, project_root=project)


def test_run_writer_cannot_escape_to_otherwise_permitted_temp_or_repo(setup):
    project, workspace, _ = setup
    public = RunWorkspace(project / "run/demo", workspace=workspace)
    public.write_once("safe.json", b"{}")
    with pytest.raises(WorkspaceViolation):
        public.write_once(project / "outside.json", b"{}")
    with pytest.raises(WorkspaceViolation):
        public.write_once("../escaped.json", b"{}")


def test_stage_failure_interrupt_resume_and_tamper_detection(setup):
    project, workspace, _ = setup
    executor = StageExecutor(project / "run/demo", workspace=workspace, implementation_sha256="b" * 64)
    calls = []
    def interrupted(scoped, attempt):
        scoped.write_once(attempt / "partial.json", b"partial")
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        executor.execute("intake", {"input_hash": "c" * 64}, dependencies={}, action=interrupted)
    def completed(scoped, attempt):
        calls.append(attempt)
        path = scoped.write_once(attempt / "manifest.json", b"fictional-manifest")
        return {"manifest": path}
    receipt = executor.execute("intake", {"input_hash": "c" * 64}, dependencies={}, action=completed)
    assert receipt["attempt"] == 2
    assert (calls[0].parent / "attempt-1/partial.json").read_bytes() == b"partial"
    failure = json.loads((calls[0].parent / "attempt-1/failure.json").read_bytes())
    assert failure["error_type"] == "KeyboardInterrupt"
    assert executor.execute("intake", {"input_hash": "c" * 64}, dependencies={}, action=completed) == receipt
    assert len(calls) == 1
    workspace.write_bytes(calls[0] / "manifest.json", b"tampered", overwrite=True)
    with pytest.raises(PublicBuildRefusal, match="public_stage_artifact_changed"):
        executor.execute("intake", {"input_hash": "c" * 64}, dependencies={}, action=completed)


def test_forged_dependencies_and_jev_control_edge_refused(setup):
    project, workspace, _ = setup
    executor = StageExecutor(project / "run/demo", workspace=workspace, implementation_sha256="b" * 64)
    def completed(scoped, attempt):
        return {"output": scoped.write_once(attempt / "output.json", b"{}")}
    fake = {"version": "mosaic-public-build/v1", "status": "succeeded", "scientifically_accepted": False,
        "stage": "intake", "fingerprint": "a" * 64, "artifacts": {}}
    with pytest.raises(PublicBuildRefusal, match="public_stage_receipt_fingerprint_changed"):
        executor.execute("capture", {}, dependencies={"intake": fake}, action=completed)
    with pytest.raises(PublicBuildRefusal, match="public_stage_dependencies_invalid"):
        executor.execute("extract", {}, dependencies={"optional_jev": fake}, action=completed)


def test_implementation_change_invalidates_stage_cache(setup):
    project, workspace, _ = setup
    def completed(scoped, attempt):
        return {"output": scoped.write_once(attempt / "output.json", b"{}")}
    first = StageExecutor(project / "run/demo", workspace=workspace, implementation_sha256="a" * 64)
    second = StageExecutor(project / "run/demo", workspace=workspace, implementation_sha256="b" * 64)
    before = first.execute("intake", {}, dependencies={}, action=completed)
    after = second.execute("intake", {}, dependencies={}, action=completed)
    assert before["fingerprint"] != after["fingerprint"]


def test_build_report_retains_missing_stages_and_refuses_orphan_success(setup):
    project, workspace, config = setup
    executor = StageExecutor(project / "run/demo", workspace=workspace, implementation_sha256="b" * 64)
    def completed(scoped, attempt):
        return {"output": scoped.write_once(attempt / "output.json", b"{}")}
    intake = executor.execute("intake", {}, dependencies={}, action=completed)
    report = executor.build_report(config, {"intake": intake})
    assert report["own_corpus_pipeline_ready"] is False
    assert report["release_gates"]["cold_start"] == "pending"
    assert {row["stage"] for row in report["stages"] if row["status"] == "not_executed"} >= {"extract", "serve"}
    capture = executor.execute("capture", {}, dependencies={"intake": intake}, action=completed)
    with pytest.raises(PublicBuildRefusal, match="public_report_dependency_missing"):
        executor.build_report(config, {"capture": capture})
    other_intake = executor.execute("intake", {"changed_source": True}, dependencies={}, action=completed)
    with pytest.raises(PublicBuildRefusal, match="public_report_dependency_lineage_mismatch"):
        executor.build_report(config, {"intake": other_intake, "capture": capture})
