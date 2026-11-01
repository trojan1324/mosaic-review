from copy import deepcopy
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import socket
import subprocess
import sys

import pymupdf
import pytest

from mosaic import public_native_runtime as runtime
from mosaic import REPO_ROOT
from mosaic.public_capture import capture_public
from mosaic.public_runtime_manifest import VERSION as MANIFEST_VERSION
from mosaic.workspace import ImmutableArtifactError, Workspace


def _pdf_bytes():
    document = pymupdf.open()
    page = document.new_page()
    for y, text in ((72, "Study 1"), (110, "Abstract"), (145, "Fictional source prose."),
                    (200, "Methods"), (235, "Fictional participants completed a scale."),
                    (290, "Results"), (325, "A fictional result is reported.")):
        page.insert_text((72, y), text)
    data = document.tobytes()
    document.close()
    return data


def _capture(tmp_path):
    workspace = Workspace(repo_root=tmp_path, temp_root=tmp_path)
    source_root = workspace.mkdir(tmp_path / "sources")
    raw = _pdf_bytes()
    workspace.write_once(source_root / "fictional.pdf", raw)
    digest = sha256(raw).hexdigest()
    row = {"paper_id": "public-" + digest, "source_pdf_sha256": digest, "byte_size": len(raw),
           "locators": ["fictional.pdf"], "source_status": "public_unreviewed_source",
           "permission": "user_declared_permitted", "scientifically_accepted": False}
    capture_root = tmp_path / "capture"
    capture_public(row, "fictional.pdf", source_root, capture_root, title="Unreviewed fictional source", workspace=workspace)
    return workspace, capture_root


def _manifest(workspace, root):
    tokenizer = (b'{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,'
                 b'"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel",'
                 b'"vocab":{"[UNK]":0},"unk_token":"[UNK]"}}')
    weights = b"fictional weights"
    workspace.write_once(root / "tokenizer.json", tokenizer)
    workspace.write_once(root / "weights" / "model-00001.safetensors", weights)
    def record(path, data): return {"path": path, "size": len(data), "sha256": sha256(data).hexdigest()}
    token = record("tokenizer.json", tokenizer)
    return {"version": MANIFEST_VERSION, "model": {"id": "fictional/local-model", "revision": "fictional-revision",
            "files": [token, record("weights/model-00001.safetensors", weights)],
            "tokenizer": {"path": token["path"], "sha256": token["sha256"]},
            "tokenizer_runtime_version": metadata.version("tokenizers"), "add_special_tokens": True},
            "runtime": {"platform": "linux", "container_digest": "a" * 64,
                        "wheels": [{"name": "xgrammar", "sha256": "b" * 64}]},
            "hardware": {"declared": {"accelerator": "fictional"}, "observed_linux_gpu_smoke_receipt": None}}


def _profile():
    return {"version": "mosaic-public-native-driver-profile/v1",
            "model": {"id": "fictional/local-model", "revision": "fictional-revision", "served_model_name": "fictional-local"},
            "source": {"permission_declaration": "user_declared_permitted"},
            "driver": {"candidate": "fictional-native", "run_id": "fictional-native-v1",
                       "max_measure_stages_per_paper": 1, "max_request_seconds": 5,
                       "request_cap": 8, "output_token_cap": 49152,
                       "wall_clock_cap_seconds": 40, "article_request_cap": 8}}


def test_prepares_exact_native_program_without_socket_or_execution(tmp_path, monkeypatch):
    workspace, capture_root = _capture(tmp_path)
    envelope = runtime.load_capture_envelope(capture_root, workspace=workspace)
    weights_root = workspace.mkdir(tmp_path / "weights")
    manifest = _manifest(workspace, weights_root)
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("preparation opened a socket"))
    result = runtime.prepare_local_program(envelope, manifest, weights_root, _profile(), "http://127.0.0.1:8000/v1",
                                           tmp_path / "prepared", workspace=workspace)
    receipt = result["receipt"]
    assert receipt["status"] == "prepared_only_execution_unvalidated"
    assert receipt["execution_api"] == "intentionally_absent"
    assert receipt["network_calls"] == receipt["model_calls"] == 0
    assert receipt["tokenizer_counter_checked"] is True
    assert receipt["local_model_manifest_sha256"] == sha256(Path(result["preflight_path"]).with_name("local-model-manifest.json").read_bytes()).hexdigest()
    assert receipt["driver_profile_sha256"] == sha256(Path(result["preflight_path"]).with_name("driver-profile.json").read_bytes()).hexdigest()
    program = json.loads(Path(result["program_path"]).read_bytes())
    assert program["fixture_mode"] is False
    assert program["context_limit"] == 98304
    assert program["papers"][0]["text_provenance"] == "native_article_text"
    assert program["recipe"]["vllm"]["local_tokenizer"]["sha256"] == manifest["model"]["tokenizer"]["sha256"]
    from mosaic.vllm.tokenizer import local_tokenizer
    assert type(local_tokenizer(program["recipe"]["vllm"])("same local tokenizer contract")) is int


def test_preparation_binds_explicit_server_context_limit(tmp_path):
    workspace, capture_root = _capture(tmp_path)
    envelope = runtime.load_capture_envelope(capture_root, workspace=workspace)
    weights_root = workspace.mkdir(tmp_path / "weights")
    profile = _profile(); profile["driver"]["context_limit"] = 32768
    result = runtime.prepare_local_program(envelope, _manifest(workspace, weights_root), weights_root, profile,
                                           "http://127.0.0.1:8000/v1", tmp_path / "prepared", workspace=workspace)
    program = json.loads(Path(result["program_path"]).read_bytes())
    saved_profile = json.loads((Path(result["program_path"]).parent / "driver-profile.json").read_bytes())
    assert program["context_limit"] == saved_profile["driver"]["context_limit"] == 32768


@pytest.mark.parametrize("url", ["https://127.0.0.1", "http://localhost:8000", "http://127.0.0.1?x=1",
                                  "http://user@127.0.0.1", "http://127.0.0.1/path", "http://127.0.0.1#x",
                                  " http://127.0.0.1", "http://127.0.0.1\n", "HTTP://127.0.0.1",
                                  "http://127.0.0.1:0", "http://127.0.0.1:99999"])
def test_refuses_nonliteral_loopback_forms(tmp_path, url):
    workspace, capture_root = _capture(tmp_path)
    envelope = runtime.load_capture_envelope(capture_root, workspace=workspace)
    weights_root = workspace.mkdir(tmp_path / "weights")
    with pytest.raises(runtime.PublicNativePreparationRefusal, match="loopback_url_refused"):
        runtime.prepare_local_program(envelope, _manifest(workspace, weights_root), weights_root, _profile(), url,
                                      tmp_path / "prepared", workspace=workspace)


def test_refuses_cross_source_binding_and_missing_permission_declaration(tmp_path):
    workspace, capture_root = _capture(tmp_path)
    envelope = runtime.load_capture_envelope(capture_root, workspace=workspace)
    weights_root = workspace.mkdir(tmp_path / "weights")
    manifest = _manifest(workspace, weights_root)
    bad = deepcopy(envelope)
    bad["prepare_upstream_input"]["source_pdf_sha256"] = "0" * 64
    with pytest.raises(runtime.PublicNativePreparationRefusal, match="capture_binding_mismatch"):
        runtime.prepare_local_program(bad, manifest, weights_root, _profile(), "http://127.0.0.1", tmp_path / "one", workspace=workspace)
    profile = _profile(); profile["source"]["permission_declaration"] = "fictional_fixture"
    with pytest.raises(runtime.PublicNativePreparationRefusal, match="permission_required"):
        runtime.prepare_local_program(envelope, manifest, weights_root, profile, "http://127.0.0.1", tmp_path / "two", workspace=workspace)


@pytest.mark.parametrize("mutation,code", [("tokenizer_version", "tokenizer_runtime_mismatch"),
                                             ("tokenizer_parse", "tokenizer_parse_refused"),
                                             ("fractional_cap", "driver_limit_invalid")])
def test_refuses_bad_tokenizer_or_noninteger_caps(tmp_path, mutation, code):
    workspace, capture_root = _capture(tmp_path)
    envelope = runtime.load_capture_envelope(capture_root, workspace=workspace)
    weights_root = workspace.mkdir(tmp_path / "weights")
    manifest, profile = _manifest(workspace, weights_root), _profile()
    if mutation == "tokenizer_version":
        manifest["model"]["tokenizer_runtime_version"] = "0.0.0"
    elif mutation == "tokenizer_parse":
        raw = b"not a tokenizer"; workspace.write_bytes(weights_root / "tokenizer.json", raw, overwrite=True)
        manifest["model"]["files"][0].update(size=len(raw), sha256=sha256(raw).hexdigest())
        manifest["model"]["tokenizer"]["sha256"] = sha256(raw).hexdigest()
    else:
        profile["driver"]["request_cap"] = 1.5
    with pytest.raises(runtime.PublicNativePreparationRefusal, match=code):
        runtime.prepare_local_program(envelope, manifest, weights_root, profile, "http://127.0.0.1", tmp_path / "prepared", workspace=workspace)


def test_repeat_is_identical_but_changed_profile_cannot_rewrite_preparation(tmp_path):
    workspace, capture_root = _capture(tmp_path)
    envelope = runtime.load_capture_envelope(capture_root, workspace=workspace)
    weights_root = workspace.mkdir(tmp_path / "weights")
    manifest, profile, output = _manifest(workspace, weights_root), _profile(), tmp_path / "prepared"
    first = runtime.prepare_local_program(envelope, manifest, weights_root, profile, "http://127.0.0.1:8000", output, workspace=workspace)
    second = runtime.prepare_local_program(envelope, manifest, weights_root, profile, "http://127.0.0.1:8000", output, workspace=workspace)
    assert first["receipt"] == second["receipt"]
    changed = deepcopy(profile); changed["driver"]["request_cap"] = 9
    with pytest.raises(ImmutableArtifactError):
        runtime.prepare_local_program(envelope, manifest, weights_root, changed, "http://127.0.0.1:8000", output, workspace=workspace)


def test_bootstrap_prepare_local_runs_real_preparation_and_reports_typed_url_refusal(tmp_path):
    workspace, capture_root = _capture(tmp_path)
    weights_root = workspace.mkdir(tmp_path / "weights")
    manifest_path = workspace.write_canonical_json(tmp_path / "manifest.json", _manifest(workspace, weights_root))
    profile_path = workspace.write_canonical_json(tmp_path / "profile.json", _profile())
    command = [sys.executable, "bootstrap.py", "prepare-local", "--capture-root", str(capture_root),
               "--model-manifest", str(manifest_path), "--weights-root", str(weights_root),
               "--driver-profile", str(profile_path), "--base-url", "http://127.0.0.1:8000/v1",
               "--output", str(tmp_path / "prepared-cli")]
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    assert completed.returncode == 0
    assert json.loads(completed.stdout)["receipt"]["status"] == "prepared_only_execution_unvalidated"
    rejected = subprocess.run([*command[:command.index("--base-url") + 1], "http://localhost:8000",
                               *command[command.index("--output"):]], cwd=REPO_ROOT, check=False,
                              capture_output=True, text=True)
    assert rejected.returncode == 2
    assert json.loads(rejected.stdout)["reason"] == "public_native_loopback_url_refused"
