"""Mocked process checks for the real Linux worker; no GPU or provider use."""
from pathlib import Path
from types import SimpleNamespace
import io
import json
import time
import pytest

from mosaic.workspace import Workspace


def test_execute_linux_smoke_records_failed_setup_without_starting_server(tmp_path, monkeypatch):
    import mosaic.public_linux_smoke as smoke
    import mosaic.public_export as export
    import platform
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(export, "verify_export", lambda root: {"version":"mosaic-public-export/v1"})
    monkeypatch.setattr(smoke.shutil, "disk_usage", lambda _: SimpleNamespace(free=smoke.MIN_FREE_BYTES))
    ws = Workspace(repo_root=tmp_path, temp_root=tmp_path)
    output, model = tmp_path / "run/out", tmp_path / "model"
    def failed(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout=b"", stderr=b"failed")
    monkeypatch.setattr(smoke.subprocess, "run", failed)
    result = smoke.execute_linux_smoke(tmp_path, model, output, time.time() + 1200)
    assert result["status"] == "failed"
    assert result["reason"] == "public_linux_smoke_download-cli_failed"
    assert (output / "outcome.json").is_file()
    assert (output / "result.zip").is_file()


@pytest.mark.parametrize("exchange_case", ["valid", "forged", "missing", "download_config_missing"])
def test_execute_linux_smoke_success_terminates_loopback_server_and_archives(tmp_path, monkeypatch, exchange_case):
    import mosaic.public_linux_smoke as smoke
    import mosaic.public_export as export
    import mosaic.public_model_inventory as inventory
    import platform
    import urllib.request
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(export, "verify_export", lambda root: {"version":"mosaic-public-export/v1"})
    monkeypatch.setattr(smoke.shutil, "disk_usage", lambda _: SimpleNamespace(free=smoke.MIN_FREE_BYTES))
    output, model = tmp_path / "run/out", tmp_path / "model"
    # This test covers model/exchange lifecycle. The orphan gate has its own
    # whole-inventory and retained-evidence tests, including actual file deltas.
    gate_trace = []
    def orphan_gate(**kwargs):
        gate_trace.append("orphan")
        kwargs["workspace"].write_once(output / "post-repair-freeze.stdout.log", b"fictional==1.0\n")
        kwargs["workspace"].write_canonical_json(output / "dependency-gate.json", {"mock": "lifecycle-only"})
        return {"mock": "lifecycle-only"}
    def orphan_pdf(**kwargs):
        gate_trace.append("pdf")
        kwargs["workspace"].write_canonical_json(output / "dependency-pdf-stage.json", {"mock": "lifecycle-only"})
        return {"mock": "lifecycle-only"}
    monkeypatch.setattr(smoke, "run_orphan_dependency_gate", orphan_gate)
    monkeypatch.setattr(smoke, "validate_orphan_pdf_stage", orphan_pdf)
    probe_proof = {"version": "mosaic-public-model-probe/v1", "request_sha256": "a" * 64,
                   "response_sha256": "b" * 64, "completion_tokens": 1,
                   "finished_epoch": 1.0, "transport_evidence": "observed_local_loopback",
                   "real_model_inference_run": True, "scientifically_accepted": False}
    monkeypatch.setattr(smoke, "verify_retained_model_probe", lambda root: probe_proof)
    def inventory_ok(declaration, root, path, *, workspace):
        workspace.write_canonical_json(path, {"model":{"id":smoke.MODEL,"revision":smoke.REVISION}})
        return {"manifest_path":str(path), "file_count":1}
    monkeypatch.setattr(inventory, "inventory_local_model", inventory_ok)
    commands = []
    dependency_state = {"freeze": 0, "check": 0}
    def completed(command, **kwargs):
        commands.append(command)
        stdout = b""
        if command[:4] == [smoke.sys.executable, "-m", "pip", "freeze"]:
            dependency_state["freeze"] += 1
            plan = smoke.load_plan()
            base = {**__import__("mosaic.public_linux_dependencies", fromlist=["BASE"]).BASE,
                    "torch": "2.13.0+cu130", "nvidia-nccl-cu13": "2.30.7"}
            if dependency_state["freeze"] >= 2:
                base.update(plan.versions)
            if dependency_state["freeze"] >= 3:
                base["pymupdf"] = "1.28.2"
            stdout = ("\n".join(f"{k}=={v}" for k, v in sorted(base.items())) + "\n").encode()
        elif command[:4] == [smoke.sys.executable, "-m", "pip", "check"]:
            dependency_state["check"] += 1
            nccl = __import__("mosaic.public_linux_dependencies", fromlist=["NCCL"]).NCCL
            otel = __import__("mosaic.public_linux_dependencies", fromlist=["OTEL"]).OTEL
            stdout = (f"{otel}\n{nccl}\n" if dependency_state["check"] == 1 else f"{nccl}\n").encode()
            return SimpleNamespace(returncode=1, stdout=stdout, stderr=b"")
        elif command[:4] == [smoke.sys.executable, "-m", "pip", "inspect"]:
            plan = smoke.load_plan()
            values = {**__import__("mosaic.public_linux_dependencies", fromlist=["BASE"]).BASE,
                      "torch": "2.13.0+cu130", "nvidia-nccl-cu13": "2.30.7"}
            if dependency_state["freeze"] >= 2:
                values.update(plan.versions)
            rows = [{"metadata": {"name": name, "version": version,
                    "requires_dist": ['nvidia-nccl-cu13==2.29.7; platform_system == "Linux"'] if name == "torch" else []}}
                    for name, version in values.items()]
            stdout = json.dumps({"installed": rows}).encode()
        if command[:2] == ["hf", "download"]:
            # The old CLI invocation consumed only *.json as --include and
            # treated the remaining patterns as explicit filenames. A zero
            # download exit code then concealed the absent config.json.
            assert command[:7] == ["hf", "download", smoke.MODEL, "--revision", smoke.REVISION, "--local-dir", str(model)]
            assert command[7:] == ["--include", "*.json", "--include", "*.safetensors",
                                   "--include", "*.model", "--include", "*.tiktoken",
                                   "--include", "*.txt", "--include", "*.jinja"]
            if exchange_case != "download_config_missing":
                (model / "config.json").write_text(json.dumps({"architectures":["Qwen3_5ForConditionalGeneration"]}))
        if "inventory_local_model" in " ".join(command):
            stdout = json.dumps({"manifest_path":str(tmp_path / "run/smoke-runtime/local-model-manifest.json"),"file_count":1}).encode()
        if command[:2] == ["nvidia-smi", "--query-gpu=name,memory.total,driver_version"]: stdout = b"NVIDIA A100, 81920 MiB, 1"
        elif command[:3] == [smoke.sys.executable, "-c", "import torch,json; print(json.dumps({'cuda_available':torch.cuda.is_available(),'device_count':torch.cuda.device_count(),'device':torch.cuda.get_device_name(0),'memory':torch.cuda.get_device_properties(0).total_memory,'torch':torch.__version__,'cuda':torch.version.cuda}))"]: stdout = json.dumps({"cuda_available":True,"device_count":1,"device":"A100","memory":80_000_000_000,"torch":"x","cuda":"x"}).encode()
        elif "mosaic.public_model_probe" in command:
            probe_root = Path(command[command.index("--output") + 1])
            probe_root.mkdir()
            (probe_root / "outcome.json").write_bytes(b'{"fictional":true}')
        elif "mosaic.public_linux_smoke" in command:
            fixture, ledger, _, _, _ = _record_exchange(tmp_path, stream=True)
            if exchange_case != "valid":
                ledger.unlink()
            stdout = json.dumps({"candidate_only": True, "scientifically_accepted": False,
                "fixture_sha256": fixture, "model_exchange": {"model_exchange_observed": True}}).encode()
        return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")
    monkeypatch.setattr(smoke.subprocess, "run", completed)
    class Child:
        stdout = io.StringIO("ready\n")
        returncode = None
        terminated = False
        def poll(self): return None
        def terminate(self): self.terminated = True; self.returncode = 0
        def wait(self, timeout): return self.returncode
    child = Child(); popen_commands = []
    def popen(command, *args, **kwargs): popen_commands.append(command); return child
    monkeypatch.setattr(smoke.subprocess, "Popen", popen)
    class Response:
        status = 200
        def read(self, limit): return b'{"data":[{"id":"screen-candidate"}]}'
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: Response())
    result = smoke.execute_linux_smoke(tmp_path, model, output, time.time() + 3500)
    if exchange_case == "download_config_missing":
        assert result["reason"] == "public_linux_smoke_download_config_missing"
        assert not popen_commands
        assert result["real_model_inference_run"] is False
        assert (output / "result.zip").is_file()
        return
    assert result["status"] == ("completed" if exchange_case == "valid" else "failed") and child.terminated
    assert result["real_model_inference_run"] is (exchange_case == "valid")
    assert result["scientifically_accepted"] is False
    assert (output / "result.zip").is_file() and (output / "outcome.json").is_file()
    server = popen_commands[0]
    assert server[server.index("--host") + 1] == "127.0.0.1"
    assert server[server.index("--max-num-seqs") + 1] == "8"
    profile = json.loads((tmp_path / "run/smoke-runtime/driver-profile.json").read_bytes())
    assert profile["driver"]["request_cap"] == 7
    assert profile["driver"]["output_token_cap"] == 49120
    assert profile["driver"]["max_request_seconds"] == 180
    assert profile["driver"]["wall_clock_cap_seconds"] == 900
    assert result["route_schedule"] == {"reserved_requests": 4, "reserved_request_seconds": 720,
                                         "driver_wall_clock_seconds": 900, "phase_cleanup_seconds": 60,
                                         "route_command_seconds": 1200}
    import_index = next(index for index, command in enumerate(commands)
                        if "vllm.entrypoints.openai.api_server" in " ".join(command))
    download_index = next(index for index, command in enumerate(commands)
                          if command[:2] == ["hf", "download"])
    assert import_index < download_index
    assert not any("http://" in " ".join(command) for command in commands)
    assert json.loads((output / "server-readiness.json").read_bytes())["served_model"] == smoke.SERVED
    assert gate_trace == ["orphan", "pdf"]
    assert result["version"] == "mosaic-public-linux-smoke/v6"


def test_route_schedule_refuses_pre_request_v5_budget_and_accepts_v6_profile():
    import mosaic.public_linux_smoke as smoke

    profile = smoke._driver_profile()
    assert smoke._route_schedule(profile)["reserved_request_seconds"] == 720
    profile["driver"]["wall_clock_cap_seconds"] = 600
    with pytest.raises(smoke.PublicLinuxSmokeRefusal, match="route_budget_infeasible"):
        smoke._route_schedule(profile)
    profile = smoke._driver_profile()
    profile["driver"]["article_request_cap"] = 3
    with pytest.raises(smoke.PublicLinuxSmokeRefusal, match="route_request_cap_infeasible"):
        smoke._route_schedule(profile)


def test_orphan_gate_refusal_prevents_pdf_model_and_gpu_work(tmp_path, monkeypatch):
    import platform
    import mosaic.public_linux_smoke as smoke
    import mosaic.public_export as export
    from mosaic.public_linux_dependencies import DependencyRefusal
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(export, "verify_export", lambda _: {})
    monkeypatch.setattr(smoke.shutil, "disk_usage", lambda _: SimpleNamespace(free=smoke.MIN_FREE_BYTES))
    calls = []
    def completed(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=b"hf-version", stderr=b"")
    monkeypatch.setattr(smoke.subprocess, "run", completed)
    def refused(**kwargs):
        raise DependencyRefusal("public_linux_smoke_orphan_reverse_dependency")
    monkeypatch.setattr(smoke, "run_orphan_dependency_gate", refused)
    monkeypatch.setattr(smoke.subprocess, "Popen", lambda *a, **kw: pytest.fail("no model process"))
    result = smoke.execute_linux_smoke(tmp_path, tmp_path / "model", tmp_path / "run/out", time.time() + 3500)
    assert result["status"] == "failed" and result["real_model_inference_run"] is False
    assert result["reason"] == "public_linux_smoke_orphan_reverse_dependency"
    assert calls == [["hf", "--version"]]
    assert (tmp_path / "run/out/result.zip").is_file()


def test_vllm_import_failure_prevents_download_and_retains_archive(tmp_path, monkeypatch):
    import platform
    import mosaic.public_linux_smoke as smoke
    import mosaic.public_export as export

    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(export, "verify_export", lambda _: {})
    monkeypatch.setattr(smoke.shutil, "disk_usage", lambda _: SimpleNamespace(free=smoke.MIN_FREE_BYTES))
    output, model = tmp_path / "run/out", tmp_path / "model"
    def gate(**kwargs):
        kwargs["workspace"].write_once(output / "post-repair-freeze.stdout.log", b"fictional==1.0\n")
        kwargs["workspace"].write_canonical_json(output / "dependency-gate.json", {"mock": True})
        return {"mock": True}
    def pdf_stage(**kwargs):
        kwargs["workspace"].write_canonical_json(output / "dependency-pdf-stage.json", {"mock": True})
        return {"mock": True}
    monkeypatch.setattr(smoke, "run_orphan_dependency_gate", gate)
    monkeypatch.setattr(smoke, "validate_orphan_pdf_stage", pdf_stage)
    calls = []
    def completed(command, **kwargs):
        calls.append(command)
        if "vllm.entrypoints.openai.api_server" in " ".join(command):
            return SimpleNamespace(returncode=1, stdout=b"", stderr=b"fictional import failure")
        if command[:2] == ["nvidia-smi", "--query-gpu=name,memory.total,driver_version"]:
            return SimpleNamespace(returncode=0, stdout=b"NVIDIA A100, 81920 MiB, 1", stderr=b"")
        if command[:3] == [smoke.sys.executable, "-c", "import torch,json; print(json.dumps({'cuda_available':torch.cuda.is_available(),'device_count':torch.cuda.device_count(),'device':torch.cuda.get_device_name(0),'memory':torch.cuda.get_device_properties(0).total_memory,'torch':torch.__version__,'cuda':torch.version.cuda}))"]:
            return SimpleNamespace(returncode=0, stdout=json.dumps({"cuda_available": True, "device_count": 1,
                "device": "A100", "memory": 80_000_000_000, "torch": "x", "cuda": "x"}).encode(), stderr=b"")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
    monkeypatch.setattr(smoke.subprocess, "run", completed)
    monkeypatch.setattr(smoke.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("server must not start"))
    result = smoke.execute_linux_smoke(tmp_path, model, output, time.time() + 3500)
    assert result["status"] == "failed"
    assert result["reason"] == "public_linux_smoke_vllm-import_failed"
    assert not any(command[:2] == ["hf", "download"] for command in calls)
    assert (output / "vllm-import.stdout.log").is_file()
    assert (output / "vllm-import.stderr.log").is_file()
    assert (output / "result.zip").is_file()


@pytest.mark.parametrize("body", [b'{"data":[{"id":"wrong"}]}', b'{"data":[]}', b'{"data":[{"id":"screen-candidate"},{"id":"extra"}]}', b'x' * 65537], ids=["wrong", "empty", "ambiguous", "oversize"])
def test_readiness_refuses_wrong_ambiguous_or_unbounded_model(body):
    import mosaic.public_linux_smoke as smoke
    class Response:
        status = 200
        def read(self, limit): return body[:limit]
        def __enter__(self): return self
        def __exit__(self, *args): pass
    with pytest.raises(smoke.PublicLinuxSmokeRefusal):
        smoke._server_readiness(lambda *args, **kwargs: Response())


def test_insufficient_disk_prevents_download_and_retains_failure(tmp_path, monkeypatch):
    import mosaic.public_linux_smoke as smoke
    import mosaic.public_export as export
    import platform
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(export, "verify_export", lambda _: {})
    monkeypatch.setattr(smoke.shutil, "disk_usage", lambda _: SimpleNamespace(free=smoke.MIN_FREE_BYTES - 1))
    monkeypatch.setattr(smoke.subprocess, "run", lambda *a, **kw: pytest.fail("No process may start"))
    result = smoke.execute_linux_smoke(tmp_path, tmp_path / "model", tmp_path / "run/out", time.time() + 3500)
    assert result["reason"] == "public_linux_smoke_disk_headroom_unavailable"
    assert result["real_model_inference_run"] is False
    assert (tmp_path / "run/out/result.zip").is_file()


def test_setup_timeout_preserves_partial_logs(tmp_path, monkeypatch):
    import mosaic.public_linux_smoke as smoke
    import mosaic.public_export as export
    import platform
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(export, "verify_export", lambda _: {})
    monkeypatch.setattr(smoke.shutil, "disk_usage", lambda _: SimpleNamespace(free=smoke.MIN_FREE_BYTES))
    def timed_out(command, **kwargs):
        raise smoke.subprocess.TimeoutExpired(command, 30, output=b"partial", stderr=b"diagnostic")
    monkeypatch.setattr(smoke.subprocess, "run", timed_out)
    result = smoke.execute_linux_smoke(tmp_path, tmp_path / "model", tmp_path / "run/out", time.time() + 3500)
    assert result["reason"] == "public_linux_smoke_download-cli_timeout"
    assert (tmp_path / "run/out/download-cli.stdout.log").read_bytes() == b"partial"
    assert result["commands"][0]["timed_out"] is True


def _record_exchange(root, *, stream=False, outcome="success"):
    import hashlib
    import mosaic.public_linux_smoke as smoke

    pdf = b"Fictional protocol-test source; not scientific evidence."
    fixture_sha = hashlib.sha256(pdf).hexdigest()
    source = root / "example-data/pdfs/fictional-build.pdf"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(pdf)
    work = "public-" + fixture_sha
    from mosaic.canonical import canonical_bytes
    packet = {"work_id": work}
    capture = {"work_id": work, "source_pdf_sha256": fixture_sha,
               "source_packet_sha256": hashlib.sha256(canonical_bytes(packet)).hexdigest()}
    capture_root = root / "run/linux-smoke/stages/capture" / ("d" * 64) / "attempt-1/papers" / work
    capture_root.mkdir(parents=True)
    (capture_root / "receipt.json").write_bytes(json.dumps({**capture,
        "version": "public-capture-source-packet/v1", "status": "source_packet_ready"}).encode())
    (capture_root / "source.json").write_bytes(canonical_bytes(packet))
    (capture_root / "source-binding.json").write_bytes(json.dumps(capture).encode())
    (capture_root / "prepare-upstream-input.json").write_bytes(json.dumps({
        "paper_id": work, "source_pdf_sha256": fixture_sha, "source": packet}).encode())
    (capture_root / "source.pdf").write_bytes(pdf)
    ledger = root / "run/linux-smoke/native" / work / ("c" * 64) / "execution/ledger/smoke.jsonl"
    ledger.parent.mkdir(parents=True)
    key = "idem-" + "b" * 64
    request = json.dumps({"model": smoke.SERVED, "stream": stream}).encode()
    response_json = {"model": smoke.SERVED, "choices": [{"index": 0, "message": {"content": "{}"},
                     "finish_reason": "stop"}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}
    if stream:
        event = {"model": smoke.SERVED, "choices": [{"index": 0, "delta": {"content": "{}"},
                 "finish_reason": "stop"}], "usage": response_json["usage"]}
        response = ("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode()
    else:
        response = json.dumps(response_json).encode()
    wire = ledger.parent / "smoke.private.responses" / key / "1"
    wire.mkdir(parents=True)
    (wire / "request.bin").write_bytes(request)
    (wire / "response.bin").write_bytes(response)
    common = {"idempotency_key": key, "attempt": 1, "route": "chat_completions",
              "target_work_id": work, "source_sha256s": [fixture_sha],
              "request_sha256": hashlib.sha256(request).hexdigest()}
    intent = {**common, "row_kind": "intent", "request_body_sha256": common["request_sha256"]}
    result = {**common, "row_kind": "result", "outcome": outcome, "http_status": 200,
              "model": smoke.SERVED, "input_tokens": 3, "output_tokens": 2,
              "response_bytes": len(response), "response_sha256": hashlib.sha256(response).hexdigest()}
    _exchange_rows(ledger, [intent, result])
    prepared = ledger.parents[2] / "prepared"
    prepared.mkdir()
    prepared_raw = json.dumps({"capture": capture}).encode()
    (prepared / "prepared-receipt.json").write_bytes(prepared_raw)
    execution = ledger.parents[1]
    intent_raw = json.dumps({"status": "execution_started", "capture": capture,
        "prepared_receipt_sha256": hashlib.sha256(prepared_raw).hexdigest()}).encode()
    (execution / "execution-intent.json").write_bytes(intent_raw)
    receipt = {"status": "terminal", "native_execution_completed": True, "scientifically_accepted": False,
               "capture": capture, "prepared_receipt_sha256": hashlib.sha256(prepared_raw).hexdigest()}
    (execution / "execution-receipt.json").write_bytes(json.dumps(receipt).encode())
    _pin_exchange_receipt(ledger)
    return fixture_sha, ledger, wire, intent, result


def _exchange_rows(ledger, rows):
    ledger.write_bytes(b"".join(json.dumps(row).encode() + b"\n" for row in rows))


def _pin_exchange_receipt(ledger):
    from hashlib import sha256
    execution = ledger.parents[1]
    path = execution / "execution-receipt.json"
    receipt = json.loads(path.read_bytes())
    receipt["artifact_hashes"] = {p.relative_to(execution).as_posix(): sha256(p.read_bytes()).hexdigest()
                                  for p in ledger.parent.rglob("*") if p.is_file()}
    path.write_bytes(json.dumps(receipt).encode())


@pytest.mark.parametrize("stream", [False, True], ids=["json", "actual-stream-protocol"])
def test_exchange_requires_bound_retained_response(tmp_path, stream):
    import mosaic.public_linux_smoke as smoke
    fixture, _, _, _, _ = _record_exchange(tmp_path, stream=stream)
    evidence = smoke._verified_model_exchange(tmp_path, fixture)
    assert evidence["model_exchange_status"] == "observed"
    assert evidence["model_exchange_observed"] is True
    assert evidence["scientifically_accepted"] is False

@pytest.mark.parametrize("case", ["missing", "refused", "empty", "unknown"])
def test_exchange_absence_and_refusal_are_not_inference(tmp_path, case):
    import mosaic.public_linux_smoke as smoke
    fixture, ledger, _, intent, result = _record_exchange(tmp_path, outcome="http_error")
    if case == "missing":
        ledger.unlink()
    elif case == "empty":
        _exchange_rows(ledger, [])
        _pin_exchange_receipt(ledger)
    elif case == "unknown":
        result["outcome"] = "unknown_outcome"
        _exchange_rows(ledger, [intent, result])
    evidence = smoke._verified_model_exchange(tmp_path, fixture)
    assert evidence["model_exchange_observed"] is False
    assert evidence["model_exchange_status"] == ("false" if case in ("empty", "refused") else "unresolved")


@pytest.mark.parametrize("case", ["request-bytes", "response-bytes", "response-usage", "response-model",
    "boolean-usage", "orphan", "duplicate", "malformed-key", "path-key", "wrong-source", "result-source",
    "invalid-row", "oversize", "incomplete-stream"])
def test_exchange_corrupt_evidence_is_unresolved(tmp_path, case):
    from hashlib import sha256
    import mosaic.public_linux_smoke as smoke
    fixture, ledger, wire, intent, result = _record_exchange(tmp_path, stream=case == "incomplete-stream")
    rows = [intent, result]
    if case in ("request-bytes", "response-bytes"):
        (wire / ("request.bin" if case == "request-bytes" else "response.bin")).write_bytes(b"changed")
    elif case in ("response-usage", "response-model"):
        response = json.loads((wire / "response.bin").read_bytes())
        if case == "response-usage":
            response["usage"]["completion_tokens"] = 99
        else:
            response["model"] = "wrong-model"
        raw = json.dumps(response).encode()
        (wire / "response.bin").write_bytes(raw)
        result.update(response_bytes=len(raw), response_sha256=sha256(raw).hexdigest())
    elif case == "boolean-usage": result["input_tokens"] = True
    elif case == "orphan": rows = [result]
    elif case == "duplicate": rows.append(intent)
    elif case == "malformed-key": intent["idempotency_key"] = []
    elif case == "path-key": intent["idempotency_key"] = "../../outside"
    elif case == "wrong-source": intent["source_sha256s"] = ["d" * 64]
    elif case == "result-source": result["source_sha256s"] = ["d" * 64]
    elif case == "invalid-row": rows = [[]]
    elif case == "oversize":
        (wire / "response.bin").write_bytes(b"x" * (8 * 1024 * 1024 + 1))
    elif case == "incomplete-stream":
        raw = (wire / "response.bin").read_bytes().replace(b"data: [DONE]\n\n", b"")
        (wire / "response.bin").write_bytes(raw)
        result.update(response_bytes=len(raw), response_sha256=sha256(raw).hexdigest())
    _exchange_rows(ledger, rows)
    _pin_exchange_receipt(ledger)
    evidence = smoke._verified_model_exchange(tmp_path, fixture)
    assert evidence["model_exchange_status"] == "unresolved"
    assert evidence["model_exchange_observed"] is False


def test_exchange_refuses_wrong_fixture_and_linked_wire_ancestor(tmp_path):
    import mosaic.public_linux_smoke as smoke
    fixture, ledger, wire, _, _ = _record_exchange(tmp_path)
    assert smoke._verified_model_exchange(tmp_path, "e" * 64)["model_exchange_observed"] is False
    renamed = wire.with_name("original-wire")
    wire.rename(renamed)
    try:
        wire.symlink_to(renamed, target_is_directory=True)
    except OSError:
        pytest.skip("OS does not permit creating a test symlink")
    assert smoke._verified_model_exchange(tmp_path, fixture)["model_exchange_status"] == "unresolved"


@pytest.mark.parametrize("case", ["wrong-capture", "changed-packet", "stale-ledger", "stale-wire",
                                 "nonterminal", "receipt-type"])
def test_exchange_requires_execution_and_capture_provenance(tmp_path, case):
    from hashlib import sha256
    import mosaic.public_linux_smoke as smoke
    fixture, ledger, wire, _, _ = _record_exchange(tmp_path)
    execution = ledger.parents[1]
    path = execution / "execution-receipt.json"
    receipt = json.loads(path.read_bytes())
    if case == "wrong-capture":
        prepared_path = execution.parent / "prepared/prepared-receipt.json"
        prepared = json.loads(prepared_path.read_bytes())
        prepared["capture"]["source_pdf_sha256"] = "f" * 64
        raw = json.dumps(prepared).encode()
        prepared_path.write_bytes(raw)
        intent_path = execution / "execution-intent.json"
        intent = json.loads(intent_path.read_bytes())
        for row in (intent, receipt):
            row["capture"] = prepared["capture"]
            row["prepared_receipt_sha256"] = sha256(raw).hexdigest()
        intent_path.write_bytes(json.dumps(intent).encode())
    elif case == "changed-packet":
        packet = next((tmp_path / "run/linux-smoke/stages/capture").glob("*/attempt-*/papers/*/source.json"))
        packet.write_bytes(b'{"work_id":"changed"}')
    elif case == "stale-ledger": receipt["artifact_hashes"]["ledger/smoke.jsonl"] = "0" * 64
    elif case == "stale-wire":
        receipt["artifact_hashes"][(wire / "response.bin").relative_to(execution).as_posix()] = "0" * 64
    elif case == "nonterminal": receipt["status"] = "started"
    elif case == "receipt-type": receipt = []
    path.write_bytes(json.dumps(receipt).encode())
    evidence = smoke._verified_model_exchange(tmp_path, fixture)
    assert evidence["model_exchange_status"] == "unresolved"
    assert evidence["model_exchange_observed"] is False


@pytest.mark.evidence_mock
def test_exchange_verifies_actual_fictional_corpus_workflow(tmp_path):
    from hashlib import sha256
    from http.server import ThreadingHTTPServer
    from threading import Thread
    import mosaic.public_linux_smoke as smoke
    from mosaic.public_workflow import run_public_corpus
    from test_public_execution import _Loopback
    from test_public_workflow import _inputs

    config, kwargs = _inputs(tmp_path)
    config["run"]["root"] = "run/linux-smoke"
    kwargs["driver_profile"]["model"]["served_model_name"] = smoke.SERVED
    # The exact GPU envelope is separately checked above; this route uses a
    # fast fictional loopback to exercise capture -> native -> driver wiring.
    assert smoke._route_schedule(smoke._driver_profile())["reserved_request_seconds"] == 720
    class Loopback(_Loopback):
        def do_GET(self):
            body = json.dumps({"data": [{"id": smoke.SERVED}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Loopback)
    thread = Thread(target=server.serve_forever)
    thread.start()
    config["runtime"]["base_url"] = f"http://127.0.0.1:{server.server_address[1]}/v1"
    raw = (tmp_path / "sources/fictional.pdf").read_bytes()
    kwargs["workspace"].write_once(tmp_path / "example-data/pdfs/fictional-build.pdf", raw)
    try:
        result = run_public_corpus(config, **kwargs, live=True)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    assert result["status"] == "candidate_corpus_served"
    evidence = smoke._verified_model_exchange(tmp_path, sha256(raw).hexdigest())
    assert evidence["model_exchange_status"] == "observed", evidence
    assert evidence["scientifically_accepted"] is False

