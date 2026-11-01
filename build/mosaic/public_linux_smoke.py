"""Execute one fictional public-build smoke on an explicitly supplied Linux GPU.

This module has no rental API or provider credential. A separate owner supplies
the GPU and supervises deletion; inference binds only to local loopback.
"""
from __future__ import annotations

from hashlib import sha256
from importlib import metadata
import io
import json
import argparse
import os
import shutil
import sys
import re
from pathlib import Path
import subprocess
import threading
import time
from typing import Any, Mapping
from zipfile import ZIP_DEFLATED, ZipFile

from mosaic.canonical import canonical_bytes
from mosaic.public_linux_dependencies import (CONFIG as DEPENDENCY_CONFIG, load_plan, parse_freeze as parse_legacy_freeze,
    run_dependency_gate, validate_pdf_stage, PDF_WHEEL, PDF_WHEEL_SHA, PDF_WHEEL_URL,
    PDF_FREEZE_VALUES, OTEL_IMPORT_CODE)
from mosaic.public_linux_orphan_dependencies import (OrphanRepairRefusal, load_plan as load_orphan_plan,
    parse_freeze as parse_orphan_freeze, run_orphan_dependency_gate, validate_orphan_pdf_stage)
from mosaic.public_model_probe import MAX_SECONDS as PROBE_MAX_SECONDS, verify_retained_model_probe
from mosaic.workspace import Workspace


VERSION = "mosaic-public-linux-smoke/v6"
IMAGE = "vllm/vllm-openai@sha256:6ee133356d49ed24cee03dd0cda3d3980279a11c107ce3e752c89e76540f41c9"
MODEL = "Qwen/Qwen3.6-27B"
REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
SERVED = "screen-candidate"
# Public upstream file metadata for the exact revision, observed 2026-10-29.
# Reserve a second tree for transfer/cache overhead plus 10 GB for runtime work.
MODEL_DOWNLOAD_BYTES = 55_586_032_434
MIN_FREE_BYTES = 2 * MODEL_DOWNLOAD_BYTES + 10_000_000_000
MAX_WORK_SECONDS = 3600
MODEL_DOWNLOAD_PATTERNS = ("*.json", "*.safetensors", "*.model", "*.tiktoken", "*.txt", "*.jinja")
# The real route freezes two upstream requests and must reserve a possible
# study-facts plus one measure request after a core response.  These bounds are
# checked before download: a later GPU cannot make an impossible finite plan
# executable.
ROUTE_MAX_REQUESTS = 4
DRIVER_MAX_REQUEST_SECONDS = 180
DRIVER_WALL_CLOCK_SECONDS = 900
PHASE_CLEANUP_SECONDS = 60
ROUTE_COMMAND_SECONDS = 1200
ROUTE_FINAL_RESERVE_SECONDS = 20
SERVER_CONTEXT_LIMIT = 32768


def model_download_command(model_root: Path) -> list[str]:
    """The pinned hf CLI accepts exactly one value per --include occurrence."""
    command = ["hf", "download", MODEL, "--revision", REVISION, "--local-dir", str(model_root)]
    for pattern in MODEL_DOWNLOAD_PATTERNS:
        command.extend(["--include", pattern])
    return command


def _driver_profile() -> dict[str, Any]:
    return {"version": "mosaic-public-native-driver-profile/v1", "model": {"id": MODEL,
        "revision": REVISION, "served_model_name": SERVED},
        "source": {"permission_declaration": "user_declared_permitted"},
        "driver": {"candidate": "public-linux-smoke-v1", "run_id": "public-linux-smoke-v1",
        "max_measure_stages_per_paper": 1, "max_request_seconds": DRIVER_MAX_REQUEST_SECONDS,
        "request_cap": 7, "output_token_cap": 49120,
        "wall_clock_cap_seconds": DRIVER_WALL_CLOCK_SECONDS, "article_request_cap": 7,
        "context_limit": SERVER_CONTEXT_LIMIT}}


def _route_schedule(profile: Mapping[str, Any]) -> dict[str, int]:
    """Validate the conservative finite-route envelope before model acquisition.

    The complete runner reserves all unknown dependent stages before its first
    send.  This deliberately mirrors that reservation instead of relying on
    typical response latency.
    """
    driver = profile.get("driver") if isinstance(profile, Mapping) else None
    _need(isinstance(driver, Mapping), "public_linux_smoke_driver_profile_refused")
    measure_stages = driver.get("max_measure_stages_per_paper")
    max_request = driver.get("max_request_seconds")
    request_cap = driver.get("request_cap")
    wall_clock = driver.get("wall_clock_cap_seconds")
    article_request_cap = driver.get("article_request_cap")
    context_limit = driver.get("context_limit")
    _need(type(measure_stages) is int and measure_stages >= 0
          and type(max_request) is int and max_request > 0
          and type(request_cap) is int and request_cap > 0
          and type(wall_clock) is int and wall_clock > 0
          and type(article_request_cap) is int and article_request_cap > 0
          and context_limit == SERVER_CONTEXT_LIMIT,
          "public_linux_smoke_driver_profile_refused")
    # findings + core + possible study facts + all permitted measure stages
    reserved_requests = 3 + measure_stages
    reserved_seconds = reserved_requests * max_request
    _need(reserved_requests == ROUTE_MAX_REQUESTS, "public_linux_smoke_route_roster_infeasible")
    _need(request_cap >= reserved_requests and article_request_cap >= reserved_requests,
          "public_linux_smoke_route_request_cap_infeasible")
    _need(wall_clock >= reserved_seconds, "public_linux_smoke_route_budget_infeasible")
    _need(ROUTE_COMMAND_SECONDS >= wall_clock + PHASE_CLEANUP_SECONDS,
          "public_linux_smoke_route_deadline_infeasible")
    return {"reserved_requests": reserved_requests, "reserved_request_seconds": reserved_seconds,
            "driver_wall_clock_seconds": wall_clock, "phase_cleanup_seconds": PHASE_CLEANUP_SECONDS,
            "route_command_seconds": ROUTE_COMMAND_SECONDS}

class PublicLinuxSmokeRefusal(ValueError):
    pass


def _need(value: bool, code: str) -> None:
    if not value:
        raise PublicLinuxSmokeRefusal(code)


def _sha(data: bytes) -> str:
    return sha256(data).hexdigest()


def _safe_failure_reason(exc: Exception) -> str:
    """Keep machine-actionable local refusal codes, never arbitrary exception text."""
    reason = str(exc)
    if reason.startswith("public_linux_smoke_"):
        return reason
    if isinstance(exc, OrphanRepairRefusal) and re.fullmatch(r"orphan_repair_[a-z0-9_]+", reason):
        return reason
    return type(exc).__name__


def _exchange_file(public_root: Path, path: Path, limit: int) -> bytes:
    from mosaic.public_build import _no_links

    _no_links(path, public_root)
    _need(path.is_file() and path.stat().st_size <= limit,
          "public_linux_smoke_exchange_file_refused")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    _need(len(raw) <= limit, "public_linux_smoke_exchange_file_refused")
    return raw


def _verified_model_exchange(public_root: Path, fixture_sha: object) -> dict[str, Any]:
    """Verify a completed exchange in this fictional run, without emitting text.

    This observes retained protocol evidence; model-weight identity and scientific
    accuracy have separate contracts. Missing or contradictory evidence is unresolved.
    """
    import re
    from mosaic.public_build import _no_links
    from mosaic.public_native_runtime import load_capture_envelope
    from mosaic.vllm.streaming import decode_chat_stream

    base = {"version": "mosaic-public-model-exchange/v1",
            "model_exchange_observed": False, "scientifically_accepted": False}

    def need(condition: bool, reason: str) -> None:
        if not condition:
            raise PublicLinuxSmokeRefusal(reason)

    def is_hash(value: object) -> bool:
        return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None

    try:
        need(is_hash(fixture_sha), "fixture_hash_invalid")
        _no_links(public_root, public_root)
        pdf = _exchange_file(public_root, public_root / "example-data/pdfs/fictional-build.pdf", 2 * 1024 * 1024)
        need(_sha(pdf) == fixture_sha, "fixture_hash_mismatch")
        work = "public-" + fixture_sha
        work_root = public_root / "run/linux-smoke/native" / work
        _no_links(work_root, public_root)
        roots = list(work_root.iterdir())
        need(len(roots) == 1 and is_hash(roots[0].name), "execution_root_ambiguous")
        native_root = roots[0]
        execution_root = native_root / "execution"
        ledger_root = execution_root / "ledger"
        _no_links(ledger_root, public_root)
        prepared_paths = list(native_root.glob("prepared/prepared-receipt.json"))
        need(len(prepared_paths) == 1, "prepared_receipt_ambiguous")
        prepared_raw = _exchange_file(public_root, prepared_paths[0], 256 * 1024)
        intent_raw = _exchange_file(public_root, execution_root / "execution-intent.json", 256 * 1024)
        receipt_raw = _exchange_file(public_root, execution_root / "execution-receipt.json", 1024 * 1024)
        prepared, execution_intent, execution_receipt = (json.loads(raw) for raw in (prepared_raw, intent_raw, receipt_raw))
        need(all(isinstance(value, dict) for value in (prepared, execution_intent, execution_receipt)),
             "execution_receipt_invalid")
        capture = prepared.get("capture") if isinstance(prepared, dict) else None
        need(isinstance(capture, dict) and capture == execution_intent.get("capture") == execution_receipt.get("capture"),
             "execution_capture_binding_invalid")
        need(capture.get("work_id") == work and capture.get("source_pdf_sha256") == fixture_sha
             and is_hash(capture.get("source_packet_sha256")), "execution_capture_identity_invalid")
        need(execution_intent.get("status") == "execution_started" and execution_receipt.get("status") == "terminal"
             and execution_receipt.get("native_execution_completed") is True
             and execution_receipt.get("scientifically_accepted") is False, "execution_not_terminal")
        need(execution_intent.get("prepared_receipt_sha256") == _sha(prepared_raw)
             and execution_receipt.get("prepared_receipt_sha256") == _sha(prepared_raw), "prepared_receipt_binding_invalid")
        capture_roots = []
        capture_stage = public_root / "run/linux-smoke/stages/capture"
        _no_links(capture_stage, public_root)
        for candidate in capture_stage.glob("*/attempt-*/papers/" + work + "/receipt.json"):
            try:
                _no_links(candidate, public_root)
                envelope = load_capture_envelope(candidate.parent.absolute(), workspace=Workspace(repo_root=public_root))
            except (OSError, ValueError):
                continue
            if envelope["capture_receipt"].get("work_id") == capture.get("work_id") and envelope["capture_receipt"].get("source_pdf_sha256") == capture.get("source_pdf_sha256") and envelope["capture_receipt"].get("source_packet_sha256") == capture.get("source_packet_sha256"):
                capture_roots.append(candidate.parent)
        need(len(capture_roots) == 1, "capture_root_ambiguous")
        ledgers = list(ledger_root.glob("*.jsonl"))
        need(len(ledgers) == 1, "ledger_ambiguous")
        ledger = ledgers[0]
        ledger_key = ledger.relative_to(execution_root).as_posix()
        artifact_hashes = execution_receipt.get("artifact_hashes")
        need(isinstance(artifact_hashes, dict), "execution_artifact_hashes_invalid")
        need(artifact_hashes.get(ledger_key) == _sha(_exchange_file(public_root, ledger, 2 * 1024 * 1024)),
             "execution_receipt_ledger_binding_invalid")
        raw_ledger = _exchange_file(public_root, ledger, 2 * 1024 * 1024)
        rows = [json.loads(line) for line in raw_ledger.splitlines() if line.strip()]
        need(len(rows) <= 32, "ledger_row_cap_exceeded")
        paired: dict[str, dict[tuple[str, int], dict]] = {"intent": {}, "result": {}}
        for row in rows:
            need(isinstance(row, dict) and row.get("row_kind") in paired, "ledger_row_invalid")
            key, attempt = row.get("idempotency_key"), row.get("attempt")
            need(isinstance(key, str) and re.fullmatch(r"idem-[0-9a-f]{64}", key) is not None
                 and type(attempt) is int and 1 <= attempt <= 2, "ledger_key_invalid")
            slot = paired[row["row_kind"]]
            need((key, attempt) not in slot, "duplicate_ledger_key")
            slot[(key, attempt)] = row
        intents, results = paired["intent"], paired["result"]
        need(intents.keys() == results.keys(), "ledger_pairing_invalid")
        verified = []
        for key, intent in intents.items():
            result = results[key]
            outcome = result.get("outcome")
            need(outcome in ("success", "http_error", "transport_error"), "exchange_outcome_unresolved")
            if outcome != "success" or intent.get("route") != "chat_completions":
                continue
            need(intent.get("target_work_id") == work and intent.get("source_sha256s") == [fixture_sha],
                 "exchange_source_invalid")
            need(all(result.get(name) == intent.get(name) for name in
                     ("route", "target_work_id", "source_sha256s", "request_sha256")), "result_binding_invalid")
            need(result.get("http_status") == 200 and result.get("model") == SERVED,
                 "exchange_response_invalid")
            need(all(type(result.get(name)) is int and result[name] > 0 for name in
                     ("input_tokens", "output_tokens", "response_bytes")), "exchange_usage_invalid")
            need(all(is_hash(value) for value in (intent.get("request_sha256"),
                 intent.get("request_body_sha256"), result.get("response_sha256"))), "exchange_digest_invalid")
            wire_root = ledger.parent / (ledger.stem + ".private.responses") / key[0] / str(key[1])
            request_raw = _exchange_file(public_root, wire_root / "request.bin", 8 * 1024 * 1024)
            response_raw = _exchange_file(public_root, wire_root / "response.bin", 8 * 1024 * 1024)
            need(all(artifact_hashes.get((wire_root / name).relative_to(execution_root).as_posix()) == _sha(raw)
                     for name, raw in (("request.bin", request_raw), ("response.bin", response_raw))),
                 "execution_receipt_wire_binding_invalid")
            need(_sha(request_raw) == intent["request_body_sha256"] == intent["request_sha256"]
                 and _sha(response_raw) == result["response_sha256"]
                 and len(response_raw) == result["response_bytes"], "wire_hash_mismatch")
            request = json.loads(request_raw)
            need(isinstance(request, dict) and request.get("model") == SERVED, "request_model_invalid")
            stream = request.get("stream", False)
            need(type(stream) is bool, "request_stream_invalid")
            response = decode_chat_stream(response_raw) if stream else json.loads(response_raw)
            need(isinstance(response, dict) and response.get("model") == SERVED, "response_model_invalid")
            usage = response.get("usage")
            need(isinstance(usage, dict), "response_usage_invalid")
            prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
            completion = usage.get("completion_tokens", usage.get("output_tokens"))
            need(type(prompt) is int and type(completion) is int and
                 prompt == result["input_tokens"] and completion == result["output_tokens"], "response_usage_invalid")
            choices = response.get("choices")
            need(isinstance(choices, list) and bool(choices) and isinstance(choices[0], dict), "response_choices_invalid")
            message = choices[0].get("message")
            need(isinstance(message, dict) and any(isinstance(message.get(field), str) and message[field]
                 for field in ("content", "reasoning_content")), "response_content_missing")
            verified.append({"request_sha256": intent["request_sha256"], "response_sha256": result["response_sha256"]})
        return {**base, "model_exchange_observed": bool(verified),
                "model_exchange_status": "observed" if verified else "false",
                "ledger_sha256": _sha(raw_ledger), "verified_exchange_hashes": verified, "reasons": []}
    except (OSError, ValueError, TypeError, UnicodeError) as exc:
        reason = str(exc) if isinstance(exc, PublicLinuxSmokeRefusal) else "exchange_artifact_invalid"
        return {**base, "model_exchange_status": "unresolved", "reasons": [reason]}


def _server_readiness(opener) -> dict[str, str]:
    """Require the intended local model before sending any fictional source."""
    from urllib.request import Request
    hashes = {}
    for endpoint in ("health", "v1/models"):
        with opener(Request("http://127.0.0.1:8000/" + endpoint), timeout=3) as response:
            _need(response.status == 200, "public_linux_smoke_server_not_ready")
            body = response.read(65537)
        _need(len(body) <= 65536, "public_linux_smoke_readiness_response_too_large")
        hashes[endpoint] = _sha(body)
        if endpoint == "v1/models":
            payload = json.loads(body)
            _need(isinstance(payload, dict) and isinstance(payload.get("data"), list)
                  and len(payload["data"]) == 1 and isinstance(payload["data"][0], dict)
                  and payload["data"][0].get("id") == SERVED,
                  "public_linux_smoke_served_model_mismatch")
    return hashes


def _archive_results(public_root: Path, output: Path, workspace: Workspace) -> Path:
    """Retain generated fictional evidence and diagnostics, excluding model bytes."""
    rows = []
    for folder in (public_root / "run", public_root / "example-data"):
        for path in sorted(folder.rglob("*")) if folder.exists() else []:
            if path.is_file() and path.name != "result.zip" and not path.is_symlink():
                rows.append((path.relative_to(public_root).as_posix(), path.read_bytes()))
    raw = io.BytesIO()
    with ZipFile(raw, "w", ZIP_DEFLATED) as archive:
        for name, data in rows:
            archive.writestr(name, data)
    return workspace.write_once(output / "result.zip", raw.getvalue())


def verify_retained_dependency_gate(public_root: Path, *, expected_config_sha256: str) -> dict[str, Any]:
    """Recheck retained dependency evidence without invoking pip or Linux imports."""
    from mosaic.public_linux_dependencies import classify_baseline, validate_metadata, validate_post

    base = public_root / "run/gpu-smoke"
    def read(name: str) -> bytes:
        return _exchange_file(public_root, base / name, 16 * 1024 * 1024)
    def obj(name: str) -> dict:
        value = json.loads(read(name))
        _need(isinstance(value, dict), "public_linux_smoke_dependency_object_invalid")
        return value
    def raw(label: str, kind: str) -> bytes:
        return read(f"{label}.{kind}.log")

    plan_raw = read("dependency-repair-plan.json")
    _need(_sha(plan_raw) == expected_config_sha256, "public_linux_smoke_dependency_plan_hash_invalid")
    plan = load_plan()
    _need(plan.config_sha256 == expected_config_sha256 and plan_raw == DEPENDENCY_CONFIG.read_bytes(),
          "public_linux_smoke_dependency_plan_drift")
    gate = obj("dependency-gate.json")
    _need(gate.get("version") == "mosaic-public-linux-dependency-gate/v1" and gate.get("image") == IMAGE
          and gate.get("plan_sha256") == expected_config_sha256 and gate.get("vllm_source_commit") == plan.source_commit
          and gate.get("scientifically_accepted") is False,
          "public_linux_smoke_dependency_gate_invalid")
    outcome = obj("outcome.json")
    _need(outcome.get("dependency_gate_sha256") == _sha(read("dependency-gate.json"))
          and outcome.get("dependency_config_sha256") == expected_config_sha256
          and outcome.get("dependency_pdf_stage_sha256") == _sha(read("dependency-pdf-stage.json")),
          "public_linux_smoke_dependency_outcome_binding_invalid")
    # Ordered receipts permit identical pip checks at different stages, while
    # requiring a separate recorded invocation for every required stage.
    labels = ["baseline-freeze", "baseline-metadata", "baseline-check", "otel-repair-install",
              "post-repair-freeze", "post-repair-metadata", "post-repair-check", "post-repair-imports",
              "install", "installed", "post-pdf-freeze", "post-pdf-check"]
    commands = outcome.get("commands")
    _need(isinstance(commands, list) and all(isinstance(item, dict) for item in commands),
          "public_linux_smoke_dependency_command_receipt_invalid")
    cursor = 0
    receipts = {}
    python = None
    for label in labels:
        receipt = obj(label + ".json")
        command = receipt.get("command")
        _need(isinstance(command, list) and all(isinstance(item, str) for item in command)
              and bool(command) and receipt.get("timed_out") is False
              and type(receipt.get("returncode")) is int
              and receipt["returncode"] == (1 if label.endswith("-check") else 0),
              "public_linux_smoke_dependency_command_receipt_invalid")
        matches = [i for i in range(cursor, len(commands)) if commands[i] == receipt]
        _need(bool(matches), "public_linux_smoke_dependency_command_order_invalid")
        cursor = matches[0] + 1
        _need(receipt.get("stdout_sha256") == _sha(raw(label, "stdout"))
              and receipt.get("stderr_sha256") == _sha(raw(label, "stderr")),
              "public_linux_smoke_dependency_command_hash_invalid")
        python = python or command[0]
        _need(bool(python) and command[0] == python, "public_linux_smoke_dependency_interpreter_drift")
        tail = command[1:]
        if label.endswith("-freeze") or label == "installed":
            expected = ["-m", "pip", "freeze", "--all"]
        elif label.endswith("-metadata"):
            expected = ["-m", "pip", "inspect", "--local"]
        elif label.endswith("-check"):
            expected = ["-m", "pip", "check"]
        elif label == "post-repair-imports":
            expected = ["-c", OTEL_IMPORT_CODE]
        else:
            lock_name = "otel-repair.lock" if label == "otel-repair-install" else "requirements-linux-smoke.lock"
            _need(bool(tail) and tail[-1].replace("\\", "/").rsplit("/", 1)[-1] == lock_name,
                  "public_linux_smoke_dependency_install_command_invalid")
            flags = (["--isolated", "--index-url", "https://pypi.org/simple", "--only-binary=:all:", "--no-cache-dir"]
                     if label == "otel-repair-install" else [])
            expected = ["-m", "pip", "install", *flags, "--no-deps", "--require-hashes", "-r", tail[-1]]
        _need(tail == expected, "public_linux_smoke_dependency_command_invalid")
        receipts[label] = receipt

    baseline = parse_legacy_freeze(raw("baseline-freeze", "stdout"))
    validate_metadata(raw("baseline-metadata", "stdout"), baseline)
    mode = classify_baseline(baseline, receipts["baseline-check"]["returncode"], raw("baseline-check", "stdout"), raw("baseline-check", "stderr"))
    repaired = parse_legacy_freeze(raw("post-repair-freeze", "stdout"))
    validate_metadata(raw("post-repair-metadata", "stdout"), repaired)
    validate_post(repaired, receipts["post-repair-check"]["returncode"], raw("post-repair-check", "stdout"), plan,
                  before=baseline, stderr=raw("post-repair-check", "stderr"))
    post_pdf = parse_legacy_freeze(raw("post-pdf-freeze", "stdout"))
    expected_pdf_lock = f"pymupdf @ {PDF_WHEEL_URL} --hash=sha256:{PDF_WHEEL_SHA}\n".encode()
    _need(read("requirements-linux-smoke.lock") == expected_pdf_lock,
          "public_linux_smoke_dependency_pdf_lock_invalid")
    _need(read("otel-repair.lock") == plan.lock_bytes() and gate.get("repair_lock_sha256") == _sha(plan.lock_bytes()),
          "public_linux_smoke_dependency_repair_lock_invalid")
    changed = {key for key in set(repaired) | set(post_pdf) if repaired.get(key) != post_pdf.get(key)}
    _need(changed <= {"pymupdf"}, "public_linux_smoke_dependency_pdf_drift")
    _need(post_pdf.get("pymupdf") in PDF_FREEZE_VALUES, "public_linux_smoke_dependency_pdf_missing")
    _need(parse_legacy_freeze(raw("installed", "stdout")) == post_pdf, "public_linux_smoke_dependency_pdf_drift")
    validate_post(post_pdf, receipts["post-pdf-check"]["returncode"], raw("post-pdf-check", "stdout"), plan,
                  stderr=raw("post-pdf-check", "stderr"))
    _need(raw("post-repair-imports", "stdout").strip() == b"otel-imports-ok",
          "public_linux_smoke_dependency_import_evidence_invalid")
    _need(gate.get("baseline_freeze_sha256") == _sha(raw("baseline-freeze", "stdout"))
          and gate.get("post_repair_freeze_sha256") == _sha(raw("post-repair-freeze", "stdout"))
          and gate.get("baseline_metadata_sha256") == _sha(raw("baseline-metadata", "stdout"))
          and gate.get("post_repair_metadata_sha256") == _sha(raw("post-repair-metadata", "stdout"))
          and gate.get("baseline_mode") == mode,
          "public_linux_smoke_dependency_gate_hash_invalid")
    pdf_stage = obj("dependency-pdf-stage.json")
    _need(pdf_stage.get("post_pdf_freeze_sha256") == _sha(raw("post-pdf-freeze", "stdout"))
          and pdf_stage.get("changed_distributions") == sorted(changed), "public_linux_smoke_dependency_pdf_stage_invalid")
    return {"version": "mosaic-public-linux-dependency-verification/v1", "plan_sha256": expected_config_sha256,
            "baseline_mode": mode, "post_pdf_changed_distributions": sorted(changed),
            "scientifically_accepted": False}


def execute_linux_smoke(public_root: Path, model_root: Path, output: Path, deadline_epoch: float) -> dict[str, Any]:
    """Run the approved fictional test on a real Linux GPU, with no rental authority.

    The external rental supervisor owns deletion at its absolute deadline. This
    worker owns setup, its local server process, and immutable evidence only.
    """
    import platform
    from urllib.request import urlopen
    from mosaic.public_export import verify_export

    _need(platform.system() == "Linux", "public_linux_smoke_linux_required")
    _need(0 < deadline_epoch - time.time() <= MAX_WORK_SECONDS, "public_linux_smoke_work_deadline_invalid")
    workspace = Workspace(repo_root=public_root)
    output = workspace.check(output); model_root = workspace.check(model_root)
    _need(not output.exists() and output.is_relative_to(public_root / "run"), "public_linux_smoke_new_run_required")
    _need(not any(model_root.is_relative_to(public_root / name) for name in ("run", "example-data")),
          "public_linux_smoke_model_inside_results_refused")
    verified = verify_export(public_root)
    workspace.mkdir(output); workspace.mkdir(model_root)
    server = None; log_thread = None; server_lines: list[str] = []; commands = []; route = None
    dependency_gate = None; dependency_pdf = None
    outcome: dict[str, Any] = {"version": VERSION, "status": "started", "image": IMAGE, "model": MODEL,
        "revision": REVISION, "export": verified, "work_deadline_epoch": deadline_epoch,
        "real_model_inference_run": False, "scientifically_accepted": False}
    workspace.write_canonical_json(output / "intent.json", outcome)
    child_env = dict(os.environ)
    for name in ("MOSAIC_CONTROL_TOKEN", "RUNPOD_API_KEY", "AI_GATEWAY_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        child_env.pop(name, None)
    child_env.update(HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HOME=str(model_root.parent / "hf-cache"),
                     PYTHONUNBUFFERED="1", VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1")

    def run(command: list[str], label: str, timeout: int = 600, *, require_success: bool = True, reserve_seconds: int = 3) -> dict:
        remaining = deadline_epoch - time.time()
        _need(remaining > reserve_seconds + 5, "public_linux_smoke_deadline_expired")
        timed_out = False
        try:
            completed = subprocess.run(command, cwd=public_root, env=child_env, capture_output=True,
                                       timeout=min(timeout, remaining - reserve_seconds), check=False)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            completed = subprocess.CompletedProcess(command, -1, exc.stdout or b"", exc.stderr or b"")
        workspace.write_once(output / (label + ".stdout.log"), completed.stdout)
        workspace.write_once(output / (label + ".stderr.log"), completed.stderr)
        receipt = {"command": command, "returncode": completed.returncode, "timed_out": timed_out,
                   "stdout_sha256": _sha(completed.stdout), "stderr_sha256": _sha(completed.stderr)}
        commands.append(receipt)
        workspace.write_canonical_json(output / (label + ".json"), receipt)
        _need(not timed_out, "public_linux_smoke_" + label + "_timeout")
        _need(not require_success or completed.returncode == 0, "public_linux_smoke_" + label + "_failed")
        return receipt

    try:
        free_bytes = shutil.disk_usage(model_root).free
        workspace.write_canonical_json(output / "disk-preflight.json", {
            "free_bytes": free_bytes, "minimum_free_bytes": MIN_FREE_BYTES,
            "expected_download_bytes": MODEL_DOWNLOAD_BYTES})
        _need(free_bytes >= MIN_FREE_BYTES, "public_linux_smoke_disk_headroom_unavailable")
        run(["hf", "--version"], "download-cli", 30)
        # Establish the full base state before any supplement.  This is a
        # fail-closed, source-bound repair, not a general pip-check bypass.
        dependency_plan = load_orphan_plan()
        dependency_gate = run_orphan_dependency_gate(run=run, python=sys.executable, output=output,
                                                     workspace=workspace, image=IMAGE, plan=dependency_plan)
        # This is the orphan-repair worker phase: use the same PEP 503
        # canonicalization as its gate and PDF validator.  The legacy parser
        # remains only for replaying the immutable pre-orphan retained gate.
        before_pdf = parse_orphan_freeze((output / "post-repair-freeze.stdout.log").read_bytes())
        # The entire base runtime is pinned by IMAGE; the one added wheel has
        # its own upstream SHA and may not mutate unrelated distributions.
        lock = workspace.write_text(output / "requirements-linux-smoke.lock",
            f"pymupdf @ {PDF_WHEEL_URL} --hash=sha256:{PDF_WHEEL_SHA}\n")
        run([sys.executable, "-m", "pip", "install", "--no-deps", "--require-hashes", "-r", str(lock)], "install", 120)
        run([sys.executable, "-m", "pip", "freeze", "--all"], "installed", 30)
        dependency_pdf = validate_orphan_pdf_stage(run=run, python=sys.executable, output=output,
                                                   workspace=workspace, before=before_pdf, plan=dependency_plan)
        outcome["dependency_gate_sha256"] = _sha((output / "dependency-gate.json").read_bytes())
        outcome["dependency_config_sha256"] = dependency_plan.config_sha256
        outcome["dependency_pdf_stage_sha256"] = _sha((output / "dependency-pdf-stage.json").read_bytes())
        run([sys.executable, "-c", "import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in ('vllm','torch','tokenizers','xgrammar','huggingface-hub','pymupdf')}))"], "required-runtime", 30)
        run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], "hardware", 30)
        run([sys.executable, "-c", "import torch,json; print(json.dumps({'cuda_available':torch.cuda.is_available(),'device_count':torch.cuda.device_count(),'device':torch.cuda.get_device_name(0),'memory':torch.cuda.get_device_properties(0).total_memory,'torch':torch.__version__,'cuda':torch.version.cuda}))"], "cuda")
        device = json.loads((output / "cuda.stdout.log").read_bytes())
        _need(device["cuda_available"] and device["device_count"] == 1 and 75e9 <= device["memory"] <= 90e9,
              "public_linux_smoke_expected_one_80gb_gpu")
        # Metadata versions do not prove this import surface is usable. This
        # bounded child check imports only route/API modules; it creates no
        # engine and starts neither a server, PDF processing, nor model fetch.
        run([sys.executable, "-c", "import vllm; import vllm.entrypoints.openai.api_server; import mosaic.public_workflow; print('vllm-route-import-ok')"],
            "vllm-import", 30)
        # Validate the complete finite request reservation before acquiring the
        # model.  The previous v5 profile promised four possible 180-second
        # requests inside a 600-second driver budget, which hard-stopped before
        # the first extraction POST.
        profile = _driver_profile()
        route_schedule = _route_schedule(profile)
        run(model_download_command(model_root), "download", 1200, reserve_seconds=1800)
        _need(deadline_epoch - time.time() > 1800, "public_linux_smoke_inventory_reserve_unavailable")
        _need((model_root / "config.json").is_file(), "public_linux_smoke_download_config_missing")
        model_config = json.loads((model_root / "config.json").read_bytes())
        _need(model_config.get("architectures") == ["Qwen3_5ForConditionalGeneration"]
              and not model_config.get("auto_map"), "public_linux_smoke_model_architecture_refused")
        declaration = {"version": "mosaic-public-model-declaration/v1", "model": {"id": MODEL,
            "revision": REVISION, "tokenizer_path": "tokenizer.json", "tokenizer_runtime_version": metadata.version("tokenizers"),
            "add_special_tokens": True}, "runtime": {"platform": "linux", "container_digest": IMAGE.split("sha256:")[1],
            "wheels": [{"name": PDF_WHEEL, "sha256": PDF_WHEEL_SHA}],
            "dependency_repair": {"plan_sha256": dependency_plan.config_sha256,
                "gate": dependency_gate, "pdf_stage": dependency_pdf,
                "action": "remove_orphan_distribution",
                "removed_distribution": "opentelemetry-exporter-otlp-common",
                "base_runtime_identity": "container_digest"},
            "installed_inventory_sha256": _sha((output / "installed.stdout.log").read_bytes())},
            "hardware": {"declared": device, "observed_linux_gpu_smoke_receipt": None}}
        runtime = workspace.mkdir(public_root / "run/smoke-runtime")
        declaration_path = workspace.write_canonical_json(runtime / "declaration.json", declaration)
        # Hashing the 55.6 GB tree (including its independent validation pass)
        # is a killable child phase, so it cannot consume the server/route reserve.
        inventory_code = ("import json,sys; from pathlib import Path; from mosaic.workspace import Workspace; "
            "from mosaic.public_model_inventory import inventory_local_model; "
            "print(json.dumps(inventory_local_model(json.loads(Path(sys.argv[1]).read_bytes()), "
            "Path(sys.argv[2]), Path(sys.argv[3]), workspace=Workspace(repo_root=Path.cwd()))))")
        run([sys.executable, "-c", inventory_code, str(declaration_path), str(model_root),
             str(runtime / "local-model-manifest.json")], "inventory", 600, reserve_seconds=1200)
        manifest_result = json.loads((output / "inventory.stdout.log").read_bytes())
        workspace.write_canonical_json(runtime / "driver-profile.json", profile)
        outcome["route_schedule"] = route_schedule
        _need(deadline_epoch - time.time() > PROBE_MAX_SECONDS + ROUTE_COMMAND_SECONDS + PHASE_CLEANUP_SECONDS,
              "public_linux_smoke_server_reserve_unavailable")
        command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(model_root),
                   "--served-model-name", SERVED, "--host", "127.0.0.1", "--port", "8000",
                   "--max-model-len", str(SERVER_CONTEXT_LIMIT), "--max-num-seqs", "8",
                   "--gpu-memory-utilization", "0.90", "--enforce-eager"]
        workspace.write_canonical_json(output / "server-command.json", {"command": command})
        server = subprocess.Popen(command, cwd=public_root, env=child_env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        def collect_log():
            for line in server.stdout:
                server_lines.append(line)
        log_thread = threading.Thread(target=collect_log, daemon=True); log_thread.start()
        ready_by = min(time.time() + 480,
                       deadline_epoch - (PROBE_MAX_SECONDS + ROUTE_COMMAND_SECONDS + PHASE_CLEANUP_SECONDS))
        ready = False; readiness = {}
        while time.time() < ready_by and server.poll() is None:
            try:
                readiness = _server_readiness(urlopen)
                ready = True
                break
            except PublicLinuxSmokeRefusal:
                raise
            except (OSError, ValueError):
                time.sleep(2)
        _need(ready, "public_linux_smoke_server_not_ready")
        workspace.write_canonical_json(output / "server-readiness.json", {"served_model": SERVED, "response_sha256": readiness})
        probe_root = output / "model-probe"
        # The subprocess gives the one request an OS-enforced wall-clock cap;
        # the retained transcript is independently recomputed before routing.
        run([sys.executable, "-m", "mosaic.public_model_probe", "run", "--output", str(probe_root),
             "--deadline-epoch", str(min(deadline_epoch - ROUTE_COMMAND_SECONDS - ROUTE_FINAL_RESERVE_SECONDS,
                                           time.time() + PROBE_MAX_SECONDS))],
            "model-probe", PROBE_MAX_SECONDS,
            reserve_seconds=ROUTE_COMMAND_SECONDS + ROUTE_FINAL_RESERVE_SECONDS)
        probe = verify_retained_model_probe(probe_root)
        outcome["model_probe"] = probe
        outcome["model_probe_outcome_sha256"] = _sha((probe_root / "outcome.json").read_bytes())
        run([sys.executable, "-m", "mosaic.public_linux_smoke", "run-public-route", "--public-root", str(public_root),
             "--model-root", str(model_root), "--deadline-epoch", str(min(deadline_epoch - ROUTE_FINAL_RESERVE_SECONDS,
                                                                            time.time() + ROUTE_COMMAND_SECONDS))],
            "route", ROUTE_COMMAND_SECONDS, reserve_seconds=ROUTE_FINAL_RESERVE_SECONDS)
        route = json.loads((output / "route.stdout.log").read_bytes())
        _need(route.get("candidate_only") is True, "public_linux_smoke_route_incomplete")
        exchange = _verified_model_exchange(public_root, route.get("fixture_sha256"))
        observed = exchange["model_exchange_observed"]
        outcome.update(status="completed" if observed else "failed", real_model_inference_run=observed,
                       route=route, model_exchange=exchange, installed_runtime_observed=True,
                       gpu_observed=device, model_inventory=manifest_result)
        if not observed:
            outcome["reason"] = "public_linux_smoke_model_exchange_not_verified"
    except Exception as exc:
        outcome.update(status="failed", reason=_safe_failure_reason(exc))
    finally:
        if server is not None:
            if server.poll() is None:
                server.terminate()
                try: server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill(); server.wait(timeout=5)
            if log_thread is not None: log_thread.join(timeout=5)
            workspace.write_text(output / "server.log", "".join(server_lines))
            outcome["owned_server_exit_code"] = server.returncode
        outcome.update(commands=commands, finished_epoch=time.time(), rental_deletion="external_supervisor_required",
                       result_archive=str(output / "result.zip"))
        workspace.write_canonical_json(output / "outcome.json", outcome)
        archive = _archive_results(public_root, output, workspace)
        outcome["result_archive"] = str(archive)
    return outcome


def run_actual_public_route(public_root: Path, model_root: Path, deadline_epoch: float) -> Mapping[str, Any]:
    """Run the bundled real public route against the already-running loopback server.

    The worker inventories its acquired model and writes a bounded driver
    profile under ``run/smoke-runtime/``. This function creates only the known
    fictional demo PDF, then calls the existing capture -> preparation ->
    native execution -> materialization -> HTTP check route.
    """
    _need(time.time() < deadline_epoch, "public_linux_smoke_deadline_expired")
    profile_path = public_root / "run/smoke-runtime" / "driver-profile.json"
    manifest_path = public_root / "run/smoke-runtime" / "local-model-manifest.json"
    _need(profile_path.is_file() and manifest_path.is_file(), "public_linux_smoke_runtime_inputs_missing")
    profile, manifest = json.loads(profile_path.read_bytes()), json.loads(manifest_path.read_bytes())
    _need(manifest.get("model", {}).get("id") == MODEL and manifest.get("model", {}).get("revision") == REVISION,
          "public_linux_smoke_model_identity_refused")
    _need(profile.get("model", {}).get("served_model_name") == SERVED
          and profile.get("driver", {}).get("request_cap") == 7
          and profile.get("driver", {}).get("output_token_cap") == 49120,
          "public_linux_smoke_driver_profile_refused")
    _route_schedule(profile)
    workspace = Workspace(repo_root=public_root, temp_root=model_root.parent)
    from mosaic.public_demo import prepare_demo
    from mosaic.public_runtime_manifest import validate_manifest
    from mosaic.public_workflow import run_public_corpus
    fixture = prepare_demo(public_root, workspace=workspace)
    checked = validate_manifest(manifest, weights_root=model_root)
    config = {"version":"mosaic-public-build/v1", "profile":"corpus",
        "sources":{"root":"example-data/pdfs", "files":["fictional-build.pdf"], "permission":"user_declared_permitted"},
        "run":{"root":"run/linux-smoke"},
        "runtime":{"kind":"local_open_weights", "base_url":"http://127.0.0.1:8000/v1", "model":MODEL,
                   "revision":REVISION, "tokenizer_revision":manifest["model"].get("tokenizer_revision", REVISION),
                   "weights_sha256":checked["model_files_sha256"], "license":"unreviewed_upstream_pointer_only"},
        "view":{"columns":["title","authors","journal","year"],"default_screen":"papers","sort":"distinct_article_count_desc"},
        "serve":{"port":8798}, "jev":{"enabled":False,"replay_only":True}}
    result = run_public_corpus(config, project_root=public_root, workspace=workspace, model_manifest=manifest,
                               weights_root=model_root, driver_profile=profile, live=True, deadline_epoch=deadline_epoch - 60)
    _need(result.get("status") == "candidate_corpus_served" and result.get("scientifically_accepted") is False,
          "public_linux_smoke_route_incomplete")
    return {"candidate_only": True, "scientifically_accepted": False, "fixture_sha256": fixture["source_pdf_sha256"],
            "build_report_path": result["build_report_path"], "materialization": result["materialization"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    route = sub.add_parser("run-public-route")
    route.add_argument("--public-root", required=True); route.add_argument("--model-root", required=True)
    route.add_argument("--deadline-epoch", required=True, type=float)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--public-root", required=True); run_parser.add_argument("--model-root", required=True)
    run_parser.add_argument("--output", required=True); run_parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        result = execute_linux_smoke(Path(args.public_root).absolute(), Path(args.model_root).absolute(),
                                    Path(args.output).absolute(), args.deadline_epoch)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "completed" else 2
    if args.command == "run-public-route":
        result = run_actual_public_route(Path(args.public_root).absolute(), Path(args.model_root).absolute(), args.deadline_epoch)
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PublicLinuxSmokeRefusal, ValueError, OSError) as exc:
        print(json.dumps({"status":"refused","reason":str(exc) if str(exc).startswith("public_") else type(exc).__name__}))
        raise SystemExit(2)
