"""Bounded local-loopback execution for a frozen public native program.

This module deliberately has no server launcher, downloader, provider fallback,
or retry wrapper.  It is the narrow bridge from an immutable preparation packet
to the existing finite complete-extraction runner.
"""
from __future__ import annotations

from hashlib import sha256
from importlib import metadata
import json
import platform
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from mosaic.canonical import canonical_bytes
from mosaic.public_native_runtime import (PublicLocalScope, _digest, _loopback,
    _no_links, _profile, _verify_tokenizer, load_capture_envelope)
from mosaic.public_runtime_manifest import validate_manifest
from mosaic.public_snapshot import corpus_report_to_public_snapshot, write_snapshot
from mosaic.vllm.ledger import VLLMBudget
from mosaic.workspace import Workspace


VERSION = "mosaic-public-native-execution/v1"
_MAX_PROTOCOL_BYTES = 1_048_576


class PublicExecutionRefusal(ValueError):
    pass


def _need(value: bool, code: str) -> None:
    if not value:
        raise PublicExecutionRefusal(code)


def _read(root: Path, name: str) -> dict[str, Any]:
    path = root / name
    _need(path.is_file() and not path.is_symlink(), "public_execution_prepared_artifact_missing")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise PublicExecutionRefusal("public_execution_prepared_artifact_invalid") from exc
    _need(isinstance(value, dict), "public_execution_prepared_artifact_invalid")
    return value


def _hash_file(root: Path, name: str, expected: Any) -> dict[str, Any]:
    value = _read(root, name)
    _need(isinstance(expected, str) and sha256((root / name).read_bytes()).hexdigest() == expected,
          "public_execution_prepared_hash_mismatch")
    return value


def _artifact_hashes(root: Path, directories: tuple[str, ...]) -> dict[str, str]:
    """Hash every persisted driver/ledger file before allowing reuse."""
    out: dict[str, str] = {}
    for name in directories:
        directory = root / name
        _need(directory.is_dir(), "public_execution_artifact_directory_missing")
        for path in sorted(directory.rglob("*")):
            if path.is_dir():
                continue
            _need(path.is_file() and not path.is_symlink(), "public_execution_artifact_link_refused")
            out[path.relative_to(root).as_posix()] = sha256(path.read_bytes()).hexdigest()
    return out


def _observation(value: Mapping[str, Any] | None, endpoint: str) -> dict[str, Any]:
    """Record an observation inventory without self-certifying model support."""
    _need(value is None or isinstance(value, Mapping), "public_execution_observation_invalid")
    item = dict(value or {})
    kind = item.get("kind", "declared_unobserved")
    _need(kind in {"mocked_loopback_protocol", "observed_local_loopback", "declared_unobserved"}, "public_execution_observation_invalid")
    _need(item.get("endpoint", endpoint) == endpoint, "public_execution_observation_endpoint_mismatch")
    inventory = item.get("inventory", {})
    _need(isinstance(inventory, Mapping), "public_execution_observation_inventory_required")
    # A mock is useful software evidence but never evidence of a model or GPU.
    if kind == "mocked_loopback_protocol":
        _need(item.get("model_execution_observed") is False, "public_execution_mock_claim_refused")
    elif kind == "observed_local_loopback":
        _need(item.get("model_execution_observed") is True, "public_execution_observed_claim_required")
    return item


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise PublicExecutionRefusal("public_execution_protocol_redirect_refused")


def _protocol_observation(endpoint: str, expected_model: str) -> dict[str, Any]:
    """Capture and bind local protocol facts; never infer GPU/model success."""
    base = endpoint.removesuffix("/v1")
    opener = build_opener(ProxyHandler({}), _NoRedirect())
    records = []
    for route in ("/health", "/v1/models"):
        try:
            with opener.open(Request(base + route, method="GET"), timeout=5) as response:  # nosec: endpoint literal-checked
                body = response.read(_MAX_PROTOCOL_BYTES + 1)
                _need(len(body) <= _MAX_PROTOCOL_BYTES, "public_execution_protocol_body_too_large")
                _need(response.status == 200, "public_execution_protocol_status_refused")
                if route == "/v1/models":
                    decoded = json.loads(body)
                    models = decoded.get("data") if isinstance(decoded, dict) else None
                    _need(isinstance(models, list) and expected_model in {item.get("id") for item in models if isinstance(item, dict)},
                          "public_execution_served_model_refused")
                records.append({"route": route, "status": response.status, "body_sha256": sha256(body).hexdigest(),
                                "bytes": len(body)})
        except PublicExecutionRefusal:
            raise
        except Exception as exc:
            raise PublicExecutionRefusal("public_execution_protocol_unavailable") from exc
    distributions = {}
    for name in ("vllm", "torch", "tokenizers", "xgrammar"):
        try:
            distributions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            distributions[name] = "not_installed"
    return {"endpoint": endpoint, "routes": records, "local_protocol_observed": True,
            "client_runtime": {"system": platform.system(), "release": platform.release(),
                               "python": platform.python_version(), "distributions": distributions}}


def run_local_program(prepared_root: Path | str, capture_root: Path | str, output: Path | str, *,
                      workspace: Workspace, deadline_epoch: float,
                      runtime_observation: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Execute exactly one prepared native program against its literal loopback server.

    ``runtime_observation`` is retained as a caller-supplied inventory.  Only a
    successful external smoke assessment can decide whether it is sufficient for
    supported-runtime status; this function always emits a candidate-only report.
    """
    root = Path(prepared_root)
    _need(root.is_absolute() and root.is_dir(), "public_execution_prepared_root_invalid")
    try:
        _no_links(root)
    except ValueError as exc:
        raise PublicExecutionRefusal("public_execution_prepared_link_refused") from exc
    prepared = _read(root, "prepared-receipt.json")
    _need(prepared.get("version") == "mosaic-public-native-preparation/v1"
          and prepared.get("status") == "prepared_only_execution_unvalidated"
          and prepared.get("execution_api") == "intentionally_absent", "public_execution_prepared_receipt_invalid")
    manifest = _hash_file(root, "local-model-manifest.json", prepared.get("local_model_manifest_sha256"))
    profile = _hash_file(root, "driver-profile.json", prepared.get("driver_profile_sha256"))
    program = _hash_file(root, "prepared-program.json", prepared.get("program_sha256"))
    preflight = _hash_file(root, "runtime-preflight.json", prepared.get("manifest_preflight_sha256"))
    envelope = load_capture_envelope(capture_root, workspace=workspace)
    receipt = envelope["capture_receipt"]
    capture = prepared.get("capture")
    _need(isinstance(capture, Mapping) and capture == {"work_id": receipt.get("work_id"),
        "source_pdf_sha256": receipt.get("source_pdf_sha256"), "source_packet_sha256": receipt.get("source_packet_sha256")},
        "public_execution_capture_changed")
    _need(receipt.get("scientifically_accepted") is False, "public_execution_candidate_claim_refused")
    weights_root = preflight.get("weights_root")
    _need(isinstance(weights_root, str), "public_execution_weights_root_missing")
    repeated_preflight = validate_manifest(manifest, weights_root=weights_root)
    _need(_digest(repeated_preflight) == _digest({k: v for k, v in preflight.items() if k != "tokenizer_checked"}),
          "public_execution_manifest_preflight_changed")
    parsed_profile = _profile(profile, manifest)
    endpoint = _loopback(prepared.get("endpoint"))
    declared_endpoint = runtime_observation.get("endpoint", endpoint) if isinstance(runtime_observation, Mapping) else endpoint
    _need(endpoint == _loopback(declared_endpoint),
          "public_execution_endpoint_changed")
    _verify_tokenizer(manifest, repeated_preflight)
    _need(program.get("fixture_mode") is False and program.get("candidate") == parsed_profile["driver"]["candidate"]
          and program.get("run_id") == parsed_profile["driver"]["run_id"], "public_execution_program_profile_changed")
    _need(program.get("context_limit") == parsed_profile["driver"]["context_limit"],
          "public_execution_program_context_limit_changed")
    _need(program.get("recipe", {}).get("vllm", {}).get("served_model_name")
          == parsed_profile["model"]["served_model_name"], "public_execution_program_model_changed")
    paper = program.get("papers", [{}])[0]
    _need(len(program.get("papers", [])) == 1 and paper.get("work_id") == receipt["work_id"]
          and paper.get("source_sha256s") == [receipt["source_pdf_sha256"]]
          and paper.get("text_provenance") == "native_article_text" and paper.get("remote") is False
          and paper.get("rights") == ["public_local_processing_user_declared"], "public_execution_program_source_changed")
    observation = _observation(runtime_observation, endpoint)
    _need(isinstance(deadline_epoch, (int, float)) and deadline_epoch > time.time(),
          "public_execution_deadline_invalid")
    out = Path(output)
    _need(out.is_absolute(), "public_execution_output_path_invalid")
    try:
        _no_links(out)
    except ValueError as exc:
        raise PublicExecutionRefusal("public_execution_output_link_refused") from exc
    execution_root = out
    limits = parsed_profile["driver"]
    budget = VLLMBudget(limits["request_cap"], limits["output_token_cap"], limits["wall_clock_cap_seconds"],
                        limits["article_request_cap"])
    start_hash = sha256(b"").hexdigest()
    binding_path = root / "execution-binding.json"
    from mosaic.adapted.k.kb_writer_lock import acquire, release
    lock = acquire(root / "execution-binding", workspace, "public native execution binding")
    try:
        if binding_path.exists():
            prior = _read(root, "execution-binding.json")
            if prior.get("output") == str(execution_root):
                intent_path = execution_root / "execution-intent.json"
                _need(intent_path.is_file() and sha256(intent_path.read_bytes()).hexdigest() == prior.get("execution_intent_sha256"),
                      "public_execution_execution_intent_changed")
                receipt_path = execution_root / "execution-receipt.json"
                _need(receipt_path.is_file(), "public_execution_prior_run_not_terminal")
                done = json.loads(receipt_path.read_bytes())
                _need(done.get("status") == "terminal", "public_execution_prior_run_not_terminal")
                aggregate_path = execution_root / "driver" / "report.json"
                report_path = execution_root / "driver" / done.get("paper_report_path", "")
                snapshot_path = execution_root / "public-snapshot.json"
                _need(aggregate_path.is_file() and report_path.is_file() and snapshot_path.is_file()
                      and sha256(aggregate_path.read_bytes()).hexdigest() == done.get("driver_report_file_sha256")
                      and sha256(report_path.read_bytes()).hexdigest() == done.get("paper_report_sha256")
                      and sha256(snapshot_path.read_bytes()).hexdigest() == done.get("snapshot_sha256"),
                      "public_execution_completed_artifact_changed")
                _need(done.get("artifact_hashes") == _artifact_hashes(execution_root, ("driver", "ledger")),
                      "public_execution_driver_or_ledger_changed")
                _need(done.get("prepared_receipt_sha256") == sha256((root / "prepared-receipt.json").read_bytes()).hexdigest()
                      and done.get("capture") == dict(capture), "public_execution_completed_binding_changed")
                return {"receipt_path": str(receipt_path), "snapshot_path": str(snapshot_path),
                        "report_path": str(report_path), "driver_root": str(execution_root / "driver"), "receipt": done}
            raise PublicExecutionRefusal("public_execution_prepared_program_already_bound")
        _need(not out.exists() or not any(out.iterdir()), "public_execution_output_not_empty")
        execution_root = workspace.mkdir(out)
        intent = {"version": VERSION, "status": "execution_started", "prepared_root": str(root),
              "prepared_receipt_sha256": sha256((root / "prepared-receipt.json").read_bytes()).hexdigest(),
              "capture": dict(capture), "endpoint": endpoint, "runtime_observation_declaration": observation,
              "budget": {"request_cap": budget.request_cap, "output_token_cap_total": budget.output_token_cap_total,
                         "wall_clock_cap_seconds": budget.wall_clock_cap_seconds,
                         "max_requests_per_article": budget.max_requests_per_article},
              "candidate_only": True, "scientifically_accepted": False, "qualified": False}
        workspace.write_once(execution_root / "execution-intent.json", canonical_bytes(intent))
        workspace.write_once(binding_path, canonical_bytes({"version": VERSION, "status": "started", "output": str(execution_root),
            "execution_intent_sha256": sha256((execution_root / "execution-intent.json").read_bytes()).hexdigest()}))
    finally:
        release(lock)
    protocol = _protocol_observation(endpoint, parsed_profile["model"]["served_model_name"])
    from scripts.complete_extraction_runner import run_program
    scope = PublicLocalScope(receipt["work_id"], receipt["source_pdf_sha256"])
    result = run_program(program, expected_program_sha256=prepared["program_sha256"], workspace=workspace,
                         receipt_root=execution_root / "driver", ledger_root=execution_root / "ledger",
                         expected_ledger_sha256=start_hash, budget=budget, scope=scope,
                         # ``PhasedRun`` reserves its fixed cleanup window separately;
                         # this adds no send/time budget beyond the frozen profile.
                         deadline_epoch=deadline_epoch + 60, base_url=endpoint, live=True, resume=False)
    paper_row = result["papers"][0]
    report = json.loads((execution_root / "driver" / paper_row["report_path"]).read_bytes())
    snapshot = corpus_report_to_public_snapshot(program["papers"][0]["plan"], report)
    snapshot_path = write_snapshot(workspace, execution_root, snapshot)
    driver_report_path = execution_root / "driver" / "report.json"
    paper_report_path = execution_root / "driver" / paper_row["report_path"]
    done = {**intent, "status": "terminal", "protocol_observation": protocol, "driver_report_sha256": sha256(canonical_bytes(result)).hexdigest(),
            "driver_report_file_sha256": sha256(driver_report_path.read_bytes()).hexdigest(),
            "paper_report_path": paper_row["report_path"], "paper_report_sha256": sha256(paper_report_path.read_bytes()).hexdigest(),
            "artifact_hashes": _artifact_hashes(execution_root, ("driver", "ledger")),
            "snapshot_sha256": sha256(snapshot_path.read_bytes()).hexdigest(), "stop_reason": result.get("stop_reason"),
            "native_execution_completed": result.get("stop_reason") is None, "real_model_inference_run": False,
            "model_execution": "mocked_loopback_protocol" if observation.get("kind") == "mocked_loopback_protocol" else "not_independently_established",
            "supported_runtime": False,
            "limitations": ["Candidate-only mechanical output; no scientific qualification.",
                            "Caller runtime inventory is recorded, not independently verified by this adapter.",
                            "Unknown outcomes remain in the append-only driver ledger and are never retried by this adapter."]}
    receipt_path = workspace.write_canonical_json(execution_root / "execution-receipt.json", done)
    # The immutable preparation-root binding permanently permits only this
    # output.  A later caller can only read the terminal receipt above.
    return {"receipt_path": str(receipt_path), "snapshot_path": str(snapshot_path), "report_path": str(paper_report_path),
            "driver_root": str(execution_root / "driver"), "receipt": done}
