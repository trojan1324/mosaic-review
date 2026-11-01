"""Bounded public fixture replay over MOSAIC's current complete driver.

This module is deliberately *not* a live inference launcher.  It accepts one
explicit, fictional public-capture packet and an injected recorded-fixture
transport, then invokes the unchanged current driver.  The resulting report is
a candidate artifact with source witnesses; it is never a qualification or a
supported cold-start runtime claim.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
from threading import Lock
import time
from typing import Any, Mapping

from mosaic.canonical import canonical_bytes
from mosaic.workspace import Workspace


VERSION = "public-runtime-replay/v1"
PUBLIC_DEMO_PDF_SHA256 = "856cf77abd227db43b2353ed00e553351fabea2c8ad075d98e23182a4b90324f"
_SHA256 = set("0123456789abcdef")


class PublicRuntimeRefusal(ValueError):
    """The supplied public packet or replay boundary was not exact."""


def _digest(value: Any) -> str:
    return sha256(canonical_bytes(value)).hexdigest()


def _need(condition: bool, code: str) -> None:
    if not condition:
        raise PublicRuntimeRefusal(code)


def _sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256


class PublicScope:
    """Accept exactly one public fixture's immutable source identity."""

    def __init__(self, *, work_id: str, source_pdf_sha256: str) -> None:
        self.work_id = work_id
        self.source_pdf_sha256 = source_pdf_sha256

    def check(self, work_id, hashes, provenance, *, remote, rights) -> None:
        _need(work_id == self.work_id, "public_runtime_work_id_refused")
        _need(hashes == [self.source_pdf_sha256], "public_runtime_source_hash_refused")
        _need(provenance == "fixture_replay", "public_runtime_provenance_refused")
        _need(remote is False, "public_runtime_remote_refused")
        _need(rights == ["fictional_fixture_local_only"], "public_runtime_rights_refused")


class _PublicDemoTransport:
    """Manufactured answers for the one hash-pinned public demo, never a model."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self._lock = Lock()

    def request(self, method, url, headers, body, timeout):
        _need(method == "POST" and url.startswith("http://127.0.0.1"), "public_runtime_fixture_wire_refused")
        wire = json.loads(body)
        schema = json.loads(wire["messages"][0]["content"].split("\nRESPONSE_SCHEMA:\n", 1)[1])
        with self._lock:
            self.requests.append(wire)
        properties = schema["properties"]
        if "findings" in properties:
            answer = {"article_kind": "original_empirical", "findings": [], "qualifications": []}
        elif "studies" in properties:
            answer = {"article_kind": "original_empirical", "studies": [{"identifier_sentence_ids": ["c001"],
                      "design": "uncertain", "design_sentence_ids": [], "population_sentence_ids": [],
                      "sample_size_sentence_ids": [], "measures": []}], "analyses": []}
        else:
            answer = {"facts": []}
        return 200, {}, canonical_bytes({"model": wire["model"], "choices": [{"finish_reason": "stop",
            "message": {"content": json.dumps(answer)}}], "usage": {"completion_tokens": 5, "prompt_tokens": 7}})


def fixture_replay_for_demo(capture_receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Return the only built-in replay: manufactured responses for public_demo.

    It is intentionally hash-pinned to the deterministic fictional PDF.  This
    convenience function cannot turn an arbitrary caller PDF into a canned
    extraction result.
    """
    capture, source, _binding, _upstream = _capture_parts(capture_receipt)
    _need(capture["source_pdf_sha256"] == PUBLIC_DEMO_PDF_SHA256,
          "public_runtime_demo_pdf_hash_refused")
    units = source.get("units")
    _need(isinstance(units, list) and units and units[0].get("unit_id") == "U001"
          and units[0].get("text") == "Study 1\n", "public_runtime_demo_source_refused")
    return {"kind": "recorded_fictional_fixture", "fixture_transport": _PublicDemoTransport(),
            "fixture_count": lambda text: len(text) // 3,
            "response_provenance": "manufactured_fixture_responses_not_model_output"}


def _capture_parts(capture_receipt: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate the path-free capture handoff used by this public adapter."""
    _need(isinstance(capture_receipt, Mapping), "public_runtime_capture_receipt_required")
    capture = capture_receipt.get("capture_receipt")
    source = capture_receipt.get("source")
    binding = capture_receipt.get("source_binding")
    upstream = capture_receipt.get("prepare_upstream_input")
    _need(all(isinstance(v, Mapping) for v in (capture, source, binding, upstream)),
          "public_runtime_capture_parts_required")
    capture, source, binding, upstream = (dict(capture), dict(source), dict(binding), dict(upstream))
    _need(capture.get("version") == "public-capture-source-packet/v1"
          and capture.get("status") == "source_packet_ready", "public_runtime_capture_not_ready")
    work_id, pdf_hash, source_hash = capture.get("work_id"), capture.get("source_pdf_sha256"), capture.get("source_packet_sha256")
    _need(isinstance(work_id, str) and _sha(pdf_hash) and _sha(source_hash)
          and work_id == "public-" + pdf_hash and capture.get("scientifically_accepted") is False,
          "public_runtime_capture_identity_invalid")
    _need(_digest(source) == source_hash and binding.get("source_packet_sha256") == source_hash,
          "public_runtime_source_packet_hash_mismatch")
    _need(source.get("work_id") == work_id and binding.get("work_id") == work_id and upstream.get("paper_id") == work_id
          and upstream.get("source_pdf_sha256") == pdf_hash and upstream.get("source") == source,
          "public_runtime_upstream_binding_mismatch")
    authorization = capture_receipt.get("runtime_authorization")
    _need(isinstance(authorization, Mapping)
          and authorization.get("kind") == "fictional_fixture_local_only"
          and authorization.get("source_pdf_sha256") == pdf_hash,
          "public_runtime_fixture_authorization_required")
    return capture, source, binding, upstream


def _profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    _need(isinstance(profile, Mapping), "public_runtime_profile_required")
    value = dict(profile)
    _need(value.get("mode") == "fictional_fixture_replay_only", "public_runtime_live_mode_refused")
    runtime = value.get("runtime")
    _need(isinstance(runtime, Mapping) and runtime.get("endpoint_policy") == "no_endpoint_in_fixture_replay"
          and runtime.get("remote") is False, "public_runtime_runtime_policy_invalid")
    driver = value.get("driver")
    _need(isinstance(driver, Mapping) and isinstance(driver.get("candidate"), str)
          and isinstance(driver.get("served_model_name"), str), "public_runtime_driver_profile_invalid")
    for key in ("max_measure_stages_per_paper", "max_request_seconds", "request_cap", "output_token_cap",
                "wall_clock_cap_seconds", "article_request_cap"):
        number = driver.get(key)
        _need(type(number) in (int, float) and math.isfinite(number) and number > 0,
              "public_runtime_driver_bound_invalid")
    _need(int(driver["max_measure_stages_per_paper"]) == driver["max_measure_stages_per_paper"],
          "public_runtime_measure_bound_invalid")
    return value


def run_extraction(capture_receipt: Mapping[str, Any], output: Path | str, *, workspace: Workspace,
                   profile: Mapping[str, Any], replay: Mapping[str, Any]) -> dict[str, Any]:
    """Replay the real complete-driver sequence for one declared fictional capture.

    ``replay`` is deliberately an in-memory test seam: it requires a recorded
    fixture transport and token counter and can never select a server URL.
    Importing the current driver is lazy because grammar compilation is a real
    runtime dependency (xgrammar and its native dependencies), not metadata we
    can safely pretend is installed.
    """
    capture, source, binding, upstream = _capture_parts(capture_receipt)
    profile = _profile(profile)
    _need(isinstance(replay, Mapping) and replay.get("kind") == "recorded_fictional_fixture",
          "public_runtime_replay_kind_required")
    transport, count = replay.get("fixture_transport"), replay.get("fixture_count")
    _need(hasattr(transport, "request") and callable(count), "public_runtime_fixture_transport_required")
    deadline = replay.get("deadline_epoch")
    if deadline is not None:
        # An explicit caller-owned absolute deadline is never extended.
        _need(isinstance(deadline, (int, float)) and deadline > time.time(), "public_runtime_deadline_invalid")

    # These imports execute the real task builders and XGrammar compiler only
    # after the public identity/replay boundary has been established.
    from mosaic.public_snapshot import corpus_report_to_public_snapshot, write_snapshot
    from mosaic.vllm.ledger import VLLMBudget
    from scripts.complete_extraction_runner import prepare_program, run_program
    from scripts.complete_extraction_sequence import prepare_upstream

    root = workspace.mkdir(output)
    recipe = {"vllm": {"served_model_name": profile["driver"]["served_model_name"]}}
    plan = prepare_upstream(recipe, upstream)
    paper = {"work_id": capture["work_id"], "plan": plan,
             "source_sha256s": [capture["source_pdf_sha256"]], "text_provenance": "fixture_replay",
             "rights": ["fictional_fixture_local_only"], "remote": False}
    limits = profile["driver"]
    program = prepare_program([paper], recipe, candidate=limits["candidate"], run_id=limits["run_id"],
                              max_measure_stages_per_paper=int(limits["max_measure_stages_per_paper"]),
                              max_workers=1, max_request_seconds=int(limits["max_request_seconds"]),
                              max_attempts=1, fixture_mode=True)
    program_sha256 = _digest(program)
    budget = VLLMBudget(int(limits["request_cap"]), int(limits["output_token_cap"]),
                        float(limits["wall_clock_cap_seconds"]), int(limits["article_request_cap"]))
    # The built-in fictional replay's allowance begins at execution so cold
    # local imports, task building and grammar preparation cannot consume it.
    # Explicit caller deadlines retain their original absolute meaning.
    if deadline is None:
        deadline = time.time() + 300
    else:
        _need(deadline > time.time(), "public_runtime_deadline_invalid")
    result = run_program(program, expected_program_sha256=program_sha256, workspace=workspace,
                         receipt_root=root / "driver", ledger_root=root / "ledger",
                         expected_ledger_sha256=sha256(b"").hexdigest(), budget=budget,
                         scope=PublicScope(work_id=capture["work_id"], source_pdf_sha256=capture["source_pdf_sha256"]),
                         deadline_epoch=deadline, live=True, fixture_transport=transport, fixture_count=count)
    _need(result.get("stop_reason") is None and len(result.get("papers", [])) == 1,
          "public_runtime_driver_not_terminal")
    report_path = workspace.check(root / "driver" / result["papers"][0]["report_path"])
    report = json.loads(report_path.read_bytes())
    snapshot = corpus_report_to_public_snapshot(plan, report)
    snapshot_path = write_snapshot(workspace, root, snapshot)
    # Only the adapter-owned, hash-pinned transport can make an observation
    # about calls.  An injected transport may itself contact a model or a
    # network endpoint; this adapter deliberately has no basis to assert
    # otherwise merely because the driver returned successfully.
    observed_fixture_calls: int | str = (
        0 if type(transport) is _PublicDemoTransport
        else "not_observed_by_adapter_fixture_transport"
    )
    receipt = {"version": VERSION, "mode": "recorded_fictional_fixture", "capture": deepcopy(capture),
               "source_binding_sha256": _digest(binding), "source_packet_sha256": _digest(source),
               "program_sha256": program_sha256, "driver_result": deepcopy(result),
               "report_sha256": sha256(report_path.read_bytes()).hexdigest(),
               "snapshot_sha256": sha256(snapshot_path.read_bytes()).hexdigest(),
               "network_calls": observed_fixture_calls, "model_calls": observed_fixture_calls,
               "scientifically_accepted": False, "qualified": False, "live_runtime_supported": False,
               "limitations": ["Manufactured fictional fixture replay; no model inference occurred.",
                               "Model-eligible source units are not full-article reader coverage.",
                               "Linux live runtime and hardware compatibility remain unverified."]}
    receipt_path = workspace.write_canonical_json(root / "public-runtime-receipt.json", receipt)
    return {"receipt_path": str(receipt_path), "snapshot_path": str(snapshot_path), "report_path": str(report_path),
            "receipt": receipt}
