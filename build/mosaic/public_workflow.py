"""Explicit public-corpus orchestration, separate from optional advisory calls.

Capture, prepared programs, execution outcomes and views are independent
artifacts. A model failure never becomes a successful article or an automatic
retry in a different execution directory.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
import threading
import time
from typing import Any, Mapping
from urllib.request import urlopen

from mosaic.public_build import PublicBuildRefusal, digest, relative_path, validate_config, _no_links
from mosaic.public_capture_build import capture_build
from mosaic.public_native_runtime import load_capture_envelope, prepare_local_program
from mosaic.public_runtime_manifest import validate_manifest
from mosaic.workspace import Workspace

VERSION = "mosaic-public-corpus-workflow/v1"
_PRIVATE = {"sealed", "references", "diagnostic", "_holdback", "ledgers", ".git", "scratch"}


def _need(condition: bool, code: str) -> None:
    if not condition:
        raise PublicBuildRefusal(code)


def _hash(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    _need(isinstance(value, dict), "public_workflow_object_required")
    return value


def _member(root: Path, name: object) -> Path:
    member = relative_path(name)
    _need(not any(p.casefold() in _PRIVATE or p.casefold().startswith(".env") for p in member.parts),
          "public_workflow_private_path_refused")
    path = root / member
    _no_links(path, root)
    return path


def _advisory_inputs(value: object, root: Path) -> list[dict]:
    _need(isinstance(value, list), "public_corpus_advisory_inputs_invalid")
    result = []
    for row in value:
        _need(isinstance(row, dict) and set(row) == {"task_path", "sidecar_path", "sidecar_sha256"},
              "public_corpus_advisory_inputs_invalid")
        result.append({**row, "task_path": str(_member(root, row["task_path"])),
                       "sidecar_path": str(_member(root, row["sidecar_path"]))})
    return result


def source_handoff(capture_root: Path, report_path: Path, snapshot_path: Path, *, workspace: Workspace) -> dict:
    """Build the shared materializer input from actual capture/runtime files."""
    envelope = load_capture_envelope(capture_root.absolute(), workspace=workspace)
    for path in (capture_root / "native-reader.json", report_path, snapshot_path):
        workspace.check(path)
        _no_links(path.absolute(), Path(path.absolute().anchor))
    cap = envelope["capture_receipt"]
    return {"capture": {
        "paper_id": cap["work_id"], "source_pdf_path": capture_root / "source.pdf",
        "source_pdf_sha256": cap["source_pdf_sha256"],
        "native_reader": _json(capture_root / "native-reader.json"),
        "capture_receipt_path": capture_root / "receipt.json",
        "capture_receipt_sha256": _hash(capture_root / "receipt.json"),
        "source_packet_path": capture_root / "source.json", "source_packet_sha256": cap["source_packet_sha256"]},
        "snapshot": {"report_path": report_path, "report_sha256": _hash(report_path),
                     "snapshot_path": snapshot_path, "snapshot_sha256": _hash(snapshot_path)}}


def load_corpus_manifest(path: Path, *, project_root: Path, workspace: Workspace) -> list[dict]:
    """Load a JSON-friendly handoff; all member paths are project-relative.

    Each article binds its capture receipt, candidate report and snapshot by
    hash before the materializer performs the deeper source/report checks.
    """
    project_root = project_root.absolute()
    _need(path.absolute().is_relative_to(project_root), "public_workflow_manifest_outside_project")
    _no_links(path.absolute(), Path(path.absolute().anchor))
    _need(not any(p.casefold() in _PRIVATE or p.casefold().startswith(".env")
                  for p in path.absolute().relative_to(project_root).parts),
          "public_workflow_private_path_refused")
    workspace.check(path)
    manifest = _json(path)
    _need(set(manifest) == {"version", "sources"} and manifest["version"] == "mosaic-public-corpus-input/v1",
          "public_corpus_manifest_invalid")
    rows = manifest["sources"]
    _need(isinstance(rows, list) and bool(rows), "public_corpus_manifest_sources_required")
    sources = []
    for row in rows:
        required = {"capture_root", "capture_receipt_sha256", "report", "snapshot"}
        optional = {"identity_review", "apa_review", "references", "measures", "advisory_sidecars"}
        _need(isinstance(row, dict) and required <= row.keys() and row.keys() <= required | optional,
              "public_corpus_manifest_source_invalid")
        capture_root = _member(project_root, row["capture_root"])
        _no_links(capture_root / "receipt.json", project_root)
        _need(_hash(capture_root / "receipt.json") == row["capture_receipt_sha256"],
              "public_corpus_manifest_capture_changed")
        bound = {}
        for key in ("report", "snapshot"):
            ref = row[key]
            _need(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "public_corpus_manifest_reference_invalid")
            target = _member(project_root, ref["path"])
            _need(_hash(target) == ref["sha256"], "public_corpus_manifest_artifact_changed")
            bound[key] = target
        handoff = source_handoff(capture_root, bound["report"], bound["snapshot"], workspace=workspace)
        handoff.update({key: deepcopy(row[key]) for key in optional if key in row})
        if "advisory_sidecars" in handoff:
            handoff["advisory_sidecars"] = _advisory_inputs(handoff["advisory_sidecars"], project_root)
        sources.append(handoff)
    return sources


def check_serving(bundle: Path, *, workspace: Workspace, output: Path) -> dict:
    """Observe a real loopback readiness response and always close our server."""
    from mosaic.public_serving import serve_public
    server = serve_public(bundle, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{server.server_address[1]}/api/display/v1/status", timeout=15) as response:
            status = json.load(response)
        receipt = {"http_status": 200, "status_response": status, "scientifically_accepted": False,
                   "scope": "local_HTTP_readiness_browser_validation_separate"}
        workspace.write_canonical_json(output, receipt)
        return receipt
    finally:
        server.shutdown()
        thread.join(10)
        server.server_close()


def _reviews(value: object) -> dict:
    if value is None:
        return {}
    _need(isinstance(value, dict) and set(value) == {"version", "papers"}
          and value["version"] == "mosaic-public-corpus-reviews/v1" and isinstance(value["papers"], dict),
          "public_corpus_reviews_invalid")
    for paper, review in value["papers"].items():
        _need(re.fullmatch(r"public-[0-9a-f]{64}", paper) is not None and isinstance(review, dict)
              and review.keys() <= {"identity_review", "apa_review", "references", "measures", "advisory_sidecars"}, "public_corpus_review_invalid")
    return deepcopy(value["papers"])


def materialization_destination(run_root: Path, view_key: str) -> tuple[Path, bool]:
    """Return a verified bundle or a fresh bounded attempt directory.

    Materialization is derived from immutable capture/execution artifacts, so a
    failed *materialization* may be tried again.  The failed directory remains
    untouched and cannot be mistaken for a completed bundle.  This does not
    apply to a model-execution attempt, which has its own one-program boundary.
    """
    base = run_root / "display" / view_key
    bundle = base / "bundle.json"
    if bundle.is_file():
        from mosaic.public_serving import _verify_bundle
        _verify_bundle(bundle)
        return base, True
    if not base.exists():
        return base, False
    for number in range(1, 17):
        candidate = base / f"attempt-{number:02d}"
        bundle = candidate / "bundle.json"
        if bundle.is_file():
            from mosaic.public_serving import _verify_bundle
            _verify_bundle(bundle)
            return candidate, True
        if not candidate.exists():
            return candidate, False
    raise PublicBuildRefusal("public_corpus_materialization_attempts_exhausted")


def run_public_corpus(config: Mapping[str, Any], *, project_root: Path, workspace: Workspace,
                      model_manifest: Mapping[str, Any], weights_root: Path, driver_profile: Mapping[str, Any],
                      reviews: Mapping[str, Any] | None = None, live: bool = False,
                      deadline_epoch: float | None = None) -> dict:
    """Capture, prepare, optionally execute, materialize and check a local corpus.

    Jev is deliberately not called here. A completed candidate can subsequently
    receive advisory sidecars through the separate public Jev command.
    """
    validate_config(dict(config), project_root=project_root)
    if deadline_epoch is not None:
        import math
        _need(isinstance(deadline_epoch, (int, float)) and math.isfinite(deadline_epoch)
              and deadline_epoch > time.time(), "public_corpus_deadline_invalid")
    _need(config["profile"] == "corpus", "public_corpus_profile_required")
    _need(config["jev"]["enabled"] is False, "public_corpus_advisory_separate_command_required")
    reviewed = _reviews(reviews)
    for review in reviewed.values():
        if "advisory_sidecars" in review:
            review["advisory_sidecars"] = _advisory_inputs(review["advisory_sidecars"], project_root)
    model_check = validate_manifest(model_manifest, weights_root=weights_root)
    declared = config["runtime"]
    _need(declared["model"] == model_check["model_id"] and declared["revision"] == model_check["revision"]
          and declared["tokenizer_revision"] == model_manifest["model"].get("tokenizer_revision", model_check["revision"])
          and declared["weights_sha256"] == model_check["model_files_sha256"], "public_corpus_runtime_binding_mismatch")
    captured = capture_build(config, project_root=project_root, workspace=workspace)
    run_root = project_root / relative_path(config["run"]["root"])
    capture_receipt = captured["stage_receipts"]["capture"]
    dispositions = [{"status": "intake_refused", **row} for row in captured["capture_summary"]["intake_refusals"]]
    sources = []
    submitted_ids = {row["paper_id"] for row in captured["capture_summary"]["sources"]}
    _need(reviewed.keys() <= submitted_ids, "public_corpus_review_source_not_selected")
    for index, row in enumerate(captured["capture_summary"]["sources"]):
        paper_id = row["paper_id"]
        if row["status"] != "source_packet_ready":
            dispositions.append({"paper_id": paper_id, "status": "capture_refused", "reason": row.get("reason")})
            continue
        ref = capture_receipt["artifacts"][f"papers/{paper_id}/receipt.json"]
        capture_root = (run_root / relative_path(ref["path"])).parent
        envelope = load_capture_envelope(capture_root.absolute(), workspace=workspace)
        identity = digest({"capture_receipt_sha256": _hash(capture_root / "receipt.json"),
                           "manifest": model_manifest, "profile": driver_profile, "endpoint": declared["base_url"]})
        native_root = run_root / "native" / paper_id / identity
        prepared_root = native_root / "prepared"
        try:
            prepared = prepare_local_program(envelope, model_manifest, weights_root, driver_profile,
                                             declared["base_url"], prepared_root, workspace=workspace)
        except (OSError, ValueError) as exc:
            reason = str(exc) if re.fullmatch(r"public_[a-z0-9_]+", str(exc)) else type(exc).__name__
            dispositions.append({"paper_id": paper_id, "status": "preparation_refused", "reason": reason})
            continue
        if not live:
            dispositions.append({"paper_id": paper_id, "status": "prepared_only",
                                 "prepared_receipt": Path(prepared["receipt_path"]).relative_to(run_root).as_posix()})
            continue
        from mosaic.public_execution import run_local_program
        try:
            execution = run_local_program(prepared_root, capture_root, native_root / "execution", workspace=workspace,
                deadline_epoch=min(time.time() + driver_profile["driver"]["wall_clock_cap_seconds"],
                                   deadline_epoch if deadline_epoch is not None else float("inf")), runtime_observation=None)
            _need(execution.get("receipt", {}).get("native_execution_completed") is True,
                  "public_corpus_native_execution_incomplete")
        except (OSError, ValueError, RuntimeError) as exc:
            reason = str(exc) if re.fullmatch(r"public_[a-z0-9_]+", str(exc)) else type(exc).__name__
            dispositions.append({"paper_id": paper_id, "status": "execution_refused", "reason": reason})
            dispositions.extend({"paper_id": later["paper_id"], "status": "not_executed_after_failure"}
                                for later in captured["capture_summary"]["sources"][index + 1:])
            failure = {"version": VERSION, "status": "execution_stopped", "dispositions": dispositions,
                       "scientifically_accepted": False, "own_corpus_pipeline_ready": False}
            workspace.write_canonical_json(run_root / "reports" / (digest(failure) + ".json"), failure)
            raise PublicBuildRefusal("public_corpus_execution_stopped") from exc
        handoff = source_handoff(capture_root, Path(execution["report_path"]), Path(execution["snapshot_path"]), workspace=workspace)
        handoff.update(reviewed.get(paper_id, {}))
        sources.append(handoff)
        dispositions.append({"paper_id": paper_id, "status": "candidate_execution_completed",
                             "execution_receipt_sha256": _hash(Path(execution["receipt_path"]))})
    result = {"version": VERSION, "status": "prepared_only" if not live else "no_materializable_candidates",
              "capture": captured, "dispositions": dispositions, "scientifically_accepted": False,
              "own_corpus_pipeline_ready": False, "real_model_inference_run": False,
              "local_model_requests_completed": False,
              "optional_jev": "separate_optional_command", "publication_approved": False}
    if live and sources:
        from mosaic.public_corpus import materialize_public_corpus
        view_key = digest({"candidates": dispositions, "reviews": reviewed, "view": config["view"],
                           "implementation": capture_receipt["implementation_sha256"]})
        output, reused_materialization = materialization_destination(run_root, view_key)
        if not reused_materialization:
            materialize_public_corpus(sources, output, workspace=workspace)
        materialized = {"bundle_path": str(output / "bundle.json"), "bundle_sha256": _hash(output / "bundle.json")}
        check_serving(Path(materialized["bundle_path"]), workspace=workspace, output=output / "readiness.json")
        result.update(status="candidate_corpus_served", materialization=materialized,
                      local_model_requests_completed=bool(sources),
                      local_candidate_workflow_completed=True,
                      limitation="Protocol completion does not independently establish the server's model weights or scientific accuracy.")
    result["prepared_source_count"] = sum(row["status"] in ("prepared_only", "candidate_execution_completed") for row in dispositions)
    result["candidate_source_count"] = len(sources)
    result["refused_source_count"] = sum(row["status"].endswith("refused") for row in dispositions)
    if not live and not result["prepared_source_count"]:
        result["status"] = "no_prepared_sources"
    report_path = run_root / "reports" / (digest(result) + ".json")
    workspace.write_canonical_json(report_path, result)
    return {**result, "build_report_path": str(report_path), "build_report_sha256": _hash(report_path)}
