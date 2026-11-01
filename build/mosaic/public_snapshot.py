"""Source-bound, candidate-only public intermediate for current corpus reports.

This adapter consumes an already-captured in-memory plan and finalized report. It
does not invoke ``Run`` or capture code, and therefore cannot consult a private
registry or sealed manifest.
"""
from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from mosaic.canonical import canonical_bytes
from mosaic.workspace import Workspace


class PublicSnapshotRefused(ValueError):
    """The supplied packet cannot be copied without losing its binding."""


BASE_VERSION = "complete-extraction-sequence/v1"
WRAPPED_VERSION = "complete-extraction-with-study-selections/v1"
SNAPSHOT_VERSION = "mosaic-public-snapshot/v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BASE_FIELDS = {"version", "paper", "findings", "core", "measures", "schema_omissions",
                "scientifically_accepted", "production_accepted", "complete_extraction_claimed"}
_WRAPPED_FIELDS = {"version", "original_report", "original_report_sha256", "study_fact_stage",
                   "statistic_annotation", "declared_unavailable_stages", "scientifically_accepted",
                   "production_accepted", "complete_extraction_claimed"}


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _need(value: bool, code: str) -> None:
    if not value:
        raise PublicSnapshotRefused(code)


def _unit_map(source: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    units = source.get("units")
    _need(isinstance(units, list) and units, "public_snapshot_source_units_missing")
    result: dict[str, dict[str, Any]] = {}
    for unit in units:
        _need(isinstance(unit, Mapping) and isinstance(unit.get("unit_id"), str), "public_snapshot_unit_id_invalid")
        _need(isinstance(unit.get("text"), str) and isinstance(unit.get("page"), int), "public_snapshot_unit_literal_invalid")
        _need(unit["unit_id"] not in result, "public_snapshot_unit_id_duplicate")
        result[unit["unit_id"]] = deepcopy(dict(unit))
    return result


def _candidate_flags(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"scientifically_accepted", "production_accepted", "qualified", "complete_extraction_claimed"}:
                _need(child is False, "public_snapshot_candidate_claim_invalid:" + key)
            _candidate_flags(child)
    elif isinstance(value, list):
        for child in value:
            _candidate_flags(child)


def _source_witnesses(value: Any, units: Mapping[str, Mapping[str, Any]], *, original_segments: bool = False) -> None:
    if isinstance(value, Mapping):
        unit_id = value.get("unit_id")
        # ``unit_id`` is also used by derived sentence/value tables (c001,
        # etc.). Only an original-text span claims an original unit namespace.
        if original_segments and unit_id is not None and {"start", "end", "text"} <= set(value):
            _need(isinstance(unit_id, str) and unit_id in units, "public_snapshot_unknown_witness_unit")
            start, end, text = value["start"], value["end"], value["text"]
            _need(isinstance(start, int) and isinstance(end, int) and isinstance(text, str),
                  "public_snapshot_witness_span_invalid")
            _need(0 <= start <= end <= len(units[unit_id]["text"])
                  and units[unit_id]["text"][start:end] == text,
                  "public_snapshot_witness_text_mismatch")
        for key, child in value.items():
            _source_witnesses(child, units, original_segments=original_segments or key == "original_segments")
    elif isinstance(value, list):
        for child in value:
            _source_witnesses(child, units, original_segments=original_segments)


def _base_and_outer(report: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if report.get("version") == BASE_VERSION:
        extras = sorted(set(report) - _BASE_FIELDS)
        _need(not extras, "public_snapshot_unknown_report_fields:" + ",".join(extras))
        return deepcopy(dict(report)), {}
    _need(report.get("version") == WRAPPED_VERSION, "public_snapshot_report_version_invalid")
    extras = sorted(set(report) - _WRAPPED_FIELDS)
    _need(not extras, "public_snapshot_unknown_wrapper_fields:" + ",".join(extras))
    original = report.get("original_report")
    _need(isinstance(original, Mapping) and _digest(original) == report.get("original_report_sha256"),
          "public_snapshot_original_report_hash_mismatch")
    base, ignored = _base_and_outer(original)
    _need(not ignored, "public_snapshot_nested_wrapper_invalid")
    stage = report.get("study_fact_stage")
    _need(isinstance(stage, Mapping) and isinstance(stage.get("status"), str), "public_snapshot_study_fact_stage_invalid")
    unavailable = report.get("declared_unavailable_stages", [])
    _need(isinstance(unavailable, list) and all(isinstance(item, Mapping) for item in unavailable),
          "public_snapshot_declared_unavailable_invalid")
    return base, deepcopy(dict(report))


def _stage(value: Any, name: str) -> Mapping[str, Any]:
    _need(isinstance(value, Mapping) and isinstance(value.get("status"), str), "public_snapshot_stage_missing:" + name)
    return value


def corpus_report_to_public_snapshot(plan: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a current base or runner-wrapped report into a non-serving candidate packet."""
    _need(isinstance(plan, Mapping) and isinstance(report, Mapping), "public_snapshot_mapping_required")
    paper, source = plan.get("paper"), plan.get("source")
    _need(isinstance(paper, Mapping) and isinstance(source, Mapping), "public_snapshot_plan_missing_paper_or_source")
    _need(isinstance(paper.get("paper_id"), str) and isinstance(paper.get("source_pdf_sha256"), str)
          and bool(_SHA256.fullmatch(paper["source_pdf_sha256"])), "public_snapshot_paper_identity_invalid")
    identity = plan.get("source_identity")
    _need(isinstance(identity, Mapping) and identity.get("source_sha256") == _digest(source),
          "public_snapshot_source_hash_mismatch")
    _need(isinstance(identity.get("source_sha256"), str) and bool(_SHA256.fullmatch(identity["source_sha256"])),
          "public_snapshot_source_identity_invalid")
    _need(identity.get("units_sha256") == _digest(source.get("units")), "public_snapshot_units_hash_mismatch")
    _need(identity.get("source_policy_sha256") == source.get("source_policy_sha256"),
          "public_snapshot_source_policy_hash_mismatch")
    units = _unit_map(source)
    base, outer = _base_and_outer(report)
    _need(base.get("scientifically_accepted") is False and base.get("production_accepted") is False
          and base.get("complete_extraction_claimed") is False, "public_snapshot_base_claim_invalid")
    if outer:
        _need(outer.get("scientifically_accepted") is False and outer.get("production_accepted") is False
              and outer.get("complete_extraction_claimed") is False, "public_snapshot_wrapper_claim_invalid")
    _need(base.get("paper") == paper, "public_snapshot_paper_identity_mismatch")
    _candidate_flags(report)
    findings, core = _stage(base.get("findings"), "findings"), _stage(base.get("core"), "core")
    measures = base.get("measures")
    _need(isinstance(measures, Mapping), "public_snapshot_stage_missing:measures")
    measure_stages = measures.get("measure_stages")
    if measure_stages is not None:
        _need(isinstance(measure_stages, list) and all(isinstance(item, Mapping) and isinstance(item.get("status"), str)
              for item in measure_stages), "public_snapshot_measure_stages_invalid")
    else:
        _need(isinstance(measures.get("status"), str), "public_snapshot_stage_missing:measures")
    _source_witnesses(report, units)
    dispositions: list[dict[str, Any]] = []
    for name, stage in (("findings", findings), ("core", core)):
        if stage["status"] != "materialized_source_bound_proposal":
            dispositions.append({"field": name, "disposition": "stage_not_materialized", "status": stage["status"], "reason": stage.get("reason")})
    if measure_stages is None:
        dispositions.append({"field": "measures", "disposition": "stage_not_materialized", "status": measures["status"], "reason": measures.get("reason")})
    else:
        for stage in measure_stages:
            if stage["status"] != "materialized_source_bound_proposal":
                dispositions.append({"field": "measure_stage", "stage_id": stage.get("id"), "disposition": "stage_not_materialized", "status": stage["status"], "reason": stage.get("reason")})
    inventory: dict[str, Any] | None = None
    studies: list[Any] = []
    analyses: list[Any] = []
    if core["status"] == "materialized_source_bound_proposal":
        materialized = core.get("materialized")
        _need(isinstance(materialized, Mapping) and isinstance(materialized.get("inventory"), Mapping), "public_snapshot_core_inventory_missing")
        inventory = deepcopy(dict(materialized["inventory"]))
        raw_studies = inventory.get("studies", [])
        _need(isinstance(raw_studies, list) and all(isinstance(item, Mapping) for item in raw_studies), "public_snapshot_studies_invalid")
        studies = deepcopy(raw_studies)
        raw_analyses = inventory.get("analyses", [])
        _need(isinstance(raw_analyses, list), "public_snapshot_analyses_invalid")
        analyses = deepcopy(raw_analyses)
    else:
        dispositions.append({"field": "inventory", "disposition": "unavailable_from_core", "status": core["status"]})
    if outer:
        dispositions.append({"field": "study_fact_stage", "disposition": "candidate_wrapper_retained", "status": outer["study_fact_stage"]["status"]})
        if "statistic_annotation" in outer:
            dispositions.append({"field": "statistic_annotation", "disposition": "candidate_wrapper_retained"})
        for item in outer.get("declared_unavailable_stages", []):
            dispositions.append({"field": "declared_unavailable_stage", "disposition": "declared_unavailable", "request_id": item.get("request_id"), "reason": item.get("reason")})
    source_packet = deepcopy(dict(source))
    return {"version": SNAPSHOT_VERSION, "paper": deepcopy(dict(paper)),
            "source": {"source_sha256": identity["source_sha256"], "units_sha256": identity.get("units_sha256"), "source_policy_sha256": identity.get("source_policy_sha256"), "units": list(units.values())},
            "source_packet_candidate": source_packet, "source_packet_sha256": _digest(source_packet),
            "source_coverage": {"packet_scope": source.get("scope", "model_eligible_body_candidate"),
                "model_eligible_units_only": True, "full_article_reader_coverage": "not_established",
                "reader_materializer_requirement": "Use the public capture package and source-binding; do not treat source.units as full article reader coverage."},
            "candidate_report": deepcopy(dict(report)), "base_report_candidate": base,
            "findings_candidate": deepcopy(dict(findings)), "core_candidate": deepcopy(dict(core)), "measures_candidate": deepcopy(dict(measures)),
            "core_inventory_candidate": inventory, "studies_candidate": studies, "analyses_candidate": analyses,
            "study_fact_candidate": deepcopy(outer.get("study_fact_stage")) if outer else None,
            "statistic_annotation_candidate": deepcopy(outer.get("statistic_annotation")) if outer else None,
            "field_dispositions": dispositions, "validation_status": "candidate_only", "scientifically_accepted": False,
            "qualified": False, "serving_artifact_status": "not_materialized",
            "downstream_seam": "public isolated capture adapter must emit the source.json packet without Run/capture_corpus_demo; corpus_report_to_search_snapshot is then required for search.sqlite/display bundle"}


def write_snapshot(workspace: Workspace, run_root: Path, snapshot: Mapping[str, Any]) -> Path:
    _need(snapshot.get("version") == SNAPSHOT_VERSION, "public_snapshot_version_invalid")
    target = workspace.check(Path(run_root) / "public-snapshot.json")
    workspace.write_canonical_json(target, dict(snapshot))
    return target
