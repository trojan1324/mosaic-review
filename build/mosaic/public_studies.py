"""Source-bound study-inspection products for public corpus views.

This deliberately accepts only the existing ``scripts.study_inspection``
inventory shape.  It does not try to turn a snapshot's candidate structures
into evidence records: that conversion belongs to the existing projector,
which has the original task/recovery witnesses needed to prove every quote.
"""
from __future__ import annotations

from hashlib import sha256
from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from mosaic.canonical import canonical_bytes

VERSION = "mosaic-study-inspection/v1"
_PUBLIC_ID = re.compile(r"public-[0-9a-f]{64}\Z")


def _digest(value: object) -> str:
    return sha256(canonical_bytes(value)).hexdigest()


def _candidate_paper(inventory: Mapping[str, Any], identity: Mapping[str, Any] | None) -> dict[str, Any]:
    paper = inventory.get("paper")
    if not isinstance(paper, Mapping) or not isinstance(paper.get("paper_id"), str):
        raise ValueError("public_study_inventory_paper_invalid")
    if identity is None:
        title, citation, status = "Source identity awaiting review", "", "bibliographic_identity_unresolved"
    else:
        title = identity.get("title")
        if not isinstance(title, str) or not title:
            raise ValueError("public_study_identity_invalid")
        citation, status = identity.get("citation", ""), "source_reviewed_identity"
    studies, analyses = inventory.get("studies"), inventory.get("analyses")
    if not isinstance(studies, list) or not isinstance(analyses, list):
        raise ValueError("public_study_inventory_records_invalid")
    names = [m.get("name") for study in studies if isinstance(study, Mapping)
             for m in study.get("measures", []) if isinstance(m, Mapping) and isinstance(m.get("name"), str)]
    variables = [role.get("name") for analysis in analyses if isinstance(analysis, Mapping)
                 for role in analysis.get("roles", []) if isinstance(role, Mapping) and isinstance(role.get("name"), str)]
    wrapper = inventory.get("public_wrapper_status", {})
    if not isinstance(wrapper, Mapping):
        raise ValueError("public_study_wrapper_status_invalid")
    return {"paper_id": paper["paper_id"], "title": title, "citation": citation,
            "identity_status": status, "study_records": len(studies), "analysis_records": len(analyses),
            "measure_names": sorted(set(names), key=str.casefold),
            "variable_names": sorted(set(variables), key=str.casefold),
            "study_fact_status": wrapper.get("study_fact", "not_supplied"),
            "statistic_annotation_status": wrapper.get("statistic_annotation", "not_supplied"),
            "raw_candidate_report_retained": wrapper.get("raw_candidate_report_retained", False)}


def _snapshot_from_handoff(snapshot_ref: Mapping[str, Any]) -> dict[str, Any]:
    """Load only a path-and-digest bound snapshot from an actual handoff."""
    path, expected = snapshot_ref.get("snapshot_path"), snapshot_ref.get("snapshot_sha256")
    if not isinstance(path, (str, Path)) or not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("public_study_snapshot_reference_invalid")
    payload = Path(path).read_bytes()
    if sha256(payload).hexdigest() != expected:
        raise ValueError("public_study_snapshot_changed")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("public_study_snapshot_invalid")
    return value


def _need(value: bool, code: str) -> None:
    if not value:
        raise ValueError(code)


def _unit_map(source: Mapping[str, Any], *, inventory_present: bool) -> dict[str, Mapping[str, Any]]:
    raw = source.get("units")
    if raw is None or raw == []:
        _need(not inventory_present, "public_study_source_units_missing")
        return {}
    _need(isinstance(raw, list), "public_study_source_units_invalid")
    result: dict[str, Mapping[str, Any]] = {}
    for unit in raw:
        _need(isinstance(unit, Mapping) and isinstance(unit.get("unit_id"), str) and unit["unit_id"]
              and isinstance(unit.get("text"), str) and type(unit.get("page")) is int and unit["page"] > 0,
              "public_study_source_unit_invalid")
        _need(unit["unit_id"] not in result, "public_study_source_unit_duplicate")
        result[unit["unit_id"]] = unit
    declared = source.get("units_sha256")
    if isinstance(declared, str):
        _need(declared == _digest(raw), "public_study_source_units_hash_mismatch")
    return result


def _validate_quote(value: Mapping[str, Any], units: Mapping[str, Mapping[str, Any]]) -> None:
    quote, segments, pages = value.get("quote"), value.get("segments"), value.get("pages")
    _need(isinstance(quote, str) and quote and isinstance(segments, list) and segments and isinstance(pages, list) and all(type(page) is int for page in pages),
          "public_study_source_witness_invalid")
    parts, derived_pages, prior_unit = [], [], None
    for segment in segments:
        _need(isinstance(segment, Mapping), "public_study_source_witness_invalid")
        unit = units.get(segment.get("unit_id"))
        start, end = segment.get("start"), segment.get("end")
        _need(unit is not None and type(start) is int and type(end) is int
              and 0 <= start < end <= len(unit["text"]) and unit["text"][start:end] == segment.get("text"),
              "public_study_source_witness_invalid")
        _need(segment.get("page") == unit["page"], "public_study_source_witness_invalid")
        if "source_locator" in segment:
            _need(segment["source_locator"] == unit.get("source_locator"), "public_study_source_witness_invalid")
        if "unit_text_sha256" in segment:
            _need(segment["unit_text_sha256"] == sha256(unit["text"].encode()).hexdigest(),
                  "public_study_source_witness_invalid")
        # ``coherent_units`` joins distinct original units with one literal
        # newline before statement selection; never silently concatenate them.
        if prior_unit is not None and prior_unit != unit["unit_id"]:
            parts.append("\n")
        parts.append(segment["text"])
        prior_unit = unit["unit_id"]
        if unit["page"] not in derived_pages:
            derived_pages.append(unit["page"])
    _need("".join(parts) == quote and pages == derived_pages, "public_study_source_witness_invalid")


def _walk(value: object, units: Mapping[str, Mapping[str, Any]]) -> None:
    if isinstance(value, Mapping):
        if "quote" in value:
            _validate_quote(value, units)
        for child in value.values():
            _walk(child, units)
    elif isinstance(value, list):
        for child in value:
            _walk(child, units)


def _validated_inventory(value: Mapping[str, Any], paper: Mapping[str, Any], source: Mapping[str, Any], units: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    _need(value.get("version") == VERSION and value.get("scientifically_accepted") is False
          and value.get("coverage_verified") is False, "public_study_inventory_not_candidate")
    inv_paper = value.get("paper")
    _need(isinstance(inv_paper, Mapping) and inv_paper.get("paper_id") == paper["paper_id"]
          and inv_paper.get("source_pdf_sha256") == paper["source_pdf_sha256"], "public_study_inventory_paper_mismatch")
    identity = value.get("source_identity")
    _need(isinstance(identity, Mapping) and identity.get("eligible_source_units_sha256") == source.get("units_sha256")
          and identity.get("source_policy_sha256") == source.get("source_policy_sha256"),
          "public_study_inventory_source_identity_mismatch")
    studies, analyses = value.get("studies"), value.get("analyses")
    _need(isinstance(studies, list) and isinstance(analyses, list), "public_study_inventory_records_invalid")
    study_ids = [row.get("id") for row in studies if isinstance(row, Mapping)]
    _need(len(study_ids) == len(studies) and all(isinstance(item, str) and item for item in study_ids)
          and len(set(study_ids)) == len(study_ids), "public_study_inventory_study_ids_invalid")
    analysis_ids = [row.get("id") for row in analyses if isinstance(row, Mapping)]
    _need(len(analysis_ids) == len(analyses) and all(isinstance(item, str) and item for item in analysis_ids)
          and len(set(analysis_ids)) == len(analysis_ids), "public_study_inventory_analysis_ids_invalid")
    for analysis in analyses:
        _need(isinstance(analysis.get("study_ids"), list) and set(analysis["study_ids"]) <= set(study_ids),
              "public_study_inventory_analysis_binding_invalid")
    _walk(value, units)
    return deepcopy(dict(value))


def _empty_inventory(paper: Mapping[str, Any]) -> dict[str, Any]:
    return {"version": VERSION, "scientifically_accepted": False, "coverage_verified": False,
            "paper": dict(paper), "studies": [], "analyses": [], "quarantine": {"remaining": 0, "original": 0},
            "source_identity": {}, "limitations": ["Core study inventory was unavailable; no study records are displayed."]}


def _build_manifest(inventories: Sequence[Mapping[str, Any]], *, identities: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, bytes]]:
    rows, files, seen = [], {}, set()
    for inventory in inventories:
        paper_id = inventory.get("paper", {}).get("paper_id")
        _need(isinstance(paper_id, str) and paper_id not in seen, "public_study_inventory_duplicate_paper")
        seen.add(paper_id)
        raw = canonical_bytes(inventory)
        row = _candidate_paper(inventory, identities.get(paper_id))
        path = "study-data/" + paper_id + ".json"
        rows.append({**row, "data_path": path, "data_sha256": sha256(raw).hexdigest()})
        files[path] = raw
    return {"version": VERSION, "papers": rows, "scientifically_accepted": False,
            "coverage_verified": False}, files


def _build_public_studies(inventories: Sequence[Mapping[str, Any],], *, identities: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Internal manifest writer; callers must first validate the source inventory."""
    for inventory in inventories:
        paper = inventory.get("paper") if isinstance(inventory, Mapping) else None
        _need(isinstance(paper, Mapping) and isinstance(paper.get("paper_id"), str)
              and _PUBLIC_ID.fullmatch(paper["paper_id"]), "public_study_direct_inventory_refused")
    return _build_manifest(inventories, identities=identities)


def build_from_public_handoffs(handoffs: Sequence[Mapping[str, Any]], *, identities: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, bytes]]:
    """The only public integration entrypoint: hash-bound actual handoffs."""
    inventories = []
    for handoff in handoffs:
        _need(isinstance(handoff, Mapping) and isinstance(handoff.get("capture"), Mapping)
              and isinstance(handoff.get("snapshot"), Mapping), "public_study_handoff_invalid")
        capture, snapshot = handoff["capture"], _snapshot_from_handoff(handoff["snapshot"])
        paper, source, candidate = snapshot.get("paper"), snapshot.get("source"), snapshot.get("core_inventory_candidate")
        _need(snapshot.get("version") == "mosaic-public-snapshot/v1" and snapshot.get("validation_status") == "candidate_only"
              and snapshot.get("scientifically_accepted") is False and isinstance(paper, Mapping) and isinstance(source, Mapping)
              and isinstance(paper.get("paper_id"), str) and _PUBLIC_ID.fullmatch(paper["paper_id"])
              and paper.get("paper_id") == capture.get("paper_id") and paper.get("source_pdf_sha256") == capture.get("source_pdf_sha256")
              and snapshot.get("source_packet_sha256") == capture.get("source_packet_sha256"),
              "public_study_snapshot_binding_invalid")
        units = _unit_map(source, inventory_present=isinstance(candidate, Mapping))
        inventory = _empty_inventory(paper) if candidate is None else _validated_inventory(candidate, paper, source, units)
        identity = identities.get(paper["paper_id"])
        if identity is not None:
            _need(isinstance(identity, Mapping) and isinstance(identity.get("title"), str) and identity["title"],
                  "public_study_identity_invalid")
        inventory["paper"] = {**dict(paper), "title": identity["title"] if identity else "Source identity awaiting review",
                              "citation": identity.get("citation", "") if identity else ""}
        inventory["public_wrapper_status"] = {"study_fact": (snapshot.get("study_fact_candidate") or {}).get("status", "not_supplied"),
                                               "statistic_annotation": (snapshot.get("statistic_annotation_candidate") or {}).get("status", "not_supplied"),
                                               "raw_candidate_report_retained": True}
        inventories.append(inventory)
    return _build_public_studies(inventories, identities=identities)
