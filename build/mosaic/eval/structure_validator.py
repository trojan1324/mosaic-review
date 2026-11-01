"""Structure-candidate validator (adapted from Q/structure_validator.py; REUSE_LEDGER R018; audit M-12).

Kept: duplicate-identifier checks per collection, unknown-source walk over
evidence ids, derivation parents, factor levels and cells, event references,
full-scale-without-items, and study-relationship endpoint checks. Changed:

* Q rejected any repeated field in `field_changes`, so a study with two
  samples or several analyses could not be represented (one `sample`/
  `analysis`/`results` slot per study). Those three fields are multi-valued
  here: repeats are allowed when each entry names its own scope, and only an
  identical repeat is flagged (M-12).
* `apply_candidate_repairs` appends to multi-valued fields instead of
  overwriting the earlier entry.
* Study relationships are read from `study_relationships` and also from the
  `study_links` / `cross_study_relationships` keys that Q's candidate adapter
  wrote but Q's validator never read (M-14, W73), after mapping each value
  through `docs/protocols/relation_vocabulary_crosswalk.json`; an unmapped
  value is an error. Direction is checked later by G5.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import jsonschema

from mosaic import REPO_ROOT

MULTI_VALUED_FIELDS = {"sample", "analysis", "results"}
CROSSWALK_PATH = REPO_ROOT / "docs" / "protocols" / "relation_vocabulary_crosswalk.json"
LINK_SOURCES = {"study_relationships": "structure-v3.schema.json:study_relationships",
                "study_links": "candidate.schema.json:study_links",
                "cross_study_relationships": "candidate.schema.json:study_links"}


def load_crosswalk(path: Path = CROSSWALK_PATH) -> tuple[set[str], dict[str, dict[str, Any]]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return set(data["vnext_study_relations"]), data["crosswalk"]


def study_edges(value: dict[str, Any]) -> list[dict[str, Any]]:
    """Every study relationship in a candidate, whichever key carried it, mapped to the v2 vocabulary."""
    vocab, crosswalk = load_crosswalk()
    edges = []
    for key, source in LINK_SOURCES.items():
        for edge in value.get(key, []) or []:
            raw = edge.get("relation")
            mapped = (crosswalk.get(source, {}).get(raw) or {}).get("to")
            edges.append({"key": key, "from_study_id": edge.get("from_study_id", value.get("study_id")),
                          "to_study_id": edge.get("to_study_id", edge.get("target_study_id")),
                          "relation": raw, "mapped_relation": mapped if mapped in vocab else None,
                          "evidence_unit_ids": edge.get("evidence_unit_ids", [])})
    return edges


def sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def validate_structure(value: dict[str, Any], unit_ids: set[str], study_ids: set[str],
                       schema: dict[str, Any] | None = None) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    flags: list[dict[str, Any]] = []
    if schema is not None:
        errors += [{"kind": "schema", "message": e.message} for e in jsonschema.Draft202012Validator(schema).iter_errors(value)]
    variables = {v["variable_id"]: v for v in value.get("variables", [])}
    events = {e["event_id"]: e for e in value.get("events", [])}
    conditions = {c["condition_id"]: c for c in value.get("condition_cells", [])}
    for key, idfield in [("variables", "variable_id"), ("events", "event_id"), ("condition_cells", "condition_id"),
                         ("measures", "measure_id")]:
        rows = value.get(key, [])
        if len({r[idfield] for r in rows}) != len(rows):
            errors.append({"kind": "duplicate_identifier", "collection": key})

    def walk(obj: Any, path: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ("evidence_unit_ids", "item_unit_ids"):
                    for uid in v:
                        if uid not in unit_ids:
                            errors.append({"kind": "unknown_source", "path": path, "id": uid})
                else:
                    walk(v, f"{path}/{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{path}/{i}")

    walk(value)
    for v in variables.values():
        for parent in v.get("based_on_variable_ids", []):
            if parent not in variables or parent == v["variable_id"]:
                errors.append({"kind": "invalid_derivation", "variable": v["variable_id"], "parent": parent})
        if v.get("acquisition") == "derived" and not v.get("based_on_variable_ids"):
            flags.append({"kind": "derived_inputs_unresolved", "variable": v["variable_id"]})
        if v.get("acquisition") == "measured" and "factor" in v.get("analysis_roles", []):
            flags.append({"kind": "observed_factor_requires_design_class_review", "variable": v["variable_id"]})
        if not v.get("evidence_unit_ids"):
            errors.append({"kind": "variable_without_source", "id": v["variable_id"]})
    levels: dict[str, set[str]] = {}
    for f in value.get("factor_levels", []):
        vid = f["variable_id"]
        if vid not in variables:
            errors.append({"kind": "unknown_factor", "id": vid})
        levels[vid] = {lvl["level_id"] for lvl in f["levels"]}
    for c in conditions.values():
        pairs = [(p["variable_id"], p["level_id"]) for p in c["factor_levels"]]
        if len({v for v, _ in pairs}) != len(pairs):
            errors.append({"kind": "duplicate_factor_in_cell", "id": c["condition_id"]})
        for vid, lid in pairs:
            if lid not in levels.get(vid, set()):
                errors.append({"kind": "unknown_level", "condition": c["condition_id"], "variable": vid, "level": lid})
    for e in events.values():
        for vid in e.get("variable_ids", []):
            if vid not in variables:
                errors.append({"kind": "unknown_event_variable", "id": vid})
        for cid in e.get("condition_ids", []):
            if cid not in conditions:
                errors.append({"kind": "unknown_event_condition", "id": cid})
    for e in value.get("event_edges", []):
        if e["from_event_id"] not in events or e["to_event_id"] not in events:
            errors.append({"kind": "dangling_event_edge", "edge": e})
    for m in value.get("measures", []):
        if m["variable_id"] not in variables:
            errors.append({"kind": "unknown_measure_variable", "id": m["variable_id"]})
        if m.get("item_availability") == "full" and not m.get("item_unit_ids"):
            errors.append({"kind": "full_scale_without_items", "id": m["measure_id"]})
    for edge in study_edges(value):
        if edge["from_study_id"] not in study_ids or edge["to_study_id"] not in study_ids or edge["from_study_id"] == edge["to_study_id"]:
            errors.append({"kind": "invalid_study_relationship", "edge": edge})
        if edge["mapped_relation"] is None:
            errors.append({"kind": "unmapped_study_relation", "edge": edge})
        if not edge["evidence_unit_ids"]:
            errors.append({"kind": "link_without_source", "edge": edge})
    seen: dict[str, list[Any]] = {}
    for change in value.get("field_changes", []):
        name = change["field"]
        if name in MULTI_VALUED_FIELDS:
            key = change.get("scope_id") or sha({k: change.get(k) for k in ("value", "evidence_unit_ids")})
            if key in seen.setdefault(name, []):
                flags.append({"kind": "duplicate_identical_repair", "field": name})
            seen[name].append(key)
        elif name in seen:
            errors.append({"kind": "duplicate_field_repair", "field": name})
        else:
            seen[name] = [name]
    return {"schema_and_graph_valid": not errors, "errors": errors, "flags": flags, "candidate_sha256": sha(value),
            "scientific_review_required": True, "critical_findings_reported": len(value.get("critical_findings", [])),
            "canonical_promotion_allowed": False}


def apply_candidate_repairs(original: dict[str, Any], structure: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(original)
    for change in structure.get("field_changes", []):
        entry = {k: change[k] for k in ("status", "value", "evidence_unit_ids") if k in change}
        if change["field"] in MULTI_VALUED_FIELDS:
            if change.get("scope_id"):
                entry["scope_id"] = change["scope_id"]
            current = out["fields"].get(change["field"])
            bucket = current if isinstance(current, list) else []
            out["fields"][change["field"]] = bucket + [entry]
        else:
            out["fields"][change["field"]] = entry
    return {"original_candidate_sha256": sha(original), "structure_sha256": sha(structure), "repaired_fields_candidate": out,
            "structure_candidate": structure, "approval": "not independently verified", "canonical_promotion_allowed": False}
