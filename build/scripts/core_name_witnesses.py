"""Repair exact name locators within a field's already-proposed citations.

This offline adapter makes no model call and verifies no semantic role. It never
searches the rest of the paper or changes the proposed name, role, or study.
Historical proposals and quarantine records remain available in the derivative.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import re
from typing import Any

from jsonschema import Draft202012Validator

from mosaic.canonical import canonical_bytes
from scripts.core_evidence_recovery import recover, source_name_mapping
from scripts.core_evidence_trial import validate

VERSION = "core-name-witnesses-v1"


def _unit_key(unit: dict[str, Any]) -> tuple[str, int, str]:
    locator, page = unit.get("source_locator"), unit.get("page")
    if not isinstance(locator, str) or not locator or type(page) is not int:
        raise ValueError("eligible_source_locator_and_page_required")
    return locator, page, hashlib.sha256(unit["text"].encode("utf-8")).hexdigest()


def _citations(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            if key.endswith("sentence_id"):
                yield item
            elif key.endswith("sentence_ids"):
                yield from item
            else:
                yield from _citations(item)
    elif isinstance(value, list):
        for item in value:
            yield from _citations(item)


def recover_cited_names(proposal: dict[str, Any], task: dict[str, Any], *,
                        eligible_units: list[dict[str, Any]],
                        source_policy_sha256: str,
                        allow_local_layout_match: bool = False) -> dict[str, Any]:
    """Create a separately versioned derivative with explicit locator repairs.

    Eligibility is supplied from a separately reviewed source-policy packet and
    matched by locator, page, and exact text hash, never by renumbered unit IDs.
    Every proposal citation must remain eligible. This is stricter than merely
    checking the fallback name witness. An invalid source packet refuses the
    derivative instead of being silently repaired here. The opt-in v2 fallback
    permits the existing unique case/layout mapping inside declared field
    citations only, after exhausting exact matches. Historical v1 is unchanged.
    """
    table = validate(task)
    Draft202012Validator(task["schema"]).validate(proposal)
    if not re.fullmatch(r"[0-9a-f]{64}", source_policy_sha256):
        raise ValueError("source_policy_hash_required")
    old_units = {unit["unit_id"]: _unit_key(unit) for unit in task["original_units"]}
    allowed_list = [_unit_key(unit) for unit in eligible_units]
    allowed = set(allowed_list)
    if len(allowed) != len(allowed_list) or not allowed <= set(old_units.values()):
        raise ValueError("eligible_source_is_not_unique_exact_subset")
    eligible_statements = {sid for sid, statement in table.items()
                           if all(old_units[s["unit_id"]] in allowed for s in statement["segments"])}
    if not set(_citations(proposal)) <= eligible_statements:
        raise ValueError("proposal_has_ineligible_source_citation")

    original = recover(proposal, task)
    changed = deepcopy(proposal)
    repairs = []

    def repair(field, name_key, evidence_key, path):
        name, original_id = field[name_key], field["name_sentence_id"]
        if source_name_mapping(name, table[original_id]["text"]) is not None:
            return
        alternatives = []
        for sid in dict.fromkeys(field[evidence_key]):
            mapping = source_name_mapping(name, table[sid]["text"])
            # Deliberately no case, punctuation or layout relaxation in this
            # fallback. Every occurrence of the exact literal is retained.
            if mapping is not None and mapping["match_type"] == "exact":
                alternatives.append({"sentence_id": sid, "mapping": mapping,
                                     "evidence": deepcopy(table[sid])})
        if alternatives:
            selected = alternatives[0]["sentence_id"]
            field["name_sentence_id"] = selected
            repairs.append({"field_path": path, "proposal_name": name,
                            "original_name_sentence_id": original_id,
                            "replacement_name_sentence_id": selected,
                            "selection_rule": "first_declared_field_citation_with_exact_literal",
                            "all_exact_alternatives": alternatives,
                            "semantic_identity_verified": False})
        elif allow_local_layout_match:
            for sid in dict.fromkeys(field[evidence_key]):
                mapping = source_name_mapping(name, table[sid]["text"])
                if mapping is not None:
                    alternatives.append({"sentence_id": sid, "mapping": mapping,
                                         "evidence": deepcopy(table[sid])})
            if alternatives:
                selected = alternatives[0]["sentence_id"]
                field["name_sentence_id"] = selected
                repairs.append({"field_path": path, "proposal_name": name,
                                "original_name_sentence_id": original_id,
                                "replacement_name_sentence_id": selected,
                                "selection_rule": "first_declared_field_citation_with_unique_case_or_layout_match",
                                "all_layout_alternatives": alternatives,
                                "semantic_identity_verified": False})

    for a, analysis in enumerate(changed["analyses"]):
        for r, role in enumerate(analysis["roles"]):
            repair(role, "variable_name", "role_sentence_ids", f"/analyses/{a}/roles/{r}")
    for s, study in enumerate(changed["studies"]):
        for m, measure in enumerate(study["measures"]):
            repair(measure, "name", "description_sentence_ids", f"/studies/{s}/measures/{m}")
    result = recover(changed, task)
    result.update(version="core-name-witnesses-v2" if allow_local_layout_match else VERSION,
                  original_recovery_version=original["version"],
                  original_proposal_sha256=hashlib.sha256(canonical_bytes(proposal)).hexdigest(),
                  original_task_sha256=hashlib.sha256(canonical_bytes(task)).hexdigest(),
                  source_policy_sha256=source_policy_sha256,
                  eligible_source_units_sha256=hashlib.sha256(canonical_bytes(eligible_units)).hexdigest(),
                  input_contains_excluded_units=len(allowed) < len(old_units),
                  original_quarantined_fields=deepcopy(original["quarantined_fields"]),
                  name_witness_repairs=repairs,
                  repair_status=("field_local_name_locator_only_not_semantic_acceptance" if allow_local_layout_match
                                 else "exact_name_locator_only_not_semantic_acceptance"))
    return result
