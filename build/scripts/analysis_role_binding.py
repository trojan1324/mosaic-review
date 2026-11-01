"""Fixed-slot, source-witnessed binding of variables to existing analyses."""
from __future__ import annotations

from copy import deepcopy
from typing import Mapping, Sequence

from jsonschema import Draft202012Validator
from scripts.source_line_schema import bounded_selection_schema


ROLE_LABELS = ("predictor", "moderator", "mediator", "outcome", "covariate")
ROLE_BINDING_INSTRUCTION = """For every analysis-by-variable slot, bind roles only when the supplied source lines support them. Baseline and later nested-model covariates remain covariates when explicitly retained. Roles are analysis-specific and are not acquisition labels. Ignore table-only values. A witness is mechanical evidence, not a guarantee that the role association is scientifically correct."""


def _unit_map(units: Sequence[Mapping]) -> dict[str, Mapping]:
    return {unit["unit_id"]: unit for unit in units if isinstance(unit, Mapping)
            and isinstance(unit.get("unit_id"), str) and isinstance(unit.get("text"), str)}


def _context(ids, available):
    return [{"unit_id": unit_id, "text": available[unit_id]["text"],
             "lines": [{'line': i, 'text': line} for i, line in enumerate(available[unit_id]['text'].splitlines(keepends=True), 1)]}
            for unit_id in dict.fromkeys(ids) if unit_id in available]


def role_binding_packet(record, units):
    """Offer immutable analysis and variable context for every fixed slot."""
    available = _unit_map(units)
    slots = []
    for analysis_index, analysis in enumerate(record.get("analyses", [])):
        cited = list(analysis.get("source_units", []))
        findings = []
        for finding in analysis.get("findings", []):
            cited.extend(finding.get("source_units", []))
            findings.append({"statement": finding.get("statement"), "source_units": list(finding.get("source_units", []))})
        for variable_index, variable in enumerate(record.get("variables", [])):
            slot_id = f"a{analysis_index + 1}_v{variable_index + 1}"
            variable_ids = list(variable.get("source_units", []))
            slots.append({"slot_id": slot_id, "analysis_index": analysis_index, "variable_index": variable_index,
                          "analysis_label": analysis.get("label"), "findings": findings,
                          "analysis_source": _context(cited, available), "variable_name": variable.get("name"),
                          "methods_source": _context(variable_ids, available)})
    return {"slots": slots}


def role_binding_schema(packet):
    """Strict schema with a required response for every analysis-variable slot."""
    properties = {}
    for slot in packet["slots"]:
        allowed = list(dict.fromkeys([row["unit_id"] for row in [*slot["analysis_source"], *slot["methods_source"]]]))
        source = {row['unit_id']: row for row in [*slot['analysis_source'], *slot['methods_source']]}
        witness = bounded_selection_schema(list(source.values()), allowed)
        properties[slot["slot_id"]] = {"type": "object", "properties": {
            "roles": {"type": "array", "items": {"type": "string", "enum": list(ROLE_LABELS)}, "uniqueItems": True, "maxItems": len(ROLE_LABELS)},
            "status": {"type": "string", "enum": ["asserted", "not_reported", "unclear"]},
            "witnesses": {"type": "array", "items": witness, "maxItems": 16}},
            "required": ["roles", "status", "witnesses"], "additionalProperties": False}
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def _witness_text(witness, source):
    lines = source[witness["unit_id"]]["text"].splitlines(keepends=True)
    first, last = witness["first_line"], witness["last_line"]
    if not 1 <= first <= last <= len(lines):
        raise ValueError("invalid_role_witness_line_range")
    copied = "".join(lines[first - 1:last])
    if not any(character.isalpha() for character in copied):
        raise ValueError("role_witness_has_no_wording")
    return copied


def apply_role_binding(record, proposal, packet, units):
    """Validate a fixed packet and change only roles of pre-existing analyses."""
    Draft202012Validator(role_binding_schema(packet)).validate(proposal)
    if role_binding_packet(record, units) != packet:
        raise ValueError("role_binding_packet_changed")
    result, changes, unresolved = deepcopy(record), [], []
    source = _unit_map(units)
    for slot in packet["slots"]:
        answer = proposal[slot["slot_id"]]
        roles, status, witnesses = answer["roles"], answer["status"], answer["witnesses"]
        if status == "asserted" and (not roles or not witnesses):
            raise ValueError("asserted_role_requires_witness")
        if status != "asserted" and (roles or witnesses):
            raise ValueError("role_status_contradiction")
        copied_witnesses = [{**witness, "copied_text": _witness_text(witness, source)} for witness in witnesses]
        analysis = result["analyses"][slot["analysis_index"]]
        variable_id = result["variables"][slot["variable_index"]]["id"]
        before = deepcopy(analysis.get("roles", []))
        if status != 'asserted':
            # An empty answer cannot retire a previously supported role.
            # A future retirement needs its own source-backed correction.
            if any(role.get('variable_id') == variable_id for role in before):
                unresolved.append({'code':'existing_role_not_reaffirmed','slot_id':slot['slot_id'], 'status':status})
            continue
        retained = [role for role in before if role.get("variable_id") != variable_id]
        after = retained + [{"variable_id": variable_id, "role": role, "source_units": [witness["unit_id"] for witness in witnesses]}
                            for role in roles]
        if after != before:
            analysis["roles"] = after
            changes.append({"slot_id": slot["slot_id"], "analysis_index": slot["analysis_index"],
                            "variable_id": variable_id, "before": before, "after": deepcopy(after),
                            "witnesses": copied_witnesses})
    return {"record": result, "changes": changes, "errors": unresolved, "mechanical_valid": not unresolved,
            "semantic_accuracy_unassessed": True, "production_accepted": False}
