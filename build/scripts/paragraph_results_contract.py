"""Paragraph-bounded wrapper for typed results extraction.

Each supplied results source unit receives one required response object.  A
response can therefore neither cite nor select a typed statistic expression
from a different paragraph.  This is a mechanical provenance guard only.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Mapping, Sequence

from jsonschema import Draft202012Validator

from scripts.section_field_contract import materialize_results, typed_results_schema, verified_count_paths
from scripts.structured_extraction_contract import audit_record
from scripts.result_block_groups import group_result_blocks


def _unit_ids(units: Sequence[Mapping]) -> list[str]:
    values = [unit.get("unit_id") if isinstance(unit, Mapping) else None for unit in units]
    if not values or len(values) != len(set(values)) or not all(isinstance(value, str) and value for value in values):
        raise ValueError("result_units_must_have_unique_nonempty_ids")
    return values


def paragraph_results_schema(units, choices, hypotheses, variables, *, study_id=None, label=None, grouped=False):
    """Return a strict, required-per-paragraph typed-results schema."""
    _unit_ids(units)
    groups = group_result_blocks(units) if grouped else [dict(group_id=u['unit_id'], units=[u], joins=[]) for u in units]
    ids = [g['group_id'] for g in groups]
    properties = {}
    for group in groups:
        unit_id = group['group_id']
        member_ids = {u['unit_id'] for u in group['units']}
        local_choices = [choice for choice in choices if choice.get("unit_id") in member_ids]
        # The inner typed schema constrains every citation to this one unit and
        # every statistic choice to expressions found there.
        properties[unit_id] = typed_results_schema(group['units'], local_choices, hypotheses, variables)
        if study_id is not None:
            properties[unit_id]['properties']['study_id'] = {'const': study_id}
        if label is not None:
            properties[unit_id]['properties']['label'] = {'const': label}
    return {"type": "object", "properties": properties, "required": ids, "additionalProperties": False}


def _direction_choice_errors(proposal, unit_id):
    """Require selected-estimate direction evidence to be a selected finding statistic."""
    errors = []
    for analysis_index, analysis in enumerate(proposal.get("analyses", [])):
        for finding_index, finding in enumerate(analysis.get("findings", [])):
            evidence = finding.get("direction_evidence")
            if isinstance(evidence, Mapping) and evidence.get("kind") == "selected_estimate":
                choice_id = evidence.get("choice_id")
                if choice_id not in finding.get("statistic_choices", []):
                    errors.append({"code": "direction_choice_not_among_finding_statistics",
                                   "path": f"/{unit_id}/analyses/{analysis_index}/findings/{finding_index}/direction_evidence"})
    return errors


def materialize_paragraph_results(base, proposal, units, choices, hypotheses, all_units, *, grouped=False, resolve_choice=None):
    """Validate and materialize every paragraph, retaining its source-unit identity."""
    schema = paragraph_results_schema(units, choices, hypotheses, base["variables"], grouped=grouped)
    Draft202012Validator(schema).validate(proposal)
    merged = deepcopy(base)
    merged["analyses"] = []
    merged["hypothesis_support"] = []
    errors, copies, checks, analysis_sources, citation_bindings = [], [], [], [], []
    seen_ids = set()
    groups = group_result_blocks(units) if grouped else [dict(group_id=u['unit_id'], units=[u], joins=[]) for u in units]
    for group in groups:
        unit_id = group['group_id']
        local_proposal = deepcopy(proposal[unit_id])
        if grouped:
            # The validated group schema already restricts selected choices to
            # this source group. Their immutable source IDs are authoritative
            # provenance; do not ask the model to repeat them perfectly in a
            # second list before copying their text.
            by_choice = {c['choice_id']: c for c in choices}
            for ai, analysis in enumerate(local_proposal['analyses']):
                for fi, finding in enumerate(analysis['findings']):
                    selected_sources = [by_choice[c]['unit_id'] for c in finding['statistic_choices']]
                    for source_id in dict.fromkeys(selected_sources):
                        if source_id not in finding['source_units']:
                            finding['source_units'].append(source_id)
                            citation_bindings.append({'group_id':unit_id, 'analysis':ai, 'finding':fi,
                                'unit_id':source_id, 'reason':'derived_from_verified_selected_statistic'})
        direction_errors = _direction_choice_errors(local_proposal, unit_id)
        # The outer schema already restricts this proposal to ``unit``.  Pass
        # the full catalogue and full source sequence here so stable choice IDs
        # retain their article-wide identity during re-verification.
        kwargs = {} if resolve_choice is None else {'resolve_choice': resolve_choice}
        result = materialize_results(base, local_proposal, units, choices, hypotheses, all_units, **kwargs)
        errors.extend(direction_errors)
        errors.extend({**error, "path": f"/{unit_id}{error['path']}"} for error in result["errors"])
        offset = len(merged['analyses'])
        for copy in result["typed_copies"]:
            parts = copy['path'].split('/')
            parts[2] = str(int(parts[2]) + offset)
            copies.append({**copy, 'path': '/'.join(parts), "source_unit_id": unit_id})
        for check in result["direction_checks"]:
            checks.append({**check, "source_unit_id": unit_id})
        for analysis in result["record"]["analyses"]:
            copied = deepcopy(analysis)
            copied["id"] = f"{unit_id}::{analysis['id']}"
            if copied["id"] in seen_ids:
                errors.append({"code": "duplicate_namespaced_analysis_id", "path": f"/{unit_id}/analyses"})
            seen_ids.add(copied["id"])
            merged["analyses"].append(copied)
            analysis_sources.append({"analysis_id": copied["id"], "source_unit_id": unit_id})
        merged["hypothesis_support"].extend(deepcopy(result["record"]["hypothesis_support"]))
        merged["gaps"].extend(item for item in deepcopy(result["record"]["gaps"]) if item not in merged["gaps"])
    audit = audit_record(merged, all_units, hypotheses)
    copied_paths = {c['path']+'/value' for c in copies} | verified_count_paths(merged, all_units)
    errors.extend(e for e in audit['errors'] if not
                  (e['code'] == 'numeric_token_not_in_source' and e['path'] in copied_paths))
    return {"record": merged, "typed_copies": copies, "direction_checks": checks,
            "analysis_source_units": analysis_sources, "errors": errors,
            "mechanical_valid": not errors, "semantic_accuracy_unassessed": True,
            "production_accepted": False, "legacy_audit": audit,
            "citation_bindings": citation_bindings,
            "result_groups": [{'group_id':g['group_id'], 'unit_ids':[u['unit_id'] for u in g['units']], 'joins':g['joins']} for g in groups]}
