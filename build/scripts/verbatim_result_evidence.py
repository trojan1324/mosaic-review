"""Verbatim numeric context for paragraph-bounded result findings.

These selections supplement typed statistics with exact source lines.  They
retain narrative counts and percentages without assigning them to conditions,
models, or statistical fields the source did not label.
"""

from __future__ import annotations

from copy import deepcopy
import re
from collections.abc import Mapping, Sequence
from typing import Any

from jsonschema import Draft202012Validator

from scripts.paragraph_results_contract import materialize_paragraph_results, paragraph_results_schema
from scripts.result_block_groups import group_result_blocks
from scripts.source_expression_choices import resolve_statistic_choice
from scripts.source_line_schema import bounded_selection_schema
from scripts.structured_extraction_contract import audit_record
from scripts.section_field_contract import verified_count_paths


def extended_results_schema(units: Sequence[Mapping], choices: Sequence[Mapping], hypotheses: Sequence[str],
                            variables: Sequence[Mapping], *, study_id: str, label: str) -> dict[str, Any]:
    """Add required per-finding source-line evidence to the grouped results schema."""
    schema = paragraph_results_schema(units, choices, hypotheses, variables, study_id=study_id, label=label, grouped=True)
    for group in group_result_blocks(units):
        finding = schema["properties"][group["group_id"]]["properties"]["analyses"]["items"]["properties"]["findings"]["items"]
        finding["properties"]["numeric_evidence"] = {
            "type": "array", "items": bounded_selection_schema(group["units"]), "maxItems": 32
        }
        finding["required"] = list(finding["properties"])
    return schema


def _copy_selection(selection: Mapping[str, Any], sources: Mapping[str, Mapping[str, Any]], *, require_digit: bool = True) -> dict[str, Any]:
    unit_id = selection["unit_id"]
    unit = sources.get(unit_id)
    if unit is None:
        raise ValueError("numeric_evidence_source_outside_group")
    lines = unit["text"].splitlines(keepends=True)
    first, last = selection["first_line"], selection["last_line"]
    if not isinstance(first, int) or not isinstance(last, int) or first < 1 or last < first or last > len(lines):
        raise ValueError("numeric_evidence_invalid_line_range")
    copied = "".join(lines[first - 1:last])
    if require_digit and not re.search(r"\d", copied):
        raise ValueError("numeric_evidence_without_printed_digit")
    start = sum(len(line) for line in lines[:first - 1])
    return {"unit_id": unit_id, "first_line": first, "last_line": last,
            "unit_start": start, "unit_end": start + len(copied), "copied_text": copied}


def materialize_extended_results(base: Mapping[str, Any], proposal: Mapping[str, Any], units: Sequence[Mapping],
                                 choices: Sequence[Mapping], hypotheses: Sequence[str], all_units: Sequence[Mapping],
                                 *, union_evidence_ranges: bool = False) -> dict[str, Any]:
    """Materialize typed results and append exact narrative numeric context.

    The input proposal is never changed.  Numeric excerpts become an ordinary
    result statistic labelled ``Reported numeric context``; their text remains
    a source excerpt, not a typed value or condition mapping.
    """
    schema = extended_results_schema(units, choices, hypotheses, base["variables"],
                                     study_id=base["study_id"], label=base["label"])
    Draft202012Validator(schema).validate(proposal)
    groups = group_result_blocks(units)
    group_sources = {group["group_id"]: {unit["unit_id"]: unit for unit in group["units"]} for group in groups}
    cleaned = deepcopy(proposal)
    # The grouped materializer preserves group/analysis/finding order. Bind
    # by that order, never by model IDs (which can duplicate or contain ::).
    selections: list[list[list[dict[str, Any]]]] = []
    for group in groups:
        group_id = group["group_id"]
        for analysis in cleaned[group_id]["analyses"]:
            analysis_selections = []
            selections.append(analysis_selections)
            for finding_index, finding in enumerate(analysis["findings"]):
                raw = finding.pop("numeric_evidence")
                if union_evidence_ranges:
                    # A wrapped sentence can have number-free continuation
                    # lines. Union only ranges the model actually selected;
                    # never expand to an unselected line or another source.
                    ordered = sorted(raw, key=lambda s: (s['unit_id'], s['first_line'], s['last_line']))
                    merged = []
                    for selection in ordered:
                        _copy_selection(selection, group_sources[group_id], require_digit=False)
                        if (merged and merged[-1]['unit_id'] == selection['unit_id']
                                and selection['first_line'] <= merged[-1]['last_line'] + 1):
                            merged[-1]['last_line'] = max(merged[-1]['last_line'], selection['last_line'])
                        else:
                            merged.append(dict(selection))
                    raw = merged
                copied = []
                seen = set()
                for selection in raw:
                    copied_selection = _copy_selection(selection, group_sources[group_id])
                    identity = (copied_selection["unit_id"], copied_selection["first_line"], copied_selection["last_line"])
                    if identity in seen:
                        raise ValueError("duplicate_numeric_evidence_within_finding")
                    seen.add(identity)
                    copied.append(copied_selection)
                analysis_selections.append(copied)
    result = materialize_paragraph_results(base, cleaned, units, choices, hypotheses, all_units,
                                           grouped=True, resolve_choice=resolve_statistic_choice)
    copies: list[dict[str, Any]] = []
    for analysis_index, analysis in enumerate(result["record"]["analyses"]):
        for finding_index, finding in enumerate(analysis["findings"]):
            for selected in selections[analysis_index][finding_index]:
                finding["statistics"].append({"label": "Reported numeric context", "value": selected["copied_text"],
                                              "source_units": [selected["unit_id"]]})
                if selected["unit_id"] not in finding["source_units"]:
                    finding["source_units"].append(selected["unit_id"])
                copies.append({"path": f"/analyses/{analysis_index}/findings/{finding_index}/statistics/{len(finding['statistics']) - 1}",
                               **selected})
    # Recheck the final shape after additions, including field length/array
    # limits. Exact copies exempt only their own legacy numeric-format checks.
    audit = audit_record(result['record'], all_units, hypotheses)
    copied_paths = {c['path']+'/value' for c in [*result['typed_copies'],*copies]} | verified_count_paths(result['record'],all_units)
    for error in audit['errors']:
        if error['code']=='numeric_token_not_in_source' and error['path'] in copied_paths:
            continue
        if error not in result['errors']:
            result['errors'].append(error)
    result['mechanical_valid'] = not result['errors']
    result['final_audit'] = audit
    return {**result, "numeric_evidence_copies": copies, "union_evidence_ranges": union_evidence_ranges,
            "numeric_evidence_raw_proposal": deepcopy(proposal)}
