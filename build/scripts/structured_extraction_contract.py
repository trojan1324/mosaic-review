"""Mechanical source-binding checks for the staged study extractor.

The model may choose offered unit identifiers and line ranges.  This module
copies the selected lines itself; it never accepts a generated quotation or
decides whether the resulting scientific interpretation is correct.
"""

from __future__ import annotations

from collections import Counter
import re
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator
from mosaic.eval.value_support import TOKEN
from mosaic.core_contract import cardinal_integer_value


def _obj(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def _string(maximum: int = 4_000) -> dict[str, Any]:
    return {"type": "string", "minLength": 1, "maxLength": maximum}


def _array(item: dict[str, Any], maximum: int, minimum: int = 0) -> dict[str, Any]:
    return {"type": "array", "items": item, "minItems": minimum,
            "maxItems": maximum}


def study_schema(unit_ids: Sequence[str], hypothesis_ids: Sequence[str]) -> dict[str, Any]:
    """Return an xgrammar-compatible strict schema constrained to source IDs."""
    if not unit_ids or len(set(unit_ids)) != len(unit_ids) or not all(isinstance(v, str) and v for v in unit_ids):
        raise ValueError("unit_ids_must_be_unique_nonempty_strings")
    if len(set(hypothesis_ids)) != len(hypothesis_ids) or not all(isinstance(v, str) and v for v in hypothesis_ids):
        raise ValueError("hypothesis_ids_must_be_unique_nonempty_strings")
    unit = {"type": "string", "enum": list(unit_ids)}
    source_units = _array(unit, 128, 1)
    selection = _obj({"unit_id": unit, "first_line": {"type": "integer", "minimum": 1},
                      "last_line": {"type": "integer", "minimum": 1}})
    selections = _array(selection, 128)
    fact = _obj({"label": _string(128), "value": _string(2_000), "source_units": source_units})
    measure = _obj({"name": _string(1_000), "source_units": source_units,
                    "items": selections, "anchors": selections,
                    "n_items": {"anyOf": [fact, {"type": "null"}]},
                    "reliability": _array(fact, 32)})
    variable = _obj({"id": _string(128), "name": _string(1_000),
                     "acquisition": {"type": "string", "enum": ["manipulated", "measured", "unclear"]},
                     "source_units": source_units, "measures": _array(measure, 32)})
    role = _obj({"variable_id": _string(128),
                 "role": {"type": "string", "enum": ["predictor", "moderator", "mediator", "outcome", "covariate"]},
                 "source_units": source_units})
    finding = _obj({"statement": _string(),
                    "direction": {"type": "string", "enum": ["positive", "negative", "null", "mixed", "unclear"]},
                    "significance": {"type": "string", "enum": ["significant", "nonsignificant", "mixed", "not_stated"]},
                    "source_units": source_units, "statistics": _array(fact, 96)})
    analysis = _obj({"id": _string(128), "label": _string(1_000), "source_units": source_units,
                     "roles": _array(role, 32), "findings": _array(finding, 96)})
    support = _obj({"hypothesis_id": {"type": "string", **({"enum": list(hypothesis_ids)} if hypothesis_ids else {})},
                    "support": {"type": "string", "enum": ["supported", "not_supported", "partial", "unclear"]},
                    "source_units": source_units})
    gap = _obj({"category": _string(128), "note": _string()})
    return _obj({"study_id": _string(128), "label": _string(1_000), "design": _array(fact, 64),
                 "variables": _array(variable, 64), "analyses": _array(analysis, 96),
                 "hypothesis_support": _array(support, 96 if hypothesis_ids else 0), "gaps": _array(gap, 96)})


def _error(errors: list[dict[str, str]], code: str, path: str) -> None:
    errors.append({"code": code, "path": path})


def _path(parts: Sequence[Any]) -> str:
    return "/" + "/".join(str(part) for part in parts)


def _numeric_value(value: str, unit_texts: list[str]) -> bool:
    """Require an entire printed numeric token in one of the asserted units."""
    wanted = list(TOKEN.finditer(value))
    if not wanted:
        # A printed item count may be a word. Preserve it, without conversion.
        return cardinal_integer_value(value) is not None and any(
            re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text) for text in unit_texts)
    for text in unit_texts:
        if value not in text:
            continue
        tokens = {(m.start(), m.end(), m.group()) for m in TOKEN.finditer(text)}
        for match in re.finditer(re.escape(value), text):
            if all((match.start() + candidate.start(), match.start() + candidate.end(), candidate.group()) in tokens for candidate in wanted):
                return True
    return False


def audit_record(record: Any, units: Sequence[Mapping[str, Any]], hypothesis_ids: Sequence[str]) -> dict[str, Any]:
    """Audit source membership, selected line copies, IDs, and numeric boundaries.

    Returned selection text is copied from the input units with original line
    endings.  ``mechanical_valid`` deliberately says nothing about extraction
    correctness or completeness.
    """
    errors: list[dict[str, str]] = []
    texts: dict[str, str] = {}
    for index, item in enumerate(units):
        if not isinstance(item, Mapping) or not isinstance(item.get("unit_id"), str) or not isinstance(item.get("text"), str):
            _error(errors, "invalid_unit", f"/units/{index}")
            continue
        if item["unit_id"] in texts:
            _error(errors, "duplicate_unit_id", f"/units/{index}")
        else:
            texts[item["unit_id"]] = item["text"]
    try:
        schema = study_schema(list(texts), hypothesis_ids)
    except ValueError as exc:
        _error(errors, str(exc), "/")
        return _report(errors, [])
    schema_errors = sorted(Draft202012Validator(schema).iter_errors(record), key=lambda item: list(item.absolute_path))
    for item in schema_errors:
        _error(errors, "schema_invalid", _path(list(item.absolute_path)))
    if schema_errors or not isinstance(record, Mapping):
        return _report(errors, [])

    resolved: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    def sources(ids: list[str], path: str) -> list[str]:
        known = []
        for unit_id in ids:
            if unit_id not in texts:
                _error(errors, "unknown_source_unit", path)
            else:
                known.append(unit_id)
        counts["source_unit_references"] += len(ids)
        return known

    def fact(item: Mapping[str, Any], path: str, numeric: bool = False) -> None:
        ids = sources(item["source_units"], path + "/source_units")
        counts["facts"] += 1
        if numeric and not _numeric_value(item["value"], [texts[item_id] for item_id in ids]):
            _error(errors, "numeric_token_not_in_source", path + "/value")

    def selections(items: list[Mapping[str, Any]], path: str) -> None:
        for index, selection in enumerate(items):
            text = texts[selection["unit_id"]]
            lines = text.splitlines(keepends=True)
            first, last = selection["first_line"], selection["last_line"]
            here = f"{path}/{index}"
            if last < first or first > len(lines) or last > len(lines):
                _error(errors, "invalid_line_range", here)
                continue
            copied = "".join(lines[first - 1:last])
            start = sum(map(len, lines[:first - 1]))
            resolved.append({"path": here, "unit_id": selection["unit_id"], "first_line": first,
                             "last_line": last, "unit_start": start, "unit_end": start + len(copied),
                             "copied_text": copied})
            counts["resolved_selections"] += 1

    for index, item in enumerate(record["design"]):
        fact(item, f"/design/{index}", item["label"] in {"analysed_n", "excluded_n"})
    variable_ids: set[str] = set()
    for index, variable in enumerate(record["variables"]):
        base = f"/variables/{index}"
        if variable["id"] in variable_ids:
            _error(errors, "duplicate_variable_id", base)
        variable_ids.add(variable["id"])
        sources(variable["source_units"], base + "/source_units")
        for measure_index, measure in enumerate(variable["measures"]):
            measure_base = f"{base}/measures/{measure_index}"
            sources(measure["source_units"], measure_base + "/source_units")
            selections(measure["items"], measure_base + "/items")
            selections(measure["anchors"], measure_base + "/anchors")
            if measure["n_items"] is not None:
                fact(measure["n_items"], measure_base + "/n_items", True)
            for reliability_index, reliability in enumerate(measure["reliability"]):
                fact(reliability, f"{measure_base}/reliability/{reliability_index}", True)
    analysis_ids: set[str] = set()
    for index, analysis in enumerate(record["analyses"]):
        base = f"/analyses/{index}"
        if analysis["id"] in analysis_ids:
            _error(errors, "duplicate_analysis_id", base)
        analysis_ids.add(analysis["id"])
        sources(analysis["source_units"], base + "/source_units")
        for role_index, role in enumerate(analysis["roles"]):
            role_base = f"{base}/roles/{role_index}"
            if role["variable_id"] not in variable_ids:
                _error(errors, "unknown_variable_link", role_base + "/variable_id")
            sources(role["source_units"], role_base + "/source_units")
        for finding_index, finding in enumerate(analysis["findings"]):
            finding_base = f"{base}/findings/{finding_index}"
            sources(finding["source_units"], finding_base + "/source_units")
            for stat_index, statistic in enumerate(finding["statistics"]):
                fact(statistic, f"{finding_base}/statistics/{stat_index}", True)
    for index, support in enumerate(record["hypothesis_support"]):
        sources(support["source_units"], f"/hypothesis_support/{index}/source_units")
    return _report(errors, resolved, counts)


def _report(errors: list[dict[str, str]], resolved: list[dict[str, Any]], counts: Counter[str] | None = None) -> dict[str, Any]:
    return {"mechanical_valid": not errors, "semantic_accuracy_unassessed": True,
            "error_count": len(errors), "reason_counts": dict(sorted(Counter(e["code"] for e in errors).items())),
            "errors": errors, "resolved_selections": resolved,
            "counts": dict(sorted((counts or Counter()).items()))}
