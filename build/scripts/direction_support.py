"""Fail-closed, mechanical support checks for asserted result directions.

This module does not determine whether an extraction is scientifically true.
It only checks whether an asserted positive or negative direction carries a
selected source witness with a directional phrase or a typed signed estimate.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any


_POSITIVE = re.compile(
    r"\b(?:increased?|higher|more|greater|positive(?:ly)?\s+(?:predict(?:ed|s)?|"
    r"associat(?:ed|es)|relat(?:ed|es))|result(?:ed|s)?\s+in\s+higher)\b",
    re.IGNORECASE,
)
_NEGATIVE = re.compile(
    r"\b(?:reduced?|decreased?|lower|less|fewer|negative(?:ly)?\s+(?:predict(?:ed|s)?|"
    r"associat(?:ed|es)|relat(?:ed|es))|result(?:ed|s)?\s+in\s+lower)\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(r"\b(?:not|no|never|neither|without)\b(?:\W+\w+){0,4}\W*$", re.IGNORECASE)
_SIGNED_LABELS = {"b", "beta", "β", "coefficient", "effect", "estimate", "slope", "r", "correlation", "difference"}
_BANNED_LABELS = {"se", "standard error", "p", "p value", "ci", "ci lower", "ci upper", "confidence interval", "df", "n"}
_NUMBER = re.compile(r"^\s*([+\-\u2212\u2013])?(?:(?:\d+(?:\.\d+)?)|(?:\.\d+))\s*$")


def direction_evidence_schema(unit_ids: Sequence[str], statistic_count: int) -> dict[str, Any]:
    """Return the narrow optional evidence field for a finding contract.

    ``direction='unclear'`` has no required witness: it preserves a reported
    relationship whose text supplies no sign.  Positive and negative findings
    must use one of these exact witnesses when validated below.
    """
    source = {"type": "string", "enum": list(unit_ids)}
    return {
        "oneOf": [
            {"type": "object", "additionalProperties": False,
             "required": ["kind", "unit_id", "first_line", "last_line"],
             "properties": {"kind": {"const": "lexical"}, "unit_id": source,
                            "first_line": {"type": "integer", "minimum": 1},
                            "last_line": {"type": "integer", "minimum": 1}}},
            {"type": "object", "additionalProperties": False,
             "required": ["kind", "unit_id", "statistic_index"],
             "properties": {"kind": {"const": "signed_estimate"}, "unit_id": source,
                            "statistic_index": {"type": "integer", "minimum": 0,
                                                "maximum": max(statistic_count - 1, -1)}}},
        ]
    }


def _unit_map(units: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for unit in units:
        if not isinstance(unit, Mapping) or not isinstance(unit.get("unit_id"), str) or not isinstance(unit.get("text"), str):
            raise ValueError("invalid_direction_source_unit")
        if unit["unit_id"] in result:
            raise ValueError("duplicate_direction_source_unit")
        result[unit["unit_id"]] = unit["text"]
    return result


def _line_text(text: str, first_line: Any, last_line: Any) -> str | None:
    if not isinstance(first_line, int) or not isinstance(last_line, int):
        return None
    lines = text.splitlines(keepends=True)
    if not 1 <= first_line <= last_line <= len(lines):
        return None
    return "".join(lines[first_line - 1:last_line])


def _lexically_supports(direction: str, text: str) -> bool:
    matcher = _POSITIVE if direction == "positive" else _NEGATIVE
    for match in matcher.finditer(text):
        # A local trailing negation is the only claim this gate makes about
        # negation.  Ambiguous language simply fails closed.
        prefix = text[max(0, match.start() - 40):match.start()]
        if not _NEGATION.search(prefix):
            return True
    return False


def _normalized_label(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def _numeric_direction(direction: str, statistic: Mapping[str, Any]) -> str | None:
    label = _normalized_label(statistic.get("label", ""))
    if label in _BANNED_LABELS or label not in _SIGNED_LABELS:
        return "direction_evidence_wrong_statistic_type"
    match = _NUMBER.fullmatch(str(statistic.get("value", "")))
    if not match:
        return "direction_evidence_not_signed_estimate"
    sign = match.group(1)
    magnitude = float(unicodedata.normalize("NFKC", str(statistic["value"])).replace("−", "-").replace("–", "-"))
    if magnitude == 0:
        return "direction_evidence_zero_estimate"
    if direction == "negative" and sign not in {"-", "−", "–"}:
        return "direction_evidence_sign_mismatch"
    if direction == "positive" and sign in {"-", "−", "–"}:
        return "direction_evidence_sign_mismatch"
    return None


def validate_direction_evidence(
    finding: Mapping[str, Any],
    direction_evidence: Mapping[str, Any] | None,
    units: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the optional evidence field for one finding.

    Only ``positive`` and ``negative`` assertions require evidence.  ``null``
    and ``unclear`` are preserved rather than recast by this mechanical gate.
    """
    direction = finding.get("direction")
    if direction not in {"positive", "negative"}:
        return {"valid": True, "errors": [], "copied_text": None}
    if not isinstance(direction_evidence, Mapping):
        return {"valid": False, "errors": ["direction_evidence_required"], "copied_text": None}
    sources = _unit_map(units)
    cited = set(finding.get("source_units", []))
    unit_id = direction_evidence.get("unit_id")
    if unit_id not in sources or unit_id not in cited:
        return {"valid": False, "errors": ["direction_evidence_source_not_cited"], "copied_text": None}
    kind = direction_evidence.get("kind")
    if kind == "lexical":
        copied = _line_text(sources[unit_id], direction_evidence.get("first_line"), direction_evidence.get("last_line"))
        if copied is None:
            return {"valid": False, "errors": ["direction_evidence_invalid_range"], "copied_text": None}
        if not _lexically_supports(direction, copied):
            return {"valid": False, "errors": ["direction_evidence_lexical_not_directional"], "copied_text": copied}
        return {"valid": True, "errors": [], "copied_text": copied}
    if kind == "signed_estimate":
        index = direction_evidence.get("statistic_index")
        statistics = finding.get("statistics", [])
        if not isinstance(index, int) or not isinstance(statistics, Sequence) or not 0 <= index < len(statistics):
            return {"valid": False, "errors": ["direction_evidence_invalid_statistic_index"], "copied_text": None}
        statistic = statistics[index]
        if not isinstance(statistic, Mapping) or unit_id not in set(statistic.get("source_units", [])):
            return {"valid": False, "errors": ["direction_evidence_source_not_cited"], "copied_text": None}
        error = _numeric_direction(direction, statistic)
        return {"valid": error is None, "errors": [] if error is None else [error], "copied_text": None}
    return {"valid": False, "errors": ["direction_evidence_kind_invalid"], "copied_text": None}
