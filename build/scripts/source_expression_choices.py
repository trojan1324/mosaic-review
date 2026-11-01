"""Source-expression retention for typed statistics and explicit unknown labels.

The fallback exists to retain a reported expression when its label is not in
the typed catalog.  It preserves the complete printed expression only; it
does not parse values into conditions or claim a scientific interpretation.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from scripts.reported_statistic_choices import build_statistic_choices as _reported_choices


_LABEL = r"[A-Za-z\u03b1-\u03c9\u0391-\u03a9][A-Za-z0-9\u00b2'\u2019._-]*"
_NUMBER = r"[+\-\u2212\u2013]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
_VALUE = _NUMBER + r"(?:\s*\(\s*SD\s*=\s*" + _NUMBER + r"\s*\))?"
_FALLBACK_RE = re.compile(
    r"(?<![\w])(?P<label>" + _LABEL + r")[ \t\r\n]*(?:\([^)]{0,80}\)[ \t\r\n]*)?(?P<comparator><=|>=|<|>|=|\u2264|\u2265)[ \t\r\n]*"
    + r"(?P<body>" + _VALUE + r"(?:\s*(?:,|and|vs\.?)\s*" + _VALUE + r")*)", re.IGNORECASE)


def _fallback_choice(unit_id: str, text: str, match: re.Match[str]) -> dict[str, Any]:
    end = match.end()
    # A terminal period belongs to prose punctuation, whereas decimal points
    # inside the body remain exactly as printed.
    if end > match.start() and text[end - 1:end] == ".":
        end -= 1
    return {"unit_id": unit_id, "start": match.start(), "end": end, "text": text[match.start():end],
            "statistic_type": "unclassified_labeled_statistic", "printed_label": match.group("label"),
            "components": [], "fallback_reason": "explicit_label_numeric_expression_unclassified"}


def build_statistic_choices(units: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return source-ordered typed choices plus non-overlapping source fallbacks."""
    typed = _reported_choices(units)
    extras: list[dict[str, Any]] = []
    for unit in units:
        unit_id, text = unit["unit_id"], unit["text"]
        existing = [choice for choice in typed if choice["unit_id"] == unit_id]
        for match in _FALLBACK_RE.finditer(text):
            candidate = _fallback_choice(unit_id, text, match)
            # A typed expression beginning at the same source location already
            # has a stricter syntax contract.  Do not duplicate it with a
            # generic fallback.
            if any(choice["start"] == candidate["start"] for choice in existing):
                continue
            nested = [choice for choice in existing if candidate["start"] <= choice["start"]
                      and choice["end"] <= candidate["end"]]
            # The fallback is an explicit whole expression (for example a
            # plural-mean comparison with parenthetical SDs).  Remove nested
            # typed fragments to keep catalog choices non-overlapping.
            if nested:
                typed = [choice for choice in typed if choice not in nested]
                existing = [choice for choice in existing if choice not in nested]
                candidate["fallback_reason"] = "explicit_group_excludes_nested_typed_fragments"
            if any(not (candidate["end"] <= choice["start"] or choice["end"] <= candidate["start"])
                   for choice in existing):
                continue
            extras.append(candidate)
            existing.append(candidate)
    choices = [*typed, *extras]
    order = {unit["unit_id"]: index for index, unit in enumerate(units)}
    choices.sort(key=lambda choice: (order[choice["unit_id"]], choice["start"], choice["end"], choice["statistic_type"]))
    for index, choice in enumerate(choices, start=1):
        choice["choice_id"] = f"stat{index:03d}"
    return choices


def resolve_statistic_choice(choice_id: str, choices: Sequence[Mapping[str, Any]],
                             units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return a full recompiled choice; source or catalog edits are rejected."""
    if not isinstance(choice_id, str):
        raise ValueError("unknown_statistic_choice")
    supplied = [choice for choice in choices if isinstance(choice, Mapping) and choice.get("choice_id") == choice_id]
    if len(supplied) != 1:
        raise ValueError("unknown_statistic_choice")
    expected = {choice["choice_id"]: choice for choice in build_statistic_choices(units)}.get(choice_id)
    if expected is None:
        raise ValueError("unknown_statistic_choice")
    if dict(supplied[0]) != expected:
        raise ValueError("statistic_choice_changed")
    return dict(expected)


def fallback_diagnostics(choices: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Expose retained unclassified expressions and overlap handling to callers."""
    return [{"choice_id": choice["choice_id"], "unit_id": choice["unit_id"], "start": choice["start"],
             "end": choice["end"], "reason": choice["fallback_reason"]}
            for choice in choices if choice.get("statistic_type") == "unclassified_labeled_statistic"]
