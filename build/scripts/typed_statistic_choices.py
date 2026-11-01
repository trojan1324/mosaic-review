"""Syntax-only catalogues of complete, typed statistic expressions.

The catalog deliberately recognizes only expressions whose printed structure
names their role.  It does not infer which variable, hypothesis, or analysis
an expression belongs to, and it leaves bare numbers unmatched.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


_NUMBER = r"[+\-\u2212\u2013]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][+\-]?\d+)?"
_SPACE = r"[ \t\r\n]*"
_CI_RE = re.compile(
    r"(?P<level>\d{1,3}%)[ \t\r\n]*(?P<label>CI|C\.I\.|confidence[ \t\r\n]+interval)[ \t\r\n]*(?:=|:)?[ \t\r\n]*[\[(]"
    + _SPACE + r"(?P<lower>" + _NUMBER + r")" + _SPACE + r"[,;]" + _SPACE
    + r"(?P<upper>" + _NUMBER + r")" + _SPACE + r"[\])]",
    re.IGNORECASE,
)
_LABEL_RE = re.compile(
    r"(?<![\w])(?P<label>B|b|\u03b2|beta|SE|p|\u03b1|alpha|\u03c1|rho|\u03c9|omega|t|F|r|R2|R\u00b2)"
    + _SPACE + r"(?:\((?P<df>" + _NUMBER + r"(?:" + _SPACE + r"," + _SPACE + _NUMBER + r")?)\))?"
    + _SPACE + r"(?P<comparator><=|>=|<|>|=|\u2264|\u2265)" + _SPACE + r"(?P<value>" + _NUMBER + r")(?!\w)",
    re.IGNORECASE,
)

_TYPE_BY_LABEL = {
    "b": "coefficient", "\u03b2": "coefficient", "beta": "coefficient",
    "se": "standard_error", "p": "p_value", "\u03b1": "reliability", "alpha": "reliability",
    "\u03c1": "reliability", "rho": "reliability", "\u03c9": "reliability", "omega": "reliability",
    "t": "t_statistic", "f": "f_statistic", "r": "correlation", "r2": "r_squared", "r\u00b2": "r_squared",
}


def _component(name: str, match: re.Match[str], group: str) -> dict[str, Any]:
    start, end = match.span(group)
    return {"name": name, "start": start, "end": end, "text": match.group(group)}


def _choice(unit_id: str, start: int, end: int, text: str, statistic_type: str,
            printed_label: str, components: list[dict[str, Any]]) -> dict[str, Any]:
    return {"unit_id": unit_id, "start": start, "end": end, "text": text[start:end],
            "statistic_type": statistic_type, "printed_label": printed_label,
            "components": components}


def _label_choice(unit_id: str, text: str, match: re.Match[str]) -> dict[str, Any]:
    label = match.group("label")
    components = [_component("value", match, "value"), _component("comparator", match, "comparator")]
    if match.group("df") is not None:
        df_start = match.start("df")
        for index, number in enumerate(re.finditer(_NUMBER, match.group("df")), start=1):
            start, end = df_start + number.start(), df_start + number.end()
            components.append({"name": f"df_{index}", "start": start, "end": end, "text": text[start:end]})
    return _choice(unit_id, match.start(), match.end(), text, _TYPE_BY_LABEL[label.casefold()], label, components)


def _ci_choice(unit_id: str, text: str, match: re.Match[str]) -> dict[str, Any]:
    return _choice(unit_id, match.start(), match.end(), text, "confidence_interval", match.group("label"),
                   [_component("confidence_level", match, "level"), _component("lower", match, "lower"),
                    _component("upper", match, "upper")])


def build_statistic_choices(units: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compile source-ordered choices for complete printed statistic expressions."""
    choices: list[dict[str, Any]] = []
    unit_ids: set[str] = set()
    for unit_index, unit in enumerate(units):
        if not isinstance(unit, Mapping) or not isinstance(unit.get("unit_id"), str) or not unit["unit_id"]:
            raise ValueError(f"invalid_unit:{unit_index}")
        if not isinstance(unit.get("text"), str):
            raise ValueError(f"invalid_unit_text:{unit_index}")
        unit_id, text = unit["unit_id"], unit["text"]
        if unit_id in unit_ids:
            raise ValueError(f"duplicate_unit_id:{unit_id}")
        unit_ids.add(unit_id)
        candidates = [_ci_choice(unit_id, text, match) for match in _CI_RE.finditer(text)]
        ci_spans = {(candidate["start"], candidate["end"]) for candidate in candidates}
        for match in _LABEL_RE.finditer(text):
            # A named statistic is permitted inside surrounding prose, but no
            # separate expression is manufactured from CI endpoints.
            if not any(start <= match.start() and match.end() <= end for start, end in ci_spans):
                candidates.append(_label_choice(unit_id, text, match))
        choices.extend(sorted(candidates, key=lambda item: (item["start"], item["end"])))
    for index, choice in enumerate(choices, start=1):
        choice["choice_id"] = f"stat{index:03d}"
    return choices


def resolve_statistic_choice(choice_id: str, choices: Sequence[Mapping[str, Any]],
                             units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return a reverified typed expression, rejecting unknown or altered input."""
    if not isinstance(choice_id, str):
        raise ValueError("unknown_statistic_choice")
    supplied = [choice for choice in choices if isinstance(choice, Mapping) and choice.get("choice_id") == choice_id]
    if len(supplied) != 1:
        raise ValueError("unknown_statistic_choice")
    canonical = {choice["choice_id"]: choice for choice in build_statistic_choices(units)}
    expected = canonical.get(choice_id)
    if expected is None:
        raise ValueError("unknown_statistic_choice")
    if dict(supplied[0]) != expected:
        raise ValueError("statistic_choice_changed")
    return dict(expected)
