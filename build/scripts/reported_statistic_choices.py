"""Additional syntax-only choices for commonly reported statistics.

This extends :mod:`typed_statistic_choices` without assigning an expression
to a study condition, variable, or substantive result.  A comma- or
``versus``-separated list is retained as one expression so its values cannot
be silently redistributed by the model.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from scripts.typed_statistic_choices import build_statistic_choices as _base_choices


_NUMBER = r"[+\-\u2212\u2013]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][+\-]?\d+)?"
_SPACE = r"[ \t\r\n]*"
_LIST_SEPARATOR = r"(?:" + _SPACE + r",(?:" + _SPACE + r"and)?" + _SPACE + r"|" + _SPACE + r"and" + _SPACE + r"|" + _SPACE + r"(?:vs\.?|versus)" + _SPACE + r")"
_NEW_LABEL = (r"M|SD|n|OR|odds[ \t\r\n]+ratio|Cohen['\u2019]s[ \t\r\n]+d|D|p|"
              r"Fisher['\u2019]s[ \t\r\n]+exact[ \t\r\n]+test|chi[\- ]?square|chi2|\u03c7(?:2|\u00b2)|z|U|"
              r"Mean[ \t\r\n]+Difference|partial[ \t\r\n]+\u03b7(?:2|\u00b2)|\u03b7(?:2|\u00b2)")
_NEW_SIMPLE_RE = re.compile(
    r"(?<![\w])(?P<label>" + _NEW_LABEL + r")" + _SPACE
    + r"(?P<comparator><=|>=|<|>|=|\u2264|\u2265)" + _SPACE
    + r"(?P<values>" + _NUMBER + r"(?:" + _LIST_SEPARATOR + _NUMBER + r"(?![\w%]))*)(?!\w)", re.IGNORECASE)
_CHI_RE = re.compile(
    r"(?<![\w])(?P<label>(?:McNemar['\u2019]s" + _SPACE + r")?(?:chi[\- ]?square|chi2|\u03c7(?:2|\u00b2)))" + _SPACE
    + r"\((?P<inside>[^)]{1,80})\)" + _SPACE + r"(?P<comparator><=|>=|<|>|=|\u2264|\u2265)" + _SPACE
    + r"(?P<value>" + _NUMBER + r")(?!\w)", re.IGNORECASE)
_DIAGNOSTIC_LABEL_RE = re.compile(r"(?<![\w])(?:Ms|" + _NEW_LABEL + r"|B|b|\u03b2|beta|SE|p|\u03b1|alpha|\u03c1|rho|\u03c9|omega|t|F|r|R2|R\u00b2)(?![\w])", re.IGNORECASE)

_TYPE = {
    "m": "mean", "sd": "standard_deviation", "n": "sample_size", "or": "odds_ratio", "odds ratio": "odds_ratio",
    "cohen's d": "cohens_d", "cohen\u2019s d": "cohens_d", "d": "cohens_d",
    "chi-square": "chi_square", "chi square": "chi_square", "chi2": "chi_square", "\u03c72": "chi_square", "\u03c7\u00b2": "chi_square",
    "p": "p_value", "fisher's exact test": "fisher_exact_test", "fisher\u2019s exact test": "fisher_exact_test",
    "z": "z_statistic", "u": "u_statistic", "mean difference": "mean_difference", "\u03b72": "eta_squared", "\u03b7\u00b2": "eta_squared",
    "partial \u03b72": "partial_eta_squared", "partial \u03b7\u00b2": "partial_eta_squared",
}


def _part(name: str, text: str, start: int, end: int) -> dict[str, Any]:
    return {"name": name, "start": start, "end": end, "text": text[start:end]}


def _values(text: str, start: int, raw: str) -> list[dict[str, Any]]:
    return [_part(f"value_{index}", text, start + match.start(), start + match.end())
            for index, match in enumerate(re.finditer(_NUMBER, raw), start=1)]


def _new_simple(unit_id: str, text: str, match: re.Match[str]) -> dict[str, Any]:
    label = match.group("label")
    components = [_part("comparator", text, *match.span("comparator"))]
    components.extend(_values(text, match.start("values"), match.group("values")))
    return {"unit_id": unit_id, "start": match.start(), "end": match.end(), "text": text[match.start():match.end()],
            "statistic_type": _TYPE[' '.join(label.casefold().split())], "printed_label": label, "components": components}


def _chi_with_details(unit_id: str, text: str, match: re.Match[str]) -> dict[str, Any]:
    components = [_part("comparator", text, *match.span("comparator")), _part("value_1", text, *match.span("value"))]
    inside_start = match.start("inside")
    for index, number in enumerate(re.finditer(_NUMBER, match.group("inside")), start=1):
        start, end = inside_start + number.start(), inside_start + number.end()
        prefix = match.group("inside")[:number.start()]
        name = "sample_n" if re.search(r"N\s*=\s*$", prefix, re.IGNORECASE) else f"df_{index}"
        components.append(_part(name, text, start, end))
    return {"unit_id": unit_id, "start": match.start(), "end": match.end(), "text": text[match.start():match.end()],
            "statistic_type": "chi_square", "printed_label": match.group("label"), "components": components}


def _validate_units(units: Sequence[Mapping[str, Any]]) -> None:
    ids: set[str] = set()
    for index, unit in enumerate(units):
        if not isinstance(unit, Mapping) or not isinstance(unit.get("unit_id"), str) or not unit["unit_id"]:
            raise ValueError(f"invalid_unit:{index}")
        if not isinstance(unit.get("text"), str):
            raise ValueError(f"invalid_unit_text:{index}")
        if unit["unit_id"] in ids:
            raise ValueError(f"duplicate_unit_id:{unit['unit_id']}")
        ids.add(unit["unit_id"])


def build_statistic_choices(units: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return source-ordered, short-ID choices for old and added typed forms."""
    _validate_units(units)
    choices = _base_choices(units)
    for unit in units:
        unit_id, text = unit["unit_id"], unit["text"]
        additions = [_chi_with_details(unit_id, text, match) for match in _CHI_RE.finditer(text)]
        chi_spans = {(choice["start"], choice["end"]) for choice in additions}
        for match in _NEW_SIMPLE_RE.finditer(text):
            if not any(start <= match.start() and match.end() <= end for start, end in chi_spans):
                additions.append(_new_simple(unit_id, text, match))
        # A grouped list supersedes the base module's single first item.  The
        # base choices remain untouched for non-overlapping legacy forms.
        for added in additions:
            choices = [choice for choice in choices if not (choice["unit_id"] == unit_id
                       and added["start"] <= choice["start"] and choice["end"] <= added["end"])]
        choices.extend(additions)
    order = {unit["unit_id"]: index for index, unit in enumerate(units)}
    choices.sort(key=lambda choice: (order[choice["unit_id"]], choice["start"], choice["end"], choice["statistic_type"]))
    for index, choice in enumerate(choices, start=1):
        choice["choice_id"] = f"stat{index:03d}"
    return choices


def resolve_statistic_choice(choice_id: str, choices: Sequence[Mapping[str, Any]],
                             units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recompile and compare every supplied field before returning a choice."""
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


def diagnose_named_statistic_coverage(units: Sequence[Mapping[str, Any]],
                                      choices: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """List named numeric-looking expressions not covered by this syntax catalog."""
    catalog = list(choices) if choices is not None else build_statistic_choices(units)
    covered = {(choice["unit_id"], choice["start"], choice["end"]) for choice in catalog}
    unmatched = []
    for unit in units:
        text = unit["text"]
        for label in _DIAGNOSTIC_LABEL_RE.finditer(text):
            window_end = min(len(text), label.end() + 100)
            # A nearby number does not make a normal word such as "or" or
            # a brand name containing "M" a statistic. Require its reporting
            # syntax immediately after the prospective label.
            tail = text[label.end():window_end]
            if not re.match(r'\s*(?:[<>=≤≥]|\.?\s*\(|of\s+[+\-−–]?\d)', tail, re.I):
                continue
            if not re.search(_NUMBER, tail):
                continue
            if not any(unit["unit_id"] == uid and start <= label.start() < end for uid, start, end in covered):
                unmatched.append({"unit_id": unit["unit_id"], "start": label.start(), "end": window_end,
                                  "text": text[label.start():window_end]})
    return {"covered_count": len(catalog), "unmatched_named_expression_count": len(unmatched),
            "unmatched_named_expressions": unmatched}
