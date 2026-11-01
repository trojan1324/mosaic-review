"""Conservative evidence guards for a study's reuse of an earlier measure.

The helpers here identify a narrow, source-backed exception to the ordinary
rule that a patch may cite only its own study.  They make no semantic claim
beyond the explicit wording that a named measure was reused.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from scripts.study_value_selections import build_value_choices


_REUSE = re.compile(
    r"\bused\s+the\s+same\s+measure\s+of\s+"
    r"(?P<measure>[a-z0-9][a-z0-9 '&/()\-]{1,120}?)\s+"
    r"as\s+in\s+(?P<label>study\s+\d+)\b",
    re.IGNORECASE,
)
_PLURAL_REUSE = re.compile(r"\bused\s+the\s+same\s+measures\s+of\b", re.IGNORECASE)
_COUNT_SUFFIX = re.compile(r"(?:\s+|[-\u2010\u2011]\s*)items?\b", re.IGNORECASE)
_N_ITEMS_PATH = re.compile(r"^/variables/\d+/measures/\d+/n_items$")
_ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
         "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
         "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
         "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
         "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}


def _normalized(value: str) -> str:
    """Normalize only text used to decide whether two labels match."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _study_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = _normalized(value)
    match = re.fullmatch(r"(?:\d+\.\s+)?(study\s+\d+)", normalized)
    return match.group(1) if match else None


def _units(value: Any) -> list[Mapping[str, Any]] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    units = list(value)
    if not all(isinstance(unit, Mapping) and isinstance(unit.get("unit_id"), str)
               and isinstance(unit.get("text"), str) for unit in units):
        return None
    return units


def _count_choices(unit: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return only literal choices immediately followed by an item-count noun."""
    text = unit["text"]
    matches: list[dict[str, Any]] = []
    for choice in build_value_choices([unit]):
        if _COUNT_SUFFIX.match(text, choice["end"]):
            matches.append(choice)
    return matches


def _positive_count_literal(value: str) -> bool:
    normalized = _normalized(value)
    if re.fullmatch(r"[1-9]\d*", normalized):
        return True
    words = normalized.replace("-", " ").split()
    if len(words) == 1:
        return _ONES.get(words[0], _TENS.get(words[0], 0)) > 0
    if len(words) == 2 and words[0] in _TENS and words[1] in _ONES:
        return _TENS[words[0]] + _ONES[words[1]] > 0
    return False


def count_choices(units: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return only positive integer/cardinal literals used as item counts.

    The result preserves the original literal selection text and offsets.  It
    deliberately excludes headings, p expressions, decimal coefficients, N,
    and alpha values even when those values appear in the same source unit.
    """
    validated = _units(units)
    if validated is None:
        raise ValueError("invalid_count_choice_units")
    return [choice for unit in validated for choice in _count_choices(unit)
            if _positive_count_literal(choice["text"])]


def build_measure_reuse_links(
    primary_units: Sequence[Mapping[str, Any]],
    reference_studies: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build explicit, unambiguous current-study-to-prior-measure links.

    ``reference_studies`` contains only other studies and uses ``label`` and
    ``units`` keys.  A link is emitted only when exactly one prior study has
    the named label, exactly one of its units contains the named measure, and
    all eligible count occurrences spell the same literal count expression.
    """
    primary = _units(primary_units)
    if primary is None or not isinstance(reference_studies, Sequence) or isinstance(reference_studies, (str, bytes)):
        raise ValueError("invalid_reuse_inputs")

    studies: list[tuple[str, list[Mapping[str, Any]]]] = []
    for study in reference_studies:
        if not isinstance(study, Mapping):
            raise ValueError("invalid_reference_study")
        label = _study_label(study.get("label"))
        units = _units(study.get("units"))
        if units is None:
            raise ValueError("invalid_reference_study")
        if label is None:
            continue
        studies.append((label, units))

    links: list[dict[str, Any]] = []
    for primary_unit in primary:
        text = primary_unit["text"]
        normalized = _normalized(text)
        # Plural wording does not identify one measure and is deliberately
        # rejected even if a singular-looking substring happens to occur.
        if _PLURAL_REUSE.search(normalized):
            continue
        matches = list(_REUSE.finditer(normalized))
        if len(matches) != 1:
            continue
        match = matches[0]
        measure_text = match.group("measure").strip()
        if " and " in measure_text or "," in measure_text:
            continue
        prior_label = _study_label(match.group("label"))
        candidates = [units for label, units in studies if label == prior_label]
        if len(candidates) != 1:
            continue
        measure_key = _normalized(measure_text)
        prior_units = [unit for unit in candidates[0] if measure_key in _normalized(unit["text"])]
        if len(prior_units) != 1:
            continue
        count_choices = _count_choices(prior_units[0])
        # A repeated literal such as "five-item ... five items" is safe; two
        # different literal expressions require a human/source interpretation.
        if not count_choices or len({_normalized(choice["text"]) for choice in count_choices}) != 1:
            continue
        links.append({
            "primary_unit_id": primary_unit["unit_id"],
            "reference_unit_id": prior_units[0]["unit_id"],
            "allowed_choice_ids": [choice["choice_id"] for choice in count_choices],
            "measure_text": measure_text,
            "prior_study_label": prior_label,
            "evidence": {
                "primary_reuse_text": match.group(0),
                "reference_measure_text": prior_units[0]["text"],
            },
        })
    return links


def reuse_allowed(
    path: str,
    choice_id: str,
    link_source_units: Sequence[str],
    primary_ids: Sequence[str],
    reuse_links: Sequence[Mapping[str, Any]],
) -> bool:
    """Return whether an external literal is allowed for this exact n-items field."""
    if not _N_ITEMS_PATH.fullmatch(path):
        return False
    if not isinstance(link_source_units, Sequence) or isinstance(link_source_units, (str, bytes)):
        return False
    primary = set(primary_ids)
    links = set(link_source_units)
    if not links or not links <= primary:
        return False
    for reuse in reuse_links:
        if not isinstance(reuse, Mapping):
            continue
        if reuse.get("primary_unit_id") in links and choice_id in reuse.get("allowed_choice_ids", []):
            return True
    return False
