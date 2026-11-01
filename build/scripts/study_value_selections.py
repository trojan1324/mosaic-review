"""Exact source choices for numeric and count facts in study packets.

The extractor receives identifiers for literal source text.  It may select a
choice, but it cannot retype a number, comparator, or count word.  Choices
are deliberately syntax-only: this module makes no claim that a printed
number is a sample size, coefficient, or item count.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


# ``ﬁ`` occurs in PDF text as a single ligature codepoint.  Keeping it in the
# selected text is necessary for a later byte-for-byte source check.
_ONES = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
         "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
         "eighteen", "nineteen")
_TENS = ("twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


def _ligature_spelling(word: str) -> str:
    return word.replace("fi", "ﬁ")


def _spelling(word: str) -> str:
    """Accept ordinary and PDF-ligature spellings, preserving what was found."""
    return "(?:" + "|".join(re.escape(value) for value in dict.fromkeys((word, _ligature_spelling(word)))) + ")"


_CARDINAL_WORDS = tuple(dict.fromkeys((*_ONES, *_TENS)))
_SIMPLE_CARDINAL = "(?:" + "|".join(_spelling(word) for word in _CARDINAL_WORDS) + ")"
_WRAP = r"(?:[-\u2010\u2011]\r?\n?|\r?\n)"
_COMPOUND_CARDINAL = "(?:" + "|".join(
    _spelling(ten) + _WRAP + _spelling(one) for ten in _TENS for one in _ONES[1:10]
) + ")"
_CARDINAL = "(?:" + _COMPOUND_CARDINAL + "|" + _SIMPLE_CARDINAL + ")"

# Do not include a leading sign in a comparator number: its sign is part of
# the printed numeric literal and is separately selectable too.
_NUMBER = r"[+\-\u2212\u2013]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][+\-]?\d+)?"
_NUMBER_RE = re.compile(r"(?<![\w.])" + _NUMBER + r"(?!\w)")
_P_EXPRESSION_RE = re.compile(r"(?<![\w])(?:[pP])\s*(?:<=|>=|<|>|=|\u2264|\u2265)\s*" + _NUMBER + r"(?!\w)")
_CARDINAL_RE = re.compile(r"(?<![\w])" + _CARDINAL + r"(?![\w])", re.IGNORECASE)

# Keep frozen requests replayable. New consumers opt into maximal, complete
# cardinals; old consumers retain their original catalogs and offset IDs.
LEGACY_VERSION = "source-values/v1"
COMPLETE_CARDINAL_VERSION = "source-values/v2"
_JOIN = r"(?:[ \t]*[-\u2010\u2011][ \t]*(?:\r?\n[ \t]*)?|\s+)"
_DIGIT_WORD = "(?:" + "|".join(_spelling(w) for w in _ONES[1:10]) + ")"
_TEN_WORD = "(?:" + "|".join(_spelling(w) for w in _TENS) + ")"
_SMALL_WORD = "(?:" + "|".join(_spelling(w) for w in _ONES[1:20]) + ")"
_UNDER_HUNDRED = "(?:" + _TEN_WORD + "(?:" + _JOIN + _DIGIT_WORD + ")?|" + _SMALL_WORD + ")"
_TAIL_JOIN = _JOIN + "(?:and" + _JOIN + ")?"
_UNDER_THOUSAND = "(?:" + _DIGIT_WORD + _JOIN + "hundred(?:" + _TAIL_JOIN + _UNDER_HUNDRED + ")?|" + _UNDER_HUNDRED + ")"


def _complete_cardinal_pattern() -> str:
    # Descending scale words prevent merging adjacent counts such as "one and
    # two" or an ascending/malformed sequence into one invented number.
    pattern = _UNDER_THOUSAND
    for scale in ("thousand", "million", "billion", "trillion"):
        pattern = "(?:" + _UNDER_THOUSAND + _JOIN + scale + "(?:" + _TAIL_JOIN + pattern + ")?|" + pattern + ")"
    return "(?:" + pattern + "|zero)"


_COMPLETE_CARDINAL_RE = re.compile(r"(?<![\w])" + _complete_cardinal_pattern() + r"(?![\w])", re.IGNORECASE)


def _choice_id(unit_id: str, start: int, end: int) -> str:
    return f"value:{unit_id}:{start}:{end}"


def _choice(unit_id: str, text: str, start: int, end: int) -> dict[str, Any]:
    return {"choice_id": _choice_id(unit_id, start, end), "unit_id": unit_id,
            "start": start, "end": end, "text": text[start:end]}


def build_value_choices(units: Sequence[Mapping[str, Any]], *, version: str = LEGACY_VERSION) -> list[dict[str, Any]]:
    """Build stable literal choices from source units.

    ``start`` and ``end`` are Python string offsets into the original supplied
    text.  ``text`` is never normalized, so ligatures and line endings survive.
    Repeated source occurrences have different offsets and therefore IDs.
    """
    if version not in (LEGACY_VERSION, COMPLETE_CARDINAL_VERSION):
        raise ValueError("unknown_value_choice_version")
    cardinal_re = _COMPLETE_CARDINAL_RE if version == COMPLETE_CARDINAL_VERSION else _CARDINAL_RE
    choices: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, unit in enumerate(units):
        if not isinstance(unit, Mapping) or not isinstance(unit.get("unit_id"), str) or not unit["unit_id"]:
            raise ValueError(f"invalid_unit:{index}")
        if not isinstance(unit.get("text"), str):
            raise ValueError(f"invalid_unit_text:{index}")
        unit_id, text = unit["unit_id"], unit["text"]
        if unit_id in seen_ids:
            raise ValueError(f"duplicate_unit_id:{unit_id}")
        seen_ids.add(unit_id)
        spans: set[tuple[int, int]] = set()
        # Comparator expressions are kept even though their numeric suffix is
        # also an offered literal.  They answer different source-selection
        # needs without collapsing punctuation or comparator direction.
        for expression in _P_EXPRESSION_RE.finditer(text):
            spans.add((expression.start(), expression.end()))
        for number in _NUMBER_RE.finditer(text):
            spans.add((number.start(), number.end()))
        for cardinal in cardinal_re.finditer(text):
            spans.add((cardinal.start(), cardinal.end()))
        choices.extend(_choice(unit_id, text, start, end) for start, end in sorted(spans))
    return choices


def resolve_value_choice(choice_id: str, choices: Sequence[Mapping[str, Any]],
                         units: Sequence[Mapping[str, Any]], *, version: str = LEGACY_VERSION) -> dict[str, Any]:
    """Reverify an offered choice and return source-selection fields only.

    A caller cannot smuggle an arbitrary selection by changing the supplied
    catalog: the catalog is compared to a fresh syntax-only rebuild of the
    current source units before resolution.
    """
    if not isinstance(choice_id, str):
        raise ValueError("unknown_choice_id")
    supplied = [choice for choice in choices if isinstance(choice, Mapping) and choice.get("choice_id") == choice_id]
    if len(supplied) != 1:
        raise ValueError("unknown_choice_id")
    canonical = {choice["choice_id"]: choice for choice in build_value_choices(units, version=version)}
    expected = canonical.get(choice_id)
    if expected is None:
        raise ValueError("unknown_choice_id")
    candidate = supplied[0]
    if dict(candidate) != expected:
        raise ValueError("value_choice_changed")
    return dict(expected)
