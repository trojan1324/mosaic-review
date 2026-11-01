"""Approved CORE fields for model generation; not a source-eligibility or quality gate.

Study relations remain a separately pending owner decision. Legacy response branches
are preserved for replay; these fields close gaps in the new live response format.
"""
from __future__ import annotations

import json
import re

from mosaic import REPO_ROOT

POLICY_PATH = REPO_ROOT / "config" / "protocols" / "core_scope_v1.json"
SCOPE_LABELS = ("sample", "analysis", "condition", "instrument_version", "administration", "hypothesis")
ATTRIBUTIONS = ("author_result", "author_interpretation", "secondary_report")
NUMBER = r"^[+−–-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?$|^[+−–-]?\.\d+(?:[eE][+-]?\d+)?$"
_CARDINAL_SMALL = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                   "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
                   "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
                   "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20}
_CARDINAL_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
                  "seventy": 70, "eighty": 80, "ninety": 90}
CARDINAL_INTEGER_VALUES = {**_CARDINAL_SMALL, **_CARDINAL_TENS,
                           **{f"{ten}-{word}": base + value for ten, base in _CARDINAL_TENS.items()
                              for word, value in _CARDINAL_SMALL.items() if 0 < value < 10}}
_CARDINAL_ALTERNATION = "|".join(sorted(CARDINAL_INTEGER_VALUES, key=lambda item: (-len(item), item)))


def _case_pattern(word: str) -> str:
    """ASCII case-insensitive spelling without unsupported regex flags."""
    return "".join(f"[{char.lower()}{char.upper()}]" if char.isalpha() else re.escape(char) for char in word)


_CARDINAL_SCHEMA_ALTERNATION = "|".join(_case_pattern(word) for word in sorted(CARDINAL_INTEGER_VALUES, key=lambda item: (-len(item), item)))
INTEGER_PATTERN = r"^(?:\d{1,3}(?:,\d{3})+|\d+)$"
N_ITEMS_LEXICAL_PATTERN = rf"^(?:\d{{1,3}}(?:,\d{{3}})+|\d+|(?:{_CARDINAL_SCHEMA_ALTERNATION}))$"
_CARDINAL_APPROX_PREFIX = re.compile(
    r"\b(?:about|approximately|approx\.?|around|roughly|nearly|at\s+least|at\s+most|more\s+than|fewer\s+than|less\s+than|over|under)\s*$",
    re.I,
)
_CARDINAL_RANGE_PREFIX = re.compile(rf"\b(?:{_CARDINAL_ALTERNATION})\s*(?:and|or|to|through|[-–])\s*$", re.I)
_CARDINAL_RANGE_SUFFIX = re.compile(rf"^\s*(?:and|or|to|through)\s+(?:{_CARDINAL_ALTERNATION})\b", re.I)


def core_policy() -> dict:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def field_definitions() -> dict:
    return {row["field"]: row for row in core_policy()["fields"]}


def cardinal_integer_value(lexical: object) -> str | None:
    """Normalized value for one closed-vocabulary English cardinal, else None.

    The lexical spelling remains source text; this only supplies its typed
    integer form for the narrow measure-item-count exception.
    """
    if not isinstance(lexical, str) or lexical != lexical.strip():
        return None
    value = CARDINAL_INTEGER_VALUES.get(lexical.lower())
    return str(value) if value is not None else None


def cardinal_integer_at(text: str, lexical: object, start: int) -> bool:
    """Whether a cardinal at one exact offset is clean source wording.

    A cardinal cannot be a substring of another word or a hyphenated compound
    tail. ``seven-item`` is the one accepted hyphen suffix because it can state
    a measure's item count. Ranges and approximate language do not support an
    exact count.
    """
    if cardinal_integer_value(lexical) is None or not isinstance(lexical, str):
        return False
    end = start + len(lexical)
    if start < 0 or text[start:end] != lexical:
        return False
    before, after = text[max(0, start - 48):start], text[end:end + 48]
    if start and (text[start - 1].isalnum() or text[start - 1] == "_" or text[start - 1] in "-–"):
        return False
    if after.startswith("-") or after.startswith("–"):
        if not re.match(r"[-–]items?(?![A-Za-z-–])", after, re.I):
            return False
    elif after[:1].isalnum() or after.startswith("_"):
        return False
    return not (_CARDINAL_APPROX_PREFIX.search(before) or _CARDINAL_RANGE_PREFIX.search(before)
                or _CARDINAL_RANGE_SUFFIX.search(after))


def cardinal_integer_in_text(text: str, lexical: object) -> bool:
    """Whether an exact cardinal lexical occurs cleanly in source text."""
    if not isinstance(lexical, str):
        return False
    for match in re.finditer(re.escape(lexical), text):
        if cardinal_integer_at(text, lexical, match.start()):
            return True
    return False


def fact_schema(unit_enum: dict, span_enum: dict, *, interval_lexicals: list[str] | None = None,
                span_enums: dict[str, dict] | None = None) -> dict:
    label = {"type": "string", "minLength": 1, "pattern": r"\S"}
    nullable = {**label, "type": ["string", "null"]}
    scope = {"type": "object", "required": list(SCOPE_LABELS),
             "properties": {key: nullable for key in SCOPE_LABELS}, "additionalProperties": False}
    branches = []
    for name, definition in field_definitions().items():
        if definition["representation"] == "interval" and interval_lexicals == []:
            # A live packet with no exact source choices must not invite the
            # model to manufacture an interval.  ``None`` retains the generic
            # contract used by historical replay and schema helpers.
            continue
        props = {"candidate_id": label, "field": {"const": name, "type": "string"},
                 "subject": label, "scope": scope,
                 "attribution": {"type": "string", "enum": list(ATTRIBUTIONS)},
                 "evidence_unit_ids": {"type": "array", "minItems": 1, "items": unit_enum}}
        representation = definition["representation"]
        if representation == "span":
            props["span_id"] = (span_enums or {}).get(name, span_enum)
        elif representation == "paraphrase":
            props["lexical"] = label
        elif representation == "interval":
            # The lexical interval is preserved exactly; its endpoints are
            # normalized independently.  A level may be absent in source text,
            # but is never silently assumed to be 95%.
            props.update(
                lexical=({"type": "string", "enum": interval_lexicals}
                         if interval_lexicals is not None
                         else {"type": "string", "minLength": 1, "maxLength": 128}),
                lower={"type": "string", "maxLength": 128, "pattern": r"^-?\d+(?:\.\d+)?$"},
                upper={"type": "string", "maxLength": 128, "pattern": r"^-?\d+(?:\.\d+)?$"},
                level={"type": ["string", "null"], "maxLength": 128,
                       "pattern": r"^-?\d+(?:\.\d+)?$"},
                statistic={"const": "CI", "type": "string"},
            )
        else:
            props.update(lexical={"type": "string", "maxLength": 128,
                                  "pattern": (N_ITEMS_LEXICAL_PATTERN if name == "n_items" else INTEGER_PATTERN)
                                  if representation == "integer" else NUMBER},
                         value={"type": "string", "maxLength": 128,
                                "pattern": r"^\d+$" if representation == "integer" else r"^-?\d+(?:\.\d+)?$"},
                         statistic=label,
                         comparator={"type": "string", "enum": ["eq", "lt", "le", "gt", "ge", "approximate"]})
        branches.append({"type": "object", "required": list(props), "properties": props,
                         "additionalProperties": False})
    return {"oneOf": branches}
