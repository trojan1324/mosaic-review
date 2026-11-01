"""Exact source-unit line-range schemas for constrained selection decoding."""
from __future__ import annotations

from typing import Mapping, Sequence


def bounded_selection_schema(units: Sequence[Mapping], allowed_ids=None):
    """Build a strict per-unit selection schema with exact line maxima.

    Range ordering remains a materialization check: JSON Schema cannot express
    the relation ``first_line <= last_line`` without complicating the bounded
    native grammar.
    """
    seen, groups = set(), {}
    permitted = None if allowed_ids is None else set(allowed_ids)
    if permitted is not None and (not permitted or not all(isinstance(value, str) and value for value in permitted)):
        raise ValueError("allowed_ids_must_be_nonempty_strings")
    for index, unit in enumerate(units):
        if not isinstance(unit, Mapping) or not isinstance(unit.get("unit_id"), str) or not unit["unit_id"]:
            raise ValueError(f"invalid_unit:{index}")
        unit_id = unit["unit_id"]
        if unit_id in seen:
            raise ValueError(f"duplicate_unit_id:{unit_id}")
        seen.add(unit_id)
        if not isinstance(unit.get("text"), str):
            raise ValueError(f"invalid_unit_text:{index}")
        line_count = len(unit["text"].splitlines(keepends=True))
        if line_count < 1:
            raise ValueError(f"empty_unit_text:{unit_id}")
        if permitted is None or unit_id in permitted:
            groups.setdefault(line_count, []).append(unit_id)
    # Units with the same line count have identical bounds. Group their IDs
    # rather than repeating the object grammar once per paragraph; the ID
    # sets are disjoint, so oneOf preserves the exact accepted selections.
    branches = []
    for line_count, unit_ids in groups.items():
        branches.append({"type": "object", "properties": {
            "unit_id": ({"const": unit_ids[0]} if len(unit_ids) == 1 else {"enum": unit_ids}),
            "first_line": {"type": "integer", "minimum": 1, "maximum": line_count},
            "last_line": {"type": "integer", "minimum": 1, "maximum": line_count}},
            "required": ["unit_id", "first_line", "last_line"], "additionalProperties": False})
    if not seen:
        raise ValueError("units_must_be_nonempty")
    if not branches:
        raise ValueError("allowed_ids_not_present_in_units")
    return {"oneOf": branches}
