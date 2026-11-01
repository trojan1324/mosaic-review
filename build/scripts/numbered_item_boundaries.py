"""Conservative expansion of already-selected numbered measure-item blocks."""
from __future__ import annotations

from copy import deepcopy
import re
from typing import Mapping, Sequence


_MARKER = re.compile(r"^\s*\d+[.)](?:\s|$)")
_LETTER = re.compile(r"[A-Za-z]")


def _blocks(lines: list[str]):
    """Return inclusive line bounds for line-start numbered blocks."""
    starts = [index for index, line in enumerate(lines) if _MARKER.match(line)]
    return [(start, (starts[index + 1] - 1) if index + 1 < len(starts) else len(lines) - 1)
            for index, start in enumerate(starts)]


def normalize_numbered_items(record: Mapping, units: Sequence[Mapping]):
    """Copy a record, widening existing item selections only when unambiguous.

    The result does not create measure items or inspect unselected prose.  Its
    line ranges are one-based and copied text retains original line endings.
    """
    result = deepcopy(record)
    source = {unit.get("unit_id"): unit.get("text") for unit in units
              if isinstance(unit, Mapping) and isinstance(unit.get("unit_id"), str)
              and isinstance(unit.get("text"), str)}
    changes, errors = [], []
    for variable_index, variable in enumerate(result.get("variables", [])):
        if not isinstance(variable, Mapping):
            errors.append({"code": "invalid_variable", "path": f"/variables/{variable_index}"})
            continue
        for measure_index, measure in enumerate(variable.get("measures", [])):
            if not isinstance(measure, Mapping):
                errors.append({"code": "invalid_measure", "path": f"/variables/{variable_index}/measures/{measure_index}"})
                continue
            for item_index, selection in enumerate(measure.get("items", [])):
                path = f"/variables/{variable_index}/measures/{measure_index}/items/{item_index}"
                if not isinstance(selection, Mapping):
                    errors.append({"code": "invalid_item_selection", "path": path})
                    continue
                unit_id = selection.get("unit_id")
                text = source.get(unit_id)
                if text is None:
                    errors.append({"code": "unknown_source_unit", "path": path})
                    continue
                first, last = selection.get("first_line"), selection.get("last_line")
                lines = text.splitlines(keepends=True)
                if not isinstance(first, int) or not isinstance(last, int) or not 1 <= first <= last <= len(lines):
                    errors.append({"code": "invalid_line_range", "path": path})
                    continue
                # Existing item scope prevents this utility from turning
                # unrelated numbered prose into a newly inferred measure item.
                contained = [(start, end) for start, end in _blocks(lines)
                             if start <= first - 1 and last - 1 <= end]
                if not contained:
                    # An unnumbered selection or a selection crossing two
                    # markers is intentionally retained without alteration.
                    if any(start <= last - 1 and end >= first - 1 for start, end in _blocks(lines)):
                        errors.append({"code": "crosses_numbered_item_boundary", "path": path})
                    continue
                start, end = contained[0]
                copied = "".join(lines[start:end + 1])
                if not _LETTER.search(copied):
                    errors.append({"code": "numbered_marker_has_no_wording", "path": path})
                    continue
                expanded = {"unit_id": unit_id, "first_line": start + 1, "last_line": end + 1}
                if dict(selection) != expanded:
                    measure["items"][item_index] = expanded
                    changes.append({"path": path, "before": dict(selection), "after": expanded,
                                    "rationale": "expanded_to_clear_numbered_item_block", "copied_text": copied})
    return {"record": result, "changes": changes, "errors": errors,
            "mechanical_valid": not errors, "semantic_accuracy_unassessed": True,
            "production_accepted": False}
