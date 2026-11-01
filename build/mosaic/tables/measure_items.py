"""Conservative selection of CORE wording from a resolved measure-item table.

DE-22 keeps a measure's printed items and response anchors in scope, but not
the statistical columns that often appear beside them. This module only
selects cells when the table itself gives an unambiguous, explicit heading.
It never infers an item column from a caption or from prose-like cell text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mosaic.tables.extract import NUMERIC


_ITEM_HEADERS = frozenset({
    "item", "items", "item wording", "item wordings", "statement", "statements",
    "question", "questions", "question wording", "question wordings",
})
_ANCHOR_HEADERS = frozenset({
    "anchor", "anchors", "response anchor", "response anchors", "response option",
    "response options", "response label", "response labels",
})


def wording_field(header_path: list[str]) -> str | None:
    """The CORE field explicitly named by a retained wording column."""
    heading = " ".join(header_path[-1].split()).casefold() if header_path else ""
    if heading in _ITEM_HEADERS:
        return "item_wording"
    if heading in _ANCHOR_HEADERS:
        return "anchor"
    return None


@dataclass(frozen=True)
class MeasureItemSelection:
    """The columns and exact cells safe to retain from one table grid."""

    column_ids: tuple[str, ...]
    cells: tuple[dict[str, Any], ...]


def select_measure_item_cells(grid: Any) -> MeasureItemSelection | None:
    """Return explicitly-labelled item/anchor cells, or refuse the whole table.

    Exact spans alone do not establish whole items: wrapped lines can each have
    an exact span. Require complete rows and either printed row separators or
    an adjacent numeric entry on every row to distinguish item boundaries.
    Two wording columns or an unspanned cell make the table ineligible.
    """
    if getattr(grid, "status", None) != "single_page_resolved":
        return None
    columns = list(getattr(grid, "columns", ()))
    def heading(column: dict[str, Any]) -> str:
        path = column.get("header_path", ()) or ()
        return " ".join(str(path[-1]).split()).casefold() if path else ""

    item = [c["column_id"] for c in columns if heading(c) in _ITEM_HEADERS]
    anchors = [c["column_id"] for c in columns if heading(c) in _ANCHOR_HEADERS]
    if len(item) > 1 or not (item or anchors):
        return None
    chosen = tuple(item + anchors)
    cells = [c for c in getattr(grid, "cells", ()) if c["position"]["column_id"] in chosen]
    if not cells or any("span" not in c or not c.get("text", "").strip() for c in cells):
        return None
    # A column headed "Items" sometimes contains only an item number. Do not let an adjacent response-anchor
    # column make that index look like wording: every retained cell must itself contain printed wording.
    if any(not any(ch.isalpha() for ch in c["text"]) for c in cells):
        return None
    rows: dict[tuple[str, ...], dict[str, dict]] = {}
    for cell in grid.cells:
        position = cell["position"]
        row = rows.setdefault(tuple(position["row_index_or_path"]), {})
        if position["column_id"] in row:
            return None
        row[position["column_id"]] = cell
    all_columns = {c["column_id"] for c in columns}
    if any(set(row) != all_columns for row in rows.values()):
        return None  # a hanging line may continue the item above; never emit it as another item
    if len(rows) > 1 and not getattr(grid, "rows_separated", False):
        boundary_columns = all_columns - set(chosen)
        if not any(all(NUMERIC.fullmatch(row[col]["text"]) for row in rows.values()) for col in boundary_columns):
            return None  # wording alone cannot distinguish separate items from wrapped lines
    return MeasureItemSelection(chosen, tuple(cells))
