"""Geometry-aware reading order (adapted from Q/source_ingestion.py order_blocks).

Kept: margin peel-off, two-column detection with width tolerance, full-width
blocks delimiting vertical bands, and the confidence labels. Changed: operates
on block dicts carrying `block_index` (and their text lines' bboxes) instead of
Q unit IDs and returns a permutation of block indices; the recursive margin
call no longer rebuilds IDs. This is the documented fix for the Colby
cross-column negation split (FINAL v2 §8.3). The gutter is not assumed to sit
at half the page width: it is found from how the horizontal text lines cover
the page, and three lines wholly on each side make two columns, as three
blocks on each side of half the width still do (a column set as one long
block is still a column; D-164). Blocks side by side on one side
of that gutter (a third column, table cells) are ordered by top y, as before,
but the page is then flagged for visual review, and so is a page where, in a
band a full-width block closes, a left block lies wholly below every right
block (a local band of columns, such as a table's label and value columns, on
a one-column page), or where a raster or drawing crosses the gutter with
column text above and below it in one band (D-191), and so is a page where a
left block lies in a vertical gap of the right side, right blocks above and
below it in its band and none beside it (a heading between two such bands;
D-202).
"""

from __future__ import annotations

from typing import Any


def _sides(lines: list[list[float]], x: float) -> tuple[list[list[float]], list[list[float]]]:
    """The lines wholly left of x and wholly right of it."""
    return [ln for ln in lines if ln[2] <= x], [ln for ln in lines if ln[0] >= x]


def _gutter(lines: list[list[float]], width: float) -> float | None:
    """Where two columns split, from how the text lines cover the page horizontally: of the x in the
    middle half of the page, the one whose lighter side holds the most line width (lines wholly on that
    side); ties go to the middle of the run of such x nearest the centre. None unless at least three lines
    on each side sit beside a line across the gutter (share some height with it), so text stacked above
    or below the other side's (a heading under the end of a column) does not make two columns."""
    xs = range(int(width * 0.25), int(width * 0.75) + 1)
    score = {x: min(sum(ln[2] - ln[0] for ln in side) for side in _sides(lines, x)) for x in xs}
    best = max(score.values())
    runs: list[list[int]] = []
    for x in (x for x in xs if score[x] == best):
        if runs and runs[-1][-1] == x - 1:
            runs[-1].append(x)
        else:
            runs.append([x])
    run = min(runs, key=lambda r: abs((r[0] + r[-1]) / 2 - width / 2))
    gutter = (run[0] + run[-1]) / 2
    left, right = _sides(lines, gutter)
    beside = [sum(any(o[1] < ln[3] and ln[1] < o[3] for o in other) for ln in side)
              for side, other in ((left, right), (right, left))]
    return gutter if min(beside) >= 3 else None


def _beside(a: dict[str, Any], side: list[dict[str, Any]]) -> bool:
    """Whether a block of horizontal text has another beside it on its side of the gutter (sharing some height,
    apart in x): a third column, or table cells. Ordering the two by top y is a guess (D-164)."""
    return bool(a["lines"]) and any(
        b is not a and b["lines"] and (b["bbox"][0] >= a["bbox"][2] or a["bbox"][0] >= b["bbox"][2])
        and b["bbox"][1] < a["bbox"][3] - 2 and b["bbox"][3] > a["bbox"][1] + 2 for b in side)


def order_blocks(blocks: list[dict[str, Any]], width: float, height: float,
                 graphics: list[list[float]] | tuple = ()) -> tuple[list[int], dict[str, Any]]:
    """`graphics`: bboxes of the page's rasters and drawing clusters, which are no blocks to order but can split
    the page as a full-width block does."""
    margins = [b for b in blocks if (b["bbox"][3] < height * 0.07 and b["bbox"][3] - b["bbox"][1] < 20)
               or (b["bbox"][1] > height * 0.94 and b["text"].strip().isdigit())]
    body = [b for b in blocks if b not in margins]
    if margins:
        ordered, meta = order_blocks(body, width, height, graphics)
        top = sorted([b for b in margins if b["bbox"][3] < height * 0.07], key=lambda b: b["bbox"][0])
        bottom = [b for b in margins if b not in top]
        return ([b["block_index"] for b in top] + ordered + [b["block_index"] for b in bottom],
                {**meta, "marginal_blocks": [b["block_index"] for b in margins]})
    gutter = _gutter([ln for b in blocks for ln in b["lines"]], width)
    mid = width / 2 if gutter is None else gutter
    tol = width * 0.015
    left = [b for b in blocks if b["bbox"][2] <= mid + tol]
    right = [b for b in blocks if b not in left and b["bbox"][0] >= mid - tol]
    wide = [b for b in blocks if b not in left and b not in right]
    two = gutter is not None or (len(left) >= 3 and len(right) >= 3)  # Q's three blocks a side still count
    if not two:
        return ([b["block_index"] for b in sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))],
                {"columns": 1, "confidence": "unresolved" if left and right else "single_column", "wide_blocks": len(wide)})
    result: list[dict[str, Any]] = []
    pending = left + right
    bands: list[list[dict[str, Any]]] = []  # the column blocks between one full-width block and the next
    guessed: list[int] = []
    for w in sorted(wide, key=lambda b: (b["bbox"][1], b["bbox"][0])):
        before = [b for b in pending if b["bbox"][3] <= w["bbox"][1] + 2]
        pending = [b for b in pending if b not in before]
        # In a band a full-width block closes, a left block wholly below every right block is read before them as
        # a column's end, after them as the text under a local table's columns: which, the page does not show.
        lows = [b for b in before if b in left]
        tops = [b["bbox"][3] for b in before if b in right]
        guessed += [b["block_index"] for b in lows if tops and b["bbox"][1] >= max(tops) - 2]
        bands.append(before)
        result.extend(sorted(before, key=lambda b: (0 if b in left else 1, b["bbox"][1], b["bbox"][0])))
        result.append(w)
    bands.append(pending)
    result.extend(sorted(pending, key=lambda b: (0 if b in left else 1, b["bbox"][1], b["bbox"][0])))
    # A left block in a vertical gap of the right side, with right blocks above and below it in its band and none
    # beside it (a heading between two local bands of columns), is read before them all as part of a column: whether
    # the right blocks above it come before it, the page does not show (D-202).
    for band in bands:
        rights = [b for b in band if b in right]
        guessed += [a["block_index"] for a in band if a in left and a["block_index"] not in guessed
                    and not any(r["bbox"][1] < a["bbox"][3] - 2 and r["bbox"][3] > a["bbox"][1] + 2 for r in rights)
                    and any(r["bbox"][3] <= a["bbox"][1] + 2 for r in rights)
                    and any(r["bbox"][1] >= a["bbox"][3] - 2 for r in rights)]
    assert sorted(b["block_index"] for b in result) == sorted(b["block_index"] for b in blocks)
    overlap = [w["block_index"] for w in wide
               if any(b["bbox"][1] < w["bbox"][3] - 2 and b["bbox"][3] > w["bbox"][1] + 2 for b in left + right)]
    overlap += [a["block_index"] for side in (left, right) for a in side if _beside(a, side)]
    # A graphic across the gutter (a full-width figure or table) with column text above and below it in one band:
    # whether each column runs on past it or the page's text runs across it first, the page does not show.
    crossing = [g for g in graphics if g[0] < mid - tol and g[2] > mid + tol
                and not (g[3] <= height * 0.07 or g[1] >= height * 0.94)
                and any(any(b["bbox"][3] <= g[1] + 2 for b in band) and any(b["bbox"][1] >= g[3] - 2 for b in band)
                        for band in bands)]
    return ([b["block_index"] for b in result],
            {"columns": 2, "confidence": "requires_visual_review" if overlap or guessed or crossing else "geometry_consistent",
             "wide_blocks": len(wide), "overlap_blocks": overlap + guessed, "crossing_graphics": len(crossing)})
