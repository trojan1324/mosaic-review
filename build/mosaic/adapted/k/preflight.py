"""Per-page routing signals (adapted from K/tools/preflight_pdf.py).

Kept: image_area_ratio, multi_column_score, table_density, equation_density
and the recommend_engine decision rule, unchanged in logic. Changed: the
functions take an already-open PyMuPDF page (no second parse), nothing writes
to kb.db or K's library, and the result is a per-page signal consumed by
v2 capture (FINAL v2 §5.1, audit M-04) instead of a disconnected report.
"""

from __future__ import annotations

import re
from typing import Any

MATH_CHARS_RE = re.compile(r"[∑∫√≈≠≤≥±×÷βγδλμσπρηθα]|(?:\b[a-zA-Z]\s*[=<>]\s*-?\d)|(?:\bp\s*[<=>]\s*0?\.\d+)")


def alpha_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]", text or ""))


def image_area_ratio(page: Any, page_dict: dict[str, Any]) -> float:
    page_area = max(1.0, float(page.rect.width * page.rect.height))
    image_area = 0.0
    for block in page_dict.get("blocks") or []:
        if block.get("type") != 1:
            continue
        bbox = block.get("bbox") or [0, 0, 0, 0]
        image_area += max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1]))
    return min(1.0, image_area / page_area)


def multi_column_score(width: float, blocks: list[tuple[float, float, float, float, str]]) -> float:
    blocks = [b for b in blocks if alpha_count(re.sub(r"\s+", " ", b[4])) >= 20]
    if len(blocks) < 6:
        return 0.0
    width = max(1.0, float(width))
    left = sum(1 for b in blocks if ((b[0] + b[2]) / 2) < width * 0.48)
    right = sum(1 for b in blocks if ((b[0] + b[2]) / 2) > width * 0.52)
    if left >= 3 and right >= 3:
        return min(1.0, (left + right) / max(12, len(blocks)))
    return 0.0


def table_density(text: str) -> float:
    lines = [line.rstrip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return 0.0
    tableish = 0
    for line in lines:
        if line.count("|") >= 2:
            tableish += 1
        elif re.findall(r"\S+\s{2,}\S+\s{2,}\S+", line):
            tableish += 1
    return min(1.0, tableish / max(8, len(lines)))


def equation_density(text: str) -> float:
    if not text:
        return 0.0
    matches = len(MATH_CHARS_RE.findall(text))
    return min(1.0, matches / max(8, len(text) / 1000))


def recommend_engine(scanned: float, empty: float, multicol: float, tables: float, equations: float) -> str:
    """K's rule, per page. Advisory: v2 records it; it does not switch parsers."""
    if equations >= 0.35:
        return "marker"
    if scanned >= 0.25 or empty >= 0.3:
        return "docling_ocr"
    if multicol >= 0.35 or tables >= 0.15:
        return "docling"
    return "pymupdf"


def page_signals(page: Any, page_dict: dict[str, Any], text: str, blocks: list[tuple]) -> dict[str, Any]:
    alpha = alpha_count(text)
    img = image_area_ratio(page, page_dict)
    empty = alpha < 80
    scanned = empty and img > 0.45
    mc = multi_column_score(page.rect.width, blocks)
    td = table_density(text)
    ed = equation_density(text)
    return {
        "empty_text": empty,
        "scanned_likelihood": scanned,
        "image_area_ratio": round(img, 3),
        "multi_column_likelihood": round(mc, 3),
        "table_density": round(td, 3),
        "equation_density": round(ed, 3),
        "recommended_engine": recommend_engine(1.0 if scanned else 0.0, 1.0 if empty else 0.0, mc, td, ed),
    }
