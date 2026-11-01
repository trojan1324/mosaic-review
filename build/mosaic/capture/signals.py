"""Text-quality signals computed per page on native text (FINAL v2 §5.2).

RUNON and the U+FFFD count come from K/tools/score_pdf_extraction.py
(REUSE_LEDGER R007), which scores a whole source after ingestion; here they
run per page before anything is promoted.

`scanned_text_layer` applies the E1 "scanned" stratum rule
(docs/E1_DEV_PROTOCOL.md §2) at capture, plus the audit's invisible-text
branches (root cause G), which E1 did not pre-register: a page whose text layer
is another engine's OCR of a page image is not born-digital text
(D-163), and a page image whose only text is a stamp in its margins has its
words in no text layer (`unread_scan`, D-189). `decoration` tells an image that holds no figure (an icon, a
banner in the margin band) from one that may (D-165).
`unmapped_glyphs` finds characters whose glyph has no faithful Unicode, so
witnesses on them are not taken as exact native text (D-166).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import pymupdf

REPLACEMENT = "�"
RUNON = re.compile(r"[A-Za-z]{25,}")
SYMBOL_FONT = re.compile(r"(?i)math|symbol")  # a math or symbol font: Cambria Math, STIX Two Math, SymbolMT
MATH_ASCII = frozenset("+-*/=<>()[]{}|,.:;'")  # ASCII a math or symbol font draws as itself: operators, delimiters
# K's ingest gate: more than 1% replacement characters rejects the page.
GARBLED_THRESHOLD = 0.01
SCAN_IMAGE_SHARE = 0.9  # E1 "scanned": one image covers at least this share of the page ...
SCAN_MIN_CHARS = 100    # ... and the page has at least this many characters of text
TOP_BAND, BOTTOM_BAND = 0.07, 0.94  # the page's top and bottom margin bands (reading order's margin peel-off)
MARGIN_BAND = 0.1  # the page's margins as a share of its height or width (the section tree's furniture bands)
DECORATION_MAX_SIDE = 12.0  # points: an image no larger than a line of type (an icon, a badge, a speck)


def decoration(bbox: tuple[float, float, float, float], height: float) -> str | None:
    """Why an image is decoration rather than a possible figure, or None: no larger than a line of type on
    either side, or wholly inside the top or bottom margin band (bbox in the unrotated cropbox frame,
    `height` the cropbox's). An image repeated on several pages is not decoration by that alone: a
    stimulus or scale reused across conditions is content (D-165)."""
    x0, y0, x1, y1 = bbox
    if max(x1 - x0, y1 - y0) <= DECORATION_MAX_SIDE:
        return f"at most {DECORATION_MAX_SIDE:g} pt on its longer side"
    if y1 <= height * TOP_BAND or y0 >= height * BOTTOM_BAND:
        return "wholly inside the page's top or bottom margin band"
    return None


def image_share(page: Any, bbox: tuple[float, float, float, float]) -> float:
    """Share of the page that an image bbox covers. get_image_info bboxes are in the unrotated
    cropbox frame, like get_text("dict") (D-061), so the page area is the cropbox's own."""
    frame = pymupdf.Rect(0, 0, page.cropbox.width, page.cropbox.height)
    r = pymupdf.Rect(bbox) & frame
    return 0.0 if r.is_empty else r.width * r.height / max(1.0, frame.width * frame.height)


def invisible_chars(page: Any) -> int:
    """Characters drawn invisible (render mode 3), the usual form of an OCR layer laid over a page image."""
    return sum(len(span["chars"]) for span in page.get_texttrace() if span["type"] == 3)


def page_image_share(page: Any) -> float:
    """Share of the page that its largest image covers."""
    return max((image_share(page, info["bbox"]) for info in page.get_image_info()), default=0.0)


def margin_text_only(page: Any) -> bool:
    """Whether every line of the page's text lies in its margins, the top or bottom tenth or a side tenth (a
    download or copyright stamp, a folio): text that is no layer of what the page shows (D-189)."""
    w, h = page.cropbox.width, page.cropbox.height  # get_text bboxes are in the unrotated cropbox frame (D-061)
    boxes = [ln["bbox"] for b in page.get_text("dict")["blocks"] if b.get("type") == 0 for ln in b["lines"]
             if "".join(s["text"] for s in ln["spans"]).strip()]
    return bool(boxes) and all(y1 <= MARGIN_BAND * h or y0 >= (1 - MARGIN_BAND) * h or x1 <= MARGIN_BAND * w
                               or x0 >= (1 - MARGIN_BAND) * w for x0, y0, x1, y1 in boxes)


def scanned_text_layer(page: Any, text: str) -> str | None:
    """Why the page's text is taken as a scan's OCR layer, or None: one image covers >= 90% of the page and
    the page has >= 100 characters of text (E1's pre-registered rule) or any invisible text (a short page of a
    scan, such as a figure page or the last page), or >= 100 characters are drawn invisible (render mode 3,
    the usual form of an OCR layer laid over a page image). The last two branches are the audit's, not
    pre-registered (D-163). Visible text that lies wholly in the page's margins is a stamp over the image, no
    layer of it (`unread_scan`, D-189)."""
    share = page_image_share(page)
    chars = len(text.strip())
    invisible = invisible_chars(page)
    if (share >= SCAN_IMAGE_SHARE and (chars >= SCAN_MIN_CHARS or invisible)) or invisible >= SCAN_MIN_CHARS:
        if not invisible and margin_text_only(page):
            return None
        return (f"the largest image covers {share:.0%} of the page; {chars} characters of text, "
                f"{invisible} of them invisible (render mode 3)")
    return None


def unread_scan(page: Any, text: str) -> str | None:
    """Why the page is a scan whose words are in no text layer, or None: one image covers >= 90% of the page and
    all of its text, none of it invisible, lies in the page's margins (a publisher's download or copyright stamp
    over a scan that has no OCR layer), so what the page shows is not read unless it is recognized (D-189)."""
    share = page_image_share(page)
    if share >= SCAN_IMAGE_SHARE and text.strip() and not invisible_chars(page) and margin_text_only(page):
        return (f"the largest image covers {share:.0%} of the page and its {len(text.strip())} characters of text "
                "all lie in the page's margins")
    return None


def replacement_ratio(text: str) -> float:
    visible = sum(1 for c in text if not c.isspace())
    if visible == 0:
        return 0.0
    return round(text.count(REPLACEMENT) / visible, 6)


def _private_use(c: str) -> bool:
    return 0xE000 <= ord(c) <= 0xF8FF or 0xF0000 <= ord(c) <= 0x10FFFD


def private_use_count(text: str) -> int:
    return sum(1 for c in text if _private_use(c))


def unmapped_glyphs(text: str, font: str, drawn_blank: bool) -> list[tuple[int, int, str]]:
    """Runs [start, end) of one span's characters whose glyph has no faithful Unicode, with the reason:
    U+FFFD, private-use code points, control characters other than TAB, LF and CR, and in a math or
    symbol font ASCII punctuation that is not an operator or delimiter (a subset font whose glyph codes,
    numbered from "!", pass through as text) or whitespace glyphs (a symbol whose map says space).
    `drawn_blank` says the span is only whitespace drawn as glyphs, not spaces PyMuPDF put in a gap."""
    symbol = bool(SYMBOL_FONT.search(font))
    runs: list[tuple[int, int, str]] = []
    for i, c in enumerate(text):
        if c == REPLACEMENT:
            reason = "replacement_character"
        elif _private_use(c):
            reason = "private_use"
        elif unicodedata.category(c) == "Cc" and c not in "\t\n\r":
            reason = "control_character"
        elif symbol and c.isascii() and not c.isalnum() and not c.isspace() and c not in MATH_ASCII:
            reason = "symbol_font_ascii"
        elif symbol and drawn_blank:
            reason = "symbol_font_whitespace"
        else:
            continue
        if runs and runs[-1][1] == i and runs[-1][2] == reason:
            runs[-1] = (runs[-1][0], i + 1, reason)
        else:
            runs.append((i, i + 1, reason))
    return runs


def runon_tokens(text: str) -> int:
    return len(RUNON.findall(text))


def is_garbled(text: str) -> bool:
    return replacement_ratio(text) > GARBLED_THRESHOLD
