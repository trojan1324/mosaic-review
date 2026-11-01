"""Ruled-table interpretation for single-page tables (M2; FINAL v2 §4.1-4.2; audit C-05, M-20, N-55).

Adapted from E's table modules (REUSE_LEDGER R021-R023): caption ownership
follows `extraction_table_ownership.py` (a caption owns its table only when it
names exactly one Study/Experiment; several or none leave ownership
unresolved); count roles follow `extraction_table_contract.py` (a bare N is a
count, never assumed to be people; means and dispersions are not counts).
E derived tables from flattened units and could not resolve multi-page or
merged-header structure; here:

* The table region runs from the caption down through the horizontal rules
  below it: a full-width top rule, a full-width rule under the column heads,
  and a full-width bottom rule. Short rules between the first two mark column
  groups (merged headers) and are used to resolve them; several heads on one
  line over one such rule head one column each. The rules lie under the
  caption's own lines, and none under a later float caption across it is the
  table's; two tables' rules side by side under one caption are refused, and
  the table ends at the next caption or note line in any case (D-194).
* Glyphs come from PyMuPDF rawdict with capture's flags, which reproduces the
  captured text character for character, so every cell is an exact code-point
  span of the page's text artifact with its geometry.
* Body columns are the x-bands of body tokens; header text above each band is
  stacked top-down, with group headers first, into its `header_path`. Stub
  columns (row labels) carry down so a row's path names its variable and its
  statistic (for example ["1. Control loss", "SD"]).
* Several complete ruled panels under one caption (Bondi Tables A2, A18;
  D-037), each its own top/header/bottom rule triple headed by a "Panel X:
  ..." title naming one study, are interpreted panel by panel: columns are
  namespaced per panel and each cell's row path is prefixed with its panel's
  title. A panel structure that is not clean triples, or titles that do not
  each name one study distinct from every other panel's, is not guessed at;
  the table stays `unresolved` as it did before.
* Anything else is recorded, not guessed: a rotated table, a caption marked
  continued, missing rules (on a scanned page, rules that are only pixels of
  the page image), a header that spans columns without a rule to
  resolve it, or a value column that holds text (wrapped text cells) gives
  `unsupported_*` or `unresolved` with no cells.
* A read grid is checked before it is kept: a body token that spans two
  columns, a value with a bracketed number under a column headed by a
  single-number statistic ("t", "SD"), or a row whose values sit under
  different sub-heads in groups that repeat the same sub-heads (Model 2 "t",
  Model 3 "B (SE)") leaves the table `unresolved` with the reason: those are
  values printed shifted across columns, and binding them would be a guess.
  So does a second printed line bound to a row path and column already
  bound: the reading could not tell the two rows apart; and a bracketed
  label-only line (a qualifier) under a row with values or its SE line, with
  no group open or under one that is not itself bracketed, set flush left or
  indented under a label whose next word would not have run past the value
  columns, which may qualify that row or head the rows below it; and a
  bracketed label beside bracketed values under a row; and numbers joined
  into one token (D-203). So do heads on different lines over one short rule,
  one of them over some but not all of the columns under it, and a row with
  values set under a full-width rule below the rows of an open group with no
  heading of its own: whether it is that group's or the whole table's is not
  shown (D-206).
* The note is the topmost block under the bottom rule that overlaps the
  table and starts "Note."/"Notes:" or is a footnote legend, reached with no
  caption or paragraph of running text between, with the footnote blocks
  stacked under it that start inside the table's columns, and a block within
  a line under it that overlaps it (its text under a "Notes:" label), set
  smaller than body text in the note's size (D-195, D-159); it is read once
  the rules are located, whether or not the body could be.
* On a page taken as a scan, vector drawings are rules only when the text is
  not a hidden layer, no other page of the PDF is such a scan, there are
  three, and they bound the table's head (D-204).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pymupdf

from mosaic.capture.signals import SCAN_IMAGE_SHARE, image_share, page_image_share, scanned_text_layer
from mosaic.inventory.section_tree import _caption_studies, is_caption_line

STAT_HEADERS = {"m", "sd", "se", "b", "β", "t", "p", "r", "α", "f", "or", "ci", "d", "mean", "median", "%"}
COUNT_HEADERS = {"n", "count", "frequency", "freq."}
SINGLE_NUMBER_HEADERS = {"t", "z", "f", "p", "sd", "se", "df"}  # a column so headed holds one number per cell
MARKS = "*†‡§⁎∗"  # significance marks, with the low asterisk U+204E and the asterisk operator U+2217
SIG = re.compile(rf"([{MARKS}]+)(?:\s*[(\[][^)\]]*[)\]])?$")  # "2.10*", "(0.065)**", "1.38** (0.16)"
# A number: a minus (as some producers print one, "-", U+00AD, U+2010: KS-DEV-05), digits with or without thousands
# separators ("12,110"), a decimal part. Numbers read as text would make a value column a row-label column (D-206).
NUM = r"(?:-\u00ad\u2010|[−–-])?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)%?"
NUMERIC = re.compile(rf"^(?:{NUM}[{MARKS}]*(?:\s*[(\[]{NUM}(?:,\s*{NUM})?[)\]])?[{MARKS}]*|[—–-]|[(\[]{NUM}(?:,\s*{NUM})?[)\]][{MARKS}]*)$")
WITH_BRACKETED = re.compile(rf"^{NUM}[{MARKS}]*\s*[(\[]")  # a value followed by a bracketed one: "−0.23* (0.09)"
CONTINUED = re.compile(r"\(?\bcont(?:inued|\.|'d)\b\)?", re.I)
FOOTNOTE = re.compile(rf"(?:[a-z]\s|[{MARKS}]+\s*p\s*[<=>≤]|[{MARKS}]+\s)")  # "a High = 0.5", "* p < .05", "† p < .10"
LEGEND = re.compile(rf"[{MARKS}]+\s*p\s*[<=>≤]")  # a significance legend: "* p < .05", "⁎⁎⁎ p < 0.001."
NOTE_START = re.compile(r"Notes?\s*[.:]", re.I)  # "Note.", "Notes:"; never "Note that ..."


@dataclass
class Glyph:
    ch: str
    offset: int
    x0: float
    y0: float
    x1: float
    y1: float
    line_dir: tuple[float, float]
    block: int = -1
    size: float = 0.0  # the type size of its span

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class Token:
    start: int
    end: int
    text: str
    x0: float
    x1: float
    y0: float
    y1: float

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class Grid:
    status: str                      # single_page_resolved | unsupported_* | unresolved
    reason: str = ""
    bbox: list[float] = field(default_factory=list)
    columns: list[dict[str, Any]] = field(default_factory=list)
    cells: list[dict[str, Any]] = field(default_factory=list)       # includes "span" (start, end) for witnesses
    notes: list[tuple[int, int, str]] = field(default_factory=list)
    reading: str = "right"           # text direction of the table: right (upright), up or down (printed sideways)
    title: list[tuple[int, int, str]] = field(default_factory=list)  # lines between the caption label and the top rule
    panels: list[dict[str, Any]] = field(default_factory=list)      # several complete ruled panels (D-037); each
                                                                     # carries panel_id, title, bbox, owner_study_local_id, ownership_basis
    stub: list[int] = field(default_factory=list)  # column indices classified as row-label (stub) columns
                                                     # (single-ruled tables only; empty for a multi-panel Grid)
    table_top: float | None = None  # where the table starts under its caption, in its reading frame: its first
    rows_separated: bool = False  # every printed body line is separated from the next by a full-width rule
    exclusion_bbox: list[float] = field(default_factory=list)  # model-input exclusion only; never a cell/ownership claim
                                    # full-width rule, or its first panel's title; a caption's own title lies above.
                                    # inf when the page has no full-width rule below the caption (a page taken as
                                    # a scan has none that is a vector drawing); None when the
                                    # page's rules were not read, or do not form top, header and bottom rules


def page_glyphs(page: pymupdf.Page, expected_text: str) -> list[Glyph]:
    """Glyphs with offsets into the captured page text; refuses if the reconstruction differs."""
    rd = page.get_text("rawdict", sort=True, flags=pymupdf.TEXTFLAGS_BLOCKS)
    glyphs: list[Glyph] = []
    pos = 0
    parts: list[str] = []
    for bi, block in enumerate(rd["blocks"]):
        if block.get("type") != 0:
            continue
        for li, line in enumerate(block["lines"]):
            if li:
                parts.append("\n")
                pos += 1
            for span in line["spans"]:
                for ch in span["chars"]:
                    x0, y0, x1, y1 = ch["bbox"]
                    glyphs.append(Glyph(ch["c"], pos, x0, y0, x1, y1, tuple(line.get("dir", (1.0, 0.0))), bi,
                                        round(float(span.get("size", 0.0)), 1)))
                    parts.append(ch["c"])
                    pos += 1
        parts.append("\n")
        pos += 1
    if "".join(parts) != expected_text:
        raise ValueError("rawdict does not reproduce the captured text; no table offsets")
    return glyphs


def horizontal_rules(page: pymupdf.Page) -> list[tuple[float, float, float]]:
    rules = []
    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                if abs(p1.y - p2.y) < 1.0 and abs(p2.x - p1.x) > 20:
                    rules.append((round((p1.y + p2.y) / 2, 1), min(p1.x, p2.x), max(p1.x, p2.x)))
            elif item[0] == "re":
                r = item[1]
                if r.height < 2.0 and r.width > 20:
                    rules.append((round((r.y0 + r.y1) / 2, 1), r.x0, r.x1))
    return _merge_rules(rules)


def vertical_rules(page: pymupdf.Page) -> list[tuple[float, float, float]]:
    """(x, y0, y1) of every vertical line: the horizontal rules of a table printed sideways."""
    rules = []
    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                if abs(p1.x - p2.x) < 1.0 and abs(p2.y - p1.y) > 20:
                    rules.append((round((p1.x + p2.x) / 2, 1), min(p1.y, p2.y), max(p1.y, p2.y)))
            elif item[0] == "re":
                r = item[1]
                if r.width < 2.0 and r.height > 20:
                    rules.append((round((r.x0 + r.x1) / 2, 1), r.y0, r.y1))
    return rules


def _merge_rules(rules: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    merged: list[tuple[float, float, float]] = []
    for y, x0, x1 in sorted(rules):  # a double rule (two lines within 3 pt) is one rule
        if merged and abs(merged[-1][0] - y) <= 3.0 and not (x1 < merged[-1][1] or x0 > merged[-1][2]):
            py, px0, px1 = merged[-1]
            merged[-1] = (py, min(px0, x0), max(px1, x1))
        else:
            merged.append((y, x0, x1))
    return merged


def _tokens(glyphs: list[Glyph], x0: float, y0: float, x1: float, y1: float, text: str = "") -> list[Token]:
    inside = [g for g in glyphs if x0 - 1 <= (g.x0 + g.x1) / 2 <= x1 + 1 and y0 < g.cy < y1 and not g.ch.isspace()]
    inside.sort(key=lambda g: g.offset)
    tokens: list[Token] = []
    for g in inside:
        t = tokens[-1] if tokens else None
        size = max(g.y1 - g.y0, 1.0)
        same_line = t is not None and abs(t.cy - g.cy) < size * 0.6
        # Words of one captured line join across the whitespace between them: one space, or a run of tabs, CRs
        # and no-break spaces as some producers set a space (KS-DEV-05), never a line break (D-185). A run spans
        # no more than such a space does (KS-DEV-05's measure at most 0.37 glyph height): two or three spaces
        # across a narrow gutter part two columns (D-203).
        between = text[t.end:g.offset] if t is not None and g.offset - t.end > 1 else ""
        adjacent = t is not None and (g.offset - t.end <= 1 or (between.isspace() and "\n" not in between))
        # Never leftwards: a glyph that starts well before the token's end is the next block on the same
        # baseline ("(0.86)" then the next row's "2. SDO"), not the next letter; kerning, accents and
        # ligatures overlap by under half a glyph height.
        contiguous = adjacent and -size * 0.5 < g.x0 - t.x1 < size * (0.4 if between else 0.8)
        if same_line and contiguous:
            t.end, t.x1 = g.offset + 1, max(t.x1, g.x1)
            t.y0, t.y1 = min(t.y0, g.y0), max(t.y1, g.y1)
        else:
            tokens.append(Token(g.offset, g.offset + 1, "", g.x0, g.x1, g.y0, g.y1))
    return tokens


def _rows(tokens: list[Token], tol: float = 3.5) -> list[list[Token]]:
    rows: list[list[Token]] = []
    for t in sorted(tokens, key=lambda t: t.cy):
        if rows and abs(sum(x.cy for x in rows[-1]) / len(rows[-1]) - t.cy) <= tol:
            rows[-1].append(t)
        else:
            rows.append([t])
    return [sorted(r, key=lambda t: t.x0) for r in rows]


def _bands(tokens: list[Token], gap: float = 2.0) -> list[tuple[float, float]]:
    bands: list[list[float]] = []
    for t in sorted(tokens, key=lambda t: t.x0):
        if bands and t.x0 <= bands[-1][1] + gap:
            bands[-1][1] = max(bands[-1][1], t.x1)
        else:
            bands.append([t.x0, t.x1])
    return [(a, b) for a, b in bands]


def _overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return min(a1, b1) - max(a0, b0) > 0.5


def count_role(header_path: list[str]) -> str:
    last = re.sub(r"\s*[(\[].*$", "", (header_path[-1] if header_path else "").strip().lower())
    if last in COUNT_HEADERS:
        return "count"
    if last in STAT_HEADERS:
        return "not_count"
    return "unknown"


def _direction(line_dir: tuple[float, float]) -> str:
    dx, dy = line_dir
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "up" if dy < 0 else "down"


def _upright(direction: str, width: float, height: float):
    """Maps between the page frame and the reading frame of a table printed sideways.

    Text read upwards is turned a quarter clockwise, (x, y) -> (height - y, x);
    text read downwards a quarter anticlockwise, (x, y) -> (y, width - x).
    Returns (bbox to reading frame, bbox back to page frame, vertical rule to reading-frame rule).
    """
    if direction == "up":
        return (lambda b: (height - b[3], b[0], height - b[1], b[2]),
                lambda b: (b[1], height - b[2], b[3], height - b[0]),
                lambda r: (r[0], height - r[2], height - r[1]))
    return (lambda b: (b[1], width - b[2], b[3], width - b[0]),
            lambda b: (width - b[3], b[0], width - b[1], b[2]),
            lambda r: (round(width - r[0], 1), r[1], r[2]))


def interpret(page: pymupdf.Page, text: str, caption_start: int, caption_end: int, caption_text: str,
              foreign: list[tuple[int, int]] | tuple = (), scanned_pdf: bool = False, body_size: float = 0.0,
              allow_text_values: bool = False) -> Grid:
    """`foreign`: spans inside [caption_start, caption_end) that are not this caption's (the lines of a caption set
    beside it that its block interleaves with its own; D-193). `scanned_pdf`: another page of the PDF is a scan with
    a hidden text layer (D-204). `body_size`: the source's body type size (the section tree's), else the page's."""
    if CONTINUED.search(caption_text):
        return Grid("unsupported_multi_page", "caption marks a continued table; continuation is not resolved")
    return _interpret_structure(page, text, caption_start, caption_end, foreign, scanned_pdf, body_size, allow_text_values)


def interpret_continuation_fragment(page: pymupdf.Page, text: str, caption_start: int, caption_end: int,
                                    foreign: list[tuple[int, int]] | tuple = (), scanned_pdf: bool = False,
                                    body_size: float = 0.0) -> Grid:
    """Interprets a continuation fragment's own page exactly as an ordinary table's page, without the
    CONTINUED short-circuit above, so a caller (`mosaic.tables.record`) can compare its structure
    against a candidate parent table before deciding whether to link and read it."""
    return _interpret_structure(page, text, caption_start, caption_end, foreign, scanned_pdf, body_size)


def _body_size(glyphs: list[Glyph]) -> float:
    """The page's body type size: the size most of its characters are set in."""
    sizes = Counter(g.size for g in glyphs if not g.ch.isspace() and g.size)
    return sizes.most_common(1)[0][0] if sizes else 0.0


def _caption_glyphs(glyphs: list[Glyph], caption_start: int, caption_end: int, foreign) -> list[Glyph]:
    return [g for g in glyphs if caption_start <= g.offset < caption_end and not g.ch.isspace()
            and not any(a <= g.offset < b for a, b in foreign)]


def _hidden_layer(page: pymupdf.Page, text: str) -> str | None:
    """Why the page's text is a hidden OCR layer, or None: invisible characters (render mode 3) that carry its text
    (at least half its characters; a few for a screen reader do not), or text painted before an image covering at
    least 90% of the page, which hides it (D-204)."""
    invisible = sum(not chr(c[0]).isspace() for span in page.get_texttrace() if span["type"] == 3 for c in span["chars"])
    chars = len("".join(text.split()))
    if invisible and 2 * invisible >= chars:
        return f"{invisible} of its {chars} characters are invisible (render mode 3)"
    painted = False
    for kind, rect in page.get_bboxlog():
        painted = painted or kind.endswith("-text")
        if painted and kind == "fill-image" and image_share(page, rect) >= SCAN_IMAGE_SHARE:
            return "its text is painted under an image that covers the page"
    return None


def _interpret_structure(page: pymupdf.Page, text: str, caption_start: int, caption_end: int, foreign=(),
                         scanned_pdf: bool = False, body_size: float = 0.0, allow_text_values: bool = False) -> Grid:
    # A scan's rules are pixels of its page image. A vector drawing over a hidden OCR layer (a highlight, a pen
    # stroke) is a reader's mark, never a printed rule, and so is one on any page of a PDF whose other pages are such
    # scans (`scanned_pdf`, D-189); a page that met the rule by visible text on a full-page image (a born-digital page
    # drawn on a background) keeps its vector rules when there are enough of them to bound a table (three) and they
    # bound its head (`suspect`, D-204). Nothing then bounds a label-only caption's title from below (inf, D-176).
    scan = scanned_text_layer(page, text)
    if (scan and (scanned_pdf or _hidden_layer(page, text) or len(horizontal_rules(page)) + len(vertical_rules(page)) < 3)) \
            or (scanned_pdf and page_image_share(page) >= SCAN_IMAGE_SHARE):
        return Grid("unresolved", "taken as a scanned page by rule (D-163): any rules would be pixels of the "
                                  "page image, not vector drawings, and its text is treated as an OCR layer; not interpreted"
                    if page_image_share(page) >= SCAN_IMAGE_SHARE else
                    f"taken as a scanned page by rule (D-163; {scan}), with no page image: its text is treated as an OCR "
                    "layer, and no vector drawing on it is taken as a rule; not interpreted", table_top=float("inf"))
    try:
        glyphs = page_glyphs(page, text)
    except ValueError as exc:
        return Grid("unresolved", str(exc))
    cap = _caption_glyphs(glyphs, caption_start, caption_end, foreign)
    if not cap:
        return Grid("unresolved", "caption glyphs not found")
    directions = {_direction(g.line_dir) for g in cap}
    if len(directions) != 1:
        return Grid("unresolved", "caption lines run in more than one direction")
    direction = directions.pop()
    body = body_size or _body_size(glyphs)
    if direction == "right":
        return _interpret_upright(glyphs, horizontal_rules(page), text, caption_start, caption_end, foreign, body,
                                  bool(scan), allow_text_values)
    if direction == "left":
        return Grid("unsupported_rotated", "table printed upside down; not interpreted")
    # A table printed sideways: read it in its own frame from the glyphs that run its way and the page's
    # vertical rules, then put every box back into the page frame.
    to_reading, to_page, rule_to_reading = _upright(direction, page.rect.width, page.rect.height)
    turned = [Glyph(g.ch, g.offset, *to_reading((g.x0, g.y0, g.x1, g.y1)), (1.0, 0.0), g.block, g.size)
              for g in glyphs if _direction(g.line_dir) == direction]
    rules = _merge_rules([rule_to_reading(r) for r in vertical_rules(page)])
    grid = _interpret_upright(turned, rules, text, caption_start, caption_end, foreign, body, bool(scan), allow_text_values)
    if grid.exclusion_bbox:
        grid.exclusion_bbox = [round(v, 2) for v in to_page(tuple(grid.exclusion_bbox))]
    if not grid.bbox:  # a table refused for shifted values keeps its located region (and so its frame)
        return grid
    grid.bbox = [round(v, 2) for v in to_page(tuple(grid.bbox))]
    for cell in grid.cells:
        cell["bbox"] = [round(v, 2) for v in to_page(tuple(cell["bbox"]))]
    for p in grid.panels:
        p["bbox"] = [round(v, 2) for v in to_page(tuple(p["bbox"]))]
    grid.reading = direction
    return grid


def _panel_body(glyphs: list[Glyph], all_rules: list[tuple[float, float, float]], text: str,
                top: tuple[float, float, float], head: tuple[float, float, float],
                bottom: tuple[float, float, float], width: float, suspect: bool = False,
                allow_text_values: bool = False) -> Grid:
    """Interprets one ruled panel's header and body between its own top, header and bottom rules: group
    headers (resolved by the spanning rule under them), row labels as the leading non-numeric columns,
    a label-less value line continuing the row above. Shared by a single-ruled table and by each panel
    of a multi-panel one (D-037)."""
    x0, x1 = min(top[1], head[1], bottom[1]), max(top[2], head[2], bottom[2])
    group_rules = [r for r in all_rules if top[0] < r[0] < head[0] and r[2] - r[1] < 0.6 * width
                   and _overlaps(r[1], r[2], x0, x1)]
    body = _tokens(glyphs, x0, head[0], x1, bottom[0], text)
    heads = _tokens(glyphs, x0, top[0], x1, head[0], text)
    if not body:
        return Grid("unresolved", "no body text between the header and bottom rules")
    for t in body + heads:
        t.text = text[t.start:t.end]
    bbox = [round(x0, 2), round(top[0], 2), round(x1, 2), round(bottom[0], 2)]
    # Numbers joined into one token ("4.10 1.20") are values of columns the reading could not part: read as a row
    # label or a text column, they would leave their columns (D-203).
    for t in body:
        parts = t.text.split()
        if len(parts) > 1 and not NUMERIC.match(t.text) and all(NUMERIC.match(p) for p in parts):
            return Grid("unresolved", f"{t.text!r} holds {len(parts)} numbers in one token; the columns they belong "
                                      "to cannot be separated", bbox)
    bands = _bands(body)
    rows = _rows(body)
    # A token that bridges two columns (a summary label printed across the stub and a value column,
    # "Weighted mean") would merge them into one band, and nothing after this could tell them apart. The bands
    # are formed without it, and again without every token at least as wide, so that a second label across the
    # same columns ("Unweighted mean") does not hold them together.
    for t in body:
        for others in ([u for u in body if u is not t], [u for u in body if u.x1 - u.x0 < t.x1 - t.x0]):
            spanned = [b for b in _bands(others) if _overlaps(t.x0, t.x1, *b)]
            if len(spanned) > 1:
                return Grid("unresolved", f"{t.text!r} spans {len(spanned)} columns of the body; "
                                          "the columns under it cannot be separated", bbox)

    def band_of(t: Token) -> int | None:
        hits = [i for i, (a, b) in enumerate(bands) if _overlaps(t.x0, t.x1, a, b)]
        return hits[0] if len(hits) == 1 else None

    def mostly_text(i: int) -> bool:
        texts = [t.text for t in body if band_of(t) == i]
        return bool(texts) and sum(bool(NUMERIC.match(x)) for x in texts) < len(texts) / 2

    # Stub columns (row labels): the leading bands, from the first, whose tokens are mostly not numbers.
    stub: list[int] = []
    while len(stub) < min(3, len(bands) - 1) and mostly_text(len(stub)):
        stub.append(len(stub))
    # The narrow measure-item retry needs every explicitly-headed column as a cell candidate.  It will retain
    # only item/anchor columns after checking their headings, so treating a leading wording column as a stub here
    # would discard the exact source span before that safety check can run.
    if allow_text_values:
        stub = []
    if suspect:  # a page taken as a scan: its vector drawings are rules only if they bound the table's head (D-204)
        for r in _rows(heads):
            if any(band_of(t) in stub and not NUMERIC.match(t.text) for t in r) and any(
                    band_of(t) not in stub and band_of(t) is not None and NUMERIC.match(t.text) for t in r):
                return Grid("unresolved", f"taken as a scanned page by rule (D-163), and its vector drawings put the row "
                                          f"{' '.join(t.text for t in r)!r} above the rule under the column heads: they "
                                          "do not bound the table, and may be marks over the scan; not interpreted")
    text_columns = [f"c{i}" for i in range(len(bands)) if i not in stub and mostly_text(i)]
    if text_columns and not allow_text_values:
        return Grid("unresolved", f"value column(s) {', '.join(text_columns)} hold text, not numbers; "
                                  "wrapped text cells are not interpreted", exclusion_bbox=bbox)
    # A text-only line with no row label inside the body is a heading (a panel title or a second row of
    # column heads); the columns under it need not line up with those above.
    inner = next((r for r in rows if not any(band_of(t) in stub for t in r) and not any(NUMERIC.match(t.text) for t in r)), None)
    if inner is not None and not allow_text_values:
        return Grid("unresolved", f"heading line {' '.join(t.text for t in inner)!r} inside the table body; "
                                  "panels with their own headings are not interpreted")
    # Header paths: group headers (resolved by the spanning rule under them) first, then stacked heads.
    group_for: dict[int, list[str]] = {i: [] for i in range(len(bands))}
    stacked: dict[int, list[Token]] = {i: [] for i in range(len(bands))}
    unresolved_heads = []
    rule_of = {id(t): next((r for r in group_rules if r[0] > t.y1 - 1 and _overlaps(t.x0, t.x1, r[1], r[2])), None)
               for t in heads}

    def cover(t: Token) -> list[int]:
        return [i for i, (a, b) in enumerate(bands) if _overlaps(t.x0, t.x1, a, b)]
    for t in sorted(heads, key=lambda t: (t.cy, t.x0)):
        covered = cover(t)
        rule = rule_of[id(t)]
        siblings = [u for u in heads if rule is not None and rule_of[id(u)] == rule and abs(u.cy - t.cy) < 3]
        if len(siblings) > 1:
            # Several heads on one line over one rule are the heads of the columns under it, one each, when every
            # column under the rule has its own ("English", "Urdu", "Math" over (1)-(3)); none is the group head of
            # all of them. Otherwise which columns each heads is not shown (D-203).
            members = [i for i, (a, b) in enumerate(bands) if rule[1] - 1 <= (a + b) / 2 <= rule[2] + 1]
            if all(len(cover(u)) == 1 for u in siblings) and sorted(cover(u)[0] for u in siblings) == members:
                stacked[covered[0]].append(t)
            else:
                return Grid("unsupported_merged_headers", f"headers {[u.text for u in siblings][:3]} sit on one line over "
                                                          "one rule; which columns each heads is not shown")
        elif rule is not None:
            members = [i for i, (a, b) in enumerate(bands) if rule[1] - 1 <= (a + b) / 2 <= rule[2] + 1]
            # A head over one rule with other heads over it, not all on one line, that lies over some but not all of the
            # columns under the rule ("Control" over (1) under "Difference ..." over (1)-(2); "Treatment effect" over
            # (2)-(3) above "Control", "OLS", "IV") may head only those columns rather than every column under the rule;
            # which columns each heads is not shown (D-206). One over all of them, or between them, is a group head.
            over = [u.text for u in heads if rule_of[id(u)] == rule]
            inside = set(covered) & set(members)
            if len(over) > 1 and inside and inside != set(members):
                return Grid("unsupported_merged_headers", f"headers {over[:3]} sit over one rule, and {t.text!r} lies over "
                                                          "some but not all of its columns; which columns each heads is "
                                                          "not shown")
            for i in members:
                group_for[i].append(t.text)
        elif len(covered) == 1:
            stacked[covered[0]].append(t)
        elif not covered:
            near = min(range(len(bands)), key=lambda i: abs((bands[i][0] + bands[i][1]) / 2 - (t.x0 + t.x1) / 2))
            stacked[near].append(t)
        else:
            unresolved_heads.append(t.text)
    if unresolved_heads:
        return Grid("unsupported_merged_headers", f"header(s) {unresolved_heads[:3]} span several columns with no rule to resolve them")
    columns = []
    for i in range(len(bands)):
        lower = " ".join(t.text for t in sorted(stacked[i], key=lambda t: (round(t.cy), t.x0)))
        path = group_for[i] + ([lower] if lower else [])
        columns.append({"column_id": f"c{i}", "header_path": path, "count_role": count_role(path) if i not in stub else "not_count"})
    cells = []
    carry: dict[int, str] = {}
    group: str | None = None
    previous: list[str] | None = None
    row_cells: list[dict[str, Any]] = []  # cells of the last row that had values
    after_row = False  # a row with values was read, and since then only its continuation lines (SEs, intervals)
    stub_left = min((t.x0 for t in body if band_of(t) in stub), default=x0)
    stub_right = max((t.x1 for t in body if band_of(t) in stub), default=x0)
    value_left = min((a for i, (a, _) in enumerate(bands) if i not in stub), default=x1)
    glyph_at = {g.offset: g for g in glyphs}
    label_right = x0  # right edge of the last label line read
    # Full-width rules inside the body, and the foot of the last line with values under the open group (D-206).
    inner_rules = [r[0] for r in all_rules if head[0] + 1 < r[0] < bottom[0] - 1 and r[2] - r[1] >= 0.6 * width
                   and _overlaps(r[1], r[2], x0, x1)]
    group_foot: float | None = None
    continuation = 1
    lines: list[list[tuple[int, dict[str, Any]]]] = []  # each printed line's value cells, with their bands

    def first_word_width(t: Token) -> float:
        word = t.text.split()[0] if t.text.split() else t.text
        first, last = glyph_at.get(t.start), glyph_at.get(t.start + len(word) - 1)
        return (last.x1 - first.x0) if first and last else t.x1 - t.x0

    def bracketed(s: str | None) -> bool:
        return bool(s) and s[0] + s[-1] in ("()", "[]")
    for n, row in enumerate(rows):
        by_band: dict[int, list[Token]] = {}
        for t in row:
            b = band_of(t)
            if b is not None:
                by_band.setdefault(b, []).append(t)
        values = [i for i in by_band if i not in stub]
        labels = [i for i in by_band if i in stub]
        line = sorted((t for i in labels for t in by_band[i]), key=lambda t: t.x0)
        wording = " ".join(t.text for t in line)
        # A bracketed line ("(0 = women, 1 = men)", "[1 = yes]") qualifies the row above or heads the rows below,
        # unless the open group is itself such a line ("(Men)" after "(Women)" names the next group; D-174).
        qualifier = bracketed(wording) and not bracketed(group)
        indented = bool(line) and line[0].x0 > stub_left + 3
        # A line wraps when its first word could not have fit on the line above: for a qualifier, only past the
        # value columns' edge shows it, since the widest label need not fill the stub column (D-203).
        wraps = indented and label_right + first_word_width(line[0]) > (value_left if qualifier else stub_right)
        opened = "with no group open" if group is None else f"under the group {group!r}"
        if labels and not values and after_row and qualifier and not (row_cells and wraps):
            # A qualifier is that row's or the heading of the rows below it ("(Managers only)"), set flush left or
            # indented, after the row's own line or its standard errors, and either reading binds the other's rows
            # under a wrong label; only a line that could not have fit on the row's own line shows it wraps.
            return Grid("unresolved", f"{wording!r} is set flush left under the row {previous} {opened}; "
                                      "it may qualify that row's label or head the rows below it" if not indented else
                        f"{wording!r} is indented under the row {previous} {opened}, and that row's label "
                        "does not fill its line; it may qualify that row's label or head the rows below it", bbox)
        if labels and values and bracketed(wording) and previous is not None and all(
                bracketed(re.sub(rf"[{MARKS}]+$", "", t.text)) for i in values for t in by_band[i]):
            # Bracketed values under a row may be its standard errors or intervals, and a bracketed label beside them
            # that row's qualifier: read as a row of their own they would leave the row they belong to (D-203).
            return Grid("unresolved", f"{wording!r} is set beside bracketed values under the row {previous}; they may "
                                      "be that row's standard errors or intervals and the line its qualifier", bbox)
        if labels and not values and row_cells and wraps:
            # An indented label-only line under a row with values is that row's wrapped label
            # ("Control Loss × Perceived" / "Harmful Intent") only when its first word could not have fit on
            # the line above: text wraps because the line is full. After a short label ("Trust") an indented
            # line is a sub-heading and names the group of the rows below it (under an open group, "(Men)" after
            # "(Women)" too), and a parenthetical one with no group open is refused above.
            for i in labels:
                carry[i] = f"{carry[i]} {' '.join(t.text for t in by_band[i])}" if i in carry else " ".join(t.text for t in by_band[i])
            previous = ([group] if group else []) + [carry[i] for i in stub if i in carry]
            for cell in row_cells:
                cell["position"]["row_index_or_path"] = previous
            label_right = max(t.x1 for t in line)
            continue
        if labels and not values:  # a flush-left label-only row names the panel or group of the rows below it
            group = " ".join(t.text for i in labels for t in by_band[i])
            carry.clear()
            previous = None
            row_cells = []
            after_row = False
            group_foot = None
            continue
        if group is not None and group_foot is not None and any(group_foot < y < min(t.y0 for t in row) for y in inner_rules):
            # A full-width rule under the rows of an open group, and then rows with no heading of their own (a table-wide
            # "Baseline Depvar (mean)" under the last panel): whether they are that group's or the whole table's the
            # page does not show (D-206).
            return Grid("unresolved", f"the row {' '.join(t.text for t in row)!r} is set under a full-width rule below the "
                                      f"rows of the group {group!r}, with no heading of its own; whether it belongs to "
                                      "that group or to the whole table is not shown", bbox)
        group_foot = max(t.y1 for t in row)
        after_row = True
        if stub and values and not labels and previous is not None:
            continuation += 1  # e.g. standard errors printed on the line under their coefficients
            path = previous + [f"(line {continuation})"]
            row_cells = []  # a wrapped label is read only directly under a labelled row
        else:
            continuation = 1
            for i in stub:
                if i in by_band:
                    carry[i] = " ".join(t.text for t in by_band[i])
                    if i == stub[0]:
                        for j in stub[1:]:
                            carry.pop(j, None)
            path = ([group] if group else []) + [carry[i] for i in stub if i in carry]
            if not path:
                path = [f"(row {n + 1})"]
            previous = path
            row_cells = []
            if labels:
                label_right = max(t.x1 for i in labels for t in by_band[i])
        lines.append([])
        for i, toks in by_band.items():
            if i in stub:
                continue
            cell_text = " ".join(t.text for t in toks)
            marks = SIG.search(cell_text)
            cell = {"position": {"row_index_or_path": path, "column_id": f"c{i}", "header_path": columns[i]["header_path"]},
                    "text": cell_text, "bbox": [round(min(t.x0 for t in toks), 2), round(min(t.y0 for t in toks), 2),
                                                round(max(t.x1 for t in toks), 2), round(max(t.y1 for t in toks), 2)]}
            if marks:
                cell["significance_markers"] = [marks.group(1)]
            if len(toks) == 1:
                cell["span"] = (toks[0].start, toks[0].end)
            cells.append(cell)
            lines[-1].append((i, cell))
            if continuation == 1:
                row_cells.append(cell)
    # Two printed lines bound to one row path under one column are two rows the reading could not tell apart
    # (a heading taken for a wrapped label files the next block's rows under the block above).
    bound: dict[tuple[tuple[str, ...], str], str] = {}
    for cell in cells:
        pos = cell["position"]
        key = (tuple(pos["row_index_or_path"]), pos["column_id"])
        if key in bound:
            return Grid("unresolved", f"{bound[key]!r} and {cell['text']!r} are both bound to row {pos['row_index_or_path']} "
                                      f"under {pos['header_path']}; the rows cannot be told apart", bbox)
        bound[key] = cell["text"]
    shifted = _shifted_values(columns, lines, stub)
    if shifted:
        return Grid("unresolved", shifted, bbox)
    separated = all(any(max(t.y1 for t in a) < y < min(t.y0 for t in b) for y in inner_rules)
                    for a, b in zip(rows, rows[1:]))
    return Grid("single_page_resolved", "", bbox, columns, cells, stub=stub, rows_separated=separated)


def _shifted_values(columns: list[dict[str, Any]], lines: list[list[tuple[int, dict[str, Any]]]],
                    stub: list[int]) -> str | None:
    """Why a read grid's values cannot stand where they were bound, or None. A value with a bracketed
    number under a column headed by a single-number statistic ("t"), or a line whose values sit under
    different sub-heads in column groups that repeat the same sub-heads (Model 2 "t" but Model 3
    "B (SE)"), is a row printed shifted across its columns (Mell Table 2); where each value belongs
    would be a guess. Heads the check does not know, and tables without repeated groups, skip it."""
    groups: dict[tuple[str, ...], list[int]] = {}
    for i, col in enumerate(columns):
        if i not in stub and len(col["header_path"]) > 1:
            groups.setdefault(tuple(col["header_path"][:-1]), []).append(i)
    for line in lines:
        if not line:
            continue
        row = line[0][1]["position"]["row_index_or_path"]
        for i, cell in line:
            head = columns[i]["header_path"]
            if head and head[-1].strip().lower() in SINGLE_NUMBER_HEADERS and WITH_BRACKETED.match(cell["text"]):
                return (f"{cell['text']!r} in row {row} is a value with a bracketed number under {head}, "
                        "a single-number column; the row's values may be shifted across columns")
        filled = {i for i, cell in line if not re.fullmatch(r"[—–-]", cell["text"])}  # a dash holds no value
        seen: dict[tuple[str, ...], tuple[tuple[str, ...], tuple[int, ...]]] = {}  # sub-heads -> first group, its filled
        for group, members in groups.items():
            subheads = tuple(columns[i]["header_path"][-1] for i in members)
            at = tuple(k for k, i in enumerate(members) if i in filled)
            if not at:
                continue
            first, first_at = seen.setdefault(subheads, (group, at))
            if first_at != at:
                return (f"row {row} has values under {[subheads[k] for k in first_at]} in {list(first)} but under "
                        f"{[subheads[k] for k in at]} in {list(group)}; the row's values may be shifted across columns")
    return None


def _legend(members: list[Glyph], text: str) -> bool:
    """A block that is a footnote legend: "* p < .05", "⁎⁎⁎ p < 0.001.", or a footnote letter set as a
    superscript ("ᵃ Difference between ..."); a block of body text that starts "a large ..." is not one, nor is
    an author footnote that opens with a mark ("* Corresponding author ...")."""
    first, *rest = sorted(members, key=lambda g: g.offset)
    if not first.ch.isalpha():
        return bool(LEGEND.match(text, first.offset))
    if not FOOTNOTE.match(text, first.offset):
        return False
    after = next((g for g in rest if not g.ch.isspace()), None)
    return after is not None and first.y1 - first.y0 < 0.85 * (after.y1 - after.y0)


def _table_notes(glyphs: list[Glyph], text: str, x0: float, x1: float, bottom_y: float,
                 body: float = 0.0) -> list[tuple[int, int, str]]:
    """The note under the bottom rule: the topmost text block within 60 pt below it that overlaps the table's
    columns and starts "Note."/"Notes:" or is a footnote legend (Mell Table 3's "* p < .05.", Colby Table 1's
    superscript a-d), plus the footnote blocks ("a ...", "* p < .05") stacked directly under it that start inside
    the table's columns, and a block that starts there within a line under it and overlaps it across (its text set
    apart from a "Notes:" label, as a float's note carries on; D-159). A note set at the page margin under a centred
    table overlaps it (Bondi A1-A17); a block stacked under that note at the margin may be body text or an author
    footnote. Body text that happens to sit
    within reach of the rule is not note text ("Note that ..." has no stop or colon), and in a table printed
    sideways the caption can fall between note blocks in text order. The topmost start is taken: text order
    follows each block's bottom edge, so a one-line legend beside a taller note comes before it.
    As for a float's notes in the section tree (D-156, D-159; D-195): the note is reached from the table, so none is
    taken past a caption or a paragraph of running text (two or more lines, all in the page's `body` size) under
    the bottom rule; a stacked block is set smaller than body text and in the size of the note above it; and a note
    set smaller than body text ends where its block runs on into lines in the body size."""
    notes = []
    blocks: dict[int, list[Glyph]] = {}
    for g in glyphs:
        blocks.setdefault(g.block, []).append(g)

    def overlaps(members: list[Glyph]) -> bool:
        return _overlaps(min(g.x0 for g in members), max(g.x1 for g in members), x0, x1)

    def lines(members: list[Glyph]) -> list[list[Glyph]]:
        out: dict[int, list[Glyph]] = {}
        for g in members:
            if not g.ch.isspace():
                out.setdefault(text.rfind("\n", 0, g.offset), []).append(g)
        return list(out.values())

    def size(members: list[Glyph]) -> float:
        return max((g.size for g in members if not g.ch.isspace()), default=0.0)

    def running(members: list[Glyph]) -> bool:
        return bool(body) and len(lines(members)) >= 2 and all(abs(size(ln) - body) <= 0.3 for ln in lines(members))

    def first_line(members: list[Glyph]) -> str:
        a = min(g.offset for g in members)
        z = text.find("\n", a)
        return text[a:z if z >= 0 else len(text)]
    starts = [b for b, members in blocks.items()
              if bottom_y < min(members, key=lambda g: g.offset).cy < bottom_y + 60 and overlaps(members)
              and (NOTE_START.match(text, min(g.offset for g in members)) or _legend(members, text))]
    if not starts:
        return notes
    chosen = [min(starts, key=lambda b: min(g.y0 for g in blocks[b]))]
    top_, edge = min(g.y0 for g in blocks[chosen[0]]), max(g.y1 for g in blocks[chosen[0]])
    if any(bottom_y < min(g.y0 for g in members) < top_ and overlaps(members)
           and (is_caption_line(first_line(members)) or running(members)) for members in blocks.values()):
        return notes
    for b, members in sorted(blocks.items(), key=lambda kv: min(g.y0 for g in kv[1])):
        b_top = min(g.y0 for g in members)
        if b in chosen or not top_ < b_top <= edge + 12 or not x0 - 1 <= min(g.x0 for g in members) <= x1 + 1:
            continue
        # Besides a footnote block, a block within a line under the note that overlaps it across carries it on, as a
        # float's note does (D-159): the note's text set as a block of its own under "Notes:" (KS-DEV-05 Table I).
        last = blocks[chosen[-1]]
        tail = max(lines(last), key=lambda ln: min(g.offset for g in ln))
        carried = bool(body) and not is_caption_line(first_line(members)) \
            and -2 <= b_top - max(g.y1 for g in last) <= max(g.y1 for g in tail) - min(g.y0 for g in tail) \
            and _overlaps(min(g.x0 for g in members), max(g.x1 for g in members),
                          min(g.x0 for g in last), max(g.x1 for g in last))
        if (FOOTNOTE.match(text, min(g.offset for g in members)) or carried) and (
                not body or (size(members) <= body - 0.3 and abs(size(members) - size(last)) <= 0.3)):
            chosen.append(b)
            top_, edge = b_top, max(edge, max(g.y1 for g in members))
    for b in chosen:
        # A note set smaller than body text ends at a line in the body size that its block runs on into: running text
        # printed close under it (D-195).
        kept = []
        for ln in sorted(lines(blocks[b]), key=lambda ln: min(g.offset for g in ln)):
            if kept and body and size(kept[0]) <= body - 0.3 and size(ln) > body - 0.3:
                break
            kept.append(ln)
        a, z = min(g.offset for g in blocks[b]), max(g.offset for g in blocks[b]) + 1
        if len(kept) < len(lines(blocks[b])):
            z = max(g.offset for ln in kept for g in ln) + 1
        notes.append((a, z, text[a:z]))
    return notes


def _panel_triples(rules: list[tuple[float, float, float]]):
    """Groups consecutive full-width rules into (top, header, bottom) triples, each of one extent (the
    same tolerance as a single table's top/head/bottom match). None when the count is not a multiple of
    three (at least two panels) or a triple's own rules do not share an extent: not a clean panel
    structure, so the caller keeps today's single-ruled-table behaviour."""
    if len(rules) < 6 or len(rules) % 3:
        return None
    triples = [tuple(rules[i:i + 3]) for i in range(0, len(rules), 3)]
    for top, head, bottom in triples:
        if not (abs(head[1] - top[1]) < 3 and abs(head[2] - top[2]) < 3
                and abs(bottom[1] - top[1]) < 3 and abs(bottom[2] - top[2]) < 3):
            return None
    return triples


def _panel_title(glyphs: list[Glyph], text: str, y_lo: float, y_hi: float, x0: float, x1: float,
                 caption_start: int, caption_end: int):
    """The text between the previous panel (or the caption) and this panel's top rule, in the panel's
    own column span ("Panel A: Study 1"); None when there is no such line."""
    hits = sorted((g for g in glyphs if y_lo - 1 < g.cy < y_hi and x0 - 5 <= g.x0 <= x1 + 5
                   and not caption_start <= g.offset < caption_end and not g.ch.isspace()), key=lambda g: g.offset)
    if not hits:
        return None
    a, b = hits[0].offset, hits[-1].offset
    return a, b + 1, text[a:b + 1]


def _multi_panel(glyphs: list[Glyph], rules: list[tuple[float, float, float]], text: str,
                 triples: list[tuple] | None, width: float,
                 caption_start: int, caption_end: int, cap_bottom: float, body: float = 0.0,
                 suspect: bool = False) -> Grid | None:
    """Several complete ruled panels stacked under one caption (Bondi Tables A2, A18; D-037): each is
    interpreted with the single-panel body logic, its columns namespaced by panel and its row paths
    prefixed with its panel's title, and it is owned by the study its title names alone. Returns None
    (the caller falls back to today's single-ruled-table interpretation) unless every panel's rules form
    a clean triple and every panel's title names exactly one study, distinct from every other panel's.
    A panel refused for values it cannot bind, its region located (a bbox), leaves the whole table
    unresolved with that panel's reason, and every panel keeps its region and its study."""
    if triples is None:
        return None
    panels: list[tuple[tuple, tuple, tuple, tuple[int, int, str], str]] = []
    prev_y = cap_bottom
    for top, head, bottom in triples:
        title = _panel_title(glyphs, text, prev_y, top[0], top[1], top[2], caption_start, caption_end)
        if title is None:
            return None
        labels = _caption_studies(title[2])
        if len(labels) != 1:
            return None
        panels.append((top, head, bottom, title, labels[0]))
        prev_y = bottom[0]
    if len({label for *_, label in panels}) != len(panels):
        return None
    columns: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    meta: list[dict[str, Any]] = []
    refused = ""
    for pi, (top, head, bottom, title, label) in enumerate(panels):
        grid = _panel_body(glyphs, rules, text, top, head, bottom, width, suspect)
        if grid.status != "single_page_resolved":
            if not grid.bbox:
                return None
            refused = refused or f"panel {title[2]!r}: {grid.reason}"
        prefix = f"p{pi}"
        for col in grid.columns:
            columns.append({**col, "column_id": f"{prefix}-{col['column_id']}"})
        for cell in grid.cells:
            pos = cell["position"]
            new_cell = dict(cell)
            new_cell["position"] = {**pos, "column_id": f"{prefix}-{pos['column_id']}",
                                    "row_index_or_path": [title[2]] + list(pos["row_index_or_path"])}
            cells.append(new_cell)
        meta.append({"panel_id": prefix, "title": title[2], "bbox": grid.bbox, "title_span": title[:2],
                     "owner_study_local_id": f"study:{label}", "ownership_basis": "explicit_single_study_caption"})
    x0 = min(p["bbox"][0] for p in meta)
    x1 = max(p["bbox"][2] for p in meta)
    last_x0, _, last_x1, last_bottom = meta[-1]["bbox"]
    notes = _table_notes(glyphs, text, last_x0, last_x1, last_bottom, body)
    if refused:
        return Grid("unresolved", refused, [x0, meta[0]["bbox"][1], x1, last_bottom], notes=notes, panels=meta)
    if not any(NUMERIC.match(c["text"]) for c in cells):
        return None
    return Grid("single_page_resolved", "", [x0, meta[0]["bbox"][1], x1, last_bottom], columns, cells, notes,
                panels=meta)


def _line_starts(glyphs: list[Glyph], text: str) -> list[tuple[Glyph, str, float, float]]:
    """Each captured line's first glyph, with the line's text and its glyphs' horizontal extent."""
    ends: dict[int, list[float]] = {}
    starts: dict[int, Glyph] = {}
    for g in glyphs:
        a = text.rfind("\n", 0, g.offset) + 1
        if g.offset == a:
            starts[a] = g
        if not g.ch.isspace():
            x = ends.setdefault(a, [g.x0, g.x1])
            x[0], x[1] = min(x[0], g.x0), max(x[1], g.x1)
    out = []
    for a, g in starts.items():
        z = text.find("\n", a)
        out.append((g, text[a:z if z >= 0 else len(text)], *ends.get(a, [g.x0, g.x1])))
    return out


def _interpret_upright(glyphs: list[Glyph], all_rules: list[tuple[float, float, float]], text: str,
                       caption_start: int, caption_end: int, foreign=(), body: float = 0.0, suspect: bool = False,
                       allow_text_values: bool = False) -> Grid:
    cap = _caption_glyphs(glyphs, caption_start, caption_end, foreign)
    cap_bottom = max(g.y1 for g in cap)
    cap_x0, cap_x1 = min(g.x0 for g in cap), max(g.x1 for g in cap)
    # The table's rules lie under its caption: not beside it (a table set in the next column), and none that lies
    # under a later float caption on the page, across it (a wider table lower down, in either column), which is that
    # caption's table's (D-194). Short rules under its column heads need not lie under the caption, only under the
    # heads they group.
    lines = _line_starts(glyphs, text)
    later = [(g.y0, lx0, lx1) for g, line, lx0, lx1 in lines if g.y0 > cap_bottom and is_caption_line(line)]
    under = [r for r in all_rules if r[0] > cap_bottom - 1
             and not any(y < r[0] and _overlaps(r[1], r[2], lx0, lx1) for y, lx0, lx1 in later)]
    rules = [r for r in under if _overlaps(r[1], r[2], cap_x0 - 5, cap_x1 + 5)]
    if not rules:
        return Grid("unresolved", "no horizontal rules below the caption", table_top=float("inf"))
    width = max(r[2] - r[1] for r in rules)
    full = [r for r in rules if r[2] - r[1] >= 0.6 * width]
    if len(full) < 3:  # the first rule may be the one under the column heads: where the table starts is unknown
        return Grid("unresolved", f"expected top, header and bottom rules; found {len(full)} full-width rule(s)",
                    table_top=None if full else float("inf"))
    top = full[0]
    beside = [r for r in full if not _overlaps(r[1], r[2], top[1], top[2])]
    if beside:  # a caption across two tables set side by side: whose rules are its own the page does not show
        return Grid("unresolved", f"full-width rules at x {top[1]:.0f}-{top[2]:.0f} and {beside[0][1]:.0f}-"
                                  f"{beside[0][2]:.0f} lie side by side under the caption; which table is its own "
                                  "is not determinable")
    # The table ends at the last rule with the top rule's extent before the next caption or note line ("TABLE 2",
    # "NOTE."; not "Note that ..."); rules in between separate panels (esttab-style regression tables).
    stop = min([g.y0 for g, line, _, _ in lines if g.y0 > top[0] and g.offset > caption_end
                and (is_caption_line(line) or NOTE_START.match(line))] + [float("inf")])
    triples = _panel_triples([r for r in full if r[0] < stop + 1])
    # A caption's own title lies above where the table starts: its top rule, or the title over its first panel.
    first = triples and _panel_title(glyphs, text, cap_bottom, *triples[0][0], caption_start, caption_end)
    table_top = min(g.y0 for g in glyphs if first[0] <= g.offset < first[1]) if first else top[0]
    panels = _multi_panel(glyphs, under, text, triples, width, caption_start, caption_end, cap_bottom, body, suspect)
    if panels is not None:
        panels.table_top = table_top
        return panels
    same = [r for r in full if abs(r[1] - top[1]) < 3 and abs(r[2] - top[2]) < 3 and top[0] < r[0] < stop + 1]
    if len(same) < 2:  # nor here: the first rule found need not be a top rule
        return Grid("unresolved", "no header and bottom rules matching the top rule's extent")
    head, bottom = same[0], same[-1]
    grid = _panel_body(glyphs, under, text, top, head, bottom, width, suspect, allow_text_values)
    # The note sits under the located bottom rule whether or not the body could be read.
    notes = _table_notes(glyphs, text, min(top[1], head[1], bottom[1]), max(top[2], head[2], bottom[2]), bottom[0], body)
    # Caption lines printed as their own blocks between the label and the top rule (a sideways table's title),
    # read whether or not the body could be: a refused table keeps the study its title names.
    title_glyphs = sorted((g for g in glyphs if cap_bottom - 1 < g.cy < top[0] and top[1] - 5 <= g.x0 <= top[2] + 5
                           and not caption_start <= g.offset < caption_end and not g.ch.isspace()), key=lambda g: g.offset)
    runs: list[list[int]] = []
    for g in title_glyphs:
        if runs and g.offset - runs[-1][1] <= 2:
            runs[-1][1] = g.offset
        else:
            runs.append([g.offset, g.offset])
    title = [(a, b + 1, text[a:b + 1]) for a, b in runs]
    if grid.status != "single_page_resolved":
        grid.notes, grid.title, grid.table_top = notes, title, table_top
        return grid
    if not allow_text_values and not any(NUMERIC.match(c["text"]) for c in grid.cells):
        return Grid("unresolved", "no numeric value cells recovered between the rules", notes=notes, table_top=table_top)
    return Grid("single_page_resolved", "", grid.bbox, grid.columns, grid.cells, notes, title=title, stub=grid.stub,
                table_top=table_top, rows_separated=grid.rows_separated)
