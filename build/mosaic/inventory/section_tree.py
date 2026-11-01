"""Deterministic section tree over Layer-1 text, in reading order (FINAL v2 §6.1 pass 1; audit M-58).

Adapted from K/tools/build_section_tree_from_pages.py, section_tree_common.py
and validate_section_tree.py (REUSE_LEDGER R015-R017). Kept: is_noise_heading,
normalize_title/title_tokens, the rule that a node runs until the next heading
of the same or a higher level, and the child-inside-parent check. Changed: K
parsed Markdown `#` headings from its own page-*.md files; here headings are
lines of captured PDF blocks (or DOCX paragraphs) whose text has a recognized
role (E's heading roles) and whose typography sets them apart from body text,
walked in each page's recorded reading order so a two-column page is read
column by column. Node extents are code-point segments of text artifacts, not
whole pages, so two studies on one page get separate regions. Nothing reads
or writes kb.db or K's outputs.

Section numbers: a number printed as its own line ("2", "|", "LITERATURE AND
THEORY"; "A.3" / "Study 2") is read with the title after it on the same
printed line, and lettered and Roman numbering take their level from the
numbering already open (a list that starts inside another style nests under
it). A number that could continue open lists at two levels continues the one
whose heading it is set like, or else the one that keeps an open study whole,
and a study whose region depends on that choice records it. A heading with no
recognized role that is typeset like this source's level-1 headings (size,
weight, case and alignment) and unlike its lower ones is a level-1 heading;
when it ends a study, both the study and the new section record that the cut
rests on typography alone.

Page furniture and footnotes are no section's text. Running heads, footers
and folios (text repeated in the top or bottom band of at least three pages)
and text set vertically in a side margin of as many pages (download stamps,
copyright notices), none of it a caption or a heading, go to a "(page
furniture)" node. A footnote (small type at the foot of its
column, starting with a number or note symbol) goes to the node whose text
holds its marker on the same page, and to a "(footnotes not attributed)"
node when the marker is in more than one section, or in none while the page
holds more than one.

A table or figure keeps its notes: a "Note."/"Notes:" or significance-legend
block below its caption, reached through the table or figure itself (never
across a paragraph of running text), is the float's text, not the text of the
section open around it, so it goes wherever the caption goes.

A page whose native text is empty or quarantined, or that capture read by
OCR because its only text is a stamp in the margins of a page scan, is read
through its OCR derivative. That text is recognized, not native, and has no typography, so
its headings, captions and notes are not read and its place does not show
which section it belongs to: it goes to a "(recognized text)" node. When the
flow resumes after such a page, or after a page from which no text was read
at all (OCR failed or did not run), the item that holds the text read next (the
open studies, or with none the level-1 section) records that its region runs
on past headings that were not read, if it was open before (D-187, D-199).

Captions set side by side on one baseline share a block: each line of such a
block belongs to the caption whose column holds it (D-193).
"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from mosaic.adapted.e.headings import SECTION_NUMBER_RE, heading_role, is_heading_text, split_section_number, study_label

WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9'-]*")
STOPWORDS = {"a", "an", "and", "are", "as", "at", "by", "for", "from", "how", "in", "into", "is", "it", "of", "on",
             "or", "the", "to", "with"}
GENERAL_DISCUSSION = re.compile(r"^(general discussion|discussion and conclusions?|conclusions?|concluding remarks)$", re.I)
EXPLICIT_GENERAL = re.compile(r"^(?:\d+(?:\.\d+)*\.?\s+)?(general discussion|discussion and conclusions?)$", re.I)
# A float label's own line: the label, then either (1) nothing else, (2) a punctuation mark before the
# title -- possibly with whitespace first, so a Nature-style "Fig. 1 | Title" legend still counts, but a
# period is never immediately followed (across any whitespace) by a second period, which would be a
# list-of-tables/contents dot leader ("Table 1 . . . . 24", "Table 1.......... 24") and not a caption --
# or (3) a "(continued)"/"--Continued" marker and then the line ends, so an ordinary sentence that happens
# to use "continued" as a verb ("Table 1 continued to show...") is not swept in. ALL-CAPS labels ("TABLE 1",
# "FIGURE 2") are matched as their own literal alternatives, not case-insensitively, so running text starting
# with a lowercase label word is never accepted (D-064). The label may follow an appendix or supplement
# prefix ("Online Appendix Table I", "Supplementary Table S2", "Extended Data Fig. 1"), and its number is
# Arabic after at most three capitals ("3", "S1", "A10", "EC1") or a Roman numeral up to XXXIX with at most
# one capital after it that is not itself a Roman digit ("I", "IV", "IIA"; not "IIII" or "VX"). The prefix, the
# table words and the number are this grammar's parts, which the table reader shares (`degrade.TABLE_CAPTION`, D-206).
FLOAT_PREFIX = r"(?:Online\s+Appendix|Appendix|Supplementary|Supplemental|Extended\s+Data)\s+"
TABLE_WORD = r"Table|TABLE"
FLOAT_NUMBER = r"[A-Z]{0,3}\d+[A-Za-z]?|X{0,3}(?:IX|IV|V?I{0,3})(?<=[IVX])(?![IVX])[A-Z]?"
FLOAT_LABEL = re.compile(
    rf"^(?:{FLOAT_PREFIX})?"
    rf"(?:{TABLE_WORD}|Figure|FIGURE|Fig\.|FIG\.|Exhibit|EXHIBIT)\s*"
    rf"(?:{FLOAT_NUMBER})"
    r"(?:\s*$"
    r"|\s*(?:\.(?!\s*\.)|[:|])\s*"
    r"|(?:\s+|\s*[-–—]\s*)\(?(?i:cont(?:inued|'d|\.))\)?\.?\s*$"
    r")"
)
# A list-of-tables/contents entry ends in a dot leader and a page number whatever separates its label from its
# title ("Table 3. Effect of X on Y ....... 42", "Table 1: Descriptives . . . 24"); such a line is never a caption.
TOC_LEADER = re.compile(r"(?:\.\s*){3,}\d+\s*$")


def is_caption_line(text: str) -> bool:
    return bool(FLOAT_LABEL.match(text.strip())) and not TOC_LEADER.search(text)


def caption_column(x0: float, label_x0s: list[float]) -> int:
    """Of captions set side by side in one block (their label lines starting at `label_x0s`), the one whose column
    holds a line starting at `x0`: the rightmost label starting at or left of it (within 2 pt), else the leftmost."""
    order = sorted(range(len(label_x0s)), key=lambda k: label_x0s[k])
    at = [k for k in order if label_x0s[k] <= x0 + 2]
    return at[-1] if at else order[0]


# The first line of a table or figure note: "Note."/"Notes:" (not "Note that ..."), or a significance legend
# ("* p < .05", "⁎⁎⁎ p < 0.001").
FLOAT_NOTE = re.compile(r"^\s*(?:(?:[Nn]otes?|NOTES?)\s*[.:]|[*∗⁎†‡]+\s*p\s*[<=>≤])")


CAPTION_STUDY = re.compile(r"\b(stud(?:y|ies)|experiments?|st\.)\s*(\d+[a-z]?(?:\s*(?:,|and|&|-|–)\s*\d+[a-z]?)*)", re.I)
CAPTION_STUDY_RANGE = re.compile(r"(\d+)(?![a-z])\s*(?:-|–)\s*(\d+)(?![a-z])", re.I)
URLISH = re.compile(r"https?://|www\.|doi\.org|\bdoi:", re.I)
APPENDIX_LABEL = re.compile(r"^(?:appendix|appendices)(?:\s+(?:[A-Z]\d*|\d+|[IVXL]+))?$", re.I)  # the word and a label only
# the word, a label, ":" or "." and a title with no sentence punctuation or brackets: "Appendix B: Pilot Study Materials"
APPENDIX_TITLED = re.compile(r"^(?:appendix|appendices)\s+(?:[A-Z]\d*|\d+|[IVXL]+)\s*[:.]\s+[A-Z][^.;:!?()\[\]]*[^.;:!?()\[\],\s]$", re.I)
FRONT_ROLES = ("abstract", "introduction", "theory")


def clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def normalize_title(value: Any) -> str:
    text = clean(value).lower().replace("&", " and ")
    text = re.sub(r"['’`]", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def title_tokens(value: Any) -> set[str]:
    return {t.lower() for t in WORD_RE.findall(clean(value)) if len(t) > 2 and t.lower() not in STOPWORDS}


def is_noise_heading(title: str) -> bool:
    """K's filter, unchanged."""
    normalized = normalize_title(title)
    if not normalized or normalized.isdigit() or len(normalized) <= 2:
        return True
    raw_words = title.split()
    if len(raw_words) > 4 and len(raw_words[0]) == 1:
        return True
    useful = title_tokens(title)
    return not useful or len("".join(useful)) < 5


@dataclass
class Segment:
    text_region_id: str
    physical_index: int | None
    start: int
    end: int


@dataclass
class Node:
    node_id: str
    source_id: str
    source_label: str
    section: str
    title: str
    role: str
    level: int
    numbered: bool
    study: tuple[str, str] | None
    heading: Segment | None
    typography: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None
    caption: str = ""
    float_studies: list[str] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    number: tuple[str, tuple[int, ...]] | None = None  # (numbering style, key), e.g. ("letter", (3, 2)) for "C.2"
    caption_lines: list[Segment] = field(default_factory=list)  # a float's own caption lines, label line first
    continues: str = ""  # a continued table float that takes its studies from the table it continues: that label


@dataclass
class SourceTree:
    source_id: str
    source_label: str
    role: str
    body_size: float
    nodes: list[Node]
    asides: list[Segment] = field(default_factory=list)  # page furniture, footnotes and OCR text: no section's flow
    problems: list[str] = field(default_factory=list)    # the tree's own structural checks (validate_tree)
    unread: list[int] = field(default_factory=list)      # pages from which no text was read, native or OCR
    # Each table whose ruled region the table module located, read or refused: its label and the blocks inside the
    # region (`record._region_claims`), whether they were claimed or stay where they print (D-207).
    table_rows: list[tuple[str, list[Segment]]] = field(default_factory=list)

    def descendants(self, node: Node) -> list[Node]:
        out, frontier = [], [node.node_id]
        while frontier:
            kids = [n for n in self.nodes if n.parent_id in frontier]
            out.extend(kids)
            frontier = [k.node_id for k in kids]
        return out

    def claim(self, node: Node, ranges: list[Segment]) -> None:
        """Move text ranges (for example a table body placed inside another study's flow) to `node`,
        cutting them out of every other node's segments."""
        for other in self.nodes:
            if other is node:
                continue
            kept: list[Segment] = []
            for s in other.segments:
                pieces = [(s.start, s.end)]
                for r in ranges:
                    if r.text_region_id != s.text_region_id:
                        continue
                    nxt = []
                    for a, b in pieces:
                        if r.end <= a or r.start >= b:
                            nxt.append((a, b))
                            continue
                        if a < r.start:
                            nxt.append((a, r.start))
                        if r.end < b:
                            nxt.append((r.end, b))
                    pieces = nxt
                kept.extend(Segment(s.text_region_id, s.physical_index, a, b) for a, b in pieces if b > a)
            other.segments = kept
        node.segments = merge_segments(node.segments + list(ranges))

    def extent(self, node: Node, skip_studies: bool = False, skip: set[str] | frozenset[str] = frozenset()) -> list[Segment]:
        """The node's own segments plus every descendant's, merged where contiguous. A caption block
        that names a study is left out: studies.py attributes it to the study it names. With
        `skip_studies`, a study nested in this node ("3 Experiments" > "3.1 Study 1") is left out too,
        with everything under it, and so is a descendant whose id is in `skip`."""
        segs = list(node.segments)
        skipped: set[str] = set()
        for d in self.descendants(node):
            if d.parent_id in skipped or (skip_studies and d.study) or d.node_id in skip:
                skipped.add(d.node_id)
            elif not d.float_studies:
                segs.extend(d.segments)
        return merge_segments(segs)


def merge_segments(segs: list[Segment]) -> list[Segment]:
    order: list[str] = []
    for s in segs:
        if s.text_region_id not in order:
            order.append(s.text_region_id)
    out: list[Segment] = []
    for tr in order:
        mine = sorted((s for s in segs if s.text_region_id == tr and s.end > s.start), key=lambda s: s.start)
        for s in mine:
            if out and out[-1].text_region_id == tr and out[-1].end >= s.start:
                out[-1].end = max(out[-1].end, s.end)
            else:
                out.append(Segment(tr, s.physical_index, s.start, s.end))
    return out


# ------------------------------------------------------------------------------ reading sequence

@dataclass
class Line:
    text_region_id: str
    physical_index: int | None
    section: str
    block_start: int
    block_end: int
    start: int
    end: int
    text: str
    typography: dict[str, Any]
    first_in_block: bool
    bbox: list[float] | None = None


def _left_edges(width: float, starts: list[tuple[float, int]]) -> tuple[float | None, float | None]:
    """A column's left edge in each half of a page: the line start (within 1 pt) under which the most
    characters begin, so that a text column outweighs table cells, figure labels and stray numbers."""
    edges = []
    for lo, hi in ((0.0, width / 2), (width / 2, float("inf"))):
        half = sorted((x, n) for x, n in starts if lo <= x < hi)
        xs = [x for x, _ in half]
        chars = [0]
        for _, n in half:
            chars.append(chars[-1] + n)
        best = max(((chars[bisect_right(xs, x + 1)] - chars[bisect_left(xs, x - 1)], -x) for x in xs), default=None)
        edges.append(-best[1] if best else None)
    return edges[0], edges[1]


def _reading_lines(objects: dict[str, dict[str, Any]], source_id: str) -> list[Line]:
    """Every line of a source's native text, page by page, blocks in recorded reading order.
    Quarantined text is skipped (it is not citable); an OCR derivative stands in for it, for a page whose
    native text is empty, and for a page capture read by OCR because its native text is no layer of what the
    page shows (a stamp in the margins of a page scan, D-189)."""
    pages = sorted(((oid, o) for oid, o in objects.items() if o["type"] == "page" and o["source_id"] == source_id),
                   key=lambda item: item[1]["physical_index"])
    regions = [o for o in objects.items() if o[1]["type"] == "text-region" and o[1]["source_id"] == source_id]
    used_ocr = {o["ocr"]["text_region_id"] for o in objects.values() if o["type"] == "disposition"
                and o.get("subject_kind") == "page" and (o.get("ocr") or {}).get("text_region_id")}
    lines: list[Line] = []
    if not pages:  # DOCX/XLSX: one text region, one "line" per paragraph
        for tr_id, tr in regions:
            if tr["status"] != "captured":
                continue
            by_index = {b["block_index"]: b for b in tr["blocks"]}
            for bi in tr.get("reading_order") or sorted(by_index):
                b = by_index[bi]
                text = tr["text"][b["start"]:b["end"]]
                lines.append(Line(tr_id, None, "supplement", b["start"], b["end"], b["start"], b["end"],
                                  text.rstrip("\n"), {"bold": b.get("bold", False), "style": b.get("style"),
                                                      "list_item": b.get("list_item", False), "size": 0.0, "italic": False,
                                                      "table_cell": b.get("kind") == "table_cell"},
                                  True))
        return lines
    for page_id, page in pages:
        section = page["printed_label"].get("section") or "main_body"
        candidates = [(tid, tr) for tid, tr in regions if tr.get("page_id") == page_id and tr["status"] == "captured"]
        if not candidates:
            continue
        # Prefer the native layer; an OCR derivative is used only when the native text is quarantined or empty, or
        # capture used it for the page.
        candidates.sort(key=lambda item: (item[0] not in used_ocr, not item[1]["text"].strip(),
                                          item[1]["derivation"]["kind"] != "native_raw"))
        tr_id, tr = candidates[0]
        by_index = {b["block_index"]: b for b in tr["blocks"]}
        # Alignment: whether a line starts at (or left of) its column's left edge ("flush") or to its right
        # (centred, indented).
        crop = page["geometry"]["cropbox"]  # the frame of the captured bboxes (D-061)
        edges = _left_edges(crop[2] - crop[0], [(ln["bbox"][0], len(tr["text"][ln["start"]:ln["end"]].strip()))
                                                for b in tr["blocks"] for ln in b.get("lines") or [] if ln.get("bbox")])
        for bi in tr.get("reading_order") or sorted(by_index):
            b = by_index[bi]
            if b.get("kind") == "image_placeholder":
                continue
            block_lines = b.get("lines") or [{"start": b["start"], "end": b["end"]}]
            for li, ln in enumerate(block_lines):
                typo = {k: ln.get(k) for k in ("size", "bold", "italic")}
                bbox = ln.get("bbox")
                if bbox:
                    edge = edges[0] if bbox[0] < (crop[2] - crop[0]) / 2 else edges[1]
                    typo["flush"] = edge is not None and bbox[0] <= edge + 2
                lines.append(Line(tr_id, page["physical_index"], section, b["start"], b["end"], ln["start"], ln["end"],
                                  tr["text"][ln["start"]:ln["end"]], typo, li == 0, bbox))
    return lines


def body_font_size(lines: list[Line]) -> float:
    weight: Counter[float] = Counter()
    for ln in lines:
        size = ln.typography.get("size") or 0.0
        if size:
            weight[size] += len(ln.text.strip())
    return weight.most_common(1)[0][0] if weight else 0.0


def _split_number(line: Line) -> tuple[str | None, str]:
    """`split_section_number` of a line, except that an Arabic number followed by a space only ("2.1 Participants")
    is read only on the first line of a block: a later line that starts with one ("5 years will report ...") is
    text wrapped onto that line, not a section number."""
    text = line.text.strip()
    number, title = split_section_number(text)
    if number and number[0].isdigit() and not line.first_in_block and not SECTION_NUMBER_RE.match(text):
        return None, text
    return number, title


def heading_of(line: Line, body_size: float, docx: bool, number: str | None = None, own_block: bool = False) -> str | None:
    """Return the heading role of a line, or None. Role headings need a recognized role and either a
    section number or typography distinct from body text (bold, italic or larger); a heading is never
    smaller than body text and never carries a URL. Bold short lines without a role are generic
    subsection headings. Only an Arabic number followed by ". " or "|" on the line itself stands in for
    distinct typography; other numbering (`_split_number`), and a `number` printed as its own line
    before this one, do not. The role, noise and punctuation tests read the title without its number. An
    explicit general-discussion heading set as a block of its own needs no distinct typography, and an
    appendix heading set in body type is one only when it opens its block and is nothing but the word and a
    label ("APPENDIX", "Appendix B"), which no sentence is (D-167)."""
    text = line.text.strip()
    if not text or len(text) > 150 or URLISH.search(text):
        return None
    typo = line.typography
    if docx:
        if typo.get("list_item"):
            return None
        distinct = bool(typo.get("bold")) or (typo.get("style") or "").lower().startswith(("heading", "title"))
    else:
        size = typo.get("size") or 0.0
        if body_size and size < body_size - 0.5:
            return None
        distinct = bool(typo.get("bold") or typo.get("italic")) or (body_size and size >= body_size + 0.9)
    numbered = number is None and SECTION_NUMBER_RE.match(text) is not None
    if number is None:
        number, title = _split_number(line)
    else:
        title = text
    role = heading_role(title, number)
    if role != "unclassified":
        if is_heading_text(title, number) and (numbered or distinct or (own_block and EXPLICIT_GENERAL.match(title))
                                               or (line.first_in_block and APPENDIX_LABEL.match(text))):
            return role
        return None
    if distinct and len(title) <= 100 and not title.endswith((".", ",", ";", ":", "?", "!")) and not is_noise_heading(title):
        return "unclassified"
    return None


def appendix_title(line: Line, body_size: float) -> bool:
    """A titled appendix heading set in body type ("Appendix B: Pilot Study Materials"): it opens its block, is no
    smaller than body text, and is the word, a label, ":" or "." and a title with no sentence punctuation, so
    "Appendix I.D). The results" is not one. It is no heading, since a paragraph can open "Appendix A. The items
    were"; it only starts the appendix section, so what follows it is `appendix`, not `main_body`
    (D-167)."""
    size = line.typography.get("size") or 0.0
    return (line.first_in_block and not (body_size and size < body_size - 0.5)
            and APPENDIX_TITLED.match(line.text.strip()) is not None)


# --------------------------------------------------------------------------------- tree building

CAPTION_RANGE_MAX_SPAN = 100  # real papers do not run hundreds of studies in one caption (D-052)


def _expand_caption_range(m: re.Match) -> str:
    """"1-3" -> "1 2 3"; a lettered endpoint (matched only as a plain digit token, see CAPTION_STUDY_RANGE's
    lookaheads) never reaches here, so "1a-1c" is left to `_caption_studies` as two literal tokens. A span
    wider than CAPTION_RANGE_MAX_SPAN (garbled/OCR digits, not a real study range) is likewise left literal
    instead of being materialized, same as the lo>=hi case."""
    lo, hi = int(m.group(1)), int(m.group(2))
    return " ".join(str(n) for n in range(lo, hi + 1)) if lo < hi <= lo + CAPTION_RANGE_MAX_SPAN else m.group(0)


def _caption_studies(text: str) -> list[str]:
    """Study labels named in a table/figure caption ("(Study 1)", "study 5", "Studies 1 and 2",
    "Studies 1-3" -> S1, S2, S3)."""
    labels: list[str] = []
    for m in CAPTION_STUDY.finditer(text):
        kind = "E" if m.group(1).lower().startswith("exp") else "S"  # "St. 1" abbreviates Study 1
        group = CAPTION_STUDY_RANGE.sub(_expand_caption_range, m.group(2))
        for num in re.findall(r"\d+[a-z]?", group, re.I):
            lab = kind + num.upper()
            if lab not in labels:
                labels.append(lab)
    return labels


# A section number set as its own line: "2", "3.1", "B", "A.3", "III", with a period after it or not.
NUMBER_LINE = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2})*|[A-Z](?:\.\d{1,2})*|[IVXLCDM]+)\.?\s*$")
ROMAN = re.compile(r"^(?=[IVXLCDM])M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})$")
ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _same_line(a: Line, b: Line) -> bool:
    """`b` continues `a`'s printed line: same text artifact, same baseline, to its right."""
    return (a.text_region_id == b.text_region_id and a.bbox is not None and b.bbox is not None
            and abs(a.bbox[3] - b.bbox[3]) < 2 and b.bbox[0] >= a.bbox[2] - 1)


def _wraps_to_next_block(title: Line, prev: Line, line: Line) -> bool:
    """`line` starts a block that carries a heading on to the next printed line: same text artifact, starting
    at the left edge of the heading's title (within 1 pt), its baseline at most 2.5 line heights below the
    heading line before it (`prev`). A LaTeX heading is often one block per printed line."""
    if not (line.first_in_block and title.bbox and prev.bbox and line.bbox
            and title.text_region_id == prev.text_region_id == line.text_region_id):
        return False
    return abs(line.bbox[0] - title.bbox[0]) <= 1 and 0 < line.bbox[3] - prev.bbox[3] <= 2.5 * (prev.bbox[3] - prev.bbox[1])


def _printed_number(lines: list[Line], i: int) -> tuple[int, str] | None:
    """A section number set as its own line at the start of a block ("2", "A.3"), followed on the same
    printed line by the heading's title, with a "|" between them or not: (the title's line, the number).
    A printed line that goes on after the title is running text that starts with a number."""
    m = NUMBER_LINE.match(lines[i].text)
    if not m or not lines[i].first_in_block:
        return None
    j = i + 1
    while j < len(lines) and lines[j].text.strip() == "|" and _same_line(lines[j - 1], lines[j]):
        j += 1
    if j == len(lines) or not _same_line(lines[j - 1], lines[j]) or NUMBER_LINE.match(lines[j].text):
        return None
    if j + 1 < len(lines) and _same_line(lines[j], lines[j + 1]):
        return None
    return j, m.group(1)


def _numbering(number: str | None, stack: list[Node]) -> tuple[str, tuple[int, ...]] | None:
    """(style, key) of a section number: "2.1" -> ("arabic", (2, 1)), "C.3" -> ("letter", (3, 3)),
    "III" -> ("roman", (3,)). A single I, V, X, L, C, D or M is a letter only where it continues an
    open lettered list ("C" after "B")."""
    if not number:
        return None
    head, *rest = number.rstrip(".").split(".")
    if not all(p.isdigit() for p in rest):
        return None
    tail = tuple(int(p) for p in rest)
    if head.isdigit():
        return "arabic", (int(head), *tail)
    letter = ord(head) - 64 if len(head) == 1 and "A" <= head <= "Z" else None
    letters_open = {n.number[1][0] for n in stack if n.number and n.number[0] == "letter"}
    if ROMAN.match(head) and not (letter and (letter in letters_open or letter - 1 in letters_open)):
        values = [ROMAN_VALUES[c] for c in head]
        return "roman", (sum(-v if v < nxt else v for v, nxt in zip(values, values[1:] + [0])), *tail)
    return ("letter", (letter, *tail)) if letter else None


def _number_level(number: tuple[str, tuple[int, ...]], stack: list[Node], signature: tuple, study: bool
                  ) -> tuple[int | None, tuple[list[Node], Node, str] | None]:
    """Level of a heading whose number is not "2.1. " or "2 |" on its own line (those keep their absolute
    level, as before), counted from the numbering open above it: a later number of an open list is that
    list's sibling when it is the list's next number or is set like that list's heading (`signature`); a
    dotted number is a child of its prefix; a list that starts (at 1, A or I) nests under the innermost
    numbered heading, or with none under an open study, which a new list never ends. A number that neither
    starts nor continues a list ("J. Smith", "0 Women", "5 years ..." in italics under "2. Study 1") is not
    taken as a number: None.
    A number that could continue open lists at different levels ("3" after "2. Study 1" > "2 Materials";
    "IV" after "III. Results" > "C." > "III.") continues the one whose last number comes right before it
    (with none, one set like it); if several do, the one whose heading is set the same way (`signature`,
    see `_signature`); if that leaves several too, a study heading continues the outermost (a study does
    not nest in another study's subsections), and any other heading the outermost that ends no open study
    the innermost keeps open.
    The choice is returned with the level as (the lists it could continue, the one taken, why)."""
    style, key = number
    same = [n for n in stack if n.number and n.number[0] == style]
    sibling = [n for n in same if len(n.number[1]) == len(key) and n.number[1][:-1] == key[:-1] and n.number[1] < key]
    options = [n for n in sibling if n.number[1][-1] == key[-1] - 1] \
        or [n for n in sibling if _alike(_signature(n.typography, n.title), signature)]
    if options:
        if len({n.level for n in options}) == 1:
            return options[0].level, None
        alike = [n for n in options if _alike(_signature(n.typography, n.title), signature)]
        if len({n.level for n in alike}) == 1:
            taken, why = alike[-1], "it is set like that heading"
        elif study:
            taken, why = options[0], "a study does not continue the numbering of another study's subsections"
        else:
            def ends(level: int) -> list[Node]:
                return [n for n in stack if n.study and n.level >= level]
            taken = next(n for n in options if ends(n.level) == ends(options[-1].level))
            why = "so that it ends no open study" if ends(taken.level) != ends(options[0].level) else "the outermost"
        return taken.level, (options, taken, why)
    parent = [n for n in same if n.number[1] == key[:-1]]
    if parent:
        return parent[-1].level + 1, None
    if key[-1] != 1:
        return None, None
    numbered = [n for n in stack if n.number]
    if numbered:
        return numbered[-1].level + 1, None
    if any(n.study for n in stack):
        return stack[-1].level + 1, None
    return len(key), None


def _signature(typography: dict[str, Any], text: str) -> tuple[float, bool, bool, bool, bool | None]:
    """Size, weight, slant, case (all capitals) and alignment (flush with its column) of a heading."""
    letters = [c for c in text if c.isalpha()]
    return (typography.get("size") or 0.0, bool(typography.get("bold")), bool(typography.get("italic")),
            len(letters) > 1 and all(c.isupper() for c in letters), typography.get("flush"))


def _alike(a: tuple, b: tuple) -> bool:
    """Two signatures (or their leading parts): the same size within 0.3 pt, and the rest equal."""
    return abs(a[0] - b[0]) <= 0.3 and a[1:] == b[1:]


PLAIN_LINES_MAX = 3  # lines of running text set like a level-1 heading before that typography says nothing


def _typeset_level_one(nodes: list[Node], candidates: list[int], signature_of, plain: list[dict[str, Any]]) -> set[int]:
    """The candidate headings (no recognized role, no number) typeset like this source's level-1 headings:
    the same size (within 0.3 pt), weight, slant, case and alignment as a heading that is level 1 by its
    number or its role (references, appendix, general discussion), where no heading that is lower by its
    number or role is typeset that way, and at most PLAIN_LINES_MAX lines that are no heading (`plain`)
    share its size, weight and slant. PDF only, since alignment needs the line geometry."""
    heads = [n for n in nodes if n.heading and n.role != "float"]
    lower = [_signature(n.typography, n.title) for n in heads if n.level > 1 and (n.number or n.role != "unclassified")]
    plain_sigs = [_signature(t, "")[:3] for t in plain]
    top = [s for s in (_signature(n.typography, n.title) for n in heads if n.level == 1 and (
        n.number or n.role in ("references", "appendix") or (n.role == "discussion" and GENERAL_DISCUSSION.match(n.title))))
           if s[4] is not None and not any(_alike(s, t) for t in lower)
           and sum(_alike(s[:3], p) for p in plain_sigs) <= PLAIN_LINES_MAX]
    return {i for i in candidates if any(_alike(_signature(*signature_of(i)), t) for t in top)}


# ------------------------------------------------------------------ page furniture and footnotes

MARGIN = 0.1             # the top, bottom and side bands of a page, as a share of its height or width
FURNITURE_MIN_PAGES = 3  # a running head, footer, folio or margin stamp repeats in its band on this many pages
FURNITURE_NOTE = (f"running heads, footers and folios (text repeated in the top or bottom tenth of at least "
                  f"{FURNITURE_MIN_PAGES} pages, numbers read as one) and text set vertically in a side margin of as "
                  "many pages (download stamps, copyright notices); kept out of every section and study")
RECOGNIZED_NOTE = ("OCR text of pages whose native text is empty, quarantined or only a stamp in the margins of a page "
                   "scan: recognized, not native, and without "
                   "typography, so its headings, captions and notes are not read; kept out of every section and study")
NOTE_SYMBOLS = "*∗⁎†‡§¶☆★"
# A footnote's first line: a number or note symbol, then its text ("8See Table A9 ...", "24 The impact ...",
# "∗tbondi@..."); not a list number ("25. Holt"), a volume ("57(5), 1334"), a significance legend ("* p < .05")
# or an equation ("2 = .06").
FOOTNOTE_START = re.compile(rf"^\s*(\d{{1,3}}|[{NOTE_SYMBOLS}]{{1,3}})\s?(?!p\s*[<=>≤])(?=[^\W\d_]|[“\"‘'])")
TABLE_NOTE = re.compile(r"^\s*Notes?\b")
# A number as a table cell or row number: "34.20", ".41", "−0.15⁎⁎⁎", "(0.86)", "96%", "2." ("2 Tenure 6.30 4.20 .41").
TABLE_NUMBER = re.compile(r"^[(\[]?[-−–]?(?:\d+(?:\.\d+)?|\.\d+)[.,;:%)\]*∗⁎†‡]*$")
SENTENCE_END = re.compile(r"[.!?][)\]’”\"']*$")  # a line that ends a sentence ("... on request.", "(see below).")


def _marker_in_text(marker: str) -> re.Pattern[str]:
    """A footnote marker in running text, set against the word before it ("expectations.8", "2013).24",
    "96%.12", "Bondi∗"); a number after a capital, a digit, a space or a single letter ("S1", "3.1", "β = .63",
    "a 2x2 design", "at t1 and t2") is not one."""
    if marker.isdigit():
        return re.compile(r"(?:(?<=[a-z]{2}[.,;:!?])|(?<=[)\]’”\"'%][.,;:!?])|(?<=[a-z]{2})|(?<=[)\]’”\"']))"
                          + marker + r"(?![\d.,]?\d)(?![^\W\d_])")
    return re.compile(rf"(?<=[^\s{NOTE_SYMBOLS}])" + re.escape(marker) + rf"(?![{NOTE_SYMBOLS}])")


def _asides(objects: dict[str, dict[str, Any]], lines: list[Line], body: float) -> tuple[set[int], list[tuple[int, str, list[int]]]]:
    """PDF lines that are no section's text. Page furniture: a block, none of whose lines is a heading or a
    caption, that is either of at most three lines in the top or bottom band of its page (by its middle) or
    in a side band with its lines set vertically (at least 10 characters), and whose text, numbers read as
    one, is in such a band on at least FURNITURE_MIN_PAGES pages (a sideways table's caption or note in the
    margin is neither). Footnotes: the blocks at the foot of their column (the lower half of
    the page, nothing below them but small type and furniture), every line smaller than body text, not a
    "Note" block and not table rows (more than half of its words numbers, TABLE_NUMBER), that start with a
    footnote marker; a block holding several footnotes is split at each
    marker line (a number must be the one after the last footnote's), a block with no marker under a
    footnote continues it, and the top block of a page's footnotes may start with lines continued from the
    page before (marker ""). Returns the furniture lines and the footnotes as (page index, marker, lines),
    in reading order."""
    blocks: dict[tuple[str, int], list[int]] = {}
    for i, ln in enumerate(lines):
        if ln.physical_index is not None and ln.bbox:
            blocks.setdefault((ln.text_region_id, ln.block_start), []).append(i)
    geometry: dict[tuple[str, int], tuple[list[float], float, float]] = {}
    for key, idx in blocks.items():
        crop = objects[objects[key[0]]["page_id"]]["geometry"]["cropbox"]
        boxes = [lines[i].bbox for i in idx]
        geometry[key] = ([min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)],
                         crop[2] - crop[0], crop[3] - crop[1])

    def visible(key) -> list[int]:
        return [i for i in blocks[key] if lines[i].text.strip()]

    band_text: dict[tuple[str, int], str] = {}
    band_pages: dict[str, set[int]] = {}
    for key in blocks:
        (x0, y0, x1, y1), w, h = geometry[key]
        texts = [lines[i].text.strip() for i in visible(key)]
        if not texts or any(is_caption_line(t) or heading_role(*reversed(split_section_number(t))) != "unclassified"
                            for t in texts):
            continue
        if not (MARGIN * h <= (y0 + y1) / 2 <= (1 - MARGIN) * h):
            band = len(texts) <= 3
        else:
            band = (x1 <= MARGIN * w or x0 >= (1 - MARGIN) * w) and sum(map(len, texts)) >= 10 and all(
                lines[i].bbox[3] - lines[i].bbox[1] > 2 * (lines[i].bbox[2] - lines[i].bbox[0]) for i in visible(key))
        if band:
            band_text[key] = re.sub(r"\d+", "#", " ".join(" ".join(texts).split()))
            band_pages.setdefault(band_text[key], set()).add(lines[blocks[key][0]].physical_index)
    furniture_blocks = {key for key, text in band_text.items() if len(band_pages[text]) >= FURNITURE_MIN_PAGES}

    def small(key) -> bool:
        return bool(visible(key)) and all(0 < (lines[i].typography.get("size") or 0) <= body - 0.5 for i in visible(key))

    def tabular(key) -> bool:  # table rows set small, not footnotes: more than half the block's words are numbers
        words = [w for i in visible(key) for w in lines[i].text.split()]
        numbers = sum(bool(TABLE_NUMBER.match(w)) for w in words)
        return numbers >= 2 and 2 * numbers > len(words)

    def below(upper, lower) -> bool:
        (a0, ay0, a1, _), _, _ = geometry[upper]
        (b0, by0, b1, _), _, _ = geometry[lower]
        return by0 > ay0 + 1 and b0 < a1 and b1 > a0

    footnotes: list[tuple[int, str, list[int]]] = []
    pages: dict[str, list[tuple[str, int]]] = {}
    for key in blocks:
        if key not in furniture_blocks:
            pages.setdefault(key[0], []).append(key)
    number = None  # the last footnote number
    for keys in pages.values():
        foot = [k for k in keys if small(k) and geometry[k][0][1] >= geometry[k][2] / 2
                and not TABLE_NOTE.match(lines[visible(k)[0]].text) and not tabular(k)
                and not any(below(k, c) and not small(c) for c in keys)]
        placed: dict[tuple[str, int], int] = {}  # a footnote block -> its last footnote
        for k in foot:
            idx, first = blocks[k], visible(k)[0]
            starts = []
            for pos, i in enumerate(idx):
                m = FOOTNOTE_START.match(lines[i].text)
                if not m or (i != first and m.group(1).isdigit() and (number is None or int(m.group(1)) != number + 1)):
                    continue
                starts.append((pos, m.group(1)))
                number = int(m.group(1)) if m.group(1).isdigit() else number
            above = [f for f in placed if below(f, k)]
            if not starts and above:  # a block with no marker under a footnote continues it
                placed[k] = placed[max(above, key=lambda f: geometry[f][0][1])]
                page, marker, prior = footnotes[placed[k]]
                footnotes[placed[k]] = (page, marker, prior + idx)
                continue
            if not starts or (idx[starts[0][0]] != first and any(below(c, k) for c in foot)):
                continue  # no marker, or lines above the first marker that are not the top of the footnotes
            page = lines[idx[0]].physical_index
            if idx[starts[0][0]] != first:
                footnotes.append((page, "", idx[:starts[0][0]]))
            else:
                starts[0] = (0, starts[0][1])
            for (pos, marker), (end, _) in zip(starts, starts[1:] + [(len(idx), "")]):
                footnotes.append((page, marker, idx[pos:end]))
            placed[k] = len(footnotes) - 1
    return {i for key in furniture_blocks for i in blocks[key]}, footnotes


def _line_span(lines: list[Line], i: int) -> Segment:
    """A line's code points with its line break, as the walk assigns them."""
    ln = lines[i]
    end = ln.block_end if i + 1 == len(lines) or not _same_block(ln, lines[i + 1]) else lines[i + 1].start
    return Segment(ln.text_region_id, ln.physical_index, ln.block_start if ln.first_in_block else ln.start, end)


def _place_asides(nodes: list[Node], lines: list[Line], furniture: set[int], footnotes: list[tuple[int, str, list[int]]],
                  recognized: set[int], source_id: str, label: str) -> None:
    """Put each footnote with the node whose text holds its marker on the same page, or, when its marker is
    not found, with the one node that holds all of that page's text (a footnote continued from the page
    before goes where that page's last footnote went, when that one broke off mid-sentence: its last line
    does not end ".", "!" or "?"); the rest go to a "(footnotes not attributed)" node
    with the reason, the furniture to a "(page furniture)" node and OCR text to a "(recognized text)" node
    with the pages it covers, all after every other node so no node id moves."""
    aside = furniture | {i for _, _, idx in footnotes for i in idx}

    def span(i: int) -> Segment:
        return _line_span(lines, i)

    held: dict[str, list[tuple[Segment, Node]]] = {}
    for n in nodes:
        for s in n.segments:
            held.setdefault(s.text_region_id, []).append((s, n))

    def holders(region: str, positions) -> dict[str, Node]:
        return {n.node_id: n for p in positions for s, n in held.get(region, []) if s.start <= p < s.end}

    loose: list[Segment] = []
    problems: list[str] = []
    last: tuple[int | None, Node | None, bool] = (None, None, False)  # the last footnote: page, node, broke off
    for page, marker, idx in footnotes:
        region = lines[idx[0]].text_region_id
        body = [ln for i, ln in enumerate(lines) if ln.text_region_id == region and i not in aside]
        if marker:
            pattern = _marker_in_text(marker)
            places = holders(region, [ln.start + m.start() for ln in body for m in pattern.finditer(ln.text)])
            if not places and len(holders(region, [ln.start for ln in body])) == 1:
                places = holders(region, [ln.start for ln in body])
            dest = next(iter(places.values())) if len(places) == 1 else None
            why = (f"its marker '{marker}' is in {len(places)} sections of its page" if places else
                   f"its marker '{marker}' is not found in the text of its page, and that text is not all in one section")
        else:  # lines above the page's first marker: the page before's last footnote, if it broke off mid-sentence
            carried = last[0] == page - 1 and last[2]
            dest = last[1] if carried else None
            why = "it continues a footnote from the page before, which " + (
                "was not attributed" if carried else "has none that breaks off mid-sentence")
        if dest is not None:
            dest.segments.extend(span(i) for i in idx)
        else:
            loose.extend(span(i) for i in idx)
            text = next(lines[i].text.strip() for i in idx if lines[i].text.strip())
            problems.append(f"footnote on page index {page} ('{text[:60]}') is not attributed: {why}")
        end = next(lines[i].text.strip() for i in reversed(idx) if lines[i].text.strip())
        last = (page, dest, not SENTENCE_END.search(end))
    pages = ", ".join(str(p) for p in sorted({lines[i].physical_index for i in recognized}))
    ocr = [f"OCR text (recognized, not native) on page index {pages}: its headings are not read, so which study or "
           "section it belongs to is not determinable from its place"]
    for title, role, segs, notes in (("(page furniture)", "furniture", [span(i) for i in sorted(furniture)], []),
                                     ("(footnotes not attributed)", "footnotes", loose, problems),
                                     ("(recognized text)", "recognized", [span(i) for i in sorted(recognized)], ocr)):
        if segs:
            nodes.append(Node(f"{label}-n{len(nodes):03d}", source_id, label, lines[0].section, title, role, 1, False,
                              None, None, segments=segs, notes=notes))


def _float_notes(objects: dict[str, dict[str, Any]], lines: list[Line], body: float, caption_block: dict[int, bool],
                 skip: set[int], headings: dict[int, str], part_of: dict[int, int]) -> dict[int, int]:
    """The note blocks of each float: {note line: its caption's first line}. A note block starts "Note."/"Notes:"
    or with a significance legend (FLOAT_NOTE) and holds no heading. In a PDF it lies below its caption on the
    same page, overlapping across what the caption and the blocks before it cover, and is reached from the
    caption in reading order through the table or figure itself: blocks below the caption's top and not below
    the note's top, none of them a caption, a heading with a recognized role or a paragraph of running text
    (two or more lines, all in the body size within 0.3 pt, one at least a third of the page wide), which
    ends the float (blocks wholly above the caption, another column read in between, are passed over). In a
    DOCX only empty paragraphs, table cells and the paragraph right after the caption lie between them. Note
    blocks directly after a note are notes too, and so is a PDF block right under a note (its top within one
    line of the note's bottom) that overlaps it across in the same type size, set smaller than body text (more
    than 0.3 pt under it). Lines in `skip` (furniture, footnotes) are passed over."""
    starts = [i for i, ln in enumerate(lines) if ln.first_in_block and i not in skip]
    members: dict[int, list[int]] = {}
    for i in starts:
        members[i] = [i]
        while members[i][-1] + 1 < len(lines) and _same_block(lines[members[i][-1]], lines[members[i][-1] + 1]):
            members[i].append(members[i][-1] + 1)

    def box(i: int) -> list[float] | None:
        boxes = [lines[k].bbox for k in members[i] if lines[k].bbox]
        return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)] \
            if boxes else None

    def running(i: int) -> bool:  # a paragraph of running text, not a table's rows or a figure's labels
        ks = [k for k in members[i] if lines[k].text.strip() and lines[k].bbox]
        crop = objects[objects[lines[i].text_region_id]["page_id"]]["geometry"]["cropbox"]
        return len(ks) >= 2 and all(abs((lines[k].typography.get("size") or 0) - body) <= 0.3 for k in ks) \
            and max(lines[k].bbox[2] - lines[k].bbox[0] for k in ks) >= (crop[2] - crop[0]) / 3

    out: dict[int, int] = {}
    for pos, c in enumerate(starts):
        if caption_block.get(c) is not True:
            continue
        cap, notes, passed = box(c), [], []
        for n in starts[pos + 1:]:
            if (lines[n].text_region_id, lines[n].physical_index) != (lines[c].text_region_id, lines[c].physical_index):
                break
            b = box(n)
            if FLOAT_NOTE.match(lines[n].text) and not any(k in headings or k in part_of for k in members[n]):
                if cap is not None and (b is None or b[1] < cap[3] - 2 or any(p[1] > b[1] + 2 for p in passed)
                                        or b[0] >= max(p[2] for p in passed + [cap])
                                        or b[2] <= min(p[0] for p in passed + [cap])):
                    break
                notes.append(n)
                continue
            if notes:  # a block right under a note (within a line), overlapping it, in its type size, continues it;
                # one in the body size may be the running text after the float, so it never does
                last, tail = box(notes[-1]), lines[members[notes[-1]][-1]]
                if cap is None or b is None or not tail.bbox or any(k in headings or k in part_of for k in members[n]) \
                        or not -2 <= b[1] - last[3] <= tail.bbox[3] - tail.bbox[1] or b[0] >= last[2] or b[2] <= last[0] \
                        or abs((lines[n].typography.get("size") or 0) - (tail.typography.get("size") or 0)) > 0.3 \
                        or any((lines[k].typography.get("size") or 0) > body - 0.3 for k in members[n] if lines[k].text.strip()):
                    break
                notes.append(n)
                continue
            if cap is not None and b is not None and b[3] <= cap[1] + 2:
                continue  # wholly above the caption: another column, read between the caption and its table
            if any(headings.get(k, "unclassified") != "unclassified" for k in members[n]) \
                    or (cap is not None and (b is None or running(n))):
                break
            if cap is None and lines[n].text.strip() and not lines[n].typography.get("table_cell") and n != c + 1:
                break  # DOCX: only empty paragraphs, table cells and the one right after the caption
            passed.append(b)
        out.update((k, c) for n in notes for k in members[n])
    return out


def build_source_tree(objects: dict[str, dict[str, Any]], source_id: str) -> SourceTree:
    source = objects[source_id]
    membership = source["membership"][0]
    label, role = membership["source_label"], membership["role"]
    lines = _reading_lines(objects, source_id)
    docx = all(ln.physical_index is None for ln in lines) if lines else False
    body = body_font_size(lines)
    furniture, footnotes = _asides(objects, lines, body)
    recognized = {i for i, ln in enumerate(lines) if objects[ln.text_region_id]["derivation"]["kind"] == "ocr_recognized"}
    # Pages from which no line is read (no usable native text, and no OCR text: OCR failed, did not run or found none)
    read = {ln.physical_index for ln in lines if ln.text.strip()}
    unread = sorted(o["physical_index"] for o in objects.values() if o["type"] == "page" and o["source_id"] == source_id
                    and o["physical_index"] not in read)
    aside = furniture | {i for _, _, idx in footnotes for i in idx} | recognized  # no section's text, never a heading

    # Table/figure captions: the first line of a block that starts "Table 2", "Fig. 4." and so on. The
    # rest of the caption block is caption, never a heading.
    caption_block: dict[int, bool] = {}
    for i, ln in enumerate(lines):
        if i in aside:
            continue
        if is_caption_line(ln.text) and (ln.first_in_block or ((i - 1) in caption_block and _same_line(lines[i - 1], ln))):
            caption_block[i] = True  # it opens its block, or is set beside another caption on its baseline
        elif not ln.first_in_block and (i - 1) in caption_block:
            caption_block[i] = False  # continuation line of a caption block
    # Each line of a caption block is its caption's: with captions set side by side the block holds several, and a
    # line (a title under a label, a wrapped line) is the one whose column holds it, whatever order the block's lines
    # come in, so that one caption's text is never another's (D-193).
    caption_of: dict[int, int] = {}  # a caption block's line -> its caption's label line
    run: list[int] = []
    for i in sorted(caption_block) + [len(lines)]:
        if run and (i == len(lines) or i != run[-1] + 1 or (caption_block[i] is True and lines[i].first_in_block)):
            labels = [k for k in run if caption_block[k] is True]
            x0s = [lines[k].bbox[0] if lines[k].bbox else 0.0 for k in labels]
            for k in run:
                caption_of[k] = k if k in labels else labels[0] if len(labels) == 1 or not lines[k].bbox else \
                    labels[caption_column(lines[k].bbox[0], x0s)]
            run = []
        if i < len(lines):
            run.append(i)
    own_lines: dict[int, list[int]] = {}
    for k, c in caption_of.items():
        own_lines.setdefault(c, []).append(k)
    headings: dict[int, str] = {}
    title_line: dict[int, int] = {}  # heading -> the line holding its title (after a number printed on its own line)
    number_of: dict[int, str] = {}   # heading -> its section number
    part_of: dict[int, int] = {}     # a heading's "|" and title line after its number, or its wrapped line -> the heading
    absolute: set[int] = set()       # "2.1. Title", "2 | Title": the number's depth is the level, as before
    for i, ln in enumerate(lines):
        if i in aside:
            continue
        if i in part_of or i in caption_block:
            if caption_block.get(i) is True:
                headings[i] = "float"
            continue
        printed = _printed_number(lines, i)
        if printed and printed[0] in caption_block:
            printed = None
        j = printed[0] if printed else i
        own_block = lines[j].first_in_block and (j + 1 == len(lines) or not _same_block(lines[j], lines[j + 1]))
        r = heading_of(lines[j], body, docx, printed[1] if printed else None, own_block)
        if not r:
            continue
        headings[i], title_line[i] = r, j
        number_of[i] = printed[1] if printed else _split_number(lines[j])[0]
        if not printed and SECTION_NUMBER_RE.match(ln.text.strip()):
            absolute.add(i)
        part_of.update((k, i) for k in range(i + 1, j + 1))
        # A heading wrapped onto more lines of its block, or onto blocks of their own at its title's left edge,
        # set the same way (size within 0.3 pt), is one heading (a line that is a heading by its role or starts
        # with a number is not a wrap).
        k = j + 1
        while (k < len(lines) and k not in caption_block and k not in aside
               and (_same_block(lines[k - 1], lines[k]) or _wraps_to_next_block(lines[j], lines[k - 1], lines[k]))
               and abs((lines[k].typography.get("size") or 0.0) - (lines[j].typography.get("size") or 0.0)) <= 0.3
               and all(lines[k].typography.get(x) == lines[j].typography.get(x) for x in ("bold", "italic"))
               and heading_role(lines[k].text.strip()) == "unclassified" and not NUMBER_LINE.match(lines[k].text)
               and split_section_number(lines[k].text.strip())[0] is None):
            part_of[k] = i
            k += 1
    note_of = _float_notes(objects, lines, body, caption_block, aside, headings, part_of)  # a note line -> its caption line

    def parts(i: int) -> list[Line]:
        return [lines[i]] + [lines[k] for k, h in part_of.items() if h == i]

    def title_text(i: int) -> str:  # the title without its number, for its role and study label
        return _split_number(lines[i])[1] if title_line[i] == i else lines[title_line[i]].text.strip()

    # Section by document position: an appendix heading, or a titled appendix line in body type, starts the
    # appendix even when the page-level split (used for folio trust) put that page in the main body.
    appendix_from = next((i for i, ln in enumerate(lines) if headings.get(i) == "appendix" or appendix_title(ln, body)), None)

    def section_of(i: int) -> str:
        return "appendix" if appendix_from is not None and i >= appendix_from else lines[i].section

    # A plain "Discussion" after the last main-body study is the paper's discussion unless an explicit
    # general-discussion heading follows; the choice is recorded on the study as an ambiguity.
    main_studies = [i for i, r in headings.items() if r == "study" and section_of(i) == "main_body"
                    and study_label(title_text(i))]
    last_study = max(main_studies) if main_studies else None
    later_general = {i for i, r in headings.items() if r != "float" and EXPLICIT_GENERAL.match(title_text(i))}

    def walk(level_one: set[int]) -> list[Node]:
        """The tree, with the headings in `level_one` (unrecognized, typeset like level-1 headings) at level 1."""
        front = Node(f"{label}-n000", source_id, label, section_of(0) if lines else "main_body",
                     "(before the first heading)", "front_matter", 1, False, None, None)
        nodes: list[Node] = [front]
        stack: list[Node] = [front]
        current = front
        floats: dict[int, Node] = {}  # caption line -> its float node
        ocr_pages: list[int] = []  # pages read only through OCR since the last line read in the flow
        unread_pages: list[int] = []  # pages from which no line was read at all since then (D-199)
        ocr_open: set[str] = set()  # the nodes open when the flow reached the first of them
        last_page = None
        for i, ln in enumerate(lines):
            if ln.physical_index is not None:
                gap = [p for p in unread if last_page is not None and last_page < p < ln.physical_index]
                if gap:
                    ocr_open = ocr_open or {n.node_id for n in stack}
                    unread_pages += [p for p in gap if p not in unread_pages]
                last_page = ln.physical_index
            if i in aside:  # placed after the walk (_place_asides)
                if i in recognized and ln.physical_index not in ocr_pages:
                    ocr_open = ocr_open or {n.node_id for n in stack}
                    ocr_pages.append(ln.physical_index)
                continue
            role_i = headings.get(i)
            if role_i == "float":
                text = ln.text.strip()
                own = [lines[k] for k in own_lines[i]]  # its lines: those of its block in its column (D-193)
                block_text = " ".join(x.text.strip() for x in own)
                node = Node(f"{label}-n{len(nodes):03d}", source_id, label, section_of(i), text, "float", 9, False, None,
                            Segment(ln.text_region_id, ln.physical_index, ln.start, ln.end), dict(ln.typography),
                            stack[-1].node_id, caption=block_text, float_studies=_caption_studies(block_text),
                            caption_lines=[Segment(x.text_region_id, x.physical_index, x.start, x.end) for x in own])
                nodes.append(node)
                floats[i] = node
                current = node  # only for this block and its notes; the flow returns to the enclosing node after it
            elif role_i:
                head = parts(i)
                text = " ".join(x.text.strip() for x in head if x.text.strip() != "|")
                title = title_text(i)
                study = study_label(title) if role_i == "study" else None
                studies_open = [n for n in stack if n.study]
                offset = studies_open[-1].level - 1 if studies_open else 0  # role levels count from the open study
                number = _numbering(number_of.get(i), stack)
                typography = {**lines[title_line[i]].typography, "flush": ln.typography.get("flush")}
                level, choice = None, None
                if number:
                    level, choice = (len(number[1]), None) if i in absolute else \
                        _number_level(number, stack, _signature(typography, text), bool(study))
                if level is None:
                    number = None
                    if i in level_one or study or role_i in ("references", "appendix"):
                        level = 1
                    elif role_i == "discussion" and GENERAL_DISCUSSION.match(title):
                        level = 1
                    elif role_i == "discussion" and last_study is not None and i > last_study \
                            and section_of(i) == "main_body" and not any(g > i for g in later_general) \
                            and normalize_title(title) == "discussion":
                        level = 1
                        prior = next((n for n in reversed(nodes) if n.study), None)
                        if prior is not None:
                            prior.notes.append(f"unnumbered 'Discussion' after the last study (page index {ln.physical_index}) "
                                               "was treated as the paper's general discussion; if it is specific to this "
                                               "study, the study's region should extend through it")
                    elif role_i in FRONT_ROLES:
                        level = offset + 2 if studies_open else 1
                    elif role_i in ("methods", "results", "discussion"):
                        level = offset + 2
                    else:
                        level = offset + 3
                notes = []
                if i in level_one:
                    for s in (n for n in stack if n.study and n.level >= level):
                        s.notes.append(f"region ends at '{text}' (page index {ln.physical_index}), a heading with no "
                                       "recognized role that is typeset like this source's level-1 headings; if that "
                                       "section belongs to this study, the study's region should extend through it")
                        notes.append(f"'{text}' is taken to end {s.title!r} only because it is typeset like this "
                                     "source's level-1 headings; which study, if any, this section belongs to is not "
                                     "determinable from headings")
                if choice:  # its number could continue lists at two levels: recorded where a study's region depends on it
                    options, taken, why = choice
                    said = (f"'{text}' (page index {ln.physical_index}) could continue the numbering of "
                            f"{' or '.join(repr(o.title) for o in options)} and is taken to continue {taken.title!r} ({why})")
                    other = " or ".join(repr(o.title) for o in options if o is not taken)
                    lo, hi = min(o.level for o in options), max(o.level for o in options)
                    for s in (n for n in stack if n.study and lo <= n.level < hi):
                        ends = s.level >= level
                        s.notes.append(f"{said}, so {'the region ends there' if ends else 'its section stays in this study'}; "
                                       "which list it continues is not determinable from its number, and if it continues "
                                       f"{other} instead, the study's region should "
                                       f"{'extend through' if ends else 'end before'} it")
                        notes.append(f"{said}; whether its section belongs to {s.title!r} is not determinable from its number")
                while stack and stack[-1].level >= level:
                    stack.pop()
                heading_end = max((x.end for x in head if x.text_region_id == ln.text_region_id), default=ln.end)
                node = Node(f"{label}-n{len(nodes):03d}", source_id, label, section_of(i), text, role_i, level,
                            number is not None, study, Segment(ln.text_region_id, ln.physical_index, ln.start, heading_end),
                            typography, stack[-1].node_id if stack else None, notes=notes, number=number)
                nodes.append(node)
                stack.append(node)
                current = node
            elif current.role == "float" and ln.first_in_block:
                current = stack[-1]  # the caption block has ended
            if ocr_pages or unread_pages:  # the flow resumes: the item that holds the text read next runs on past
                # unread headings: every open study (a study holds its nested ones), or with none open the level-1
                # section, when it was open before those pages (D-187, D-199)
                said = [f"page index {', '.join(str(p) for p in ocr_pages)}, whose text is read only through OCR "
                        "(recognized, not native) and whose headings are not read"] if ocr_pages else []
                said += [f"page index {', '.join(str(p) for p in unread_pages)}, from which no text was read (no usable "
                         "native text and no OCR text), so whose headings are not read"] if unread_pages else []
                holders = [n for n in stack if n.study] or [n for n in stack if n.level == 1]
                for n in holders:
                    if n.node_id in ocr_open:
                        n.notes.append(f"its region runs on past {' and '.join(said)}; text after that page may "
                                       "belong to another study or section")
                ocr_pages, unread_pages, ocr_open = [], [], set()
            if ln.physical_index is None:  # DOCX paragraph: the whole block
                seg_start, seg_end = ln.block_start, ln.block_end
            else:
                seg_start = ln.start if (role_i and role_i != "float") or not ln.first_in_block else ln.block_start
                seg_end = ln.block_end if (i + 1 == len(lines) or not _same_block(ln, lines[i + 1])) else lines[i + 1].start
            (floats[note_of[i]] if i in note_of else floats[caption_of[i]] if caption_of.get(i) in floats
             else current).segments.append(Segment(ln.text_region_id, ln.physical_index, seg_start, seg_end))
        return nodes

    nodes = walk(set())
    level_one = _typeset_level_one(nodes, [i for i, r in headings.items() if r == "unclassified" and not number_of.get(i)],
                                   lambda i: ({**lines[title_line[i]].typography, "flush": lines[i].typography.get("flush")},
                                              " ".join(x.text for x in parts(i))),
                                   [ln.typography for i, ln in enumerate(lines) if ln.text.strip() and i not in headings
                                    and i not in part_of and i not in caption_block and i not in aside])
    if level_one:
        nodes = walk(level_one)
    _place_asides(nodes, lines, furniture, footnotes, recognized, source_id, label)
    for n in nodes:
        n.segments = merge_segments(n.segments)
    tree = SourceTree(source_id, label, role, body, nodes, merge_segments([_line_span(lines, i) for i in sorted(aside)]))
    tree.problems, tree.unread = validate_tree(tree), unread
    return tree


def _same_block(a: Line, b: Line) -> bool:
    return a.text_region_id == b.text_region_id and a.block_start == b.block_start and not b.first_in_block


def validate_tree(tree: SourceTree) -> list[str]:
    """K's structural checks that still apply: every non-root node has a parent that exists and
    comes earlier, and a heading with no following text is reported."""
    problems = []
    ids = [n.node_id for n in tree.nodes]
    for pos, n in enumerate(tree.nodes[1:], start=1):
        if n.parent_id is not None and (n.parent_id not in ids or ids.index(n.parent_id) >= pos):
            problems.append(f"{n.node_id}: parent {n.parent_id} missing or later in reading order")
        if n.level > 2:
            continue
        body_chars = sum(s.end - s.start for s in tree.extent(n)) - ((n.heading.end - n.heading.start) if n.heading else 0)
        if body_chars <= 1:
            problems.append(f"{n.node_id} '{n.title}': heading with no following text")
    return problems
