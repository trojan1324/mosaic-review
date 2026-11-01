"""Printed page labels from page geometry, trusted per section (audit M-10, M-57).

Adapted from K/tools/extract_spatial_page_labels.py (zone_of, page_candidates,
LABEL_RE and BAND kept as written) and K/tools/page_resolution.py
(_label_is_trustworthy). Changes and why:

* Candidates come from an open PyMuPDF page dict; nothing writes _meta.yaml.
* Trust is decided **within an artifact-membership section** (for example the
  main body versus a bound appendix), never over the whole file. K's
  discriminator asks whether ANY page in the document carries a folio distinct
  from its index; an unpaginated appendix then inherits trust from the main
  body and a stray number in its margin is accepted as a folio (M-10).
* A label is only ever a string visibly printed on that page. A page without a
  candidate gets value null; a sequence is never used to invent a label.
* `legacy_whole_file_trust` reproduces K's rule so a test can show the
  difference on the same fixture.
* Sequence consistency counts three label kinds -- plain digits, a single
  letter followed by digits ("A1", an appendix style), and roman numerals
  (front matter) -- each in its own numeric range so they can't be confused
  with one another (D-051).
* `folio_candidates` drops three or more label-like tokens side by side in a
  margin band (a table row), and a page with two numbers on the section's
  sequence takes the one where the section's folios are printed
  (D-167).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

BAND = 0.12
LABEL_RE = re.compile(r"^(?:p(?:age)?\.?\s*)?([A-Za-z]?\d{1,5}|[ivxlcdmIVXLCDM]{1,12})$")
POSITIONS = {"top-left": "top_left", "top-center": "top_center", "top-right": "top_right",
             "bottom-left": "bottom_left", "bottom-center": "bottom_center", "bottom-right": "bottom_right"}
MIN_SUPPORT = 3


def zone_of(bbox, width: float, height: float) -> str | None:
    x0, y0, x1, y1 = bbox
    ymid = (y0 + y1) / 2
    if ymid <= height * BAND:
        vertical = "top"
    elif ymid >= height * (1 - BAND):
        vertical = "bottom"
    else:
        return None
    frac = ((x0 + x1) / 2) / width if width else 0.5
    horizontal = "left" if frac < 0.33 else ("right" if frac > 0.67 else "center")
    return f"{vertical}-{horizontal}"


def page_candidates(page_dict: dict[str, Any], width: float, height: float) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for block in page_dict.get("blocks", []):
        for line in block.get("lines", []):
            text = "".join(s.get("text", "") for s in line.get("spans", [])).strip()
            if not text or len(text) > 24:
                continue
            zone = zone_of(line["bbox"], width, height)
            if not zone:
                continue
            match = LABEL_RE.match(text)
            if not match:
                continue
            found.append({"position": POSITIONS[zone], "text": match.group(1), "bbox": [round(v, 2) for v in line["bbox"]]})
    return found


def folio_candidates(page_dict: dict[str, Any], width: float, height: float) -> list[dict[str, Any]]:
    """`page_candidates` less any three or more side by side: a table row in the margin band (column heads,
    a row of numbers), not a folio, which stands alone or beside one other number (D-167)."""
    found = page_candidates(page_dict, width, height)

    def row(c: dict[str, Any]) -> int:  # candidates at c's height, c included
        return sum(1 for o in found if o["bbox"][1] < c["bbox"][3] and c["bbox"][1] < o["bbox"][3])

    return [c for c in found if row(c) < 3]


_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
_ROMAN_NUMERALS = [(1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"),
                    (50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
# Disjoint numeric ranges per label kind, so a section's dominant (offset, count) can only be
# won by candidates of the SAME kind -- and, for letter-prefixed labels, the SAME letter. A
# stray "B3" can't be counted as part of an "A1".."A5" sequence, and a roman numeral never
# collides with a page number that happens to share its digits (audit capture finding #1).
_LETTER_BASE = 100_000  # past LABEL_RE's plain-digit range (at most 5 digits => 99999)
_ROMAN_BASE = _LETTER_BASE + 27 * 100_000  # past every letter bucket (A-Z)


def _int_to_roman(n: int) -> str:
    out = []
    for value, sym in _ROMAN_NUMERALS:
        count, n = divmod(n, value)
        out.append(sym * count)
    return "".join(out)


def _roman_to_int(label: str) -> int | None:
    s = label.lower()
    total, prev = 0, 0
    for ch in reversed(s):
        val = _ROMAN_VALUES.get(ch)
        if val is None:
            return None
        total += -val if val < prev else val
        prev = max(prev, val)
    # Only a canonical (minimal, well-formed) roman numeral is trusted as a folio -- LABEL_RE's
    # character class also matches non-canonical junk like "IIII" or "VV", which this rejects.
    # The upper bound also rejects "MMMM" and longer M-runs: every other repeatable symbol is
    # capped at 3 by a subtractive-pair entry in _ROMAN_NUMERALS (e.g. 4 -> "iv"), but M has no
    # symbol above it, so _int_to_roman would otherwise round-trip "mmmm" as canonical.
    return total if 0 < total < 4000 and _int_to_roman(total) == s else None


def _as_int(label: str) -> int | None:
    """Sequence-consistency key for a printed label: plain digits, a single-letter-prefixed
    appendix label ("A1", "A2", ...) or a roman numeral (front matter). See the range comment
    above for why the three kinds never collide."""
    if label.isdigit():
        return int(label)
    if len(label) >= 2 and label[0].isalpha() and label[1:].isdigit():
        return _LETTER_BASE + (ord(label[0].upper()) - ord("A")) * 100_000 + int(label[1:])
    roman = _roman_to_int(label)
    return _ROMAN_BASE + roman if roman is not None else None


@dataclass
class SectionResult:
    name: str
    pages: list[int]
    offset: int | None
    support: int
    labels: dict[int, dict[str, Any]] = field(default_factory=dict)


def resolve_section(name: str, page_indices: list[int], candidates: dict[int, list[dict[str, Any]]]) -> SectionResult:
    """Decide labels for one section. offset = printed number - physical index."""
    offsets = Counter()
    for idx in page_indices:
        for cand in candidates.get(idx, []):
            n = _as_int(cand["text"])
            if n is not None:
                offsets[n - idx] += 1
    offset, support = (offsets.most_common(1)[0] if offsets else (None, 0))
    consistent = offset is not None and support >= MIN_SUPPORT and support >= 0.5 * sum(1 for i in page_indices if candidates.get(i))
    result = SectionResult(name, list(page_indices), offset if consistent else None, support)
    # Where the section's folios are printed: a page with two numbers on the sequence (a section number and
    # the folio) takes the one there (D-167).
    where = Counter(c["position"] for i in page_indices for c in candidates.get(i, [])
                    if consistent and _as_int(c["text"]) == i + offset)
    for idx in page_indices:
        cands = candidates.get(idx, [])
        if not cands:
            result.labels[idx] = {"value": None, "status": "none", "position": None, "section": name,
                                  "method": "spatial_six_position;no_candidate"}
            continue
        if consistent:
            matching = [c for c in cands if _as_int(c["text"]) == idx + offset]
            if matching:
                c = max(matching, key=lambda m: where[m["position"]])
                result.labels[idx] = {"value": c["text"], "status": "verified", "position": c["position"], "section": name,
                                      "method": f"spatial_six_position;section_sequence_offset={offset};support={support}"}
                continue
            result.labels[idx] = {"value": None, "status": "ambiguous_or_conflicting", "position": None, "section": name,
                                  "method": "spatial_six_position;candidate_conflicts_with_section_sequence"}
            continue
        # Visible candidate but no corroborating sequence in this section.
        distinct = sorted({c["text"] for c in cands})
        if len(distinct) == 1:
            c = cands[0]
            result.labels[idx] = {"value": c["text"], "status": "inferred", "position": c["position"], "section": name,
                                  "method": "spatial_six_position;uncorroborated_in_section"}
        else:
            result.labels[idx] = {"value": None, "status": "ambiguous_or_conflicting", "position": None, "section": name,
                                  "method": "spatial_six_position;multiple_candidates_no_sequence"}
    return result


def resolve_labels(sections: dict[str, list[int]], candidates: dict[int, list[dict[str, Any]]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for name, pages in sections.items():
        out.update(resolve_section(name, pages, candidates).labels)
    return out


def citable(label: dict[str, Any]) -> str | None:
    """Only a verified visible label is citable; None never means page 1."""
    return label["value"] if label.get("status") == "verified" else None


def legacy_whole_file_trust(candidates: dict[int, list[dict[str, Any]]]) -> dict[int, str | None]:
    """K's page_resolution._label_is_trustworthy applied to first candidates.

    A label distinct from its raw (1-based) index is trusted outright; an
    identity-mapped label is trusted only if no page in the WHOLE file carries
    a distinct folio. Kept only to demonstrate audit M-10.
    """
    first = {idx: (c[0]["text"] if c else None) for idx, c in candidates.items()}
    n_distinct = sum(1 for idx, lab in first.items() if lab is not None and lab != str(idx + 1))
    out: dict[int, str | None] = {}
    for idx, lab in first.items():
        if not lab:
            out[idx] = None
        elif lab != str(idx + 1):
            out[idx] = lab
        else:
            out[idx] = lab if n_distinct == 0 else None
    return out
