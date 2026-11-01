"""Deterministic span catalog and selection-time re-derivation (FINAL v2 §6.3; audit M-65, N-28).

The catalog for a text artifact is built from Layer-1 output before any model
call. Every entry has one of five kinds or the forced-review state
`unclassified`:

* `whole_unit_minus_marker`: an item-like unit (a list item, a DOCX list
  paragraph, a standalone question) without its leading list marker;
* `numbered_parenthetical`: a scale anchor such as "1 (extremely unlikely)";
* `quoted_substring`: the text inside a pair of double quotes (or guarded
  single curly quotes), without a trailing comma, semicolon or colon;
* `table_cell`: supplied by the table module (M2), never guessed here;
* `figure_caption`: a unit that starts "Figure N" / "Fig. N";
* `unclassified`: any other unit, as a whole, flagged for review. There is no
  silent whole-paragraph default: selecting such a span routes the result to
  review.

`selection_schema` publishes the span IDs as an enum, so an invented ID is a
schema violation at generation time. `select` re-derives the substring from
the current text artifact and rejects any drift. Only native text can be
catalogued; OCR output and quarantined text are refused outright.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from mosaic.package import actor
from mosaic.validation import registry

NATIVE_KINDS = ("native_raw", "docx_body", "xlsx_cells")
LIST_MARKER = re.compile(r"^(?:\(?\d{1,3}[.)]|\(?[a-z][.)]|\(?(?:i|ii|iii|iv|v|vi|vii|viii|ix|x)[.)]|[•▪◦●○■□*–—-])\s+")
ANCHOR = re.compile(r"(?<![\d.,])(\d{1,2})\s*(?:=\s*)?\(([A-Za-z][^()\n]{0,60}(?:\n[^()\n]{0,60})?)\)")
FIGURE_CAPTION = re.compile(r"^(?:Figure|Fig\.)\s*[A-Z]?\d+[A-Za-z]?(?:\s*$|[.:|]\s*)")
SENTENCE_BREAK = re.compile(r"[.!?]\s+[A-Z]")
QUOTES = (("“", "”"), ("\"", "\""), ("‘", "’"))
TRAILING_PUNCT = ",;:"
_CURLY_SINGLE_BOUNDARY = frozenset(".?!,;:)]}")
_CURLY_SINGLE_CONNECTOR = re.compile(r"^\s*(?:and|or)\s*$", re.I)


class CatalogError(ValueError):
    """Raised with a stable reason code as its message."""


@dataclass(frozen=True)
class Selection:
    span_id: str
    kind: str
    text_region_id: str
    start: int
    end: int
    text: str
    text_sha256: str
    needs_review: bool


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def span_id(text_region_id: str, start: int, end: int) -> str:
    return f"span:{text_region_id.rsplit('-', 1)[1][:12]}@{start}-{end}"


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _quoted(text: str, start: int, end: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    unit = text[start:end]
    for open_q, close_q in QUOTES:
        if open_q == "‘":
            out.extend((start + s, start + e) for s, e in _curly_single_quoted(unit))
            continue
        pos = 0
        while True:
            a = unit.find(open_q, pos)
            if a < 0:
                break
            b = unit.find(close_q, a + 1)
            if b < 0:
                break
            s, e = a + 1, b
            while e > s and unit[e - 1] in TRAILING_PUNCT:
                e -= 1
            if e - s >= 3 and any(c.isalpha() for c in unit[s:e]) and e - s <= 600:
                out.append((start + s, start + e))
            pos = b + 1
    return out


def _curly_single_quoted(unit: str) -> list[tuple[int, int]]:
    """Find only unambiguous ``‘…’`` interiors.

    The right curly apostrophe is also used in contractions and possessives.
    A candidate closer therefore needs a strong printed boundary: end of the
    unit, outside punctuation/closers, or an ``and``/``or`` bridge to the next
    opening quote.  If another opener arrives first, the same-style nesting is
    ambiguous and is skipped through its next strong closer rather than guessed.
    """
    out: list[tuple[int, int]] = []
    pos = 0
    while True:
        opening = unit.find("‘", pos)
        if opening < 0:
            break
        if opening > 0 and unit[opening - 1].isalnum():
            pos = opening + 1
            continue
        next_opening = unit.find("‘", opening + 1)
        limit = next_opening if next_opening >= 0 else len(unit)
        closer = _strong_curly_single_closer(unit, opening + 1, limit, next_opening)
        if closer is not None:
            s, e = opening + 1, closer
            while e > s and unit[e - 1] in TRAILING_PUNCT:
                e -= 1
            if e - s >= 3 and any(c.isalpha() for c in unit[s:e]) and e - s <= 600:
                out.append((s, e))
            pos = closer + 1
            continue
        if next_opening < 0:
            break
        # A same-style opener before a qualified close may be nested or may
        # follow a malformed quote.  Do not turn its tail into a candidate.
        skipped = _strong_curly_single_closer(unit, next_opening + 1, len(unit), None)
        pos = (skipped + 1) if skipped is not None else len(unit)
    return out


def _strong_curly_single_closer(unit: str, start: int, limit: int, next_opening: int | None) -> int | None:
    """Return the first boundary-qualified closer in ``[start, limit)``."""
    for candidate in range(start, limit):
        if unit[candidate] != "’":
            continue
        after = candidate + 1
        if after == len(unit) or unit[after] in _CURLY_SINGLE_BOUNDARY:
            return candidate
        elif next_opening is not None and _CURLY_SINGLE_CONNECTOR.match(unit[after:next_opening]):
            return candidate
        elif unit[after].isalpha():
            continue  # an apostrophe inside a word
        elif (candidate > 0 and unit[candidate - 1].lower() == "s"
              and re.match(r"\s+(?!(?:and|or)\b)[A-Za-z]", unit[after:], re.I)):
            continue  # a possible plural possessive before its following word
        else:
            return None  # an unresolved earlier close must not swallow later prose
    return None


def _entry(text: str, tr_id: str, kind: str, start: int, end: int, block_index: int) -> dict[str, Any]:
    entry = {"span_id": span_id(tr_id, start, end), "kind": kind, "start": start, "end": end,
             "text_sha256": _sha(text[start:end]), "block_index": block_index}
    if kind == "unclassified":
        entry["needs_review"] = True
    return entry


def unit_entries(text: str, tr_id: str, block: dict[str, Any]) -> list[dict[str, Any]]:
    start, end = _trim(text, block["start"], block["end"])
    if end <= start or block.get("kind") == "image_placeholder":
        return []
    unit = text[start:end]
    bi = block["block_index"]
    entries: list[dict[str, Any]] = []
    if FIGURE_CAPTION.match(unit):
        entries.append(_entry(text, tr_id, "figure_caption", start, end, bi))
    else:
        marker = LIST_MARKER.match(unit)
        standalone_question = (unit.endswith("?") and unit.count("?") == 1 and len(unit) <= 400
                               and not SENTENCE_BREAK.search(unit))
        if marker and end - (start + marker.end()) >= 3:
            entries.append(_entry(text, tr_id, "whole_unit_minus_marker", start + marker.end(), end, bi))
        elif block.get("list_item") or standalone_question:
            entries.append(_entry(text, tr_id, "whole_unit_minus_marker", start, end, bi))
    for m in ANCHOR.finditer(unit):
        entries.append(_entry(text, tr_id, "numbered_parenthetical", start + m.start(), start + m.end(), bi))
    for s, e in _quoted(text, start, end):
        entries.append(_entry(text, tr_id, "quoted_substring", s, e, bi))
    if not entries:
        entries.append(_entry(text, tr_id, "unclassified", start, end, bi))
    return entries


def build_catalog(text_region_id: str, region: dict[str, Any],
                  table_cells: list[tuple[int, int, int]] | None = None) -> dict[str, Any]:
    """Catalog every unit of one native text artifact. `table_cells` are (start, end, block_index)
    triples from the table module; nothing here infers a cell."""
    if region["status"] != "captured" or region["derivation"]["kind"] not in NATIVE_KINDS:
        raise CatalogError("source_not_native_text")
    text = region["text"]
    if _sha(text) != region["text_sha256"]:
        raise CatalogError("text_hash_mismatch")
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for block in region["blocks"]:
        for entry in unit_entries(text, text_region_id, block):
            if entry["span_id"] not in seen:
                seen.add(entry["span_id"])
                entries.append(entry)
    for start, end, bi in table_cells or []:
        entry = _entry(text, text_region_id, "table_cell", start, end, bi)
        if entry["span_id"] not in seen:
            seen.add(entry["span_id"])
            entries.append(entry)
    catalog = {"catalog_version": "mosaic.span-catalog/1", "text_region_id": text_region_id,
               "text_sha256": region["text_sha256"], "built_by": actor("deterministic", "mosaic.inventory.span_catalog"),
               "entries": entries}
    errors = registry.validate("span-catalog", catalog)
    if errors:
        raise CatalogError(f"catalog_invalid: {errors[:3]}")
    return catalog


def catalog_sha256(catalog: dict[str, Any]) -> str:
    from mosaic.canonical import canonical_bytes
    return hashlib.sha256(canonical_bytes(catalog)).hexdigest()


def selection_schema(catalogs: list[dict[str, Any]], *, many: bool = False,
                     block_indices: set[int] | None = None) -> dict[str, Any]:
    """Enum-constrained output schema over the span IDs offered to the model."""
    ids = [e["span_id"] for c in catalogs for e in c["entries"]
           if block_indices is None or e.get("block_index") in block_indices]
    one = {"type": "string", "enum": ids}
    return {"type": "array", "items": one, "uniqueItems": True} if many else one


def select(catalog: dict[str, Any], selected_id: str, text_region_id: str, region: dict[str, Any]) -> Selection:
    """Re-derive a selected span from the current text artifact and reject any drift."""
    if text_region_id != catalog["text_region_id"]:
        raise CatalogError("wrong_text_artifact")
    if region["status"] != "captured" or region["derivation"]["kind"] not in NATIVE_KINDS:
        raise CatalogError("source_not_native_text")
    entry = next((e for e in catalog["entries"] if e["span_id"] == selected_id), None)
    if entry is None:
        raise CatalogError("unknown_span_id")
    text = region["text"]
    if _sha(text) != catalog["text_sha256"] or region["text_sha256"] != catalog["text_sha256"]:
        raise CatalogError("catalog_stale")
    if not 0 <= entry["start"] < entry["end"] <= len(text):
        raise CatalogError("invalid_span_bounds")
    piece = text[entry["start"]:entry["end"]]
    if _sha(piece) != entry["text_sha256"] or span_id(text_region_id, entry["start"], entry["end"]) != entry["span_id"]:
        raise CatalogError("span_drift")
    return Selection(entry["span_id"], entry["kind"], text_region_id, entry["start"], entry["end"], piece,
                     entry["text_sha256"], entry["kind"] == "unclassified")
