"""Q copier adapter: legacy UTF-16 spans to code-point witnesses (FINAL v2 §4.1, §10.1; audit M-09).

Adapted from Q/next-steps/source_span_copy.py and grouped_source_copy.py
(REUSE_LEDGER R013, R014). Kept: UTF-16LE slicing that rejects a split
surrogate pair; the catalog drift check (span id must equal
`unit_id@start-end` and the slice must equal the catalogued text); refusal of
image-derived or unverified units; one witness per occurrence (Q's grouped
copy keeps occurrences apart, which is `join = none`, not a join). Changed:

* Offsets are labelled. Q's span ids carry UTF-16 offsets into a unit quote;
  the adapter converts them to code points and then to the captured text
  artifact, and verifies the converted substring against the artifact.
* Legacy span ids are re-issued as `legacy-utf16:<unit_id>@<start>-<end>`,
  because Q's ids disagree with the code-point offsets it returned beside them.
* Page identity is rebuilt from the fresh capture (zero-based physical index,
  display ordinal, printed label with its status); Q's one-based page integer
  is only cross-checked.
* DOCX units: Q's quote joined `w:t` runs without separators (M-11), so a
  position in Q's quote is mapped onto the captured paragraph by skipping the
  separators capture inserted, and the mapping is verified.
* Exact-text selections (unit id plus wording, no offsets) must match exactly
  one occurrence unless an occurrence index is given (T03 ambiguous case).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from mosaic.inventory.witnesses import text_witness
from mosaic.package import PackageBuilder, actor

INSERTED_SEPARATORS = "\t\n‑"
Q_PDF_UNIT = re.compile(r"^(?P<label>[a-z0-9]+)_p(?P<page>\d{3})_b(?P<block>\d{3})$")
Q_DOCX_UNIT = re.compile(r"^(?P<label>[a-z0-9]+)_para(?P<para>\d{4})$")


class CopyError(ValueError):
    """Raised with a stable reason code as its message."""


@dataclass(frozen=True)
class UnitRef:
    unit_id: str
    text_region_id: str
    start: int  # offset of the unit's first character in the text artifact
    end: int    # end of the unit's text (PDF: including the block's trailing LF, as in Q's quote)
    docx: bool
    physical_index: int | None


@dataclass(frozen=True)
class CopyResult:
    witness_id: str
    legacy_id: str
    text_region_id: str
    start: int
    end: int
    text: str
    located_by: str = "exact"


def legacy_index(objects: dict[str, dict[str, Any]]) -> dict[str, UnitRef]:
    """Q unit id -> the captured block that Layer 1 mapped it to (identical text verified there)."""
    out: dict[str, UnitRef] = {}
    for tr_id, region in objects.items():
        if region["type"] != "text-region" or region["status"] != "captured":
            continue
        docx = region["derivation"]["kind"] == "docx_body"
        page = objects.get(region.get("page_id") or "", {})
        for b in region["blocks"]:
            uid = b.get("legacy_unit_id")
            if not uid:
                continue
            end = b["end"] - 1 if docx else b["end"]  # a DOCX unit excludes the paragraph-end LF
            out[uid] = UnitRef(uid, tr_id, b["start"], end, docx, page.get("physical_index"))
    return out


def utf16_to_codepoints(text: str, start_utf16: int, end_utf16: int) -> tuple[int, int]:
    raw = text.encode("utf-16-le")
    if not 0 <= start_utf16 < end_utf16 <= len(raw) // 2:
        raise CopyError("invalid_span_bounds")
    try:
        prefix = raw[:start_utf16 * 2].decode("utf-16-le")
        piece = raw[start_utf16 * 2:end_utf16 * 2].decode("utf-16-le")
    except UnicodeDecodeError:
        raise CopyError("split_unicode_character") from None
    return len(prefix), len(prefix) + len(piece)


def _docx_position(captured: str, q_pos: int) -> int:
    """Captured offset of Q position `q_pos`, skipping separators that capture inserted."""
    seen = 0
    for i, ch in enumerate(captured):
        if ch in INSERTED_SEPARATORS:
            continue
        if seen == q_pos:
            return i
        seen += 1
    if seen == q_pos:
        return len(captured)
    raise CopyError("invalid_span_bounds")


def _locate(builder: PackageBuilder, ref: UnitRef, q_quote: str, cp_start: int, cp_end: int) -> tuple[int, int]:
    region = builder.objects[ref.text_region_id]
    captured = region["text"][ref.start:ref.end]
    if not ref.docx:
        if captured != q_quote:
            raise CopyError("unit_text_differs_from_capture")
        return ref.start + cp_start, ref.start + cp_end
    s = _docx_position(captured, cp_start)
    e = _docx_position(captured, cp_end)
    while e > s and captured[e - 1] in INSERTED_SEPARATORS:
        e -= 1  # a separator is never the last character of a copied span
    stripped = "".join(ch for ch in captured[s:e] if ch not in INSERTED_SEPARATORS)
    if stripped != q_quote[cp_start:cp_end]:
        raise CopyError("separator_mapping_mismatch")
    return ref.start + s, ref.start + e


def _check_page(ref: UnitRef, unit_id: str, q_page: int | None) -> None:
    m = Q_PDF_UNIT.match(unit_id)
    if m and ref.physical_index is not None and int(m.group("page")) != ref.physical_index + 1:
        raise CopyError("page_mismatch")
    if q_page is not None and ref.physical_index is not None and q_page != ref.physical_index + 1:
        raise CopyError("page_mismatch")


def copy_q_span(builder: PackageBuilder, index: dict[str, UnitRef], entry: dict[str, Any], q_units: dict[str, dict[str, Any]],
                *, attribution_role: str, located_by: str = "exact") -> CopyResult:
    """Copy one Q catalog entry {span_id, unit_id, start_utf16, end_utf16, text} into a witness."""
    unit_id = entry["unit_id"]
    a, z = entry["start_utf16"], entry["end_utf16"]
    if entry["span_id"] != f"{unit_id}@{a}-{z}":
        raise CopyError("catalog_source_drift")
    unit = q_units.get(unit_id)
    if unit is None:
        raise CopyError("unknown_unit")
    kind = unit.get("kind", "")
    if "image" in kind or "unverified" in kind:
        raise CopyError("source_not_native_text")
    ref = index.get(unit_id)
    if ref is None:
        raise CopyError("unit_not_in_capture")
    _check_page(ref, unit_id, unit.get("page"))
    quote = unit["quote"]
    cp_start, cp_end = utf16_to_codepoints(quote, a, z)
    if quote[cp_start:cp_end] != entry["text"]:
        raise CopyError("catalog_source_drift")
    start, end = _locate(builder, ref, quote, cp_start, cp_end)
    legacy_id = f"legacy-utf16:{unit_id}@{a}-{z}"
    transform = "Q UTF-16 unit offsets converted to code points in the captured text"
    if located_by != "exact":
        transform += "; the requested wording was located by a whitespace-normalized match and the exact source span copied"
    witness_id = text_witness(builder, [(ref.text_region_id, start, end)], attribution_role=attribution_role,
                              legacy_id=legacy_id, created_by=actor("deterministic", "mosaic.inventory.q_copier"),
                              retrieval_transform=transform)
    return CopyResult(witness_id, legacy_id, ref.text_region_id, start, end, builder.objects[ref.text_region_id]["text"][start:end],
                      located_by)


def _whitespace_hits(quote: str, wanted: str) -> list[tuple[int, int]]:
    """Spans of `quote` whose whitespace-collapsed form equals `wanted` collapsed (line breaks from PDF
    wrapping differ between a model's rendering and the source); the source span itself is returned."""
    chars: list[str] = []
    where: list[int] = []
    for i, ch in enumerate(quote):
        if ch.isspace():
            if chars and chars[-1] != " ":
                chars.append(" ")
                where.append(i)
        else:
            chars.append(ch)
            where.append(i)
    collapsed = "".join(chars)
    target = " ".join(wanted.split())
    if not target:
        return []
    return [(where[m.start()], where[m.start() + len(target) - 1] + 1) for m in re.finditer(re.escape(target), collapsed)]


def copy_q_exact_text(builder: PackageBuilder, index: dict[str, UnitRef], unit_id: str, exact_text: str,
                      q_units: dict[str, dict[str, Any]], *, attribution_role: str, occurrence: int | None = None) -> CopyResult:
    """Copy wording given as (unit id, exact text) without offsets: one occurrence, or an explicit index."""
    unit = q_units.get(unit_id)
    if unit is None:
        raise CopyError("unknown_unit")
    quote = unit["quote"]
    hits = [(m.start(), m.end()) for m in re.finditer(re.escape(exact_text), quote)] if exact_text else []
    located_by = "exact"
    if not hits:
        hits = _whitespace_hits(quote, exact_text)
        located_by = "whitespace_normalized"
    if not hits:
        raise CopyError("exact_text_not_in_unit")
    if occurrence is None and len(hits) > 1:
        raise CopyError("ambiguous_occurrence")
    if (occurrence or 0) >= len(hits):
        raise CopyError("occurrence_out_of_range")
    s, e = hits[occurrence or 0]
    a = len(quote[:s].encode("utf-16-le")) // 2
    z = a + len(quote[s:e].encode("utf-16-le")) // 2
    return copy_q_span(builder, index, {"span_id": f"{unit_id}@{a}-{z}", "unit_id": unit_id, "start_utf16": a,
                                        "end_utf16": z, "text": quote[s:e]}, q_units, attribution_role=attribution_role,
                       located_by=located_by)
