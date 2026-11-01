"""Text witnesses with verified joins (FINAL v2 §4.1, §4.3; audit M-08, M-09).

A witness is materialized only from code-point spans into immutable text
artifacts of one source. The copied text is sliced by code, never supplied by
a model. A multi-span witness must declare how its spans join, and the join is
checked against the artifacts before anything is written:

* `adjacent_no_gap`: span k ends exactly where span k+1 starts in the same
  artifact, or span k ends its page's text and span k+1 opens the next page's
  (whitespace aside); copied text is the plain concatenation.
* `line_break`: only whitespace containing a line break lies between the
  spans, or they sit at the end and start of consecutive pages; joined with
  LF.
* `omitted_material_marker`: real text lies between the spans; joined with
  " […] " so the result is never presented as continuous.
* `none`: separate occurrences of identical wording cited together; the
  copied text is that wording once. Different wording needs separate witnesses.
"""

from __future__ import annotations

from typing import Any

from mosaic.canonical import sha256_hex
from mosaic.inventory.span_catalog import NATIVE_KINDS
from mosaic.package import PackageBuilder, actor

OMISSION = " […] "
NATIVE = {"native_raw": "native", "docx_body": "native", "xlsx_cells": "native", "ocr_recognized": "recognized"}


class WitnessError(ValueError):
    """Raised with a stable reason code as its message."""


def _state(region: dict[str, Any]) -> str:
    """A native text layer that is a scan's OCR layer holds recognized words, not typeset ones (D-163)."""
    return "recognized" if region["derivation"].get("ocr_text_layer") else NATIVE[region["derivation"]["kind"]]


def _unmapped(region: dict[str, Any], start: int, end: int) -> bool:
    """Whether a span overlaps a character whose glyph has no faithful Unicode (the text region's
    `glyph_flags`): its text is not what the page shows, so the witness needs the same visual
    confirmation as recognized text (D-166)."""
    return any(f["start"] < end and start < f["end"] for f in region.get("glyph_flags", []))


def _page_of(objects: dict[str, dict[str, Any]], region: dict[str, Any]) -> dict[str, Any] | None:
    page_id = region.get("page_id")
    if not page_id:
        return None
    page = objects[page_id]
    return {"page_id": page_id, "physical_index": page["physical_index"], "display_ordinal": page["display_ordinal"],
            "printed_label": page["printed_label"]}


def _block_at(region: dict[str, Any], offset: int) -> dict[str, Any] | None:
    return next((b for b in region["blocks"] if b["start"] <= offset < max(b["end"], b["start"] + 1)), None)


def _geometry(region: dict[str, Any], start: int, end: int) -> list[list[float]]:
    boxes = []
    for b in region["blocks"]:
        for ln in b.get("lines", []):
            if ln["start"] < end and start < ln["end"] and "bbox" in ln:
                boxes.append(ln["bbox"])
    return boxes


def _check_join(objects, spans: list[tuple[str, int, int]], join: str) -> None:
    for (ra, sa, ea), (rb, sb, eb) in zip(spans, spans[1:]):
        a, b = objects[ra], objects[rb]
        if join == "none":
            if a["text"][sa:ea] != b["text"][sb:eb]:
                raise WitnessError("join_none_needs_identical_occurrences")
            continue
        if ra == rb:
            if sb < ea:
                raise WitnessError("spans_overlap_or_out_of_order")
            gap = a["text"][ea:sb]
            if join == "adjacent_no_gap" and gap:
                raise WitnessError("adjacent_no_gap_has_gap")
            if join == "line_break" and (gap.strip() or "\n" not in gap):
                raise WitnessError("line_break_gap_is_not_a_line_break")
            if join == "omitted_material_marker" and not gap.strip():
                raise WitnessError("omission_marker_without_omitted_text")
            continue
        # Different artifacts: allowed only across consecutive pages of one source, span k closing its
        # page and span k+1 opening the next; anything else is an omission.
        pa, pb = a.get("page_id"), b.get("page_id")
        consecutive = (pa and pb and a["source_id"] == b["source_id"]
                       and objects[pb]["physical_index"] == objects[pa]["physical_index"] + 1)
        edge = consecutive and not a["text"][ea:].strip() and not b["text"][:sb].strip()
        if join in ("adjacent_no_gap", "line_break") and not edge:
            raise WitnessError("cross_page_join_not_at_page_edges")
        if join == "omitted_material_marker" and edge:
            raise WitnessError("omission_marker_without_omitted_text")


def text_witness(builder: PackageBuilder, spans: list[tuple[str, int, int]], *, attribution_role: str,
                 join: str | None = None, legacy_id: str | None = None, provenance_ids: list[str] | None = None,
                 retrieval_transform: str | None = None, created_by: dict[str, str] | None = None) -> str:
    objects = builder.objects
    if not spans:
        raise WitnessError("no_spans")
    regions = []
    for tr_id, start, end in spans:
        region = objects.get(tr_id)
        if region is None or region["type"] != "text-region":
            raise WitnessError("unknown_text_artifact")
        if region["status"] != "captured":
            raise WitnessError("text_artifact_not_citable")  # quarantined or superseded text
        if region["derivation"]["kind"] not in NATIVE:
            raise WitnessError("text_artifact_not_citable")
        if not 0 <= start < end <= len(region["text"]):
            raise WitnessError("invalid_span_bounds")
        regions.append(region)
    if len({r["source_id"] for r in regions}) != 1:
        raise WitnessError("spans_from_different_sources")
    if len({_state(r) for r in regions}) != 1:
        raise WitnessError("mixed_transcription_states")
    if len(spans) > 1:
        if join is None:
            raise WitnessError("multi_span_needs_join")
        _check_join(objects, spans, join)
    pieces = [objects[tr]["text"][s:e] for tr, s, e in spans]
    if len(spans) == 1:
        copied = pieces[0]
    elif join == "none":
        copied = pieces[0]
    elif join == "adjacent_no_gap":
        copied = "".join(pieces)
    elif join == "line_break":
        copied = "\n".join(pieces)
    else:
        copied = OMISSION.join(pieces)
    if not copied.strip():
        raise WitnessError("empty_copy")
    first = regions[0]
    source = objects[first["source_id"]]
    body = builder.envelope("witness", "materialized", created_by or actor("deterministic", "mosaic.inventory.witnesses"),
                            sorted({tr for tr, _, _ in spans} | set(provenance_ids or [])))
    page = _page_of(objects, first)
    body.update(
        witness_kind="text_span", artifact_sha256=source["artifact"]["sha256"], source_id=first["source_id"],
        page=page, native_locator=None if page else (_block_at(first, spans[0][1]) or {}).get("native_locator"),
        spans=[{"text_artifact_id": tr, "start_codepoint": s, "end_codepoint": e} for tr, s, e in spans],
        copied_text=copied, copied_text_sha256=sha256_hex(copied.encode("utf-8")),
        transcription_state="recognized" if any(_unmapped(objects[tr], s, e) for tr, s, e in spans) else _state(first),
        attribution_role=attribution_role,
    )
    if len(spans) > 1:
        body["join"] = join
    geometry = [box for tr, s, e in spans for box in _geometry(objects[tr], s, e)]
    if geometry:
        body["geometry"] = geometry
    if legacy_id:
        body["legacy_id"] = legacy_id
    if retrieval_transform:
        body["retrieval_transform"] = retrieval_transform
    return builder.add(body)


def table_cell_witness(builder: PackageBuilder, text_region_id: str, start: int, end: int, *, table_id: str,
                       row_index_or_path: Any, column_id: str, header_path: list[str], bbox: list[float] | None = None,
                       context_witness_ids: list[str] | None = None, created_by: dict[str, str] | None = None) -> str:
    """A witness for one table cell: exact span of the page text plus the full cell locator (FINAL v2 §4.1).

    Table cells come only from native ruled-table interpretation (M2), never from OCR-recognized
    text, so the citability check is narrower than `text_witness`'s: same allowlist as
    `span_catalog.NATIVE_KINDS`.
    """
    objects = builder.objects
    region = objects[text_region_id]
    if region["status"] != "captured" or region["derivation"]["kind"] not in NATIVE_KINDS:
        raise WitnessError("text_artifact_not_citable")
    if not 0 <= start < end <= len(region["text"]):
        raise WitnessError("invalid_span_bounds")
    copied = region["text"][start:end]
    body = builder.envelope("witness", "materialized", created_by or actor("deterministic", "mosaic.tables"),
                            [text_region_id, table_id])
    body.update(witness_kind="table_cell", artifact_sha256=objects[region["source_id"]]["artifact"]["sha256"],
                source_id=region["source_id"], page=_page_of(objects, region), native_locator=None,
                spans=[{"text_artifact_id": text_region_id, "start_codepoint": start, "end_codepoint": end}],
                cell={"table_id": table_id, "row_index_or_path": row_index_or_path, "column_id": column_id,
                      "header_path": header_path},
                copied_text=copied, copied_text_sha256=sha256_hex(copied.encode("utf-8")),
                transcription_state="recognized" if _unmapped(region, start, end) else _state(region),
                attribution_role="author_result")
    if bbox:
        body["geometry"] = [bbox]
    if context_witness_ids:
        body["context_witness_ids"] = context_witness_ids
    return builder.add(body)
