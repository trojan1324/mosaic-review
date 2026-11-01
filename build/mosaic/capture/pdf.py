"""Native PDF capture with per-page dispositions (HANDOFF §6.3; audit M-01, M-02, C-04, M-10, N-60).

For every physical page: geometry, a native text artifact built from PyMuPDF
blocks (block text = lines joined by "\\n" plus a trailing "\\n", identical to
PyMuPDF's blocks mode, so Q's legacy unit IDs map exactly), line geometry and
fonts, reading order (Q's geometry rule), quality signals, six-position folio
candidates, 150/300 dpi renders and a per-character geometry sidecar. Every
page gets a disposition, including zero-text and garbled pages, which are
never dropped: garbled native text is kept as a quarantined text artifact with
its hash before any OCR replacement is considered. A scanned page's text layer
is another engine's OCR, so its text artifact says so (`ocr_text_layer`) and its
page image is not recorded as a figure (D-163); in a PDF with such pages, a page on
a page image with any text is one too. A page scan whose only text is a stamp in its
margins, and an image-only one, are read by OCR, and their page image is the scan,
not a figure (D-189). Every other image is an
`image_region`, which mosaic.inventory.figures later marks pending as decoration
(icons, specks, banners and logos in the margin bands), joins to a figure caption
or marks pending (D-165). Characters whose glyph has no faithful
Unicode (a control character, a math font's glyph passed through as "!", a symbol
set as a small image between a line's characters) are recorded as `glyph_flags`
on the native text (D-166).
Folio trust splits the pages into a main body and an appendix only at a line where
the section tree starts the appendix, and not where one folio sequence runs
across it (D-167).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf

from mosaic.adapted.k.page_labels import folio_candidates, resolve_labels, resolve_section
from mosaic.adapted.k.preflight import page_signals
from mosaic.adapted.q.reading_order import order_blocks
from mosaic.canonical import sha256_hex
from mosaic.capture import signals as sig
from mosaic.capture.ocr import ocr_png
from mosaic.inventory.section_tree import Line, appendix_title, body_font_size, heading_of, is_caption_line
from mosaic.package import PackageBuilder, actor

ENGINE = "pymupdf"
ENGINE_VERSION = pymupdf.VersionBind
SIDECAR_FORMAT = "JSON array of [unicode_code_point, x0, y0, x1, y1] per character from PyMuPDF rawdict, in reading-sort order"


@dataclass
class PageRecord:
    index: int
    page_id: str
    text_region_id: str | None
    ocr_text_region_id: str | None
    disposition_id: str
    disposition: str
    label: dict[str, Any]
    legacy_units: dict[str, tuple[int, int]] = field(default_factory=dict)  # unit_id -> (start, end) in native text


@dataclass
class CaptureResult:
    source_id: str
    pages: list[PageRecord]
    image_region_ids: list[str]
    legacy_ok: bool


BOLD_NAME = re.compile(r"(?i)(bold|black|heavy|demibold|medium|[.\-]B$|,B$)")
ITALIC_NAME = re.compile(r"(?i)(italic|oblique|[.\-]I$|[.\-]It$|,I$)")


def _typography(spans: list[dict[str, Any]]) -> dict[str, Any]:
    """Line size and weight/slant from the non-space spans (PyMuPDF flags: 2 italic, 16 bold; many
    embedded fonts only say it in the name)."""
    visible = [s for s in spans if s.get("text", "").strip()]
    if not visible:
        return {"size": 0.0, "bold": False, "italic": False}
    return {"size": round(max(float(s.get("size", 0.0)) for s in visible), 1),
            "bold": all(bool(s.get("flags", 0) & 16) or bool(BOLD_NAME.search(s.get("font", ""))) for s in visible),
            "italic": all(bool(s.get("flags", 0) & 2) or bool(ITALIC_NAME.search(s.get("font", ""))) for s in visible)}


def _block_text(block: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    parts: list[str] = []
    lines: list[dict[str, Any]] = []
    pos = 0
    for i, line in enumerate(block.get("lines", [])):
        text = "".join(s.get("text", "") for s in line.get("spans", []))
        lines.append({"start": pos, "end": pos + len(text), "bbox": [round(v, 2) for v in line["bbox"]],
                      "fonts": sorted({s.get("font", "") for s in line.get("spans", []) if s.get("text", "").strip()}),
                      **_typography(line.get("spans", []))})
        parts.append(text)
        pos += len(text) + 1  # the "\n" after every line
    return "\n".join(parts) + "\n", lines


def _flow_lines(block: dict[str, Any]) -> list[list[float]]:
    """Bboxes of a block's upright horizontal lines with visible text, the lines a text column is made of;
    sideways text (a rotated table, a margin watermark) does not mark a column (D-164)."""
    return [[round(v, 2) for v in ln["bbox"]] for ln in block.get("lines", [])
            if ln["dir"][0] > 0 and abs(ln["dir"][1]) < 0.1 and "".join(s.get("text", "") for s in ln.get("spans", [])).strip()]


def _horizontal_rule_boxes(page: pymupdf.Page) -> list[list[float]]:
    """Persist only long, thin vector rules as capture geometry.

    They are layout evidence, never extracted table content.  A later source
    fence can require them before withholding a captionless repeated-column
    grid, avoiding a guess from text layout alone.
    """
    frame = page.cropbox
    rows = []
    for drawing in page.get_drawings():
        rect = pymupdf.Rect(drawing["rect"])
        if (rect.width < frame.width * 0.55 or rect.height > 2.0
                or not frame.contains(rect) or len(drawing.get("items", [])) != 1):
            continue
        item = drawing["items"][0]
        if item[0] == "l":
            start, end = item[1], item[2]
            if abs(start.y - end.y) > 1.0:
                continue
        elif item[0] != "re":
            continue  # a long thin drawn rectangle can also be a horizontal rule
        rows.append([round(rect.x0, 2), round(rect.y0, 2), round(rect.x1, 2), round(rect.y1, 2)])
    return [list(row) for row in sorted(set(tuple(row) for row in rows), key=lambda row: (row[1], row[0], row[2]))]


def _blank_symbol_spans(page_dict: dict[str, Any]) -> list[pymupdf.Rect]:
    """Boxes of the spans in a math or symbol font that hold only whitespace."""
    return [pymupdf.Rect(s["bbox"]) for b in page_dict.get("blocks", []) if b.get("type") == 0
            for ln in b.get("lines", []) for s in ln.get("spans", [])
            if not s.get("text", "").strip() and sig.SYMBOL_FONT.search(s.get("font", ""))]


def _glyph_flags(block: dict[str, Any], pos: int, drawn_blanks: list[pymupdf.Rect]) -> list[dict[str, Any]]:
    """Glyph-mapping failures in a block's spans as code-point ranges of the page text, the block starting at
    `pos` (offsets as in `_block_text`). A blank symbol-font span counts only where a glyph is drawn: PyMuPDF
    also inserts a space at a gap between glyphs, which TEXT_INHIBIT_SPACES leaves out (D-166)."""
    flags = []
    for line in block.get("lines", []):
        for s in line.get("spans", []):
            text = s.get("text", "")
            drawn = not text.strip() and any(pymupdf.Rect(s["bbox"]).intersects(r) for r in drawn_blanks)
            flags += [{"start": pos + a, "end": pos + b, "reason": reason}
                      for a, b, reason in sig.unmapped_glyphs(text, s.get("font", ""), drawn)]
            pos += len(text)
        pos += 1  # the "\n" after every line
    return flags


def _inline_image_flags(lines: list[dict[str, Any]], images: list[tuple[float, ...]]) -> list[dict[str, Any]]:
    """Glyph flags on the lines of one block (offsets in the page text) at whose height a small image sits with the
    block's text on both sides of it: a symbol set as a picture ("p < .01" with its "<" drawn as an image), so
    the text there lacks a character the page shows (D-166)."""
    flags: list[dict[str, Any]] = []
    for x0, y0, x1, y1 in images:
        row = [ln for ln in lines if ln["bbox"][1] <= (y0 + y1) / 2 <= ln["bbox"][3]]
        if any(ln["bbox"][0] < x0 for ln in row) and any(ln["bbox"][2] > x1 for ln in row):
            for ln in row:
                flag = {"start": ln["start"], "end": ln["end"], "reason": "inline_image"}
                if ln["end"] > ln["start"] and flag not in flags:
                    flags.append(flag)
    return flags


def _tree_line(line: dict[str, Any]) -> Line:
    """A captured line as the section tree reads it: its text and typography."""
    spans = line.get("spans", [])
    return Line("", None, "", 0, 0, 0, 0, "".join(s.get("text", "") for s in spans), _typography(spans), True)


def _sections(page_dicts: list[dict[str, Any]], heights: list[float],
              candidates: dict[int, list[dict[str, Any]]]) -> dict[str, list[int]]:
    """Split pages at the first page whose top third opens a block with a line at which the section tree starts
    the appendix (an appendix heading, or a titled appendix line in body type; not a sentence that begins
    "Appendix", nor a float caption such as "Appendix Table A1:", which the tree reads as a caption first), unless
    folios on both sides of that page run in one sequence: they are then one section for folio trust (D-167)."""
    n = len(page_dicts)
    blocks = [[b for b in pd.get("blocks", []) if b.get("type") == 0] for pd in page_dicts]
    body = body_font_size([_tree_line(ln) for page in blocks for b in page for ln in b.get("lines", [])])
    whole = resolve_section("main_body", list(range(n)), candidates).labels
    folios = [i for i, label in whole.items() if label["status"] == "verified"]
    for idx in range(1, n):
        if min(folios, default=n) < idx <= max(folios, default=-1):
            continue  # the folio sequence runs on across this page
        for block in blocks[idx]:
            first = _tree_line((block.get("lines") or [{}])[0])
            if block["bbox"][1] <= heights[idx] / 3 and not is_caption_line(first.text) and (
                    heading_of(first, body, False) == "appendix" or appendix_title(first, body)):
                return {"main_body": list(range(idx)), "appendix": list(range(idx, n))}
    return {"main_body": list(range(n))}


def capture_pdf(builder: PackageBuilder, source_id: str, source_label: str, path: Path, *,
                renders: bool = True, ocr: bool = True, sections: dict[str, list[int]] | None = None) -> CaptureResult:
    doc = pymupdf.open(path)
    created = actor("deterministic", "mosaic.capture.pdf")
    # Who made the PDF, as its metadata states it: on a scan, the likeliest name for the OCR engine.
    maker = {k: (doc.metadata or {}).get(k) or None for k in ("creator", "producer")}
    named = ", ".join(f"PDF {k} {v!r}" for k, v in maker.items() if v) or "no PDF creator or producer stated"
    raw: list[dict[str, Any]] = []
    for idx, page in enumerate(doc):
        # Same flags as PyMuPDF's blocks mode, so block indices match Q's unit IDs;
        # images are captured separately from get_image_info.
        pd = page.get_text("dict", sort=True, flags=pymupdf.TEXTFLAGS_BLOCKS)
        blocks_mode = page.get_text("blocks", sort=True)
        raw.append({"page": page, "dict": pd, "blocks_mode": blocks_mode, "images": page.get_image_info()})
    page_dicts = [r["dict"] for r in raw]
    # get_text("dict") bboxes are in the pre-rotation, cropbox-relative frame; page.rect
    # swaps width/height under a 90/270 rotation, so everything that compares those bboxes
    # with a page size (the appendix heading band, folio zones, the column split) uses the
    # cropbox's own (unrotated) dimensions (audit capture finding #0; D-061).
    heights = [float(r["page"].cropbox.height) for r in raw]
    # Scans by rule (D-163), and pages whose words are in no text layer (a margin stamp over a page image; D-189).
    # In a PDF with scanned pages, a page on a page image with any text is a scan too, however short its text and
    # whether its layer is visible or not: a figure page or last page of the scan (D-189).
    texts = ["".join(_block_text(b)[0] for b in r["dict"].get("blocks", []) if b.get("type") == 0) for r in raw]
    scans = [sig.scanned_text_layer(r["page"], t) for r, t in zip(raw, texts)]
    unread = [sig.unread_scan(r["page"], t) for r, t in zip(raw, texts)]
    ruled = sum(bool(s) for s in scans)
    for i, r in enumerate(raw):
        share = sig.page_image_share(r["page"])
        if ruled and not scans[i] and not unread[i] and texts[i].strip() and share >= sig.SCAN_IMAGE_SHARE:
            scans[i] = (f"the largest image covers {share:.0%} of the page; {len(texts[i].strip())} characters of text, "
                        f"on a page image in a PDF {ruled} of whose pages are taken as scans by rule")
    candidates = {i: folio_candidates(r["dict"], r["page"].cropbox.width, r["page"].cropbox.height) for i, r in enumerate(raw)}
    sections = sections or _sections(page_dicts, heights, candidates)
    labels = resolve_labels(sections, candidates)
    render_dir = builder.renders_dir() / source_label
    records: list[PageRecord] = []
    image_regions: list[str] = []
    legacy_ok = True

    for idx, r in enumerate(raw):
        page, pd = r["page"], r["dict"]
        text_parts: list[str] = []
        blocks: list[dict[str, Any]] = []
        legacy: dict[str, tuple[int, int]] = {}
        pos = 0
        nonempty_spans = 0
        glyph_flags: list[dict[str, Any]] = []
        drawn_blanks = (_blank_symbol_spans(page.get_text("dict", flags=pymupdf.TEXTFLAGS_BLOCKS | pymupdf.TEXT_INHIBIT_SPACES))
                        if _blank_symbol_spans(pd) else [])
        small = sorted({tuple(info["bbox"]) for info in r["images"]
                        if max(info["bbox"][2] - info["bbox"][0], info["bbox"][3] - info["bbox"][1]) <= sig.DECORATION_MAX_SIDE})
        for bi, block in enumerate(pd.get("blocks", [])):
            if block.get("type") != 0:
                blocks.append({"block_index": bi, "start": pos, "end": pos, "kind": "image_placeholder",
                               "bbox": [round(v, 2) for v in block["bbox"]]})
                continue
            btext, lines = _block_text(block)
            glyph_flags += _glyph_flags(block, pos, drawn_blanks)
            nonempty_spans += sum(1 for ln in block.get("lines", []) for s in ln.get("spans", []) if s.get("text", "").strip())
            entry = {"block_index": bi, "start": pos, "end": pos + len(btext), "kind": "text",
                     "bbox": [round(v, 2) for v in block["bbox"]],
                     "lines": [{**ln, "start": ln["start"] + pos, "end": ln["end"] + pos} for ln in lines]}
            glyph_flags += _inline_image_flags(entry["lines"], small)
            # Q's unit ids enumerate PyMuPDF blocks mode; verify identical text before mapping.
            bm = r["blocks_mode"][bi] if bi < len(r["blocks_mode"]) else None
            if btext.strip():
                if bm is not None and bm[6] == 0 and bm[4] == btext:
                    uid = f"{source_label}_p{idx + 1:03d}_b{bi:03d}"
                    entry["legacy_unit_id"] = uid
                    legacy[uid] = (pos, pos + len(btext))
                else:
                    legacy_ok = False
            blocks.append(entry)
            text_parts.append(btext)
            pos += len(btext)
        text = "".join(text_parts)
        order_input = [{"block_index": b["block_index"], "bbox": b["bbox"], "text": text[b["start"]:b["end"]],
                        "lines": _flow_lines(pd["blocks"][b["block_index"]])}
                       for b in blocks if b["kind"] == "text" and text[b["start"]:b["end"]].strip()]
        # Rasters and drawing clusters, which can cross a column gutter (a full-width figure or table; D-191). A drawing
        # over half the page or more (a page background, a frame around the text) is left out before clustering: with
        # the rest it would join every drawing into one cluster the size of the page (D-202).
        frame = pymupdf.Rect(0, 0, page.cropbox.width, page.cropbox.height)
        drawings = [d for d in page.get_drawings() if (pymupdf.Rect(d["rect"]) & frame).get_area() < frame.get_area() / 2]
        graphics = [list(info["bbox"]) for info in r["images"]] + (
            [list(c) for c in page.cluster_drawings(clip=frame, drawings=drawings)] if order_input and drawings else [])
        order, layout = order_blocks(order_input, page.cropbox.width, page.cropbox.height, graphics) if order_input else ([], {"confidence": "single_column"})
        others = [b["block_index"] for b in blocks if b["block_index"] not in order]
        reading = order + others
        pre = page_signals(page, pd, text, [tuple(b[:5]) for b in r["blocks_mode"] if len(b) > 6 and b[6] == 0])
        signals = {
            "char_count": len(text),
            "nonempty_span_count": nonempty_spans,
            "zero_text": nonempty_spans == 0,
            "replacement_char_ratio": sig.replacement_ratio(text),
            "private_use_char_count": sig.private_use_count(text),
            "image_count": len(page.get_images()),
            "drawing_count": len(page.get_drawings()),
            "runon_tokens": sig.runon_tokens(text),
            "reading_order_confidence": layout.get("confidence", "single_column"),
            "recommended_engine": pre["recommended_engine"],
            "visually_flagged": bool(len(page.get_images()) or len(page.get_drawings()) >= 5 or len(text.strip()) < 100),
        }
        render_records = []
        png300 = None
        if renders:
            for dpi in (150, 300):
                png = page.get_pixmap(dpi=dpi).tobytes("png")
                digest = hashlib.sha256(png).hexdigest()
                rel = f"renders/{source_label}/p{idx:04d}_{dpi}.png"
                builder.workspace.write_once(render_dir / f"p{idx:04d}_{dpi}.png", png)
                render_records.append({"dpi": dpi, "sha256": digest, "relative_path": rel})
                if dpi == 300:
                    png300 = png
        chars = []
        for block in page.get_text("rawdict", sort=True).get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    for ch in span.get("chars", []):
                        # Code points, not strings: rawdict passes unmapped glyphs through as lone
                        # surrogates (the other modes show U+FFFD), which are not valid UTF-8.
                        chars.append([ord(ch["c"]), *[round(v, 2) for v in ch["bbox"]]])
        sidecar = json.dumps(chars, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        sidecar_sha = hashlib.sha256(sidecar).hexdigest()
        builder.workspace.write_once(render_dir / f"p{idx:04d}_chars.json", sidecar)
        page_body = builder.envelope("page", "captured", created, [source_id])
        page_body.update(
            source_id=source_id, physical_index=idx, display_ordinal=idx + 1, printed_label=labels[idx],
            label_candidates=candidates[idx],
            geometry={"width": round(page.rect.width, 3), "height": round(page.rect.height, 3), "rotation": int(page.rotation),
                      "mediabox": [round(v, 3) for v in page.mediabox], "cropbox": [round(v, 3) for v in page.cropbox],
                      "horizontal_rule_boxes": _horizontal_rule_boxes(page)},
            signals=signals, renders=render_records,
            char_geometry_sidecar={"sha256": sidecar_sha, "relative_path": f"renders/{source_label}/p{idx:04d}_chars.json", "chars": len(chars),
                                   "format": SIDECAR_FORMAT},
        )
        page_id = builder.add(page_body)
        garbled = sig.is_garbled(text)
        scan, margins_only = scans[idx], unread[idx]
        page_scan = sig.page_image_share(page) >= sig.SCAN_IMAGE_SHARE  # one image is the whole page
        tr_status = "quarantined" if garbled else "captured"
        tr = builder.envelope("text-region", tr_status, created, [page_id])
        derivation = {"kind": "native_raw", "engine": ENGINE, "engine_version": ENGINE_VERSION, "parent_text_region_id": None,
                      "separator_policy": "block text = lines joined by LF + trailing LF (PyMuPDF blocks mode); blocks concatenated in sort order"}
        if scan:  # another engine recognized these words from the page image; witnesses on them are `recognized`
            derivation["ocr_text_layer"] = {"pdf_creator": maker["creator"], "pdf_producer": maker["producer"], "basis": scan}
        tr.update(source_id=source_id, page_id=page_id, derivation=derivation,
                  text=text, text_sha256=sha256_hex(text.encode("utf-8")), blocks=blocks, reading_order=reading,
                  reading_order_method="q.geometry_column_order_v1")
        if glyph_flags:  # characters that are not what the page shows; witnesses on them are `recognized`
            tr["glyph_flags"] = glyph_flags
        tr_id = builder.add(tr)
        ocr_tr_id = None
        disp = builder.envelope("disposition", "recorded", created, [page_id])
        page_identity = {"page_id": page_id, "physical_index": idx, "display_ordinal": idx + 1, "printed_label": labels[idx]}
        disp.update(subject_kind="page", subject_ref=page_id, page_identity=page_identity)
        flags = []
        if signals["zero_text"]:
            flags.append("zero_text")
        if garbled:
            flags.append("replacement_characters_over_1pct")
        if signals["private_use_char_count"]:
            flags.append("private_use_glyphs")
        if signals["runon_tokens"]:
            flags.append("runon_tokens")
        if signals["reading_order_confidence"] in ("requires_visual_review", "unresolved"):
            flags.append("reading_order_review")
        if scan:
            flags.append("scanned_ocr_text_layer")
        if margins_only:
            flags.append("page_image_text_in_margins")
        if glyph_flags:
            flags.append("unmapped_glyphs")
        needs_ocr = signals["zero_text"] or garbled or bool(margins_only)
        # Why the native text does not stand for the page, and, on a page that is one image, that the image is the scan.
        native_is = ("empty" if signals["zero_text"] else "garbled" if garbled else
                     f"only a stamp over a page image with no OCR layer ({margins_only})")
        as_scan = "; its page image is the scan, not a figure" if page_scan else ""
        if needs_ocr:
            result = ocr_png(png300 or page.get_pixmap(dpi=300).tobytes("png")) if ocr else None
            if result is None:
                disp.update(disposition="pending", reason="OCR required but disabled for this run" + (
                    f": native text {native_is}" if margins_only else "") + as_scan,
                    ocr={"status": "ocr_unavailable", "engine": None, "engine_version": None})
            elif result.status == "ocr_succeeded":
                otr = builder.envelope("text-region", "captured", actor("deterministic", "mosaic.capture.ocr"), [page_id, tr_id])
                otr.update(source_id=source_id, page_id=page_id,
                           derivation={"kind": "ocr_recognized", "engine": result.engine, "engine_version": result.engine_version or "unknown",
                                       "parent_text_region_id": tr_id, "transform": "OCR of the 300 dpi render"},
                           text=result.text, text_sha256=sha256_hex(result.text.encode("utf-8")),
                           blocks=[{"block_index": 0, "start": 0, "end": len(result.text), "kind": "text"}], reading_order=[0],
                           reading_order_method="ocr_engine_order")
                ocr_tr_id = builder.add(otr)
                disp.update(disposition="processed", reason=f"native text {native_is}; OCR text used as a recognized derivative" + as_scan,
                            ocr={"status": "ocr_succeeded", "engine": result.engine, "engine_version": result.engine_version, "text_region_id": ocr_tr_id})
            else:
                disp.update(disposition="unavailable_or_illegible",
                            reason=("no native text; " if signals["zero_text"] else "garbled native text; " if garbled else
                                    f"native text {native_is}; ") + (result.detail or result.status) + as_scan,
                            ocr={"status": result.status, "engine": result.engine, "engine_version": result.engine_version})
            if garbled:
                disp["quarantine"] = {"content_sha256": sha256_hex(text.encode("utf-8")), "relative_path": f"objects/{tr_id}.json",
                                      "signal": f"replacement_char_ratio={signals['replacement_char_ratio']}", "text_region_id": tr_id}
                if disp["disposition"] != "processed":
                    disp["disposition"] = "quarantined"
        else:
            disp.update(disposition="processed", ocr={"status": "not_needed", "engine": None, "engine_version": None})
            if scan:  # a rule's outcome, not an inspection: a born-digital page on a full-page background image meets it too
                disp["reason"] = (f"taken as a scanned page by rule ({scan}): its native text is treated as a third-party "
                                  f"OCR layer ({named}) and recorded as recognized text, not born-digital text, and its "
                                  "page image as the scan, not a figure")
        if flags:
            disp["signals"] = flags
        disp_id = builder.add(disp)
        for info in r["images"]:
            x0, y0, x1, y1 = [round(v, 2) for v in info["bbox"]]
            if (scan or needs_ocr) and sig.image_share(page, info["bbox"]) >= sig.SCAN_IMAGE_SHARE:
                continue  # the page scan itself, with a text layer or read by OCR: the page object and its renders stand for it
            if x1 > x0 and y1 > y0:  # decoration too: mosaic.inventory.figures marks it pending as such
                fig = builder.envelope("figure-equation", "captured", created, [page_id])
                fig.update(source_id=source_id, page_id=page_id, kind="image_region", label=None, bbox=[x0, y0, x1, y1],
                           caption_witness_id=None, crop_sha256=None, recognition_status="not_attempted")
                image_regions.append(builder.add(fig))
        records.append(PageRecord(idx, page_id, tr_id, ocr_tr_id, disp_id, disp["disposition"], labels[idx], legacy))
    doc.close()
    return CaptureResult(source_id, records, image_regions, legacy_ok)
