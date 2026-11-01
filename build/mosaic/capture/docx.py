"""DOCX capture that preserves manual line breaks and tabs (audit M-11; FINAL v2 §3.2).

Adapted from the DOCX branch of Q/source_ingestion.py, which joined `w:t`
runs with no separator and so turned "item1<w:br/>item2" into "item1item2"
before hashing. Here each paragraph's children are visited in document order:
`w:t` text is copied, `w:tab` becomes "\\t", `w:br` and `w:cr` become "\\n",
`w:noBreakHyphen` becomes U+2011; each paragraph ends with "\\n". Paragraphs
inside tables carry a docx_table_cell locator. Legacy unit IDs
`<label>_para<NNNN>` enumerate every `w:p` in document order exactly as Q did,
and the mapping is accepted only if removing the inserted separators gives
Q's joined text.
"""

from __future__ import annotations

import hashlib
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mosaic.canonical import sha256_hex
from mosaic.package import PackageBuilder, actor

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
SEPARATORS = {W + "tab": "\t", W + "br": "\n", W + "cr": "\n", W + "noBreakHyphen": "‑"}
# Text-bearing parts outside word/document.xml. Capture v1 reads only the body,
# so each part that exists gets an explicit `excluded` disposition (M-02).
OTHER_TEXT_PARTS = re.compile(r"^word/(footnotes|endnotes|comments|header\d*|footer\d*)\.xml$")


@dataclass
class DocxResult:
    source_id: str
    text_region_id: str
    paragraphs: int
    embedded_media: list[str]
    legacy_units: dict[str, tuple[int, int]] = field(default_factory=dict)
    legacy_joined_text: dict[str, str] = field(default_factory=dict)
    disposition_ids: list[str] = field(default_factory=list)


def paragraph_text(p: ET.Element) -> tuple[str, str]:
    """Return (text with explicit separators, Q-style joined w:t text).

    `w:tab` inside paragraph properties (`w:pPr/w:tabs`) defines a tab stop, not
    a character, so property subtrees are skipped.
    """
    out: list[str] = []
    joined: list[str] = []
    properties = {id(el) for ppr in p.iter(W + "pPr") for el in ppr.iter()}
    for el in p.iter():
        if id(el) in properties:
            continue
        if el.tag == W + "t":
            out.append(el.text or "")
            joined.append(el.text or "")
        elif el.tag in SEPARATORS:
            out.append(SEPARATORS[el.tag])
    return "".join(out), "".join(joined)


def _locate_tables(body: ET.Element) -> dict[int, dict[str, int]]:
    """Map id(w:p) to its table-cell locator for paragraphs inside tables."""
    where: dict[int, dict[str, int]] = {}
    for ti, tbl in enumerate(body.iter(W + "tbl")):
        for ri, tr in enumerate(tbl.findall(W + "tr")):
            for ci, tc in enumerate(tr.findall(W + "tc")):
                for p in tc.iter(W + "p"):
                    where.setdefault(id(p), {"table_index": ti, "row": ri, "column": ci})
    return where


def _on(el: ET.Element | None) -> bool:
    return el is not None and el.get(W + "val", "true") not in ("0", "false", "off")


def _paragraph_format(p: ET.Element) -> dict[str, Any]:
    """Style id, list membership and all-bold runs, for heading and item detection. A list
    paragraph's number is rendered by Word from w:numPr and is not part of the captured text."""
    ppr = p.find(W + "pPr")
    style_el = ppr.find(W + "pStyle") if ppr is not None else None
    style = style_el.get(W + "val") if style_el is not None else None
    runs = [r for r in p.iter(W + "r") if "".join(t.text or "" for t in r.iter(W + "t")).strip()]
    return {"style": style,
            "list_item": (ppr is not None and ppr.find(W + "numPr") is not None) or (style or "").lower().startswith("list"),
            "bold": bool(runs) and all(_on(r.find(f"{W}rPr/{W}b")) for r in runs)}


def capture_docx(builder: PackageBuilder, source_id: str, source_label: str, path: Path) -> DocxResult:
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("word/document.xml"))
        media = {n: z.read(n) for n in sorted(z.namelist()) if n.startswith("word/media/")}
        other_parts = sorted(n for n in z.namelist() if OTHER_TEXT_PARTS.match(n))
    body = root.find(W + "body")
    tables = _locate_tables(body)
    text_parts: list[str] = []
    blocks: list[dict[str, Any]] = []
    legacy: dict[str, tuple[int, int]] = {}
    joined_map: dict[str, str] = {}
    pos = 0
    for i, p in enumerate(root.iter(W + "p")):
        text, joined = paragraph_text(p)
        ptext = text + "\n"
        loc = tables.get(id(p))
        locator = ({"kind": "docx_table_cell", "paragraph_index": i, **loc} if loc
                   else {"kind": "docx_paragraph", "paragraph_index": i})
        block = {"block_index": i, "start": pos, "end": pos + len(ptext), "kind": "table_cell" if loc else "paragraph",
                 "native_locator": locator, **_paragraph_format(p)}
        if joined.strip():
            uid = f"{source_label}_para{i + 1:04d}"
            if text.replace("\t", "").replace("\n", "").replace("‑", "") == joined.replace("‑", ""):
                block["legacy_unit_id"] = uid
                legacy[uid] = (pos, pos + len(text))
                joined_map[uid] = joined
        blocks.append(block)
        text_parts.append(ptext)
        pos += len(ptext)
    full = "".join(text_parts)
    tr = builder.envelope("text-region", "captured", actor("deterministic", "mosaic.capture.docx"), [source_id])
    tr.update(source_id=source_id, page_id=None,
              derivation={"kind": "docx_body", "engine": "zipfile+xml.etree", "engine_version": "stdlib",
                          "parent_text_region_id": None,
                          "separator_policy": "w:tab->TAB; w:br,w:cr->LF; w:noBreakHyphen->U+2011; paragraph end->LF"},
              text=full, text_sha256=sha256_hex(full.encode("utf-8")), blocks=blocks,
              reading_order=[b["block_index"] for b in blocks], reading_order_method="document_order")
    tr_id = builder.add(tr)
    created = actor("deterministic", "mosaic.capture.docx")
    disposition_ids = []
    body_disp = builder.envelope("disposition", "recorded", created, [source_id, tr_id])
    body_disp.update(subject_kind="source", subject_ref=f"{source_id}#word/document.xml", disposition="processed")
    disposition_ids.append(builder.add(body_disp))
    for name in other_parts:
        d = builder.envelope("disposition", "recorded", created, [source_id])
        d.update(subject_kind="attachment", subject_ref=f"{source_id}#{name}", disposition="excluded",
                 reason=f"text-bearing part {name} is not captured by DOCX capture v1 (word/document.xml only)")
        disposition_ids.append(builder.add(d))
    for name, data in media.items():
        d = builder.envelope("disposition", "recorded", created, [source_id])
        d.update(subject_kind="attachment", subject_ref=f"{source_id}#{name}", disposition="pending",
                 reason=(f"embedded media {name} (sha256 {hashlib.sha256(data).hexdigest()}, {len(data)} bytes) is kept "
                         "inside the source artifact; DOCX figure capture and recognition are not implemented in capture v1"))
        disposition_ids.append(builder.add(d))
    return DocxResult(source_id, tr_id, len(blocks), list(media), legacy, joined_map, disposition_ids)
