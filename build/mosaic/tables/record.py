"""Record every captioned table: interpreted when the recipe reactivates tables (M2), otherwise degraded (M1).

For an interpreted table (`single_page_resolved`) the table object holds its
columns (header paths, count roles), its cells and notes, each numeric cell
gets a `table_cell` witness with the full locator, and when the caption names
exactly one study the table's blocks are claimed for that study in the
section tree before the inventory is built, so values in the table body are
in-scope for the study the caption names rather than the study whose text
happens to surround the table (C-05 value-to-study; E's caption ownership).
A table of several complete ruled panels (D-037) carries a `panels` record
instead: each panel's blocks are claimed for the study its own title names,
so a cell's scope is its panel's study regardless of what the table-level
caption alone would resolve to.
A table that cannot be interpreted keeps the M1 graceful path: an
`unsupported` table object with the reason and a pending region disposition;
when its rules were located, the note under them is recorded as well.
A caption that is only its label ("Table 1") takes the block printed directly
under it, above where the table starts, as its title, read or not (once the
page's rules were read), and the study that title names owns the table as
for any caption.
A table refused because its values sit shifted across its columns was still
located by its rules, so its region and note are claimed for the study its
caption (a sideways table's separate title included) names exactly as a read
table's are, or each panel's for the study its own title names; no cell is
recorded.
The float node holds what the table object records as the table's own text,
read or refused, owned or not: its caption block, a title printed in a block
of its own and the blocks of its recorded notes, so they go wherever the
caption goes (D-152); its ruled region goes with them only when the caption
names one study. A caption that names a study the inventory does not have
("Experiment 1" in a work headed Study 1 and Study 2) owns nothing, and the
inventory never places its float by where it prints.

A continued caption whose label matches an earlier table in the same source
always records `continuation.parent_table_id` pointing at that root table,
even when the fragment itself cannot be read (D-066). The fragment's
own page is read and linked (`continues_table`) only when the parent is
itself `single_page_resolved`, the fragment resolves `single_page_resolved`
on its own page, its column count, header paths and stub (row-label) column
positions match the parent's exactly, and its caption does not name a
different study than the parent's owner, nor an ambiguous set of several
studies at once; otherwise it stays `unresolved_continuation` with a reason
naming the failed condition. A read fragment's cells carry the fragment's
own table id, never the parent's (MOSAIC objects are write-once and
content-addressed), and inherit the parent's owner only when the fragment's
own caption names no study at all; its float then names the parent's
studies, so the inventory places its rows with the parent's.

A DOCX float's table has no page (`page_ids` empty; D-068): it
gets the same graceful-degradation object, with a reason read from the raw
`<w:tbl>` (row count, gridSpan/vMerge/nested-table presence, stdlib
zipfile+ElementTree only) naming what a reader would have to handle. No DOCX
grid interpretation is attempted (v0).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
import io
import hashlib
import zipfile
from typing import Any

import pymupdf

from mosaic.capture.docx import W as DOCX_W
from mosaic.inventory.section_tree import Node, Segment, SourceTree, _caption_studies, caption_column
from mosaic.inventory.witnesses import WitnessError, table_cell_witness, text_witness
from mosaic.package import PackageBuilder, actor
from mosaic.tables.degrade import CONTINUED, REASON, TABLE_CAPTION
from mosaic.tables.extract import NUMERIC, Grid, _hidden_layer, interpret, interpret_continuation_fragment
from mosaic.tables.measure_items import select_measure_item_cells

CREATED = actor("deterministic", "mosaic.tables.record")
DOCX_REASON_PREAMBLE = ("DOCX table interpretation is not built: mosaic.tables.extract reads PDF ruled-table "
                        "geometry only, so this table's own <w:tbl> is not read into cells and no value from it "
                        "is accepted")


NOT_IN_INVENTORY = "explicit_single_study_caption_not_in_inventory"


def ownership(float_studies: list[str], known: set[str] | None = None) -> tuple[str | None, str]:
    """E's rule (extraction_table_ownership.py): only one explicit study in the caption owns the table. A study
    the inventory does not have (`known`: the labels of the work's study headings; "Experiment 1" in a work headed
    Study 1 and Study 2) owns nothing: no inventory item is that study."""
    if len(float_studies) == 1 and known is not None and float_studies[0] not in known:
        return None, NOT_IN_INVENTORY
    if len(float_studies) == 1:
        return f"study:{float_studies[0]}", "explicit_single_study_caption"
    if len(float_studies) > 1:
        return None, "multiple_explicit_study_captions_no_override"
    return None, "no_explicit_study_caption_no_override"


def _caption_witness(builder: PackageBuilder, own: list[Segment], extra: list[tuple[str, int, int]] = ()) -> str | None:
    text = builder.objects[own[0].text_region_id]["text"]
    spans = []
    for s in own:
        a, b = s.start, s.end
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if b > a:
            spans.append((s.text_region_id, a, b))
    if not spans:
        return None
    spans += list(extra)
    join = None
    if len(spans) > 1:  # a separate title block: a line break when nothing lies between, else a marked omission
        gaps = [text[e1:s2] for (_, _, e1), (_, s2, _) in zip(spans, spans[1:])]
        join = "line_break" if all(not g.strip() and "\n" in g for g in gaps) else "omitted_material_marker"
    try:
        return text_witness(builder, spans, attribution_role="author_report", created_by=CREATED,
                            retrieval_transform="table caption block", join=join)
    except WitnessError:
        return None


def _title_under(region: dict[str, Any], node: Node, table_top: float) -> list[tuple[int, int, str]]:
    """The title of a caption that is only its label ("Table 1"), printed in a block of its own directly under
    the label ("Wording of Three Factors of Study 1", KS-DEV-01): the nearest block below the label's block,
    after it in the text, overlapping it and starting within one of its line heights, and above where the
    table starts (`Grid.table_top`: its first full-width rule, or the title over its first panel), so column
    heads and panel titles are never a title; on a page with no full-width rule below the label (inf:
    KS-DEV-01's rules exist only in its scan, and no vector drawing on a page taken as a scan is a rule) nothing
    bounds it. The caller asks only when the page's rules
    were read and, if there are any, form the table's top, header and bottom rules (with no top rule, the first
    rule is the one under the column heads). Empty for a caption that has more than its label, and for a label
    not set horizontally (a sideways title is read from the table's own frame). Labels set side by side on one
    baseline share a block: each is measured from its own line, and of a title block under them it takes only the
    lines in its own column (D-193)."""
    cap = (node.caption or node.title).strip()
    m = TABLE_CAPTION.match(cap)
    if not m or cap[m.end():].strip(" .:|"):
        return []
    blk = next((b for b in region["blocks"] if b.get("bbox") and b["start"] <= node.heading.start < b["end"]), None)
    if blk is None:
        return []
    lines = [ln for ln in blk.get("lines") or [] if ln.get("bbox")]
    label = next((ln for ln in lines if ln["start"] <= node.heading.start <= ln["end"]), None)
    x0, y0, x1, y1 = label["bbox"] if label else blk["bbox"]
    if x1 - x0 <= y1 - y0:
        return []
    below = [b for b in region["blocks"] if b.get("bbox") and b["start"] >= blk["end"]
             and min(b["bbox"][2], x1) - max(b["bbox"][0], x0) > 0.5 and y1 - 1 <= b["bbox"][1] <= y1 + (y1 - y0)
             and b["bbox"][1] < table_top - 1]
    if not below:
        return []
    title = min(below, key=lambda b: b["bbox"][1])
    beside = [ln["bbox"][0] for ln in lines if ln is not label and region["text"][ln["start"]:ln["end"]].strip()]
    if label is not None and beside:  # other captions' labels on this one's baseline: only this column's lines
        spans = [(ln["start"], ln["end"]) for ln in title.get("lines") or []
                 if ln.get("bbox") and caption_column(ln["bbox"][0], [x0] + beside) == 0]
    else:
        spans = [(title["start"], title["end"])]
    out: list[tuple[int, int, str]] = []
    for a, z in spans:
        if out and not region["text"][out[-1][1]:a].strip():  # lines printed one after another: one span
            a = out.pop()[0]
        while a < z and region["text"][z - 1].isspace():
            z -= 1
        if z > a:
            out.append((a, z, region["text"][a:z]))
    return out


def _notes(builder: PackageBuilder, tr_id: str, grid: Grid) -> list[dict[str, Any]]:
    notes = []
    for a, b, text in grid.notes:
        try:
            wid = text_witness(builder, [(tr_id, a, b)], attribution_role="author_report", created_by=CREATED,
                               retrieval_transform="table note")
            notes.append({"text": text, "witness_id": wid})
        except WitnessError:
            notes.append({"text": text})
    return notes


def _page_identity(page: dict[str, Any], page_id: str) -> dict[str, Any]:
    return {"page_id": page_id, "physical_index": page["physical_index"], "display_ordinal": page["display_ordinal"],
            "printed_label": page["printed_label"]}


def _region_claims(region: dict[str, Any], tr_id: str, page: dict[str, Any], grid: Grid) -> list[Segment]:
    """The blocks of a located table's ruled region, for the study its caption names (D-018). Its note and title
    are claimed with the float whether or not the region is (`_blocks_holding`)."""
    x0, y0, x1, y1 = grid.bbox
    return [Segment(tr_id, page["physical_index"], blk["start"], blk["end"]) for blk in region["blocks"]
            if blk.get("bbox") and x0 - 2 <= (blk["bbox"][0] + blk["bbox"][2]) / 2 <= x1 + 2
            and y0 - 2 <= (blk["bbox"][1] + blk["bbox"][3]) / 2 <= y1 + 2]


def _blocks_holding(region: dict[str, Any], tr_id: str, page: dict[str, Any], spans: list[tuple[int, int, str]]) -> list[Segment]:
    """The blocks that hold any of these spans (a table's recorded notes, its separately printed title): the
    table object's text, so the float node holds them and they go wherever its caption goes (D-152). A block that
    also holds other text (the titles of captions set side by side) gives only the spans (D-193)."""
    out = []
    for blk in region["blocks"]:
        mine = [(max(a, blk["start"]), min(b, blk["end"])) for a, b, _ in spans if blk["start"] < b and a < blk["end"]]
        if not mine:
            continue
        rest = [ch for k, ch in enumerate(region["text"][blk["start"]:blk["end"]], start=blk["start"])
                if not ch.isspace() and not any(a <= k < b for a, b in mine)]
        out += [Segment(tr_id, page["physical_index"], blk["start"], blk["end"])] if not rest else \
            [Segment(tr_id, page["physical_index"], a, b) for a, b in mine]
    return out


def _claim_panels(tree: SourceTree, node: Node, region: dict[str, Any], tr_id: str, page: dict[str, Any],
                  grid: Grid) -> None:
    """Each panel owns only its own blocks (its title line down through its bottom rule), for the study its title
    names (D-037); the table-level owner (None here: the caption itself names no study) is left alone. The title
    line lies above the panel's top rule, so it is claimed by its span (D-196). A panel naming a study the inventory
    does not have keeps its node and rows, which the inventory then places nowhere by where they print (D-182)."""
    for p in grid.panels:
        px0, py0, px1, py1 = p["bbox"]
        claims = [Segment(tr_id, page["physical_index"], blk["start"], blk["end"])
                  for blk in region["blocks"] if blk.get("bbox")
                  and px0 - 2 <= (blk["bbox"][0] + blk["bbox"][2]) / 2 <= px1 + 2
                  and py0 - 2 <= (blk["bbox"][1] + blk["bbox"][3]) / 2 <= py1 + 2]
        if claims and p.get("title_span"):
            claims += _blocks_holding(region, tr_id, page, [(*p["title_span"], p["title"])])
        if not claims:
            continue
        panel_node = Node(f"{node.node_id}-{p['panel_id']}", node.source_id, node.source_label,
                          node.section, p["title"], "float", node.level, False, None, None,
                          parent_id=node.node_id, float_studies=[p["owner_study_local_id"].split(":", 1)[1]])
        tree.nodes.append(panel_node)
        tree.claim(panel_node, claims)


def _root_link_reason(root_body: dict[str, Any]) -> str | None:
    """None when `root_body` is a valid link target for a continuation fragment; otherwise why it is not.
    `record_tables`'s own storage invariant (a root is only ever stored from the `resolved` branch, which
    unconditionally sets `structure_status='single_page_resolved'` on every body it stores) means this
    never fires through the normal document-reading path today; kept, and unit-tested directly against a
    hand-built `root_body`, as a guard against a future change to that invariant silently treating an
    unresolved table as a valid link target."""
    if root_body["structure_status"] != "single_page_resolved":
        return "parent table is not single_page_resolved"
    return None


def _block_index_at(region: dict[str, Any], pos: int) -> int | None:
    for blk in region["blocks"]:
        if blk["start"] <= pos < blk["end"]:
            return blk["block_index"]
    return None


def _docx_table_after(root: ET.Element, caption_block_index: int | None,
                      claimed: set[int]) -> tuple[ET.Element | None, bool]:
    """First unclaimed top-level <w:tbl> whose own first paragraph comes after the caption's paragraph in
    document order (the same global w:p enumeration mosaic.capture.docx.capture_docx uses for block_index);
    marks it claimed so a second caption in the same source cannot re-pair it. The flag says whether a
    following table was passed over because an earlier caption had already claimed it."""
    body = root.find(DOCX_W + "body")
    if body is None or caption_block_index is None:
        return None, False
    positions = {id(p): i for i, p in enumerate(root.iter(DOCX_W + "p"))}
    skipped_claimed = False
    for ti, tbl in enumerate(body.findall(DOCX_W + "tbl")):
        first_p = next(tbl.iter(DOCX_W + "p"), None)
        if first_p is None or positions.get(id(first_p), -1) <= caption_block_index:
            continue
        if ti in claimed:
            skipped_claimed = True
            continue
        claimed.add(ti)
        return tbl, skipped_claimed
    return None, skipped_claimed


def _docx_table_facts(tbl: ET.Element) -> dict[str, Any]:
    """Row count and gridSpan/vMerge/nested-table presence, read directly from the raw XML; not a grid
    interpretation (no header/body split, no logical columns, no cells)."""
    rows = tbl.findall(DOCX_W + "tr")
    gridspan_rows: list[int] = []
    vmerge = False
    nested = False
    for ri, tr in enumerate(rows):
        for tc in tr.findall(DOCX_W + "tc"):
            span = tc.find(f"{DOCX_W}tcPr/{DOCX_W}gridSpan")
            if span is not None and int(span.get(DOCX_W + "val", "1")) > 1 and ri not in gridspan_rows:
                gridspan_rows.append(ri)
            if tc.find(f"{DOCX_W}tcPr/{DOCX_W}vMerge") is not None:
                vmerge = True
            if next(tc.iter(DOCX_W + "tbl"), None) is not None:
                nested = True
    return {"rows": len(rows), "gridspan_rows": gridspan_rows, "vmerge": vmerge, "nested": nested}


def _docx_unsupported_reason(facts: dict[str, Any] | None, skipped_claimed: bool = False) -> str:
    if facts is None:
        if skipped_claimed:
            return DOCX_REASON_PREAMBLE + "; every <w:tbl> after this caption is already paired with an earlier caption"
        return DOCX_REASON_PREAMBLE + "; no <w:tbl> follows this caption in the document"
    rows = facts["gridspan_rows"]
    if rows:
        shown = ", ".join(str(r) for r in rows[:8])
        if len(rows) > 8:
            shown += f" (+{len(rows) - 8} more)"
        span_bit = f"row(s) {shown} hold a cell spanning more than one grid column (e.g. a wide value cell or a full-row sub-panel header)"
    else:
        span_bit = "no row spans more than one grid column"
    vmerge_bit = "vertically merged cells (w:vMerge) are present" if facts["vmerge"] else "no vertically merged cells"
    nested_bit = "a nested table is present" if facts["nested"] else "no nested table"
    return f"{DOCX_REASON_PREAMBLE}: a reader would have to handle {facts['rows']} rows; {span_bit}; {vmerge_bit}; {nested_bit}."


def record_tables(builder: PackageBuilder, trees: dict[str, SourceTree], *, interpret_tables: bool,
                  measure_items_only: bool = False, source_regions: list[dict[str, Any]] | None = None,
                  source_bytes: dict[str, bytes] | None = None) -> list[str]:
    objects = builder.objects
    table_ids: list[str] = []
    docs: dict[str, pymupdf.Document] = {}
    scans: dict[str, set[int]] = {}
    # (source_id, label's prefix words, label number) -> (root table's id, root table's own body, root table's stub
    # column indices, the studies its float names), updated as nodes are visited in document reading order; a
    # continuation fragment that itself links keeps the id pointing at the true ROOT (never an intermediate
    # fragment's id) while refreshing the body/stub used for the next fragment's comparison to this, the most
    # recently resolved page.
    continuation_roots: dict[tuple[str, ...], tuple[str, dict[str, Any], list[int], list[str]]] = {}
    docx_roots: dict[str, ET.Element] = {}
    claimed_docx_tables: dict[str, set[int]] = {}
    known = {n.study[1] for t in trees.values() for n in t.nodes if n.study}  # the inventory's studies (studies.py)
    def source_data(sid: str) -> bytes | None:
        if source_bytes is None:
            return None
        raw = source_bytes.get(sid)
        if not isinstance(raw, bytes) or hashlib.sha256(raw).hexdigest() != objects[sid]["artifact"]["sha256"]:
            raise ValueError("table_source_bytes_hash_mismatch")
        return raw
    try:
        for sid, tree in trees.items():
            for node in tree.nodes:
                if node.role != "float" or not node.heading or not TABLE_CAPTION.match(node.title):
                    continue
                tr_id = node.heading.text_region_id
                region = objects[tr_id]
                page_id = region.get("page_id")
                is_docx = not page_id and region["derivation"]["kind"] == "docx_body"
                if not page_id and not is_docx:
                    continue
                page = objects[page_id] if page_id else None
                # The caption is its own lines (the float node also holds the notes under the table): its block from
                # its label line, and of a block that captions set side by side share, only the lines in its column;
                # lines printed one after another are one span, and the lines between them are another caption's
                # (D-193).
                cap_block = next(b for b in region["blocks"] if b["start"] <= node.heading.start < b["end"])
                own: list[Segment] = []
                for s in sorted(node.caption_lines, key=lambda s: s.start) or [
                        Segment(tr_id, node.heading.physical_index, node.heading.start, cap_block["end"])]:
                    if own and not region["text"][own[-1].end:s.start].strip():
                        own[-1] = Segment(tr_id, s.physical_index, own[-1].start, s.end)
                    else:
                        own.append(Segment(tr_id, s.physical_index, s.start, s.end))
                cap_start, cap_end = own[0].start, own[-1].end
                foreign = [(a.end, b.start) for a, b in zip(own, own[1:])]
                grid = None
                selected_measure_cells: tuple[dict[str, Any], ...] = ()
                if (interpret_tables or measure_items_only) and page is not None:
                    if sid not in docs:
                        raw = source_data(sid)
                        docs[sid] = pymupdf.open(stream=raw, filetype="pdf") if raw is not None else pymupdf.open(objects[sid]["artifact"]["acquisition_path"])
                        # The pages capture took as scans whose text is a hidden layer: on any other page of such a
                        # PDF that is a page image, no vector drawing is a rule either (D-189, D-204).
                        scans[sid] = {objects[o["page_id"]]["physical_index"] for o in objects.values()
                                      if o["type"] == "text-region" and o["source_id"] == sid and o.get("page_id")
                                      and o["derivation"].get("ocr_text_layer") and _hidden_layer(
                                          docs[sid][objects[o["page_id"]]["physical_index"]], o["text"])}
                    grid = interpret(docs[sid][page["physical_index"]], region["text"], cap_start, cap_end,
                                     node.caption or node.title, foreign=foreign,
                                     scanned_pdf=bool(scans[sid] - {page["physical_index"]}), body_size=tree.body_size)
                    # Inspect every resolved grid as well as the two ordinary refusals that text-only measure tables
                    # reach. The retry exposes a leading wording column that the numeric reader otherwise files as a
                    # stub, then retains it only if an explicit item/anchor heading proves the DE-22 exception.
                    selected = select_measure_item_cells(grid)
                    if selected is None and (grid.status == "single_page_resolved" or (
                            grid.status == "unresolved" and ("hold text, not numbers" in grid.reason
                                                            or "no numeric value cells" in grid.reason))):
                        measure_grid = interpret(docs[sid][page["physical_index"]], region["text"], cap_start, cap_end,
                                                 node.caption or node.title, foreign=foreign,
                                                 scanned_pdf=bool(scans[sid] - {page["physical_index"]}), body_size=tree.body_size,
                                                 allow_text_values=True)
                        selected = select_measure_item_cells(measure_grid)
                        if selected is not None:
                            grid = measure_grid
                    if selected is not None:
                        grid.columns = [c for c in grid.columns if c["column_id"] in selected.column_ids]
                        grid.cells = list(selected.cells)
                        selected_measure_cells = selected.cells
                    elif measure_items_only and grid.status == "single_page_resolved":
                        # Keep the located float for its image and study ownership, but publish no data-table
                        # cells into a model-screen package. The narrow item path above is the only exception.
                        grid = Grid("unresolved", "data table cells excluded by capture scope", bbox=grid.bbox,
                                    notes=grid.notes, reading=grid.reading, title=grid.title, panels=grid.panels,
                                    stub=grid.stub, table_top=grid.table_top)
                title: list[tuple[int, int, str]] = []
                if grid is not None and grid.reading != "right":
                    # A sideways table's title can be its own block after the label ("Table 4" / "OLS Results
                    # (Study 2)"); the caption, and the study it names, include it, whether the located table was
                    # read or refused (only a located sideways table is turned back to the page frame).
                    title = grid.title
                elif grid is not None and grid.table_top is not None:  # where the table starts is known; read or not,
                    title = _title_under(region, node, grid.table_top)  # a label-only caption's title is its caption
                if title:
                    node.float_studies = _caption_studies(" ".join([node.caption or node.title] + [t for _, _, t in title]))
                owner, basis = ownership(node.float_studies, known)
                caption_witness = _caption_witness(builder, own, [(tr_id, a, b) for a, b, _ in title])
                caption: dict[str, Any] = {"text": " ".join([node.caption or node.title] + [t.strip() for _, _, t in title])}
                if caption_witness:
                    caption["witness_id"] = caption_witness
                # The label as printed, with its prefix, as figures._name names a float (D-201): "TABLE V", "Online
                # Appendix Table I" (not "Table I"). A continuation is found by prefix and number, so "TABLE 1" and
                # "Table 1 (continued)" are one table (D-206).
                m = TABLE_CAPTION.match(node.title)
                label = " ".join(m.group(0).split())
                key = (sid, *m.group("prefix").split(), m.group("label"))
                continued = bool(CONTINUED.search(caption["text"]))
                continuation: dict[str, Any] = {"status": "unresolved_continuation" if continued else "none", "parent_table_id": None}
                link_reason: str | None = None
                if continued and interpret_tables and not measure_items_only and page is not None:  # a DOCX caption has no page to read
                    root = continuation_roots.get(key)
                    if root is None:
                        link_reason = "no earlier table with this label in this source to link to"
                    else:
                        root_id, root_body, root_stub, root_studies = root
                        continuation["parent_table_id"] = root_id  # always recorded once a root is found,
                                                                    # even if the fragment below is not read
                        link_reason = _root_link_reason(root_body)
                        if link_reason is None:
                            frag_grid = interpret_continuation_fragment(
                                docs[sid][page["physical_index"]], region["text"], cap_start, cap_end, foreign=foreign,
                                scanned_pdf=bool(scans[sid] - {page["physical_index"]}), body_size=tree.body_size)
                            if frag_grid.status != "single_page_resolved":
                                link_reason = f"fragment does not resolve on its own page ({frag_grid.status}: {frag_grid.reason})"
                            elif len(frag_grid.columns) != len(root_body["columns"]):
                                link_reason = "fragment's column count differs from the parent table's"
                            elif [c["header_path"] for c in frag_grid.columns] != [c["header_path"] for c in root_body["columns"]]:
                                link_reason = "fragment's header_path differs from the parent table's"
                            elif frag_grid.stub != root_stub:
                                link_reason = "fragment's stub (row-label) column positions differ from the parent table's"
                            elif owner is not None and owner != root_body["owner_study_local_id"]:
                                link_reason = "fragment's own caption names a different study than the parent table's owner"
                            elif owner is None and basis == "multiple_explicit_study_captions_no_override":
                                # ownership() returns owner=None for two different reasons (its own `basis`
                                # distinguishes them): no study named at all (safe to inherit), or two or more
                                # named at once -- an ambiguous, unresolved claim, not an absence of one. The
                                # ambiguous case is never treated as "no study named".
                                link_reason = "fragment's own caption names multiple studies, ambiguous against the parent table's owner"
                            elif owner is None and basis == NOT_IN_INVENTORY:
                                link_reason = ("fragment's own caption names a study the inventory does not have, "
                                               "ambiguous against the parent table's owner")
                            else:
                                grid = frag_grid
                                if owner is None:  # inherit the parent's owner only when the fragment names no study,
                                    # and its float names the parent's studies, so the inventory places it with them
                                    owner, basis = root_body["owner_study_local_id"], f"continuation_of_{root_body['ownership_basis']}"
                                    node.float_studies, node.continues = list(root_studies), label
                                continuation = {"status": "continues_table", "parent_table_id": root_id,
                                                "column_mapping": {fc["column_id"]: pc["column_id"]
                                                                    for fc, pc in zip(frag_grid.columns, root_body["columns"])}}
                body = builder.envelope("table", "captured" if grid and grid.status == "single_page_resolved" else "unsupported",
                                        CREATED, [x for x in (page_id, caption_witness) if x])
                body.update(source_id=sid, page_ids=[page_id] if page_id else [], label=label, caption=caption,
                            owner_study_local_id=owner, ownership_basis=basis, continuation=continuation)
                resolved = grid is not None and grid.status == "single_page_resolved"
                if resolved:
                    cells = [{k: v for k, v in c.items() if k != "span"} for c in grid.cells]
                    body.update(structure_status="single_page_resolved", columns=grid.columns, cells=cells,
                                notes=_notes(builder, tr_id, grid))
                    if grid.panels:  # a panel naming a study the inventory lacks says so (its id is kept: the schema
                        # needs one), as a caption's owner does (D-182, D-196)
                        body.update(panels=[{**{k: v for k, v in p.items() if k not in ("bbox", "title_span")},
                                             **({"ownership_basis": NOT_IN_INVENTORY}
                                                if p["owner_study_local_id"].split(":", 1)[1] not in known else {})}
                                            for p in grid.panels])
                elif is_docx:
                    if sid not in docx_roots:
                        raw = source_data(sid)
                        with zipfile.ZipFile(io.BytesIO(raw) if raw is not None else objects[sid]["artifact"]["acquisition_path"]) as z:
                            docx_roots[sid] = ET.fromstring(z.read("word/document.xml"))
                    tbl, skipped_claimed = _docx_table_after(docx_roots[sid], _block_index_at(region, node.heading.start),
                                                             claimed_docx_tables.setdefault(sid, set()))
                    reason = _docx_unsupported_reason(_docx_table_facts(tbl) if tbl is not None else None, skipped_claimed)
                    body.update(structure_status="unresolved", unsupported_reason=reason, columns=[], cells=[])
                    if tbl is not None:  # its cell paragraphs are its rows, as a located PDF table's region is (D-207)
                        cells = {id(q) for q in tbl.iter(DOCX_W + "p")}
                        at = {i for i, q in enumerate(docx_roots[sid].iter(DOCX_W + "p")) if id(q) in cells}
                        tree.table_rows.append((label, [Segment(tr_id, None, blk["start"], blk["end"])
                                                        for blk in region["blocks"] if blk.get("block_index") in at]))
                else:
                    status = grid.status if grid is not None else ("unsupported_multi_page" if continued else "unresolved")
                    reason = link_reason or ((grid.reason or grid.status) if grid is not None else REASON)
                    body.update(structure_status=status, unsupported_reason=reason, columns=[], cells=[])
                    if grid is not None and grid.notes:  # its rules were located: the note under them is its note
                        body.update(notes=_notes(builder, tr_id, grid))
                table_id = builder.add(body)
                table_ids.append(table_id)
                if source_regions is not None:
                    # Separate model-input metadata; never alter native capture or
                    # the objects made by historical deterministic recipes.
                    source_regions.append({"table_id": table_id, "text_region_id": tr_id,
                                           "bbox": (list(grid.bbox or grid.exclusion_bbox)
                                                    if grid is not None and (grid.bbox or grid.exclusion_bbox) else None)})
                if resolved and continuation["status"] in ("none", "continues_table"):
                    # Root-chasing: the id always stays the true root's (never an intermediate fragment's),
                    # while the body/stub used for the NEXT fragment's comparison come from this, the most
                    # recently resolved page under this label.
                    continuation_roots[key] = (continuation.get("parent_table_id") or table_id, body, grid.stub,
                                                        list(node.float_studies))
                # The rows go with the float only when the table names one study (its owner, or one the inventory
                # does not have, whose float the inventory then places nowhere by its print position).
                claim_rows = len(node.float_studies) == 1
                disp = builder.envelope("disposition", "recorded", CREATED, [table_id])
                if resolved:
                    disp.update(subject_kind="region", subject_ref=table_id, disposition="processed",
                                page_identity=_page_identity(page, page_id))
                    context = [x for x in [caption_witness] + [n.get("witness_id") for n in body["notes"]] if x]
                    for cell in grid.cells:
                        if "span" in cell and (NUMERIC.match(cell["text"]) or cell in selected_measure_cells):
                            pos = cell["position"]
                            table_cell_witness(builder, tr_id, *cell["span"], table_id=table_id,
                                               row_index_or_path=pos["row_index_or_path"], column_id=pos["column_id"],
                                               header_path=pos["header_path"], bbox=cell.get("bbox"),
                                               context_witness_ids=context, created_by=CREATED)
                    if grid.panels:
                        _claim_panels(tree, node, region, tr_id, page, grid)
                    elif claim_rows:
                        tree.claim(node, _region_claims(region, tr_id, page, grid))
                else:
                    disp.update(subject_kind="region", subject_ref=table_id, disposition="pending",
                                reason=f"table not interpreted ({body['structure_status']}); text captured as ordinary blocks")
                    if page is not None:
                        disp.update(page_identity=_page_identity(page, page_id))
                    if grid is not None and grid.bbox:
                        # Refused for values it could not bind, but its ruled region was located: the region still
                        # belongs to the study its caption (or each panel's title) names, as a read table's does (D-018).
                        if grid.panels:
                            _claim_panels(tree, node, region, tr_id, page, grid)
                        elif claim_rows:
                            tree.claim(node, _region_claims(region, tr_id, page, grid))
                if grid is not None and page is not None:
                    # The float holds what the table object records as its own, read or refused, owned or not: its
                    # separately printed title and its notes go wherever its caption goes (D-152).
                    tree.claim(node, _blocks_holding(region, tr_id, page, title + grid.notes))
                    if grid.bbox:  # the inventory flags a non-empirical section that holds its rows (D-207)
                        tree.table_rows.append((label, _region_claims(region, tr_id, page, grid)))
                builder.add(disp)
    finally:
        for d in docs.values():
            d.close()
    return table_ids
