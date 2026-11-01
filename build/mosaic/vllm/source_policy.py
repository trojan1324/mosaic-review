"""Auditable model-input fence for detected floats (DE-22).

Capture remains immutable. Native units touching excluded material are
withheld; verified body-line fragments and exact item/anchor table cells are
offered separately with new identities. This is not proof every float was detected or
that the remaining prose is scientifically eligible: that audit stays open.
"""
from __future__ import annotations

from collections import Counter
import math
import re
import statistics
from types import SimpleNamespace

from mosaic.canonical import canonical_bytes, sha256_hex
from mosaic.inventory.q_copier import UnitRef, legacy_index
from mosaic.tables.measure_items import wording_field
from mosaic.validation import registry

VERSION = "mosaic.model-source-policy/5"
# Same neighbourhood as figure inventory's graphic-to-caption association.
FIGURE_TEXT_NEAR = 36.0
_TABLE_NOTE = re.compile(r"^\s*notes?\s*[:.]", re.IGNORECASE)


def conservative_source_policy(builder):
    """Direct backend callers lacking inventory geometry withhold float pages.

    The orchestrator supplies the richer persisted policy. Never silently use
    unrestricted legacy text when a caller bypasses that orchestration.
    """
    from mosaic.inventory.studies import build_trees
    regions = []
    for tid, table in builder.of_type("table"):
        for tr, region in builder.of_type("text-region"):
            if region.get("page_id") in table["page_ids"]:
                regions.append({"table_id": tid, "text_region_id": tr, "bbox": None})
    trees = build_trees(builder)
    # A direct caller may have captured a caption but not recorded its table.
    for tree in trees.values():
        for node in tree.nodes:
            if node.role == "float" and node.heading:
                regions.append({"table_id": "unrecorded", "text_region_id": node.heading.text_region_id, "bbox": None})
    return build_source_policy(builder, trees, regions)


def _overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _near(a, b):
    return max(b[0] - a[2], a[0] - b[2], b[1] - a[3], a[1] - b[3], 0) <= FIGURE_TEXT_NEAR


def _figure_text_seeds(objects, figure):
    """Graphics alone exclude neither external axis labels nor caption columns.

    These boxes only seed a conservative *exclusion* for nearby small type;
    they never classify it as body text or alter the captured figure object.
    """
    seeds = [figure["bbox"]] if figure.get("bbox") else []
    witness = objects.get(figure.get("caption_witness_id"), {})
    for span in witness.get("spans", []):
        region = objects.get(span["text_artifact_id"], {})
        if region.get("page_id") != figure.get("page_id"):
            continue
        for block in region.get("blocks", []):
            for line in block.get("lines", []):
                if (line.get("bbox") and span["start_codepoint"] < line["end"]
                        and line["start"] < span["end_codepoint"]):
                    seeds.append(line["bbox"])
    return seeds


def _adjacent_figure_lines(region, figure, seeds, body_size, headings, *, recover_side_caption=False):
    """Withhold uncertain nearby type, without claiming it is figure content.

    Small lines connect caption continuations and notes. A one-line block in
    body-sized type can also be an axis label; only its proximity to the
    original graphic seeds that exclusion. Running body paragraphs do not
    extend the neighbourhood. A mixed block is withheld whole by the caller.
    """
    body_fonts = {font for block in region["blocks"] if len(block.get("lines", [])) >= 3
                  for ln in block.get("lines", []) if abs(ln.get("size", 0) - body_size) <= 0.3
                  for font in ln.get("fonts", [])}
    pending, selected = [], []
    for block in region["blocks"]:
        lines = block.get("lines", [])
        for line in lines:
            if not line.get("bbox") or not 0 < line.get("size", 0):
                continue
            if (line["size"] <= body_size - 0.3 or
                    (recover_side_caption and body_fonts and line.get("fonts")
                     and not body_fonts.intersection(line["fonts"]))):
                pending.append(line)
            elif (len(lines) == 1 and line["size"] <= body_size + 0.3 and figure.get("bbox")
                  and _near(line["bbox"], figure["bbox"])
                  and not any(a < line["end"] and line["start"] < z for a, z in headings)):
                selected.append(line)
    seeds = seeds + [line["bbox"] for line in selected]
    while pending:
        nearby = [line for line in pending if any(_near(line["bbox"], seed) for seed in seeds)]
        if not nearby:
            break
        selected.extend(nearby)
        seeds.extend(line["bbox"] for line in nearby)
        pending = [line for line in pending if line not in nearby]
    return selected


def _line_rows(block):
    """Group horizontal lines that share a printed baseline.

    This is geometry only.  It never looks at a cell's value, header, or
    wording: a row is evidence of a grid only when it repeats three or more
    aligned columns across several baselines.
    """
    rows = []
    for line in block.get("lines", []):
        box = line.get("bbox")
        if not box or len(box) != 4 or not all(math.isfinite(v) for v in box):
            continue
        y = (box[1] + box[3]) / 2
        for anchor, members in rows:
            if abs(anchor - y) <= 1.5:
                members.append(line)
                break
        else:
            rows.append([y, [line]])
    return rows


def _column_anchor_counts(rows):
    """Return how many distinct printed rows use each aligned x position."""
    anchors = []
    for baseline, lines in rows:
        starts = []
        for line in lines:
            x = line["bbox"][0]
            for i, value in enumerate(starts):
                if abs(value - x) <= 3.0:
                    starts[i] = min(value, x)
                    break
            else:
                starts.append(x)
        if len(starts) < 2:
            continue
        for x in starts:
            for item in anchors:
                if abs(item[0] - x) <= 3.0:
                    item[1].add(round(baseline, 2))
                    break
            else:
                anchors.append([x, {round(baseline, 2)}])
    return sorted((x, len(ys)) for x, ys in anchors)


def _unlabeled_grid_boxes(objects):
    """Locate an uncaptioned ruled-like text grid from retained line geometry.

    Captioned tables are handled by ``record_tables``.  This intentionally
    smaller fallback covers a table body that has no recognized caption but
    has three repeated column positions on at least four printed rows.  A wide
    block alone, multiline prose, and separate newspaper-style columns do
    not meet this predicate.  The returned boxes only *withhold* source;
    they do not make a table object, read a cell, or create an item exception.
    """
    boxes = {}
    for tr_id, region in objects.items():
        if region.get("type") != "text-region" or not region.get("page_id"):
            continue
        page = objects.get(region["page_id"], {})
        geometry = page.get("geometry", {})
        width = geometry.get("width")
        if not isinstance(width, (int, float)) or not math.isfinite(width) or width <= 0:
            continue
        rules = []
        for rule in geometry.get("horizontal_rule_boxes", []):
            if (not isinstance(rule, list) or len(rule) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in rule)
                    or rule[2] <= rule[0] or rule[3] - rule[1] > 2.0):
                continue
            rules.append(rule)
        rows = []
        for block in region.get("blocks", []):
            box = block.get("bbox")
            if (block.get("kind") != "text" or not box or len(box) != 4
                    or not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]
                    or box[2] - box[0] < width * 0.55):
                continue
            for baseline, line_group in _line_rows(block):
                starts = []
                for line in line_group:
                    x = line["bbox"][0]
                    if not any(abs(old - x) <= 3.0 for old in starts):
                        starts.append(x)
                if len(starts) >= 3:
                    rows.append((baseline, line_group, block))
        rows.sort(key=lambda row: row[0])
        groups, current = [], []
        for row in rows:
            if current and row[0] - current[-1][0] > 28.0:
                groups.append(current)
                current = []
            current.append(row)
        if current:
            groups.append(current)
        grids = []
        for group in groups:
            anchors = _column_anchor_counts([(baseline, lines) for baseline, lines, _ in group])
            # At least three aligned columns each appear on four distinct
            # rows. This prevents a two-column prose layout, a hanging-indent
            # paragraph, or a single narrative list from becoming a table.
            # Two-column uncaptioned layouts remain unresolved rather than
            # risking a false positive source exclusion.
            if sum(count >= 4 for _, count in anchors) < 3:
                continue
            selected = {id(block): block for _, _, block in group}
            grid_box = [min(b["bbox"][0] for b in selected.values()), min(b["bbox"][1] for b in selected.values()),
                        max(b["bbox"][2] for b in selected.values()), max(b["bbox"][3] for b in selected.values())]
            # Repeated columns alone also occur in some prose layouts.  Require
            # captured vector-rule evidence: at least three long horizontal
            # rules aligned to the grid, with one at its upper boundary and one
            # at its lower boundary.  Old captures without this geometry stay
            # unresolved rather than being retroactively reclassified.
            def aligned(rule):
                overlap = max(0.0, min(rule[2], grid_box[2]) - max(rule[0], grid_box[0]))
                return overlap >= (grid_box[2] - grid_box[0]) * 0.8
            aligned_rules = [rule for rule in rules if aligned(rule)]
            ys = [(rule[1] + rule[3]) / 2 for rule in aligned_rules]
            if (sum(grid_box[1] - 28.0 <= y <= grid_box[3] + 28.0 for y in ys) < 3
                    or not any(grid_box[1] - 28.0 <= y <= grid_box[1] + 28.0 for y in ys)
                    or not any(grid_box[3] - 28.0 <= y <= grid_box[3] + 28.0 for y in ys)):
                continue
            grids.append((selected, grid_box))
            for block in selected.values():
                boxes.setdefault(tr_id, []).append((block["bbox"], "unlabeled_grid_geometry"))
        for selected, grid_box in grids:
            note_bottom = grid_box[3]
            for block in region.get("blocks", []):
                box = block.get("bbox")
                if (id(block) in selected or block.get("kind") != "text" or not box or len(box) != 4
                        or not all(math.isfinite(v) for v in box)
                        or not 0 <= box[1] - note_bottom <= 28.0):
                    continue
                # A table note has an explicit printed label and smaller type;
                # following small-font prose without that label stays eligible.
                text = region["text"][block["start"]:block["end"]]
                grid_sizes = [line.get("size", 0) for grid in selected.values() for line in grid.get("lines", [])
                              if line.get("size", 0) > 0]
                note_sizes = [line.get("size", 0) for line in block.get("lines", []) if line.get("size", 0) > 0]
                overlap = max(0.0, min(box[2], grid_box[2]) - max(box[0], grid_box[0]))
                if (not _TABLE_NOTE.match(text) or not grid_sizes or not note_sizes
                        or max(note_sizes) >= min(grid_sizes) - 0.2
                        or overlap < min(box[2] - box[0], grid_box[2] - grid_box[0]) * 0.8):
                    continue
                boxes.setdefault(tr_id, []).append((box, "unlabeled_grid_note"))
    return boxes


def _compact_note_grid_boxes(objects):
    """Fence a narrow class of compact two-column grids with an explicit note.

    This is intentionally stricter than the three-column fallback: it needs a
    short band of aligned vector rules, an explicitly labelled smaller note,
    and compact repeated right-edge cells.  It therefore does not treat an
    ordinary two-column paragraph as a data grid merely because a page uses
    rules and a note nearby.
    """
    boxes = {}
    for tr_id, region in objects.items():
        if region.get("type") != "text-region" or not region.get("page_id"):
            continue
        geometry = objects.get(region["page_id"], {}).get("geometry", {})
        rules = [rule for rule in geometry.get("horizontal_rule_boxes", [])
                 if isinstance(rule, list) and len(rule) == 4 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in rule)
                 and rule[2] > rule[0] and rule[3] - rule[1] <= 2.0]
        rules.sort(key=lambda rule: ((rule[1] + rule[3]) / 2, rule[0]))
        text_blocks = [block for block in region.get("blocks", []) if block.get("kind") == "text" and block.get("bbox")]
        notes = [block for block in text_blocks if _TABLE_NOTE.match(region["text"][block["start"]:block["end"]])]
        for note in notes:
            note_box = note["bbox"]
            note_top = note_box[1]
            bottom_index = next((i for i in range(len(rules) - 1, -1, -1)
                                 if 0 <= note_top - (rules[i][1] + rules[i][3]) / 2 <= 28.0), None)
            if bottom_index is None:
                continue
            previous_note_end = max((block["bbox"][3] for block in notes if block is not note and block["bbox"][3] <= note_box[1]), default=-math.inf)
            band = [rules[bottom_index]]
            for rule in reversed(rules[:bottom_index]):
                prior_y = (band[-1][1] + band[-1][3]) / 2
                y = (rule[1] + rule[3]) / 2
                if y <= previous_note_end or prior_y - y > 60.0:
                    break
                overlap = max(0.0, min(rule[2], band[0][2]) - max(rule[0], band[0][0]))
                if overlap < min(rule[2] - rule[0], band[0][2] - band[0][0]) * 0.8:
                    break
                band.append(rule)
            if len(band) < 3:
                continue
            top, bottom = band[-1], band[0]
            fence = [max(rule[0] for rule in band), (top[1] + top[3]) / 2,
                     min(rule[2] for rule in band), (bottom[1] + bottom[3]) / 2]
            if fence[2] <= fence[0] or fence[3] <= fence[1]:
                continue
            note_overlap = max(0.0, min(note_box[2], fence[2]) - max(note_box[0], fence[0]))
            if note_overlap < min(note_box[2] - note_box[0], fence[2] - fence[0]) * 0.8:
                continue
            inside = [block for block in text_blocks if fence[0] <= block["bbox"][0] and block["bbox"][2] <= fence[2]
                      and fence[1] <= block["bbox"][1] and block["bbox"][3] <= fence[3]]
            rows = []
            for block in inside:
                rows.extend(_line_rows(block))
            left, right = [], []
            for baseline, lines in rows:
                for side, x, line in [("left", line["bbox"][0], line) for line in lines] + [("right", line["bbox"][2], line) for line in lines]:
                    anchors = left if side == "left" else right
                    for anchor in anchors:
                        if abs(anchor["x"] - x) <= 4.0:
                            anchor["baselines"].add(round(baseline, 2))
                            anchor["widths"].append(line["bbox"][2] - line["bbox"][0])
                            break
                    else:
                        anchors.append({"x": x, "baselines": {round(baseline, 2)}, "widths": [line["bbox"][2] - line["bbox"][0]]})
            span = fence[2] - fence[0]
            left = [anchor for anchor in left if len(anchor["baselines"]) >= 4]
            right = [anchor for anchor in right if len(anchor["baselines"]) >= 4 and statistics.median(anchor["widths"]) <= span * 0.2]
            if not any(abs(r["x"] - l["x"]) >= span * 0.45
                       and len(l["baselines"] & r["baselines"]) >= 4 for l in left for r in right):
                continue
            grid_sizes = [line.get("size", 0) for block in inside for line in block.get("lines", []) if line.get("size", 0) > 0]
            note_sizes = [line.get("size", 0) for line in note.get("lines", []) if line.get("size", 0) > 0]
            if not grid_sizes or not note_sizes or max(note_sizes) >= min(grid_sizes) - 0.2:
                continue
            boxes.setdefault(tr_id, []).append((fence, "compact_note_grid_geometry"))
            boxes.setdefault(tr_id, []).append((note_box, "compact_note_grid_note"))
    return boxes


def _source_constraints(objects, trees, table_regions, *, side_caption_recovery=False):
    denied = {}
    headings = {}
    uncertain = set()
    boxes = {}

    def deny(tr, start, end, reason):
        denied.setdefault(tr, []).append((start, end, reason))

    for tree in trees.values():
        for s in tree.asides:
            deny(s.text_region_id, s.start, s.end, "aside_or_furniture")
        for node in tree.nodes:
            if node.heading and node.role in {"study", "methods", "results", "discussion", "introduction", "theory", "abstract"}:
                h = node.heading
                headings.setdefault(h.text_region_id, []).append((h.start, h.end))
            if node.role in {"float", "references", "furniture", "footnotes", "recognized"}:
                for s in node.segments:
                    deny(s.text_region_id, s.start, s.end, "non_body_section")
        for _, segments in tree.table_rows:
            for s in segments:
                deny(s.text_region_id, s.start, s.end, "table_region")
    for row in table_regions:
        tr = row["text_region_id"]
        if objects[tr].get("page_id"):
            if row["bbox"] is None:
                uncertain.add(tr)
            else:
                boxes.setdefault(tr, []).append((row["bbox"], "table_region"))

    # Some PDFs print an appendix grid without a recognizable "Table" caption.
    # The captioned-table path above remains authoritative; this fallback is
    # strictly a source fence for repeated aligned rows and an explicit note.
    for tr, entries in _unlabeled_grid_boxes(objects).items():
        boxes.setdefault(tr, []).extend(entries)
    for tr, entries in _compact_note_grid_boxes(objects).items():
        boxes.setdefault(tr, []).extend(entries)

    # Recovery may add text derivatives on an already captured page. A
    # derivative must never replace the native region in the page lookup:
    # figure geometry excludes overlapping text in every copy of that page.
    page_regions = {}
    for oid, o in objects.items():
        if o["type"] == "text-region" and o.get("page_id"):
            page_regions.setdefault(o["page_id"], []).append(oid)
    side_caption_regions = set()
    dispositions = {}
    body_sizes = {tree.source_id: tree.body_size for tree in trees.values()}
    for o in objects.values():
        if o["type"] == "disposition":
            dispositions.setdefault(o.get("subject_ref"), []).append(o.get("reason", ""))
    for oid, o in objects.items():
        if o["type"] != "figure-equation" or o.get("kind") not in {"figure", "image_region"}:
            continue
        if o["kind"] == "image_region" and any(
                "decoration (" in r or "the page's background" in r for r in dispositions.get(oid, [])):
            continue
        seeds = _figure_text_seeds(objects, o) if o["kind"] == "figure" else []
        for tr in page_regions.get(o.get("page_id"), []):
            if o.get("bbox") is None:
                uncertain.add(tr)
            else:
                boxes.setdefault(tr, []).append((o["bbox"], "figure_region"))
            region = objects[tr]
            body_size = body_sizes.get(region["source_id"], 0)
            side_caption = side_caption_recovery and bool(o.get("bbox")) and any(
                (seed[2] < o["bbox"][0] or o["bbox"][2] < seed[0])
                and min(seed[3], o["bbox"][3]) > max(seed[1], o["bbox"][1])
                and _near(seed, o["bbox"]) for seed in seeds[1:])
            if side_caption:
                side_caption_regions.add(tr)
            if o["kind"] == "figure":
                for line in _adjacent_figure_lines(region, o, seeds, body_size, headings.get(tr, []),
                                                   recover_side_caption=side_caption):
                    deny(tr, line["start"], line["end"], "figure_adjacent_text")

    # Newly recovered side-caption pages can contain a separately captured
    # small-type table note. Keep that uncertain text out, too. Applying this
    # beyond recovered pages would newly suppress previously admitted prose.
    for row in table_regions:
        tr, box = row["text_region_id"], row["bbox"]
        if not box or tr not in side_caption_regions:
            continue
        region = objects[tr]
        body_size = body_sizes.get(region["source_id"], 0)
        for line in _adjacent_figure_lines(region, {"bbox": box}, [box], body_size, headings.get(tr, [])):
            if line["size"] <= body_size - 0.3:
                deny(tr, line["start"], line["end"], "table_adjacent_small_text")

    # DOCX capture already records exact XML table membership, even without a
    # caption. Reuse that immutable locator; no source file is reopened here.
    for tr, region in objects.items():
        if region["type"] != "text-region" or region["derivation"]["kind"] != "docx_body":
            continue
        for block in region["blocks"]:
            if block.get("kind") == "table_cell" or block.get("native_locator", {}).get("kind") == "docx_table_cell":
                deny(tr, block["start"], block["end"], "docx_table")

    return denied, uncertain, boxes


def _unit_rows(objects, denied, uncertain, boxes):
    units = []
    for uid, ref in sorted(legacy_index(objects).items()):
        region = objects[ref.text_region_id]
        reasons = {r for start, end, r in denied.get(ref.text_region_id, [])
                   if ref.start < end and start < ref.end}
        if region["derivation"]["kind"] not in {"native_raw", "docx_body"} or region["derivation"].get("ocr_text_layer"):
            reasons.add("non_native_text")
        if ref.text_region_id in uncertain:
            reasons.add("float_bounds_unresolved")
        block = next(b for b in region["blocks"] if b.get("legacy_unit_id") == uid)
        for box, reason in boxes.get(ref.text_region_id, []):
            if not block.get("bbox"):
                reasons.add("block_geometry_unresolved")
            elif _overlap(block["bbox"], box):
                reasons.add(reason)
        units.append({"unit_id": uid, "text_region_id": ref.text_region_id,
                      "start": ref.start, "end": ref.end,
                      "status": "withheld" if reasons else "body_candidate", "reasons": sorted(reasons)})
    return units


def _body_fragments(objects, units, *, side_caption_recovery=False):
    """Re-derive native body-line runs from immutable capture, never stored claims.

    This deliberately narrow recovery excludes every recorded table page and
    independently rebuilds section/figure exclusions. A parent stays withheld.
    Unclassified gaps, small type and missing geometry cannot join a run.
    """
    parents = {u["unit_id"] for u in units if u["status"] == "withheld"
               and u["reasons"] == ["figure_adjacent_text"]}
    if not parents:
        return []
    from mosaic.inventory.studies import build_trees
    trees = build_trees(SimpleNamespace(objects=objects))
    denied, uncertain, boxes = _source_constraints(objects, trees, [], side_caption_recovery=side_caption_recovery)
    table_pages = {pid for o in objects.values() if o["type"] == "table" for pid in o["page_ids"]}
    body_sizes = {tree.source_id: tree.body_size for tree in trees.values()}
    rows = []
    for parent in _unit_rows(objects, denied, uncertain, boxes):
        if parent["unit_id"] not in parents or parent["reasons"] != ["figure_adjacent_text"]:
            continue
        tr = parent["text_region_id"]
        region = objects[tr]
        if (region["status"] != "captured" or region["derivation"]["kind"] != "native_raw"
                or not region.get("page_id") or region["page_id"] in table_pages):
            continue
        text = region["text"]
        if sha256_hex(text.encode()) != region["text_sha256"]:
            raise ValueError("model_source_fragment_capture_changed")
        block = next(b for b in region["blocks"] if b.get("legacy_unit_id") == parent["unit_id"])
        body_size = body_sizes.get(region["source_id"], 0)
        lines = block.get("lines", [])
        # Capture order/offsets must be intact; do not sort malformed lines
        # into apparently valid evidence or offer overlapping copies.
        if any(not parent["start"] <= ln["start"] < ln["end"] <= parent["end"]
               or (i and lines[i - 1]["end"] > ln["start"]) for i, ln in enumerate(lines)):
            continue
        runs = []
        current = None
        for line in lines:
            start, end, box = line["start"], line["end"], line.get("bbox")
            valid = (box and len(box) == 4 and all(math.isfinite(v) for v in box)
                     and box[0] < box[2] and box[1] < box[3] and body_size > 0
                     and abs(line.get("size", 0) - body_size) <= 0.3
                     and text[start:end].strip()
                     and not any(a < end and start < z for a, z, _ in denied.get(tr, []))
                     and not any(_overlap(box, b) for b, _ in boxes.get(tr, [])))
            if not valid or (current and text[current[1]:start].strip()):
                current = None
            if valid:
                if current is None:
                    current = [start, end]
                    runs.append(current)
                else:
                    current[1] = end
        for start, end in runs:
            row = {"parent_unit_id": parent["unit_id"], "text_region_id": tr, "start": start, "end": end,
                   "text_sha256": sha256_hex(text[start:end].encode())}
            rows.append({"unit_id": "fragment:" + sha256_hex(canonical_bytes(row)), **row})
    return rows


def build_source_policy(builder, trees, table_regions):
    objects = builder.objects
    units = _unit_rows(objects, *_source_constraints(objects, trees, table_regions, side_caption_recovery=True))

    table_ids = {r["table_id"] for r in table_regions}
    cells = []
    for wid, witness in sorted(builder.of_type("witness")):
        if witness.get("witness_kind") != "table_cell" or witness["cell"]["table_id"] not in table_ids:
            continue
        field = wording_field(witness["cell"]["header_path"])
        if field is None or len(witness["spans"]) != 1:
            raise ValueError("model_source_unexpected_table_cell")
        span = witness["spans"][0]
        cells.append({"unit_id": "cell:" + wid, "witness_id": wid, "field": field,
                      "text_region_id": span["text_artifact_id"], "start": span["start_codepoint"],
                      "end": span["end_codepoint"]})
    policy = {"version": VERSION, "units": units, "cells": cells,
              "fragments": _body_fragments(objects, units, side_caption_recovery=True),
              "source_audit": "unassessed", "counts": dict(Counter(u["status"] for u in units))}
    validate_source_policy(objects, policy)
    return policy


def validate_source_policy(objects, policy):
    if registry.validate("model-source-policy", policy):
        raise ValueError("model_source_policy_invalid")
    index = legacy_index(objects)
    if len(policy["units"]) != len(index) or {u["unit_id"] for u in policy["units"]} != set(index):
        raise ValueError("model_source_policy_unit_inventory_changed")
    for row in policy["units"]:
        ref = index[row["unit_id"]]
        if (row["text_region_id"], row["start"], row["end"]) != (ref.text_region_id, ref.start, ref.end):
            raise ValueError("model_source_policy_unit_span_changed")
        if (row["status"] == "withheld") != bool(row["reasons"]):
            raise ValueError("model_source_policy_inconsistent_status")
    if policy["counts"] != dict(Counter(u["status"] for u in policy["units"])):
        raise ValueError("model_source_policy_counts_changed")
    if policy["version"] in {"mosaic.model-source-policy/2", "mosaic.model-source-policy/3", "mosaic.model-source-policy/4", VERSION}:
        fragments = _body_fragments(objects, policy["units"], side_caption_recovery=policy["version"] == VERSION)
        if policy["fragments"] != fragments:
            raise ValueError("model_source_fragment_evidence_changed")
    seen = set()
    for row in policy["cells"]:
        w = objects.get(row["witness_id"], {})
        if row["unit_id"] in seen or row["unit_id"] != "cell:" + row["witness_id"]:
            raise ValueError("model_source_cell_identity_changed")
        seen.add(row["unit_id"])
        span = {"text_artifact_id": row["text_region_id"], "start_codepoint": row["start"], "end_codepoint": row["end"]}
        region = objects.get(row["text_region_id"], {})
        text = region.get("text", "")[row["start"]:row["end"]]
        if (w.get("witness_kind") != "table_cell" or w.get("spans") != [span]
                or wording_field(w.get("cell", {}).get("header_path", [])) != row["field"]
                or not text or text != w.get("copied_text") or sha256_hex(text.encode()) != w.get("copied_text_sha256")):
            raise ValueError("model_source_cell_evidence_changed")


def policy_index(objects, policy):
    validate_source_policy(objects, policy)
    index = legacy_index(objects)
    result = {u["unit_id"]: index[u["unit_id"]] for u in policy["units"] if u["status"] == "body_candidate"}
    for row in policy["cells"] + policy.get("fragments", []):
        region = objects[row["text_region_id"]]
        page = objects.get(region.get("page_id"), {})
        result[row["unit_id"]] = UnitRef(row["unit_id"], row["text_region_id"], row["start"], row["end"],
                                         False, page.get("physical_index"))
    return result


def policy_sha256(policy):
    return sha256_hex(canonical_bytes(policy))


def witness_source_status(objects, policy, witness_ids, family, predicate):
    """Check each assertion/alternative's own spans against what was offered.

    A body candidate still needs scientific review. Exact table wording has a
    narrower field allowlist; no table span can support an ordinary number.
    """
    if policy is None:
        return "unassessed"
    body = [u for u in policy["units"] if u["status"] == "body_candidate"] + policy.get("fragments", [])
    field = {("measure", "item_wording"): "item_wording", ("measure", "anchor"): "anchor"}.get((family, predicate))
    saw_cell = False
    for wid in witness_ids:
        for span in objects[wid].get("spans", []):
            tr, start, end = span["text_artifact_id"], span["start_codepoint"], span["end_codepoint"]
            if any(u["text_region_id"] == tr and u["start"] <= start < end <= u["end"] for u in body):
                continue
            if any(c["text_region_id"] == tr and c["start"] == start and c["end"] == end
                   and c["field"] == field for c in policy["cells"]):
                saw_cell = True
                continue
            return "outside_offered_source"
    return "measure_table_wording" if saw_cell else "body_candidate"
