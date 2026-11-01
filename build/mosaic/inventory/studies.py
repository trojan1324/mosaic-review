"""Study inventory from the section trees (FINAL v2 §4.2 study inventory, §6.1; HANDOFF §6.4).

Deterministic headings propose boundaries; nothing here resolves an
ambiguous one. A study's region is the extent of its main-body heading node,
plus same-label heading nodes in the article's own appendix. Same-label
sections in a separate supplement file are added as proposals and make the
study `ambiguous`, with the reason written out, because a label match across
artifacts (whose version relation may be unresolved) is not evidence of
identity. Supplement sections with no study label become `supplemental_study`
items whose attribution is recorded as unresolved; the studies their body text
names are recorded as boundary evidence only. Every top-level region of every
source ends up in some inventory item, and this is checked: captured text that
no item holds is recorded on the paper-level item, which is then ambiguous.
Page indices restart in each file, so `physical_pages` is written only for an
item whose pages lie in one source; one spanning files lists them by source in
its evidence.

An article with no "Study N"/"Experiment N" heading gets one study, from its
first Method or Results heading through its last Results section, marked
ambiguous ("one study assumed"). With no such headings either, a main article
whose text reports a sample size and a test statistic has its paper-level
item, and every section reporting a test statistic, marked ambiguous: the
empirical content was not located. A top-level section whose heading names a
pretest, a pilot or a supplementary experiment is an ambiguous pretest or
pilot candidate. A source's page furniture is a non-empirical item of its
own, and its footnotes whose marker was not placed are another, marked
ambiguous, and so is its OCR text, whose headings are not read.

A table or figure goes with the study its caption names, notes included. In
a work with studies, one whose caption names none stays in the study open at
its caption only when that study holds all the other text on its page (a
DOCX has no pages, so there it stays);
otherwise its caption and notes go to a "(floats not attributed)" item,
marked ambiguous with the reason, and so do the shared caption and notes of
a table whose panels name studies, and every table or figure whose caption
names only studies the inventory does not have ("Experiment 1" in a work
headed Study 1 and Study 2), with the rows of such a table when the table
module located it. The rows of a table whose caption names no study stay
where they print; a non-empirical section (references, an appendix, the
front matter and so on) that holds text inside a table's located region is
`ambiguous`, naming the table (D-207).

Any item whose boundary falls on a page whose reading order capture left
unresolved or for visual review is `ambiguous`, naming the page. Page
furniture and footnotes are placed by rule, and tables and figures by their
captions, not by reading order, so they count on neither side of such a
boundary, and their own items have none (D-186, D-197). A study's item holds
the doubts recorded on any of its heading nodes (D-198), and a main article's
page from which no text was read makes the paper-level item ambiguous (D-199).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from mosaic.inventory.section_tree import (FURNITURE_NOTE, RECOGNIZED_NOTE, Line, Node, Segment, SourceTree,
                                           _caption_studies, _reading_lines, build_source_tree, merge_segments,
                                           normalize_title)
from mosaic.package import PackageBuilder, actor

NONEMPIRICAL_ROLES = {"abstract", "introduction", "theory", "discussion", "references", "appendix", "front_matter",
                      "furniture", "footnotes", "recognized"}
# A heading that names a pretest, a pilot or a supplementary experiment anywhere in it ("Detailed Description
# of Supplementary Experiment"); one that starts "Pretest"/"Pilot" is already a study heading (PILOT_STUDY_RE).
PRETEST_HEADING = re.compile(r"\b(?:pre-?test(?:s|ing)?|pilot|supplementary experiments?)\b", re.I)
# Empirical content in running text: a sample size ("n = 198") and a test statistic with its degrees of
# freedom ("t(36) = 2.73", "F(1, 94) = 8.94").
SAMPLE_SIZE = re.compile(r"\b[Nn]\s*=\s*\d+(?:,\d{3})*")
TEST_STATISTIC = re.compile(
    r"(?<![A-Za-z])(?:t|F|r|z|χ\s*2|χ²)\s*\(\s*\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?)?\s*\)\s*[=<>]\s*[-−]?\s*(?:\d+(?:\.\d+)?|\.\d+)")
FLOATS_NOTE = ("the caption and notes of each table or figure that names no study and prints where its place does not "
               "show which study it belongs to, or names only studies the inventory does not have; a table's rows are "
               "here with its caption only when the table module located the table and its caption, or for a panel's "
               "rows that panel's title, names one study the inventory does not have, and otherwise stay in the text "
               "around it (D-018, D-171, D-196)")


@dataclass
class InventoryResult:
    study_ids: dict[str, str]  # local_id -> object id
    trees: dict[str, SourceTree]


def _spans(segments) -> list[dict[str, Any]]:
    return [{"text_artifact_id": s.text_region_id, "start_codepoint": s.start, "end_codepoint": s.end} for s in segments]


def _pages(objects: dict[str, dict[str, Any]], segments) -> dict[str, list[int]]:
    """The segments' physical page indices by source: each file's indices start at 0."""
    out: dict[str, set[int]] = {}
    for s in segments:
        if s.physical_index is not None:
            out.setdefault(objects[s.text_region_id]["source_id"], set()).add(s.physical_index)
    return {sid: sorted(pages) for sid, pages in out.items()}


def _where(node: Node) -> str:
    page = f" page index {node.heading.physical_index}" if node.heading and node.heading.physical_index is not None else ""
    return f"{node.source_label}{page} ({node.section}) '{node.title}'"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", normalize_title(text)).strip("-")[:40] or "section"


def _first(pattern: re.Pattern[str], objects: dict[str, dict[str, Any]], segments: list[Segment]) -> str | None:
    """The first match of `pattern` in these segments, quoted, with its page index."""
    for s in segments:
        m = pattern.search(objects[s.text_region_id]["text"], s.start, s.end)
        if m:
            return f"'{m.group(0)}'" + (f", page index {s.physical_index}" if s.physical_index is not None else "")
    return None


def _named_studies(objects: dict[str, dict[str, Any]], segments: list[Segment], labels: list[str]) -> list[dict[str, str]]:
    """Boundary evidence for each known study that a region's text names ("In Study 2, we adapted ...").
    A mention is recorded, never used to attribute the region (D-011)."""
    first: dict[str, int | None] = {}
    for s in segments:
        for lab in _caption_studies(objects[s.text_region_id]["text"][s.start:s.end]):
            if lab in labels and lab not in first:
                first[lab] = s.physical_index
    return [{"kind": "section_tree", "text": f"body text names {lab}"
             + (f" (first on page index {p})" if p is not None else "") + "; not used to attribute the region"}
            for lab, p in first.items()]


def _rows_held(objects: dict[str, dict[str, Any]], tree: SourceTree, segments: list[Segment], role: str) -> list[str]:
    """A note for each table whose located region (`SourceTree.table_rows`, claimed or not) holds text of these
    segments, a non-empirical section's: a table whose caption names no study keeps its rows where they print
    (D-182 (3)), and they are not a references, appendix or other non-empirical section's own text (D-207). A table
    whose region was not located gives no note: nothing shows which text is its rows. A DOCX table's rows are the
    cell paragraphs of the <w:tbl> its caption was matched to; a DOCX has no pages, so its note gives none."""
    return [f"holds the rows of {label}"
            + (f" (page index {rows[0].physical_index})" if rows and rows[0].physical_index is not None else "")
            + f", printed inside this {role.replace('_', ' ')} section" for label, rows in tree.table_rows
            if any(not ch.isspace() for s in segments for r in rows if r.text_region_id == s.text_region_id
                   for ch in objects[s.text_region_id]["text"][max(s.start, r.start):min(s.end, r.end)])]


def _method_results_span(tree: SourceTree) -> list[Node]:
    """Nodes from the first Method or Results heading outside the appendix and any study through the end of
    the last Results heading's subsections (the last Method heading's when there is none), widened so that no
    node's subsections are split; nodes inside a study are left out. Descendants follow their node."""
    by_id = {n.node_id: n for n in tree.nodes}

    def in_study(n: Node | None) -> bool:
        while n is not None:
            if n.study:
                return True
            n = by_id.get(n.parent_id or "")
        return False

    hits = [k for k, n in enumerate(tree.nodes) if n.role in ("methods", "results") and n.section != "appendix"
            and not in_study(n)]
    if not hits:
        return []
    end = max((k for k in hits if tree.nodes[k].role == "results"), default=hits[-1])
    k = hits[0]
    while k <= end:
        end = max(end, k + len(tree.descendants(tree.nodes[k])))
        k += 1
    return [n for n in tree.nodes[hits[0]:end + 1] if not in_study(n) and not n.float_studies]


def _floats_not_attributed(objects: dict[str, dict[str, Any]], tree: SourceTree, taken: set[str],
                           unknown: dict[str, list[str]]) -> list[tuple[Node, str]]:
    """The tables and figures whose caption names no inventory study and whose place does not show which study
    they belong to, each with what the inventory says of it. Such a float stays with the study open at its
    caption (a study by label, the assumed single study, whose nodes are `taken`, or a top-level pretest or pilot
    candidate) only when that study holds all the other text on the float's pages (a DOCX float has no page, so
    it stays); a float whose panels name studies (D-037) never stays, since its shared caption and notes belong
    to none of them alone, and nor does one whose caption names only studies the inventory does not have
    (`unknown`: "Experiment 1" in a work headed Study 1 and Study 2), since its caption says it belongs to
    another study than any here, and which is not determinable."""
    by_id = {n.node_id: n for n in tree.nodes}

    def study_of(n: Node | None) -> str | None:
        while n is not None:
            if n.study:
                return f"study:{n.study[1]}"
            if n.node_id in taken:
                return "study:single"
            if n.parent_id is None and n.heading and PRETEST_HEADING.search(n.title):
                return f"the candidate '{n.title}'"
            n = by_id.get(n.parent_id or "")
        return None

    out: list[tuple[Node, str]] = []
    for f in tree.nodes:  # a panel naming only studies the inventory does not have goes with its rows (D-196)
        if f.role == "float" and not f.heading and f.node_id in unknown:
            out.append((f, f"names {', '.join(unknown[f.node_id])}, which is no study of the inventory"))
    for f in tree.nodes:
        if f.role != "float" or not f.heading or f.float_studies:
            continue
        pages = {s.physical_index for s in f.segments if s.physical_index is not None}
        home = study_of(by_id.get(f.parent_id or ""))
        panels = sorted({lab for n in tree.nodes if n.parent_id == f.node_id and n.role == "float"
                         for lab in n.float_studies + unknown.get(n.node_id, [])})
        others = {study_of(n) for n in tree.nodes if n.role not in ("float", "furniture", "footnotes") and any(
            s.physical_index in pages and objects[s.text_region_id]["text"][s.start:s.end].strip() for s in n.segments)}
        if f.node_id in unknown:
            out.append((f, f"names {', '.join(unknown[f.node_id])}, which is no study of the inventory"))
            continue
        if panels:
            reason = f"its panels name {', '.join(panels)}, so its shared caption and notes belong to none of them alone"
        elif home is None:
            top = by_id[f.parent_id]
            while top.parent_id:
                top = by_id[top.parent_id]
            reason = f"it lies outside any study (in '{top.title}')"
        elif others - {home}:
            reason = (f"it prints inside {home}, but its page also holds "
                      + ", ".join(sorted(x or "text outside any study" for x in others - {home})))
        else:
            continue
        out.append((f, f"names no study and {reason}"))
    return out


def _uncovered(objects: dict[str, dict[str, Any]], trees: dict[str, SourceTree], regions: list[dict[str, Any]]) -> list[str]:
    """Captured text that no inventory item holds, one entry per text region: coverage is checked, not assumed."""
    held = {oid: bytearray(len(o["text"])) for oid, o in objects.items()
            if o["type"] == "text-region" and o["status"] == "captured" and o["source_id"] in trees}
    for r in regions:
        if r["text_artifact_id"] in held:
            held[r["text_artifact_id"]][r["start_codepoint"]:r["end_codepoint"]] = b"\1" * (r["end_codepoint"] - r["start_codepoint"])
    out = []
    for oid, mask in held.items():
        region = objects[oid]
        lost = sum(1 for ch, m in zip(region["text"], mask) if not m and not ch.isspace())
        if lost:
            page = f" page index {objects[region['page_id']]['physical_index']}" if region.get("page_id") else ""
            out.append(f"{trees[region['source_id']].source_label}{page} ({oid}): {lost} characters of captured text are "
                       "in no inventory item")
    return out


def _aside_ranges(trees: dict[str, SourceTree]) -> dict[str, list[tuple[int, int]]]:
    """The code points of page furniture, footnotes and OCR text, by text region: text the section tree keeps out of
    the flow, placed by rule, not by reading order; and of every table and figure (caption, notes and claimed rows),
    which the inventory places by its caption, not by where it is read (D-197)."""
    out: dict[str, list[tuple[int, int]]] = {}
    for tree in trees.values():
        for s in tree.asides + [s for n in tree.nodes if n.role == "float" for s in n.segments]:
            out.setdefault(s.text_region_id, []).append((s.start, s.end))
    return out


def _page_entries(objects: dict[str, dict[str, Any]], trees: dict[str, SourceTree],
                  asides: dict[str, list[tuple[int, int]]]) -> dict[str, tuple[Line, Line | None]]:
    """For each native text region the section tree reads on a page whose reading order is unresolved or needs
    visual review: the page's first line read and the line read just before the page (None on a source's first
    page). Lines of whitespace do not count, and nor do page furniture and footnotes (`asides`): a running head or
    folio read first, or a footnote read last on the page before, says nothing of where a boundary falls."""
    out: dict[str, tuple[Line, Line | None]] = {}
    for sid in trees:
        before = None
        for ln in (ln for ln in _reading_lines(objects, sid) if ln.text.strip() and not any(
                a <= ln.start < b for a, b in asides.get(ln.text_region_id, ()))):
            region = objects[ln.text_region_id]
            page = objects.get(region.get("page_id") or "")
            if (page and region["derivation"]["kind"] == "native_raw" and ln.text_region_id not in out
                    and page["signals"]["reading_order_confidence"] in ("unresolved", "requires_visual_review")):
                out[ln.text_region_id] = (ln, before)
            before = ln
    return out


def _unsettled_pages(objects: dict[str, dict[str, Any]], segs, entries: dict[str, tuple[Line, Line | None]],
                     asides: dict[str, list[tuple[int, int]]]) -> list[str]:
    """The pages on which an item's boundary falls while the page's reading order is unresolved or needs visual
    review: the item holds some of the page's text, not all, or it holds the page's first line read but not the
    line read before the page, or that line but none of the page. Which side of the boundary the page's text
    falls on follows an order capture could not confirm (D-164); a heading read first on such a page
    may have text before it that belongs to the item it closes. Page furniture and footnotes are placed by rule,
    whatever the reading order, so they count on neither side (`asides`)."""

    def holds(ln: Line | None) -> bool:
        return ln is not None and any(s.text_region_id == ln.text_region_id and s.start <= ln.start < s.end for s in segs)

    out = []
    for tr_id in dict.fromkeys([s.text_region_id for s in segs] + [t for t, (_, b) in entries.items() if holds(b)]):
        if tr_id not in entries:
            continue
        region, (first, before) = objects[tr_id], entries[tr_id]
        page = objects[region["page_id"]]
        mask = bytearray(len(region["text"]))  # 1: the item's text, 2: furniture or footnotes
        for s in segs:
            if s.text_region_id == tr_id:
                mask[s.start:s.end] = b"\1" * (s.end - s.start)
        for a, b in asides.get(tr_id, ()):
            mask[a:b] = b"\2" * (b - a)
        held = {m for ch, m in zip(region["text"], mask) if not ch.isspace()}
        partial = 1 in held and 0 in held
        if partial or (before is not None and holds(first) != holds(before)):
            label = objects[region["source_id"]]["membership"][0]["source_label"]
            out.append(f"boundary falls on {label} page index {page['physical_index']} (printed "
                       f"{page['printed_label'].get('value') or 'none'}), whose reading order is "
                       f"{page['signals']['reading_order_confidence']}")
    return out


def build_trees(builder: PackageBuilder) -> dict[str, SourceTree]:
    objects = builder.objects
    return {sid: build_source_tree(objects, sid) for sid, o in objects.items() if o["type"] == "source"
            and any(x["type"] == "text-region" and x["source_id"] == sid for x in objects.values())}


def build_inventory(builder: PackageBuilder, trees: dict[str, SourceTree] | None = None) -> InventoryResult:
    """`trees` may be passed in after the table module has let caption owners claim their table bodies."""
    objects = builder.objects
    created = actor("deterministic", "mosaic.inventory.studies")
    sources = [(oid, o) for oid, o in objects.items() if o["type"] == "source"]
    trees = trees if trees is not None else build_trees(builder)
    main_ids = [sid for sid, o in sources if o["membership"][0]["role"] == "main_article" and sid in trees]
    relation = {sid: o["version"]["relation_to_published"] for sid, o in sources}
    asides = _aside_ranges(trees)
    entries = _page_entries(objects, trees, asides)
    out: dict[str, str] = {}

    def add(local_id: str, label: str, kind: str, segments, evidence, status: str, ambiguity: list[str],
            parent: str | None = None, by_rule: bool = False) -> None:
        """`by_rule`: an item whose text is placed by rule, not by reading order (page furniture, footnotes, floats
        placed by their captions), so no boundary of it falls on a page by that page's reading order."""
        segs = merge_segments(list(segments))
        boundary: dict[str, Any] = {"regions": _spans(segs)}
        pages = _pages(objects, segs)
        if len(pages) > 1:  # `physical_pages` has no source key: an item spanning files lists its pages by source
            evidence = [*evidence, {"kind": "section_tree", "text": "physical pages by source: " + "; ".join(
                f"{t.source_label} {pages[sid]}" for sid, t in trees.items() if sid in pages)}]
        else:
            boundary["physical_pages"] = next(iter(pages.values()), [])
        ambiguity = ambiguity + ([] if by_rule else _unsettled_pages(objects, segs, entries, asides))
        body = builder.envelope("study", "proposed", created, sorted({s.text_region_id for s in segs}))
        body.update(local_id=local_id, label=label, kind=kind, parent_local_id=parent,
                    inventory_status="ambiguous" if ambiguity else status, boundary=boundary,
                    boundary_evidence=evidence, source_ids=sorted({objects[s.text_region_id]["source_id"] for s in segs}))
        if ambiguity:
            body["ambiguity"] = "; ".join(ambiguity)
        out[local_id] = builder.add(body)

    # ---------------------------------------------------------------- studies by label
    study_nodes: list[tuple[SourceTree, Node]] = [(t, n) for t in trees.values() for n in t.nodes if n.study]
    labels: list[str] = []
    for _, n in study_nodes:
        if n.study[1] not in labels:
            labels.append(n.study[1])
    # A caption that names an existing study is attributed to it; one naming no known study stays in the flow,
    # unless it names only studies the inventory does not have.
    unknown: dict[str, list[str]] = {}
    for t in trees.values():
        for n in t.nodes:
            if n.float_studies and not any(lab in labels for lab in n.float_studies):
                unknown[n.node_id] = list(n.float_studies)
            n.float_studies = [lab for lab in n.float_studies if lab in labels]
    # No "Study N" heading: the nodes of the one study assumed (added below), known here so floats can be placed.
    span = [n for sid in main_ids for n in _method_results_span(trees[sid])] \
        if not any(n.study[0] == "study" for _, n in study_nodes) else []
    heads = [n for n in span if n.heading and n.role in ("methods", "results")]
    taken: set[str] = {n.node_id for n in span} if heads else set()  # left out of the sections they sit in
    # A table or figure that names no study leaves the study around it unless its place shows it belongs there; in a
    # work with no study, one that names a study the inventory does not have still never stays where it prints.
    held = {sid: [(f, said) for f, said in _floats_not_attributed(objects, t, taken, unknown)
                  if labels or heads or f.node_id in unknown] for sid, t in trees.items()}
    loose = {f.node_id for floats in held.values() for f, _ in floats}
    used_nodes: set[str] = set()
    for label in labels:
        nodes = [(t, n) for t, n in study_nodes if n.study[1] == label]
        primary = [(t, n) for t, n in nodes if t.source_id in main_ids and n.section == "main_body"]
        ambiguity: list[str] = []
        if not primary:
            primary = nodes[:1]
            ambiguity.append(f"no main-body heading for {label}; first heading is {_where(primary[0][1])}")
        head_tree, head = primary[0]
        segments = list(head_tree.extent(head, skip=loose))
        evidence = [{"kind": "heading", "text": _where(head)}]
        ambiguity.extend(head.notes)
        for t, n in nodes:
            if n is head:
                continue
            segments.extend(t.extent(n, skip=loose))
            evidence.append({"kind": "heading", "text": _where(n)})
            ambiguity.extend(x for x in n.notes if x not in ambiguity)  # its region's own doubts (D-198)
            if t.source_id == head_tree.source_id and n.section != head.section:
                continue  # the article's own appendix, headed with this study's label
            if t.source_id == head_tree.source_id:
                ambiguity.append(f"'{label}' is headed more than once in the same section: {_where(n)}")
            else:
                ambiguity.append(f"supplement section {_where(n)} attributed by heading label only "
                                 f"(version relation of {t.source_label}: {relation[t.source_id]})")
        for t in trees.values():
            for n in t.nodes:
                if label not in n.float_studies:
                    continue
                segments.extend(n.segments)
                evidence.append({"kind": "heading", "text": f"caption {_where(n)} continues {n.continues}, whose caption "
                                 f"names {label}" if n.continues else f"caption {_where(n)} names {label}"})
                if t.source_id != head_tree.source_id:
                    ambiguity.append(f"supplement caption {_where(n)} names {label} "
                                     f"(version relation of {t.source_label}: {relation[t.source_id]})")
        for t, n in nodes:
            used_nodes.add(n.node_id)
        kind, parent = _kind(head.study, labels)
        add(f"study:{label}", head.title, kind, segments, evidence, "discovered", ambiguity, parent)

    # ------------------------------------------ no "Study N" heading: one study assumed, or none located
    whole = [seg for sid in main_ids for n in trees[sid].nodes for seg in n.segments]
    unlocated: list[str] = []    # empirical content in a main article with no study, method or results heading
    if not any(n.study[0] == "study" for _, n in study_nodes):
        if heads:
            used_nodes |= taken
            add("study:single", f"one study assumed ('{heads[0].title}' to '{heads[-1].title}')", "study",
                [seg for n in span if n.node_id not in loose for seg in n.segments],
                [{"kind": "heading", "text": _where(n)} for n in heads],
                "discovered", [f"no study heading in the article: one study is assumed, from its method and results "
                               f"headings {[n.title for n in heads[:6]]}; whether these sections report one study or "
                               "several, and whether study text lies outside them, is not determinable from headings"])
        else:
            sample, statistic = _first(SAMPLE_SIZE, objects, whole), _first(TEST_STATISTIC, objects, whole)
            if sample and statistic:
                unlocated.append(f"the main article has no study, method or results heading, but its text reports a "
                                 f"sample size ({sample}) and a test statistic ({statistic}); its empirical content "
                                 "was not located, so no study is proposed")

    # ------------------------------------------------ top-level non-study regions of every source
    for sid, tree in trees.items():
        is_main = sid in main_ids
        for n in tree.nodes:
            if n.level != 1 or n.study or n.node_id in used_nodes:
                continue
            extent = tree.extent(n, skip_studies=True, skip=taken | loose)  # a study nested in it has its own item
            if not extent:
                continue
            local = f"study:sec/{tree.source_label}/{_slug(n.title)}-{n.node_id.rsplit('-n', 1)[1]}"
            children = [d.title for d in tree.descendants(n) if d.heading]
            evidence = [{"kind": "heading" if n.heading else "section_tree", "text": _where(n)}]
            if n.role in ("furniture", "recognized"):
                evidence.append({"kind": "section_tree", "text": FURNITURE_NOTE if n.role == "furniture" else RECOGNIZED_NOTE})
            notes = list(n.notes)
            kind, label = "nonempirical_section", n.title
            if not is_main and n.role not in NONEMPIRICAL_ROLES - {"front_matter"}:
                kind = "supplemental_study"
                label = n.title if n.heading else f"{tree.source_label} content without a study heading"
                notes.insert(0, f"supplement {tree.source_label} region '{n.title}'"
                             + (f" with sections {children[:6]}" if children else "")
                             + " has no study label; which study it belongs to is not determinable from headings")
            pretest = PRETEST_HEADING.search(n.title) if n.heading else None
            if pretest:
                kind = "pilot" if pretest.group(0).lower() == "pilot" else "pretest"
                notes.insert(0, f"heading '{n.title}' names a {pretest.group(0).lower()}, so the section is a candidate "
                                f"{kind}; whether it is one, and which study it serves, is not determinable from headings")
            if not is_main and kind != "nonempirical_section":
                evidence += _named_studies(objects, extent, labels)
            statistic = _first(TEST_STATISTIC, objects, extent) if unlocated and is_main else None
            if statistic:
                notes.append(f"its text reports a test statistic ({statistic}) but no study, method or results heading "
                             "was found; whether it holds an empirical study is not determinable from headings")
            if kind == "nonempirical_section" and n.role in NONEMPIRICAL_ROLES:
                notes += _rows_held(objects, tree, extent, n.role)
            add(local, label, kind, extent, evidence, "discovered", notes, by_rule=n.role in ("furniture", "footnotes"))
    for sid, floats in held.items():
        if floats:
            add(f"study:sec/{trees[sid].source_label}/floats-not-attributed", "(floats not attributed)", "nonempirical_section",
                [seg for f, _ in floats for seg in f.segments], [{"kind": "section_tree", "text": FLOATS_NOTE}], "discovered",
                [f"{_where(f)} {said}; which study it belongs to, if any, is not determinable from "
                 "its caption or its place" for f, said in floats], by_rule=True)

    # ----------------------------------------------------------------------- paper level
    title = next((o["work"].get("title") for _, o in sources if o["membership"][0]["role"] == "main_article"), "") or builder.work_id
    regions = [r for oid in out.values() for r in builder.get(oid)["boundary"]["regions"]] + _spans(whole)
    unread = [f"{trees[sid].source_label} page index {', '.join(str(p) for p in trees[sid].unread)}: no text was read "
              "from it (no usable native text and no OCR text), so what it holds is in no item" for sid in main_ids
              if trees[sid].unread]  # D-199
    add("study:paper", title, "paper_level", whole, [{"kind": "section_tree", "text": "the whole main article"}],
        "discovered", unlocated + unread + _uncovered(objects, trees, regions))
    return InventoryResult(out, trees)


def _kind(study: tuple[str, str], labels: list[str]) -> tuple[str, str | None]:
    kind, label = study
    if kind in ("pilot", "pretest"):
        return kind, None
    m = re.match(r"^([SE])(\d+)([A-Z])$", label)
    if m and f"{m.group(1)}{m.group(2)}" in labels:
        return "substudy", f"study:{m.group(1)}{m.group(2)}"
    return "study", None
