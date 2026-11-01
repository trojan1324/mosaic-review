"""Record every captioned figure, and read nothing from it (FINAL v2 §5.2; D-165).

For each figure caption the section tree finds on a PDF page ("Figure 2.",
"Fig. 3 |", "FIGURE A1", "Figure II:", "Supplementary Figure 1.") there is one
`figure-equation` object of kind `figure`: its label, a witness on the
caption's own lines, and a bbox that is the union of the graphics next to the
caption. Graphics are the raster image regions capture recorded, less
decoration (icons, specks, banners in the margin bands), and PyMuPDF's clusters
of vector drawings, less a page background or text frame (drawn or raster) and
less any in the page's top or bottom margin band (running heads and rules).
Graphics within NEAR points of each other form one graphic unless a float
caption lies between them, each holds the graphic nearest to a different
caption (figures set side by side), or they lie along the reading directions of
different captions; graphics along no caption's direction join the one group
along a caption within reach of them, and none when two are. A caption lies
along a graphic's direction only when no nearer caption on that side overlaps
it across the direction, and a graphic along two captions on one side (a key
across two figures) joins none and settles neither (D-200). A raster over half
the page with a caption on it is the background only when it covers nearly all
of the page or holds other text; otherwise the caption is printed on it and
that figure is not located (D-201). A graphic goes to one caption only, within
NEAR points along the caption's reading direction (above or below a horizontal
caption, beside a sideways one), in the assignment that locates the most
captions; a caption whose graphic that does not settle takes none. When no
graphic qualifies the bbox is null and the disposition says why. Every figure
gets a `pending` region disposition, since its content is not interpreted, and
so does every raster image region, saying it is decoration or the page's
background, naming the figure that takes it in (which also lists it in its
provenance) or the captions it lies next to (or, along no caption's direction,
the located figure it lies within reach of), or saying that no figure caption
lies next to it. A DOCX caption has no
page; the DOCX image itself is capture's pending attachment.
"""

from __future__ import annotations

import re
from typing import Any

import pymupdf

from mosaic.capture.signals import BOTTOM_BAND, TOP_BAND, decoration
from mosaic.capture.signals import SCAN_IMAGE_SHARE, image_share
from mosaic.inventory.section_tree import Node, SourceTree, is_caption_line
from mosaic.inventory.witnesses import WitnessError, text_witness
from mosaic.package import PackageBuilder, actor

CREATED = actor("deterministic", "mosaic.inventory.figures")
# A figure float's label, in the section tree's float-label grammar (FLOAT_LABEL, D-151) for the figure words: an
# appendix or supplement prefix, then an Arabic number after at most three capitals ("3", "A1", "EC1") or a Roman
# numeral with at most one capital after it ("II", "IIA").
FIGURE_CAPTION = re.compile(
    r"^(?P<prefix>(?:(?:Online\s+Appendix|Appendix|Supplementary|Supplemental|Extended\s+Data)\s+)?)"
    r"(?:Figure|FIGURE|Fig\.|FIG\.)\s*"
    r"(?P<label>[A-Z]{0,3}\d+[A-Za-z]?|X{0,3}(?:IX|IV|V?I{0,3})(?<=[IVX])(?![IVX])[A-Z]?)")
# Any float's label, in the same grammar with every float word, as the page prints it ("Online Appendix Table 2").
FLOAT_NAME = re.compile(
    r"^(?:(?:Online\s+Appendix|Appendix|Supplementary|Supplemental|Extended\s+Data)\s+)?"
    r"(?:Table|TABLE|Figure|FIGURE|Fig\.|FIG\.|Exhibit|EXHIBIT)\s*"
    r"(?:[A-Z]{0,3}\d+[A-Za-z]?|X{0,3}(?:IX|IV|V?I{0,3})(?<=[IVX])(?![IVX])[A-Z]?)")
NEAR = 36.0  # points: about three lines of body text


def _figure_label(title: str) -> str | None:
    """"Figure 2", "Figure II", "Supplementary Figure 1" ("Fig." written out), or None for a caption of another kind."""
    m = FIGURE_CAPTION.match(title)
    return " ".join(m.group("prefix").split() + ["Figure", m.group("label")]) if m else None


def _union(boxes: list[list[float]]) -> list[float]:
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _gaps(a: list[float], b: list[float]) -> tuple[float, float]:
    """Horizontal and vertical distance between two boxes (0 on an axis where they overlap)."""
    return max(0.0, b[0] - a[2], a[0] - b[2]), max(0.0, b[1] - a[3], a[1] - b[3])


def _meets(a: list[float], b: list[float]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _caption_lines(region: dict[str, Any], node: Node) -> list[dict[str, Any]]:
    """The caption's own lines, as the section tree reads them (its block from its label line; of a block that
    captions set side by side share, the lines in its column, D-193), up to the first blank one (a caption block
    can run on into a graphic's blank placeholder lines and the text after it)."""
    own = {s.start for s in node.caption_lines}
    start = node.heading.start
    block = next(b for b in region["blocks"] if b["start"] <= start < b["end"])
    out = []
    for ln in block.get("lines") or [{"start": block["start"], "end": block["end"]}]:  # OCR text: one line, no bbox
        if ln["end"] <= start or (own and ln["start"] not in own):
            continue
        text = region["text"][ln["start"]:ln["end"]]
        if not text.strip() or (out and not own and is_caption_line(text)):
            break
        out.append(ln)
    return out


def _caption_witness(builder: PackageBuilder, region_id: str, lines: list[dict[str, Any]]) -> str | None:
    """A witness on the caption's lines: lines printed one after another are one span, and lines another caption's
    line separates (captions set side by side) are joined with a marked omission."""
    text = builder.objects[region_id]["text"]
    spans: list[list[int]] = []
    for ln in lines:
        if spans and not text[spans[-1][1]:ln["start"]].strip():
            spans[-1][1] = ln["end"]
        else:
            spans.append([ln["start"], ln["end"]])
    for s in spans:
        while s[1] > s[0] and text[s[1] - 1].isspace():
            s[1] -= 1
    try:
        return text_witness(builder, [(region_id, a, b) for a, b in spans], attribution_role="author_report",
                            created_by=CREATED, retrieval_transform="figure caption lines",
                            join="omitted_material_marker" if len(spans) > 1 else None)
    except WitnessError:
        return None


def _backdrop(rect: pymupdf.Rect, frame: pymupdf.Rect, captions: list[list[float]]) -> bool:
    """A drawing over half the page or more with a float caption inside it: the page's background or a frame
    around its text, not a figure's graphic."""
    return (rect & frame).get_area() >= frame.get_area() / 2 and any(rect.contains(pymupdf.Rect(c)) for c in captions)


def _graphics(page: pymupdf.Page, images: list[tuple[list[float], str]],
              captions: list[list[float]]) -> list[tuple[list[float], str | None]]:
    """The page's raster image regions (with their ids) and vector drawing clusters, less page furniture in the
    top or bottom margin band. A page background or text frame is left out before clustering: clustered with
    the rest it would join every graphic on the page into one. Coordinates are the unrotated cropbox frame,
    like the text's (D-061)."""
    w, h = page.cropbox.width, page.cropbox.height
    frame = pymupdf.Rect(0, 0, w, h)
    drawings = [d for d in page.get_drawings() if not _backdrop(pymupdf.Rect(d["rect"]), frame, captions)]
    clusters = [[round(v, 2) for v in r] for r in page.cluster_drawings(clip=frame, drawings=drawings)]
    return [(box, oid) for box, oid in images + [(c, None) for c in clusters]
            if not (box[3] <= h * TOP_BAND or box[1] >= h * BOTTOM_BAND)]


def _side(caption: list[float], sideways: bool, box: list[float]) -> str:
    """Which side of a graphic a caption facing it lies on: above or below it, or (sideways) left or right."""
    if sideways:
        return "left" if caption[0] + caption[2] < box[0] + box[2] else "right"
    return "above" if caption[1] + caption[3] < box[1] + box[3] else "below"


def _facing(graphics: list[tuple[list[float], str | None]],
            captions: list[tuple[list[float], bool]]) -> list[dict[str, set[int]]]:
    """For each graphic, by side, the captions it lies along the reading direction of (over or under a horizontal
    caption, beside a sideways one) at any distance, less any caption behind a nearer one on that side that overlaps
    it across the direction: a full-width caption below a figure's own caption faces that caption, not the figure's
    graphic (D-200)."""
    out = []
    for g, _ in graphics:
        near = [(ci, _side(c, sideways, g), d) for ci, (c, sideways) in enumerate(captions)
                if (d := _distance(c, sideways, g)) is not None]
        sides: dict[str, set[int]] = {}
        for ci, side, d in near:
            c, sideways = captions[ci]
            if not any(cj != ci and sj == side and dj < d and (
                    min(captions[cj][0][3], c[3]) > max(captions[cj][0][1], c[1]) if sideways
                    else min(captions[cj][0][2], c[2]) > max(captions[cj][0][0], c[0])) for cj, sj, dj in near):
                sides.setdefault(side, set()).add(ci)
        out.append(sides)
    return out


def _faced(graphics: list[tuple[list[float], str | None]], captions: list[tuple[list[float], bool]]) -> list[frozenset[int]]:
    """For each graphic, the captions it faces (`_facing`), on any side."""
    return [frozenset(ci for s in sides.values() for ci in s) for sides in _facing(graphics, captions)]


def _straddles(graphics: list[tuple[list[float], str | None]], captions: list[tuple[list[float], bool]]) -> list[set[int]]:
    """For each graphic, the captions of a side on which it faces more than one (a key across two figures set side by
    side, each captioned below): which figure it belongs to, the page does not settle, so it joins none (D-200)."""
    return [{ci for s in sides.values() if len(s) > 1 for ci in s} for sides in _facing(graphics, captions)]


def _groups(graphics: list[tuple[list[float], str | None]], captions: list[tuple[list[float], bool]]) -> list[list[int]]:
    """Graphics within NEAR points of each other, with no caption lying between them, form one group when they lie
    along the reading direction of a common caption, or both of none, unless each already holds the graphic
    nearest to a different caption (two figures set side by side, each captioned below its own graphic). A graphic
    along one caption never joins one along only other captions (another figure's legend, or its graphic across a
    column gutter with its caption out of reach). A group along no caption (tick labels left of a narrow caption,
    a key beside a chart) then joins the one group along a caption within reach of it, and none when two are:
    which figure it belongs to the page does not settle. A graphic that faces two captions on one side joins nothing.
    The groups do not depend on the order of the graphics."""
    parent = list(range(len(graphics)))
    faced = _faced(graphics, captions)
    straddle = _straddles(graphics, captions)
    claims: dict[int, set[int]] = {}  # group root -> the captions whose nearest graphic it holds
    for ci, (c, sideways) in enumerate(captions):
        near = [(d, gi) for gi, (g, _) in enumerate(graphics) if (d := _distance(c, sideways, g)) is not None and d <= NEAR]
        if near:
            claims.setdefault(min(near)[1], set()).add(ci)

    def root(i: int) -> int:
        while parent[i] != i:
            i = parent[i]
        return i

    def reach(a: list[float], b: list[float]) -> bool:  # within NEAR, with no caption between them
        span = _union([a, b])
        return max(_gaps(a, b)) <= NEAR and not any(_meets(c, span) and not _meets(c, a) and not _meets(c, b)
                                                    for c, _ in captions)

    for i, (a, _) in enumerate(graphics):
        for j in range(i + 1, len(graphics)):
            ri, rj = root(i), root(j)
            if (ri != rj and not (claims.get(ri) and claims.get(rj)) and (faced[i] & faced[j] or not (faced[i] or faced[j]))
                    and not straddle[i] and not straddle[j] and reach(a, graphics[j][0])):
                parent[rj] = ri
                claims[ri] = claims.get(ri, set()) | claims.pop(rj, set())
    groups: dict[int, list[int]] = {}
    for i in range(len(graphics)):
        groups.setdefault(root(i), []).append(i)
    along = {r: list(g) for r, g in groups.items() if any(faced[k] for k in g)}  # as they stand, before any joins
    for r in [r for r in groups if r not in along]:
        near = {q for q, g in along.items() for k in groups[r] for m in g if reach(graphics[k][0], graphics[m][0])}
        if len(near) == 1:
            groups[near.pop()] += groups.pop(r)
    return list(groups.values())


def _distance(caption: list[float], sideways: bool, box: list[float]) -> float | None:
    """How far a graphic lies from a caption along the caption's reading direction, or None when it is not
    beside the caption's lines (above or below a horizontal caption, left or right of a sideways one)."""
    gx, gy = _gaps(caption, box)
    return (gx if gy == 0 else None) if sideways else (gy if gx == 0 else None)


def _located(options: dict[int, list[int]]) -> int:
    """How many captions a largest one-to-one assignment of captions to graphic groups locates (augmenting paths)."""
    owner: dict[int, int] = {}

    def place(ci: int, seen: set[int]) -> bool:
        for gi in options[ci]:
            if gi not in seen:
                seen.add(gi)
                if gi not in owner or place(owner[gi], seen):
                    owner[gi] = ci
                    return True
        return False
    return sum(place(ci, set()) for ci in options)


def _side_caption_options(options, captions, graphics, groups, on_image, texts):
    """Offer unclaimed graphics beside horizontal captions, without guessing.

    Existing above/below associations take precedence. New options still pass
    through the global one-to-one assignment, so iteration order cannot settle
    competing captions or graphics. Native text in the gap blocks association.
    """
    result = {ci: list(gis) for ci, gis in options.items()}
    claimed = {gi for gis in options.values() for gi in gis}
    for ci, (node, lines, box, sideways) in enumerate(captions):
        if result[ci] or ci in on_image or not box or sideways or not _figure_label(node.title):
            continue
        for gi, group in enumerate(groups):
            if gi in claimed:
                continue
            g = _union([graphics[k][0] for k in group])
            gx, gy = _gaps(box, g)
            overlap = min(box[3], g[3]) - max(box[1], g[1])
            if not (0 < gx <= NEAR and gy == 0 and overlap >= min(12., (box[3] - box[1]) / 2)):
                continue
            bridge = ([g[2], max(box[1], g[1]), box[0], min(box[3], g[3])]
                      if g[2] < box[0] else
                      [box[2], max(box[1], g[1]), g[0], min(box[3], g[3])])
            if any(_meets(t, bridge) for t in texts):
                continue
            span = _union([box, g])
            if any(cj != ci and other[2] and _meets(other[2], span)
                   and not _meets(other[2], box) and not _meets(other[2], g)
                   for cj, other in enumerate(captions)):
                continue
            result[ci].append(gi)
    return result


def _assign(options: dict[int, list[int]]) -> tuple[dict[int, int], dict[int, list[int]]]:
    """Each caption's graphic group (`options`: the groups within NEAR of it), one group per caption, from the
    assignments that locate the most captions: a caption takes the group every one of them gives it. Where they
    differ (a graphic within reach of two captions, or two graphics within reach of one), the page does not settle
    which graphic is the caption's own, and it takes none; the second result lists the groups it might take.
    Taking the nearest pair first instead can give a caption the graphic that is the only one within reach of the
    next caption, and the nearer graphic is not always the caption's own (tick labels and an axis title can set a
    chart's caption farther from it than from the next chart)."""
    most = _located(options)
    taken: dict[int, int] = {}
    unsettled: dict[int, list[int]] = {}
    for ci, gis in options.items():
        rest = {c: g for c, g in options.items() if c != ci}
        fits = [gi for gi in gis if 1 + _located({c: [x for x in g if x != gi] for c, g in rest.items()}) == most]
        if len(fits) == 1 and _located(rest) < most:
            taken[ci] = fits[0]
        elif fits:
            unsettled[ci] = fits
    return taken, unsettled


def _name(title: str) -> str:
    """A float caption's label as the page prints it: "Figure 2", "Table 1", "Online Appendix Table 2" (D-201)."""
    m = FLOAT_NAME.match(title)
    return _figure_label(title) or (" ".join(m.group(0).split()) if m else " ".join(title.split()[:2]).rstrip(".:|"))


def record_figures(builder: PackageBuilder, trees: dict[str, SourceTree], *,
                   source_bytes: dict[str, bytes] | None = None) -> list[str]:
    objects = builder.objects
    figure_ids: list[str] = []
    for sid, tree in trees.items():
        if objects[sid]["artifact"]["media_type"] != "application/pdf":
            continue
        images: dict[str, list[tuple[list[float], str]]] = {}
        for oid, o in builder.of_type("figure-equation"):
            if o["source_id"] == sid and o["kind"] == "image_region":
                images.setdefault(o["page_id"], []).append((o["bbox"], oid))
        floats: dict[str, list] = {}
        for node in tree.nodes:
            if node.role == "float" and node.heading is not None:
                region = objects[node.heading.text_region_id]
                if region.get("page_id"):
                    floats.setdefault(region["page_id"], []).append(node)
        if source_bytes is not None:
            from hashlib import sha256
            raw = source_bytes.get(sid)
            if not isinstance(raw, bytes) or sha256(raw).hexdigest() != objects[sid]["artifact"]["sha256"]:
                raise ValueError("figure_source_bytes_hash_mismatch")
            doc = pymupdf.open(stream=raw, filetype="pdf")
        else:
            doc = pymupdf.open(objects[sid]["artifact"]["acquisition_path"])
        try:
            for page_id in dict.fromkeys([*floats, *images]):
                page = objects[page_id]
                identity = {"page_id": page_id, "physical_index": page["physical_index"],
                            "display_ordinal": page["display_ordinal"], "printed_label": page["printed_label"]}
                captions = []  # (node, caption lines, their box or None, set sideways)
                for node in floats.get(page_id, []):
                    lines = _caption_lines(objects[node.heading.text_region_id], node)
                    boxes = [ln["bbox"] for ln in lines if "bbox" in ln]
                    captions.append((node, lines, _union(boxes) if boxes else None,
                                     bool(boxes) and boxes[0][3] - boxes[0][1] > boxes[0][2] - boxes[0][0]))
                figures_here = any(_figure_label(c[0].title) for c in captions)
                pdf_page = doc[page["physical_index"]]
                decor = {oid: why for box, oid in images.get(page_id, [])
                         if (why := decoration(box, pdf_page.cropbox.height))}
                placed = [(c[2], c[3]) for c in captions if c[2]]
                at = [ci for ci, c in enumerate(captions) if c[2]]  # placed caption -> its index in `captions`
                # A raster over half the page or more with a caption on it is the page's background, as a drawing is,
                # when it covers nearly all of the page or holds text besides the captions' own lines (D-201).
                # Otherwise the caption is printed on the image, and which part of it is the figure's graphic the page
                # does not show: the image joins no figure, and that caption takes no graphic.
                frame = pymupdf.Rect(0, 0, pdf_page.cropbox.width, pdf_page.cropbox.height)
                big = {oid: box for box, oid in images.get(page_id, []) if oid not in decor
                       and _backdrop(pymupdf.Rect(box), frame, [box for box, _ in placed])}
                own = {ln["start"] for c in captions for ln in c[1]}
                texts = [ln["bbox"] for c in captions[:1] for b in objects[c[0].heading.text_region_id]["blocks"]
                         for ln in b.get("lines") or [] if ln.get("bbox") and ln["start"] not in own
                         and objects[c[0].heading.text_region_id]["text"][ln["start"]:ln["end"]].strip()]
                backdrop = {oid for oid, box in big.items() if image_share(pdf_page, box) >= SCAN_IMAGE_SHARE or any(
                    pymupdf.Rect(box).contains(pymupdf.Point((t[0] + t[2]) / 2, (t[1] + t[3]) / 2)) for t in texts)}
                printed_on = {oid: [ci for ci, c in enumerate(captions) if c[2] and pymupdf.Rect(box).contains(pymupdf.Rect(c[2]))]
                              for oid, box in big.items() if oid not in backdrop}
                on_image = {ci for cis in printed_on.values() for ci in cis}
                content = [(box, oid) for box, oid in images.get(page_id, []) if oid not in decor and oid not in big]
                graphics = _graphics(pdf_page, content, [box for box, _ in placed]) if figures_here else []
                groups = _groups(graphics, placed)
                straddle = _straddles(graphics, placed)
                options: dict[int, list[int]] = {}  # caption -> the groups within NEAR of it, nearest first
                for ci, (_, _, box, sideways) in enumerate(captions):
                    near = []
                    for gi, group in enumerate(groups):
                        if ci in on_image or any(straddle[k] for k in group):
                            continue  # its caption is on an image; a graphic across two captions settles neither
                        g = _union([graphics[k][0] for k in group])
                        d = _distance(box, sideways, g) if box else None
                        span = _union([box, g]) if box else g
                        # no other float caption between them (a figure over a table: each caption keeps its own)
                        if d is not None and d <= NEAR and not any(
                                _meets(c, span) and not _meets(c, box) and not _meets(c, g) for c, _ in placed):
                            near.append((d, gi))
                    options[ci] = [gi for _, gi in sorted(near)]
                options = _side_caption_options(options, captions, graphics, groups, on_image, texts)
                taken, unsettled = _assign(options)
                names = [_name(c[0].title) for c in captions]
                unsure: dict[str, list[str]] = {}  # image region id -> the captions it lies next to, none settled
                for ci, gis in unsettled.items():
                    for gi in gis:
                        for k in groups[gi]:
                            if graphics[k][1]:
                                unsure.setdefault(graphics[k][1], []).append(names[ci])
                for k, cis in enumerate(straddle):  # a key across two figures' columns, each captioned on its side
                    if graphics[k][1] and cis:
                        unsure[graphics[k][1]] = [names[at[c]] for c in sorted(cis)]
                # A group along no caption's reading direction, within reach of a located figure's graphic (a legend
                # beside a chart under a narrow caption), may be that figure's: the page does not settle it.
                faced = _faced(graphics, placed)
                for gi, group in enumerate(groups):
                    if gi in taken.values() or any(faced[k] for k in group):
                        continue
                    g = _union([graphics[k][0] for k in group])
                    for ci, tg in taken.items():
                        t = _union([graphics[k][0] for k in groups[tg]])
                        if max(_gaps(g, t)) <= NEAR and not any(_meets(c, _union([g, t])) and not _meets(c, g)
                                                                and not _meets(c, t) for c, _ in placed):
                            for k in group:
                                if graphics[k][1] and names[ci] not in unsure.get(graphics[k][1], []):
                                    unsure.setdefault(graphics[k][1], []).append(names[ci])
                part_of: dict[str, str] = {}  # image region id -> the label of the figure that takes it in
                for ci, (node, lines, box, _) in enumerate(captions):
                    label = _figure_label(node.title)
                    if not label:
                        continue  # a table caption: it only keeps its own graphics from a figure
                    region_id = node.heading.text_region_id
                    wid = _caption_witness(builder, region_id, lines)
                    group = [graphics[k] for k in groups[taken[ci]]] if ci in taken else []
                    ids = [oid for _, oid in group if oid]
                    part_of.update((oid, label) for oid in ids)
                    fig = builder.envelope("figure-equation", "captured", CREATED, [page_id] + ([wid] if wid else []) + ids)
                    fig.update(source_id=sid, page_id=page_id, kind="figure", label=label,
                               bbox=_union([g for g, _ in group]) if group else None, caption_witness_id=wid,
                               crop_sha256=None, recognition_status="not_attempted")
                    fig_id = builder.add(fig)
                    figure_ids.append(fig_id)
                    reason = "figure not interpreted: nothing is read from its graphic"
                    if not group:
                        rivals = [names[c] for c, gis in options.items() if c != ci and set(gis) & set(unsettled.get(ci, []))]
                        reason += "; graphic not located: " + (
                            "the caption's text has no position on the page" if box is None else
                            "the page is taken as a scan (D-163), so the figure would be pixels of the page image"
                            if objects[region_id]["derivation"].get("ocr_text_layer") else
                            "the caption is printed on a raster image that covers half the page or more, and which part "
                            "of that image is the figure's graphic the page does not show"
                            if ci in on_image else
                            f"a graphic within {NEAR:g} pt of the caption along its reading direction is also within reach "
                            f"of another caption ({', '.join(rivals)}), and the page does not settle which is its own"
                            if rivals else
                            f"more than one graphic lies within {NEAR:g} pt of the caption along its reading direction, "
                            "and the page does not settle which is its own"
                            if ci in unsettled else
                            f"no raster image or cluster of vector drawings lies within {NEAR:g} pt of the caption "
                            "along its reading direction")
                    disp = builder.envelope("disposition", "recorded", CREATED, [fig_id])
                    disp.update(subject_kind="region", subject_ref=fig_id, disposition="pending", reason=reason,
                                page_identity=identity)
                    builder.add(disp)
                for _, oid in images.get(page_id, []):
                    disp = builder.envelope("disposition", "recorded", CREATED, [oid])
                    disp.update(subject_kind="region", subject_ref=oid, disposition="pending", page_identity=identity,
                                reason=f"image not interpreted; decoration ({decor[oid]}), not part of any figure"
                                if oid in decor else
                                "image not interpreted; the page's background (it covers half the page or more with a "
                                "float caption on it), not part of any figure"
                                if oid in backdrop else
                                f"image not interpreted; the caption of {' and '.join(names[c] for c in printed_on[oid])} "
                                "is printed on it, and which part of it is a figure's graphic the page does not show"
                                if oid in printed_on else
                                f"image not interpreted; part of the graphic of {part_of[oid]} on this page"
                                if oid in part_of else
                                f"image not interpreted; it may be part of the graphic of {' or '.join(unsure[oid])} on "
                                "this page, which the page does not settle"
                                if oid in unsure else "image not interpreted; no figure caption lies next to it")
                    builder.add(disp)
        finally:
            doc.close()
    return figure_ids
