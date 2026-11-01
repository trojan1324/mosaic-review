"""Lossless native-package reader projection, separate from model eligibility."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from mosaic.package import load_package
from mosaic.workspace import Workspace


VERSION = "public-native-reader/v1"
OWN_ARTICLE_METADATA_LANES = frozenset({"front_matter", "body"})


def is_own_article_metadata_unit(unit: object) -> bool:
    """Whether a native-reader unit may witness this article's own metadata.

    References and appendices remain available to their dedicated review paths,
    but cannot establish the identity or APA supplements of the containing
    article merely because a cited string is literal.
    """
    return isinstance(unit, dict) and unit.get("lane") in OWN_ARTICLE_METADATA_LANES


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_public_reader(package_root: Path, source_pdf_sha256: str, output: Path, *, workspace: Workspace) -> dict[str, Any]:
    """Project every native text block verbatim; never apply source-policy loss."""
    package = load_package(package_root)
    pages = {key: item for key, item in package.objects.items() if item.get("type") == "page"}
    sources = [item for item in package.objects.values() if item.get("type") == "source"]
    if len(sources) != 1 or sources[0].get("artifact", {}).get("sha256") != source_pdf_sha256:
        raise ValueError("public_reader_source_pdf_mismatch")
    if not any(item.get("sha256") == source_pdf_sha256 for item in package.manifest.get("inputs", [])):
        raise ValueError("public_reader_manifest_input_mismatch")
    regions = [(key, value) for key, value in package.objects.items() if value.get("type") == "text-region"]
    regions.sort(key=lambda item: (pages[item[1]["page_id"]]["display_ordinal"], item[0]))
    units, abstract, section, seen, previous_page = [], None, "front_matter", set(), None
    for region_id, region in regions:
        page = pages[region["page_id"]]
        if previous_page is not None and page["display_ordinal"] != previous_page and section == "table":
            section = "body"  # a table boundary cannot silently consume a new page
        previous_page = page["display_ordinal"]
        blocks = {block["block_index"]: block for block in region["blocks"]}
        for block_index in region.get("reading_order", sorted(blocks)):
            block = blocks[block_index]
            if block["kind"] != "text":
                continue
            quote = region["text"][block["start"]:block["end"]]
            token = quote.strip().casefold()
            if not (0 <= block["start"] <= block["end"] <= len(region["text"])):
                raise ValueError("public_reader_offset_invalid")
            unit_id = block["legacy_unit_id"]
            if unit_id in seen:
                raise ValueError("public_reader_duplicate_unit_id")
            seen.add(unit_id)
            lane = "front_matter" if section == "front_matter" else ("references" if section == "references" else "appendix" if section == "appendix" else "table_region_candidate" if section == "table" else "body")
            default_visible = section not in {"front_matter", "table"}
            visibility_reason = "body_after_abstract" if default_visible else "before_abstract_or_table_boundary"
            if token == "abstract":
                if abstract is None:
                    abstract = unit_id
                section, lane, default_visible, visibility_reason = "body", "abstract_heading", False, "standalone_section_heading"
            elif token in {"methods", "results"}:
                section, lane, default_visible, visibility_reason = "body", "section_heading", False, "standalone_section_heading"
            elif token == "references": section, lane, default_visible = "references", "references_heading", False
            elif token == "appendix": section, lane, default_visible = "appendix", "appendix_heading", False
            elif re.fullmatch(r"table\s+\d+(?:\s*[.:]\s*.+)?", token):
                section, lane, default_visible, visibility_reason = "table", "table_region_candidate", False, "standalone_table_caption_candidate"
            if abstract is None:
                visibility_reason = "before_first_literal_abstract"
            elif lane.endswith("_heading"):
                visibility_reason = "standalone_section_heading"
            elif lane in {"references", "appendix"}:
                visibility_reason = "retained_" + lane
            elif lane == "table_region_candidate" and visibility_reason != "standalone_table_caption_candidate":
                visibility_reason = "candidate_table_region"
            units.append({"unit_id": unit_id, "page": page["display_ordinal"],
                "physical_page_id": region["page_id"], "text_region_id": region_id,
                "start": block["start"], "end": block["end"], "quote": quote,
                "quote_sha256": _sha(quote), "source_pdf_sha256": source_pdf_sha256,
                "lane": lane, "default_visible": default_visible and abstract is not None,
                "visibility_reason": visibility_reason, "ordinal": len(units)})
    payload = {"version": VERSION, "package_manifest_id": package.manifest_id,
        "source_pdf_sha256": source_pdf_sha256, "scope": "native_capture_full_package_candidate",
        "units": units, "default_mask": {"status": "candidate_literal_abstract_heading" if abstract else "refused_no_abstract_heading", "anchor_unit_id": abstract},
        "scientifically_accepted": False}
    workspace.write_canonical_json(output / "native-reader.json", payload)
    return payload
