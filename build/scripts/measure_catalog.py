"""A source-bound, deliberately conservative measure-catalog projection.

The projection is a presentation adapter over candidate study inventories.  It
does not decide that an excerpt is a complete scale, split prose into items,
or merge similarly named measures.  Each study/measure occurrence remains a
separate proposed version with its own source witnesses.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from mosaic.adapted.e.restricted_instruments import restricted_holder
from mosaic.canonical import canonical_bytes
from mosaic.workspace import Workspace
from scripts.source_decoding import subspan, validate_slice


VERSION = "mosaic-measure-catalog/v1"
INVENTORY_VERSION = "mosaic-study-inspection/v1"
PROPOSED = "model_proposed_not_independently_reviewed"
_HASH = re.compile(r"[0-9a-f]{64}\Z")


def _digest(value: Any) -> str:
    return sha256(canonical_bytes(value)).hexdigest()


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def _statement(value: Any) -> dict[str, Any]:
    """Copy one already source-bound statement after checking its witness shape."""
    _require(isinstance(value, Mapping), "measure_evidence_statement_required")
    quote = value.get("quote")
    pages = value.get("pages")
    segments = value.get("segments")
    _require(isinstance(quote, str) and quote.strip(), "measure_evidence_quote_required")
    _require(isinstance(pages, list) and pages and all(type(page) is int and page > 0 for page in pages),
             "measure_evidence_pages_required")
    _require(isinstance(segments, list) and segments, "measure_evidence_segments_required")
    copied_segments = []
    seen_pages = []
    for segment in segments:
        _require(isinstance(segment, Mapping), "measure_evidence_segment_required")
        text = segment.get("text")
        start, end = segment.get("start"), segment.get("end")
        page, locator = segment.get("page"), segment.get("source_locator")
        unit_hash = segment.get("unit_text_sha256")
        _require(isinstance(text, str) and text, "measure_evidence_segment_text_required")
        _require(type(start) is int and type(end) is int and 0 <= start < end,
                 "measure_evidence_segment_offsets_required")
        _require(end - start == len(text), "measure_evidence_segment_length_mismatch")
        _require(type(page) is int and page > 0 and isinstance(locator, str) and locator,
                 "measure_evidence_segment_source_required")
        _require(isinstance(unit_hash, str) and _HASH.fullmatch(unit_hash) is not None,
                 "measure_evidence_segment_hash_required")
        _require(page in pages, "measure_evidence_cross_page")
        if page not in seen_pages:
            seen_pages.append(page)
        copied_segments.append({"unit_id": segment.get("unit_id"), "start": start, "end": end,
                                "text": text, "page": page, "source_locator": locator,
                                "unit_text_sha256": unit_hash})
        if "text_derivation" in segment:
            validate_slice(segment)
            copied_segments[-1]["text_derivation"] = deepcopy(segment["text_derivation"])
    _require(seen_pages == pages, "measure_evidence_pages_mismatch")
    # Study inspection preserves a multi-segment statement with LF separators;
    # a single segment (or segments already containing their separator) may also
    # concatenate exactly.  Anything else is a changed/foreign witness.
    parts = [segment["text"] for segment in copied_segments]
    _require(quote in {"".join(parts), "\n".join(parts)}, "measure_evidence_quote_segments_mismatch")
    return {"quote": quote, "segments": copied_segments, "pages": list(pages)}


def _evidence(value: Any) -> list[dict[str, Any]]:
    _require(isinstance(value, list), "measure_evidence_list_required")
    return [_statement(statement) for statement in value]


def _field(value: Any, name: str) -> list[dict[str, Any]]:
    _require(isinstance(value, Mapping), f"measure_{name}_field_required")
    status = value.get("status")
    _require(status in {PROPOSED, "not_extracted"}, f"measure_{name}_status_invalid")
    evidence = _evidence(value.get("evidence"))
    _require((status == PROPOSED) == bool(evidence), f"measure_{name}_status_evidence_mismatch")
    return evidence


def _paper(value: Any) -> dict[str, str]:
    _require(isinstance(value, Mapping), "measure_inventory_paper_required")
    paper_id, title = value.get("paper_id"), value.get("title")
    citation, pdf_hash = value.get("citation", ""), value.get("source_pdf_sha256")
    _require(isinstance(paper_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", paper_id) is not None,
             "measure_inventory_paper_id_invalid")
    _require(isinstance(title, str) and title.strip() and isinstance(citation, str), "measure_inventory_paper_metadata_invalid")
    _require(isinstance(pdf_hash, str) and _HASH.fullmatch(pdf_hash) is not None,
             "measure_inventory_pdf_hash_invalid")
    return {"paper_id": paper_id, "title": title, "citation": citation, "source_pdf_sha256": pdf_hash}


def _rights(name: str, aliases: list[str], name_evidence: list[dict[str, Any]],
            description_evidence: list[dict[str, Any]], item_evidence: list[dict[str, Any]]) -> dict[str, Any]:
    # Source prose naming an instrument is an alias for the rights check.  This
    # deliberately over-withholds rather than taking a chance with a licensed form.
    source_names = [statement["quote"] for statement in name_evidence + description_evidence + item_evidence]
    holder = restricted_holder({"label": name, "aliases": aliases + source_names})
    return {"holder": holder, "display_items": holder is None, "reproduction_permitted": False,
            "clearance": "not_established",
            "statement": ("Item-bearing excerpts are withheld for this restricted instrument."
                          if holder else "No restricted match is not permission to reproduce a complete scale; this preview shows candidate source excerpts only.")}


def _version_id(paper: Mapping[str, str], study_id: str, measure_id: str, name: str,
                source_identity: Mapping[str, Any]) -> str:
    body = {"paper_id": paper["paper_id"], "source_pdf_sha256": paper["source_pdf_sha256"], "study_id": study_id, "measure_id": measure_id,
            "name": name, "source_identity": source_identity}
    return "measure-version-" + _digest(body)[:24]


def _inventory_versions(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    _require(inventory.get("version") == INVENTORY_VERSION, "measure_inventory_version_invalid")
    _require(inventory.get("scientifically_accepted") is False and inventory.get("coverage_verified") is False,
             "measure_inventory_must_remain_unpromoted")
    paper = _paper(inventory.get("paper"))
    source_identity = inventory.get("source_identity")
    _require(isinstance(source_identity, Mapping), "measure_inventory_source_identity_required")
    for key in ("original_task_sha256", "recovered_sha256", "eligible_source_units_sha256", "source_policy_sha256"):
        _require(isinstance(source_identity.get(key), str) and _HASH.fullmatch(source_identity[key]) is not None,
                 "measure_inventory_source_identity_invalid")
    studies = inventory.get("studies")
    _require(isinstance(studies, list), "measure_inventory_studies_required")
    versions = []
    for study in studies:
        _require(isinstance(study, Mapping) and study.get("status") == PROPOSED, "measure_study_unpromoted_required")
        study_id = study.get("id")
        _require(isinstance(study_id, str) and study_id, "measure_study_id_required")
        measures = study.get("measures")
        _require(isinstance(measures, list), "measure_study_measures_required")
        for measure in measures:
            _require(isinstance(measure, Mapping) and measure.get("status") == PROPOSED,
                     "measure_version_unpromoted_required")
            measure_id, name = measure.get("id"), measure.get("name")
            _require(isinstance(measure_id, str) and measure_id and isinstance(name, str) and name.strip(),
                     "measure_version_identity_required")
            name_evidence = _evidence(measure.get("name_evidence"))
            _require(name_evidence, "measure_name_evidence_required")
            description_evidence = _field(measure.get("description"), "description")
            item_evidence = _field(measure.get("items"), "items")
            raw_aliases = measure.get("aliases", [])
            _require(isinstance(raw_aliases, list) and all(isinstance(alias, str) for alias in raw_aliases),
                     "measure_aliases_invalid")
            rights = _rights(name, list(raw_aliases), name_evidence, description_evidence, item_evidence)
            layout_evidence = measure.get('name_layout_evidence')
            if measure_id in source_identity.get('name_layout_field_ids', []):
                _require(layout_evidence is not None, 'measure_name_layout_evidence_stripped')
            if layout_evidence is not None:
                from scripts.core_rendered_names import validate_bound_layout_evidence
                validate_bound_layout_evidence(layout_evidence, name=name, name_evidence=name_evidence,
                    source_identity=source_identity, pdf_sha256=paper['source_pdf_sha256'])
            repeated_evidence = measure.get('repeated_name_evidence')
            if measure_id in source_identity.get('repeated_name_field_ids', []):
                _require(repeated_evidence is not None, 'measure_repeated_name_evidence_stripped')
            if repeated_evidence is not None:
                from scripts.core_repeated_names import validate_bound_repeated_evidence, VERSION as repeated_version
                _require(source_identity.get('repeated_name_policy') == repeated_version,
                         'measure_repeated_name_policy_changed')
                validate_bound_repeated_evidence(repeated_evidence, name=name, name_evidence=name_evidence)
            # No candidate prose is parsed into individual items, stems, anchors,
            # counts, or reliability values.  Those need separately selected exact
            # spans and their own future validation route.
            restricted = not rights["display_items"]
            versions.append({
                "id": _version_id(paper, study_id, measure_id, name, source_identity),
                "name": name, "paper": deepcopy(paper), "study_id": study_id, "status": PROPOSED,
                "name_evidence": [] if restricted else name_evidence,
                "description_evidence": [] if restricted else description_evidence,
                "item_evidence": [] if restricted else item_evidence,
                "items": [], "question_stems": [], "response_scale": [],
                "reported_item_count": None,
                "coverage": "withheld" if restricted else ("uncertain" if item_evidence else "reference"),
                "rights": rights,
                **({'name_layout_evidence': deepcopy(layout_evidence)}
                   if layout_evidence is not None and not restricted else {}),
                **({'repeated_name_evidence': deepcopy(repeated_evidence)}
                   if repeated_evidence is not None and not restricted else {}),
            })
    return versions


def build_catalog(inventories: Sequence[Mapping[str, Any]], *, api_snapshot_id: str | None = None) -> dict[str, Any]:
    """Create a display/search projection without merging label matches into identities."""
    _require(isinstance(inventories, Sequence) and not isinstance(inventories, (str, bytes)),
             "measure_inventories_required")
    _require(api_snapshot_id is None or (isinstance(api_snapshot_id, str) and api_snapshot_id),
             "measure_api_snapshot_id_invalid")
    versions = []
    paper_ids = set()
    for inventory in inventories:
        produced = _inventory_versions(inventory)
        versions.extend(produced)
        paper_ids.update(version["paper"]["paper_id"] for version in produced)
    versions.sort(key=lambda row: (row["name"].casefold(), row["paper"]["paper_id"], row["study_id"], row["id"]))
    _require(len({row["id"] for row in versions}) == len(versions), "measure_version_id_collision")
    # A browsing group is only a convenience for the UI.  Exact label equality
    # deliberately does not assert that two records are the same instrument.
    groups: dict[str, list[str]] = {}
    for version in versions:
        groups.setdefault(version["name"], []).append(version["id"])
    browse_groups = [{"label": label, "version_ids": ids} for label, ids in sorted(groups.items(), key=lambda row: row[0].casefold())]
    snapshot_basis = {"version": VERSION, "api_snapshot_id": api_snapshot_id, "versions": versions}
    return {"version": VERSION, "snapshot_id": "measure-catalog-" + _digest(snapshot_basis),
            "api_snapshot_id": api_snapshot_id, "papers": len(paper_ids), "versions": versions,
            "browse_groups": browse_groups,
            "scientifically_accepted": False, "coverage_verified": False,
            "limitations": [
                "Each record is a proposed study-specific measure occurrence, not a canonical instrument identity.",
                "Item evidence is a preserved candidate excerpt, not a claim that all scale items were captured.",
                "Missing fields mean not extracted in this preview, not absent from the source paper.",
            ]}


def attach_exact_measure_fields(version: Mapping[str, Any], *, items: Sequence[Mapping[str, Any]] = (),
                                question_stems: Sequence[Mapping[str, Any]] = (),
                                response_scale: Sequence[Mapping[str, Any]] = (),
                                reported_item_count: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Attach separately validated exact spans without promoting completeness.

    The caller must provide already evidence-shaped excerpts.  A count needs its
    own source witness and is represented as an exact lexical value, never a
    count inferred from the length of ``items``.
    """
    out = deepcopy(dict(version))
    _require(out.get("status") == PROPOSED and out.get("rights", {}).get("display_items") is True,
             "measure_fields_require_displayable_unpromoted_version")
    allowed = [*out.get("name_evidence", []), *out.get("description_evidence", []),
               *out.get("item_evidence", []), *out.get("field_context_evidence", [])]
    _require(allowed, "measure_fields_require_original_evidence")

    def source_bound(value: Mapping[str, Any]) -> dict[str, Any]:
        copied = _statement(value)
        for selected in copied["segments"]:
            matches = []
            for statement in allowed:
                for original in statement["segments"]:
                    same_source = all(selected.get(key) == original.get(key) for key in
                                      ("unit_id", "page", "source_locator", "unit_text_sha256"))
                    if not same_source or not (original["start"] <= selected["start"] < selected["end"] <= original["end"]):
                        continue
                    relative_start = selected["start"] - original["start"]
                    relative_end = selected["end"] - original["start"]
                    expected = subspan(original, relative_start, relative_end)
                    if (expected["text"] == selected["text"]
                            and expected.get("text_derivation") == selected.get("text_derivation")):
                        matches.append(original)
            _require(matches, "measure_field_not_subspan_of_version_evidence")
        return copied

    for key, values in (("items", items), ("question_stems", question_stems), ("response_scale", response_scale)):
        _require(isinstance(values, Sequence) and not isinstance(values, (str, bytes)), "measure_exact_fields_invalid")
        out[key] = [source_bound(value) for value in values]
    if reported_item_count is not None:
        _require(isinstance(reported_item_count, Mapping) and isinstance(reported_item_count.get("value"), str)
                 and reported_item_count["value"].strip() and "evidence" in reported_item_count,
                 "measure_reported_item_count_invalid")
        count_evidence = source_bound(reported_item_count["evidence"])
        value = reported_item_count["value"]
        _require(re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", count_evidence["quote"]) is not None,
                 "measure_reported_item_count_not_literal")
        out["reported_item_count"] = {"value": value, "evidence": count_evidence}
    if out["items"] or out["question_stems"] or out["response_scale"]:
        out["coverage"] = "partial"
    return out


def write_catalog(inventories: Sequence[Mapping[str, Any]], output: Path, *, workspace: Workspace,
                  template: Path | None = None, api_snapshot_id: str | None = None) -> dict[str, Any]:
    """Write the small-preview measure page and its immutable JSON payload."""
    catalog = build_catalog(inventories, api_snapshot_id=api_snapshot_id)
    target = workspace.check(output)
    template = template or Path(__file__).resolve().parents[1] / "web" / "measure-catalog.html"
    html = Path(template).read_text(encoding="utf-8")
    html = html.replace("__SOURCE_TEXT_SCRIPT__", (Path(__file__).resolve().parents[1] / "web" / "source-text.js").read_text(encoding="utf-8"))
    _require(html.count("__MEASURE_CATALOG__") == 1, "measure_template_catalog_marker")
    payload = json.dumps(catalog, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    workspace.write_once(target / "measures.html", html.replace("__MEASURE_CATALOG__", payload).encode("utf-8"))
    workspace.write_once(target / "measure-catalog.json", canonical_bytes(catalog))
    return {"papers": catalog["papers"], "versions": len(catalog["versions"]),
            "snapshot_id": catalog["snapshot_id"], "scientifically_accepted": False,
            "coverage_verified": False}
