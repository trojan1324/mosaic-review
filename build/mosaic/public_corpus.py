"""Conservative, review-bound multi-source public corpus materializer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from mosaic.canonical import canonical_bytes
from mosaic.discovery_display_v1 import (
    format_apa_citation,
    is_displayable_measure_label,
    is_generic_method_label,
)
from mosaic.public_serving import (VERSION as SERVING_VERSION, PublicServingRefusal,
    _MASKED_LANES, _copy_frontend, _native_reader, _resolve_input, _sha256, _stage_envelope)
from mosaic.public_reader import is_own_article_metadata_unit
from mosaic.workspace import Workspace
from mosaic.public_studies import build_from_public_handoffs
from mosaic.public_jev import PublicJevRefusal, replay_public_advisory
from scripts.research_search_index import build_index

VERSION = "mosaic-public-corpus/v1"
_FIELDS = ("title", "authors", "journal", "year")
_GENERIC = frozenset({"method", "methods", "survey", "questionnaire", "interview", "interviews", "scale", "item", "items", "data", "analysis"})
_APA_SUPPLEMENTS = frozenset({"volume", "issue", "pages_verbatim", "article_number", "doi"})


def _witness(value: object, units: Mapping[str, Mapping[str, Any]], sha: str, *, own_article_metadata: bool = False) -> bool:
    """Accept only an exact literal span, bound to an original captured unit."""
    if not isinstance(value, Mapping): return False
    unit, text = units.get(value.get("unit_id")), value.get("text")
    if not isinstance(unit, Mapping) or not isinstance(text, str) or not text: return False
    if own_article_metadata and not is_own_article_metadata_unit(unit): return False
    if value.get("source_pdf_sha256") != sha or value.get("quote_sha256") != unit.get("quote_sha256"): return False
    start, end = value.get("start"), value.get("end")
    if type(start) is not int or type(end) is not int or start < unit["start"] or end > unit["end"] or start >= end: return False
    return unit["quote"][start-unit["start"]:end-unit["start"]] == text


_REVIEWER_TYPES = frozenset({"caller_attested_source_review", "agent_source_layout_review", "human_source_review"})


def _receipt(review: Mapping[str, Any], status: str, sha: str, payload: object) -> bool:
    """Verify a caller-attested receipt; it is provenance, never human proof."""
    receipt = review.get("receipt")
    if (review.get("status") != status or review.get("reviewer_type") not in _REVIEWER_TYPES
            or not isinstance(review.get("version"), str) or not review["version"] or not isinstance(receipt, Mapping)
            or not isinstance(review.get("receipt_sha256"), str)):
        return False
    expected = {"source_pdf_sha256": sha, "reviewer_type": review["reviewer_type"], "version": review["version"],
                "reviewed_payload_sha256": hashlib.sha256(canonical_bytes(payload)).hexdigest()}
    return dict(receipt) == expected and hashlib.sha256(canonical_bytes(expected)).hexdigest() == review["receipt_sha256"]


def _reviewed_identity(review: object, units: Mapping[str, Mapping[str, Any]], sha: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not isinstance(review, Mapping): return None, {"reason": "identity_review_receipt_missing_or_invalid"}
    fields = review.get("fields")
    if not isinstance(fields, Mapping): return None, {"reason": "identity_fields_missing"}
    values: dict[str, Any] = {}
    for name in _FIELDS:
        field = fields.get(name)
        if not isinstance(field, Mapping) or not _witness(field.get("witness"), units, sha, own_article_metadata=True):
            return None, {"reason": "identity_literal_witness_incomplete", "field": name}
        value = field.get("value")
        if field["witness"].get("text") != str(value): return None, {"reason": "identity_value_not_literal", "field": name}
        values[name] = value.strip() if isinstance(value, str) else value
    if not all(isinstance(values[n], str) and values[n] for n in ("title", "authors", "journal")) or type(values["year"]) is not int or not 1500 <= values["year"] <= 2200:
        return None, {"reason": "identity_complete_printed_fields_invalid"}
    if not _receipt(review, "source_reviewed_complete", sha, fields): return None, {"reason": "identity_review_receipt_missing_or_invalid"}
    return {**values, "authors_witness": dict(fields["authors"]["witness"]),
            "reviewer_type": review["reviewer_type"], "receipt_sha256": review["receipt_sha256"],
            "review_version": review["version"]}, {}


def _reviewed_apa(review: object, units: Mapping[str, Mapping[str, Any]], sha: str,
                  admitted_byline_witness: Mapping[str, Any], admitted_title: str) -> tuple[dict[str, Any] | None, str | None]:
    """Keep APA rendering separate from printed-identity admission.

    The supplemental payload cannot replace source-reviewed title/byline or
    provenance fields.  Its own receipt makes a rendering choice auditable.
    """
    if not isinstance(review, Mapping):
        return None, "apa_review_missing"
    authors = review.get("authors_apa")
    title_apa = review.get("title_apa")
    byline_witness = review.get("byline_witness")
    raw_supplements = review.get("supplements", {})
    if (not isinstance(authors, list) or not authors or not all(isinstance(item, str) and item.strip() for item in authors)
            or not isinstance(title_apa, str) or not title_apa.strip() or title_apa.casefold() != admitted_title.casefold()
            or not _witness(byline_witness, units, sha, own_article_metadata=True) or not isinstance(raw_supplements, Mapping)
            or set(raw_supplements) - _APA_SUPPLEMENTS):
        return None, "apa_review_payload_invalid"
    if dict(byline_witness) != dict(admitted_byline_witness):
        return None, "apa_review_byline_witness_mismatch"
    supplements: dict[str, Any] = {}
    for name, field in raw_supplements.items():
        if not isinstance(field, Mapping) or "value" not in field or not _witness(field.get("witness"), units, sha, own_article_metadata=True):
            return None, "apa_review_supplement_witness_invalid"
        if str(field["value"]) != field["witness"].get("text"):
            return None, "apa_review_supplement_not_literal"
        supplements[name] = field["value"]
    payload = {"authors_apa": authors, "title_apa": title_apa, "byline_witness": dict(byline_witness),
               "supplements": raw_supplements}
    if not _receipt(review, "source_reviewed_apa_rendering", sha, payload):
        return None, "apa_review_receipt_missing_or_invalid"
    return {"authors_apa": authors, "title_apa": title_apa, "supplements": supplements,
            "reviewer_type": review["reviewer_type"], "receipt_sha256": review["receipt_sha256"],
            "review_version": review["version"]}, None


def _reference(ref: object, paper_id: str, units: Mapping[str, Mapping[str, Any]], sha: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not isinstance(ref, Mapping) or not isinstance(ref.get("literal_reference"), str):
        return None, {"paper_id": paper_id, "reason": "literal_reference_witness_missing"}
    witness = {**dict(ref.get("witness", {})), "text": ref["literal_reference"]}
    if not _witness(witness, units, sha): return None, {"paper_id": paper_id, "reason": "literal_reference_witness_invalid", "candidate": dict(ref)}
    result = {"reference_id": "reference:" + hashlib.sha256(canonical_bytes([paper_id, witness])).hexdigest(), "source_paper_id": paper_id, "literal_reference": ref["literal_reference"], "witness": witness}
    target = ref.get("target_paper_id")
    if isinstance(target, str) and target: result["target_paper_id"] = target
    if ref.get("edge_status") == "reviewed_edge" and isinstance(target, str) and target:
        result["edge_status"] = "reviewed_edge"; result["edge_review"] = ref.get("edge_review")
    return result, {"paper_id": paper_id, "reason": "literal_reference_retained", **result}


def _measure(value: object, paper_id: str, units: Mapping[str, Mapping[str, Any]], sha: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not isinstance(value, Mapping): return None, {"paper_id": paper_id, "reason": "measure_not_object"}
    label = value.get("label"); witness = {**dict(value.get("witness", {})), "text": value.get("literal_evidence", "")}
    if (not isinstance(label, str) or not label.strip() or label.casefold().strip() in _GENERIC
            or value.get("generic_artifact") is True or not is_displayable_measure_label(label)
            or is_generic_method_label(label)):
        return None, {"paper_id": paper_id, "reason": "measure_withheld_generic_or_missing", "candidate": dict(value)}
    if not _witness(witness, units, sha): return None, {"paper_id": paper_id, "reason": "measure_literal_witness_invalid", "candidate": dict(value)}
    review = value.get("review")
    raw_items = value.get("item_texts", [])
    if not isinstance(raw_items, list):
        return None, {"paper_id": paper_id, "reason": "measure_item_texts_invalid", "candidate": dict(value)}
    items = []
    for item in raw_items:
        if not _witness(item, units, sha):
            return None, {"paper_id": paper_id, "reason": "measure_item_text_witness_invalid", "candidate": dict(value)}
        items.append(dict(item))
    if len({(item["unit_id"], item["start"], item["end"]) for item in items}) != len(items):
        return None, {"paper_id": paper_id, "reason": "measure_item_text_duplicate", "candidate": dict(value)}
    payload = {"label": label.strip(), "witness": witness}
    if raw_items:
        payload["items"] = items
    if not isinstance(review, Mapping) or not _receipt(review, "source_reviewed_meaningful_instrument", sha, payload):
        return None, {"paper_id": paper_id, "reason": "measure_requires_explicit_review", "candidate": dict(value)}
    return {"label": label.strip(), "paper_id": paper_id, "witness": witness, "items": items,
            "pages": [units[witness["unit_id"]]["page"]], "reviewer_type": review["reviewer_type"],
            "receipt_sha256": review["receipt_sha256"], "review_version": review["version"]}, {"paper_id": paper_id, "reason": "reviewed_measure_admitted"}


def materialize_public_corpus(sources: Sequence[Mapping[str, Any]], output: Path | str, *, workspace: Workspace) -> dict[str, Any]:
    """Make a hash-bound local bundle. Candidate discovery never equals review."""
    output = workspace.check(output)
    if output.exists() or not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)) or not sources: raise PublicServingRefusal("public_corpus_sources_invalid")
    workspace.mkdir(output)
    raw=[]; default=[]; catalog=[]; source_passages=[]; assets=[]; reports=[]; report_bodies={}; identities={}; apa_reviews={}; identity_queue=[]; reference_queue=[]; measure_queue=[]; refs=[]; reviewed_measures=[]; pages={}; seen=set(); advisories=[]; copied_artifacts=[]
    for item in sources:
        if not isinstance(item, Mapping) or not isinstance(item.get("capture"), Mapping) or not isinstance(item.get("snapshot"), Mapping): raise PublicServingRefusal("public_corpus_source_invalid")
        capture, snapshot = item["capture"], item["snapshot"]; sha=str(capture.get("source_pdf_sha256", "")); paper_id="public-"+sha
        if len(sha) != 64 or capture.get("paper_id", paper_id) != paper_id or paper_id in seen: raise PublicServingRefusal("public_corpus_paper_binding_invalid")
        pdf=_resolve_input(workspace,capture.get("source_pdf_path"),"public_source_pdf_missing")
        if pdf.suffix.casefold() != ".pdf" or _sha256(pdf) != sha or not pdf.read_bytes().startswith(b"%PDF-"): raise PublicServingRefusal("public_source_pdf_hash_mismatch")
        native=capture.get("native_reader"); native=_resolve_input(workspace,native,"public_native_reader_missing") if isinstance(native,(str,Path)) else native
        reader=_native_reader(native,sha); report,report_sha=_stage_envelope(capture,snapshot,workspace=workspace,paper_id=paper_id,source_sha=sha,package_manifest_id=reader["package_manifest_id"],reader=reader)
        workspace.copy_file(pdf,output/"sources"/(paper_id+".pdf"),expected_sha256=sha); workspace.copy_file(report,output/"candidate-reports"/(paper_id+".json"),expected_sha256=report_sha)
        copied_artifacts.extend(["sources/" + paper_id + ".pdf", "candidate-reports/" + paper_id + ".json"])
        snapshot_path = _resolve_input(workspace, snapshot.get("snapshot_path"), "public_snapshot_missing")
        snapshot_sha = snapshot.get("snapshot_sha256")
        workspace.copy_file(snapshot_path, output / "snapshots" / (paper_id + ".json"), expected_sha256=snapshot_sha)
        copied_artifacts.append("snapshots/" + paper_id + ".json")
        for number, advisory in enumerate(item.get("advisory_sidecars", [])):
            if not isinstance(advisory, Mapping): raise PublicServingRefusal("public_advisory_invalid")
            try:
                sidecar=replay_public_advisory(advisory.get("task_path"),advisory.get("sidecar_path"),expected_sidecar_sha256=advisory.get("sidecar_sha256"))
            except PublicJevRefusal as exc: raise PublicServingRefusal("public_advisory_replay_invalid") from exc
            if sidecar.get("pdf_sha256") != sha or sidecar.get("snapshot_sha256") != snapshot.get("snapshot_sha256"):
                raise PublicServingRefusal("public_advisory_source_binding_invalid")
            sidecar_path=_resolve_input(workspace,advisory.get("sidecar_path"),"public_advisory_sidecar_missing")
            task_path=_resolve_input(workspace,advisory.get("task_path"),"public_advisory_task_missing")
            sidecar_name=f"{paper_id}-{number}.json"; task_name=f"{paper_id}-{number}-task.json"
            workspace.copy_file(sidecar_path,output/"advisories"/sidecar_name,expected_sha256=advisory["sidecar_sha256"])
            workspace.copy_file(task_path,output/"advisories"/task_name)
            copied_artifacts.extend(["advisories/" + sidecar_name, "advisories/" + task_name])
            advisories.append({"paper_id":paper_id,"task_id":sidecar["task_id"],"sidecar_path":"advisories/"+sidecar_name,"task_path":"advisories/"+task_name,"sidecar_sha256":advisory["sidecar_sha256"],"advice":sidecar["advice"],"kind":"post_extraction_advisory_only","may_not_gate_visibility":True,"may_not_modify_extraction":True,"scientifically_accepted":False})
        report_bodies[paper_id]=json.loads(report.read_bytes()); units={u["unit_id"]:u for u in reader["units"]}; anchor_id=reader["default_mask"].get("anchor_unit_id"); anchor=next((u["ordinal"] for u in reader["units"] if u["unit_id"]==anchor_id),None); rows=[]
        for unit in reader["units"]:
            row={"id":f"passage:{paper_id}:{unit['unit_id']}","paper_id":paper_id,**unit,"package_manifest_id":reader["package_manifest_id"],"scientifically_accepted":False}; raw.append(row); rows.append(row)
            if anchor is not None and unit["ordinal"]>anchor and unit["default_visible"] and unit["lane"] not in _MASKED_LANES: default.append(row)
        identity, reason=_reviewed_identity(item.get("identity_review"),units,sha)
        if identity is None:
            identity_queue.append({"paper_id":paper_id,"source_pdf_sha256":sha,"status":"review_required",**reason}); title="Source identity awaiting review"; status="public_unreviewed_source"
        else:
            identities[paper_id]=identity
            apa, apa_reason = _reviewed_apa(item.get("apa_review"), units, sha, identity["authors_witness"], identity["title"])
            if apa is None:
                identity_queue.append({"paper_id": paper_id, "status": "review_required", "reason": apa_reason})
            else:
                apa_reviews[paper_id]=apa
            title=identity["title"]; status="admitted_source_packet"
        units_sha=hashlib.sha256(canonical_bytes(reader["units"])).hexdigest()
        catalog.append({"paper_id":paper_id,"title":title,"citation":None,"source_pdf_sha256":sha,"source_units_sha256":units_sha,"source_inventory_status":status,**({"authors":[],"journal":identity["journal"],"year":identity["year"]} if identity else {})})
        source_passages.append({"paper_id":paper_id,"title":title,"citation":None,"source_pdf_sha256":sha,"source_units_sha256":units_sha,"passages":[{"unit_id":r["unit_id"],"text":r["quote"],"page":r["page"]} for r in rows]})
        assets.append({"paper_id":paper_id,"source_pdf_sha256":sha,"availability":"local_file_present","path":"sources/"+paper_id+".pdf"}); reports.append({"paper_id":paper_id,"path":"candidate-reports/"+paper_id+".json","sha256":report_sha}); pages[paper_id]=sorted({r["page"] for r in rows})
        reference_candidates=list(item.get("references",[]))
        reference_candidates.extend({"literal_reference":u["quote"],"witness":{"unit_id":u["unit_id"],"text":u["quote"],"start":u["start"],"end":u["end"],"source_pdf_sha256":sha,"quote_sha256":u["quote_sha256"]}} for u in reader["units"] if u["lane"]=="references")
        for ref in reference_candidates:
            accepted,queued=_reference(ref,paper_id,units,sha); reference_queue.append(queued)
            if accepted:
                prior = next((row for row in refs if row["reference_id"] == accepted["reference_id"]), None)
                if prior is None:
                    refs.append(accepted)
                elif accepted.get("edge_status") == "reviewed_edge" and prior != accepted:
                    raise PublicServingRefusal("public_reference_conflicting_review")
        report_measure_candidate=report_bodies[paper_id].get("original_report",{}).get("measures")
        if report_measure_candidate not in (None, {"status":"not_started"}): measure_queue.append({"paper_id":paper_id,"reason":"candidate_report_measure_requires_literal_source_review","candidate":report_measure_candidate})
        for measure in item.get("measures",[]):
            accepted,queued=_measure(measure,paper_id,units,sha); measure_queue.append(queued)
            if accepted: reviewed_measures.append(accepted)
        seen.add(paper_id)
    bib=[]
    for paper_id, identity in sorted(identities.items()):
        apa = apa_reviews.get(paper_id)
        if apa is None:
            continue
        record={"bibliographic_provenance":"printed_source_verified", "record_type":"journal_article",
                "authors":apa["authors_apa"], "year":identity["year"], "title_verbatim":identity["title"],
                "title_apa":apa["title_apa"], "journal":identity["journal"], **apa["supplements"]}
        citation,reason=format_apa_citation(record)
        if citation is None: identity_queue.append({"paper_id":paper_id,"status":"review_required","reason":"apa_incomplete_or_invalid","apa_reason":reason}); continue
        bib.append({"paper_id":paper_id,"title":identity["title"],"authors":apa["authors_apa"],"journal":identity["journal"],"year":identity["year"],"citation_apa":citation,"identity_review":{k:identity[k] for k in ("reviewer_type","receipt_sha256","review_version")},"apa_review":{k:apa[k] for k in ("reviewer_type","receipt_sha256","review_version")}})
    groups={}
    for row in reviewed_measures: groups.setdefault(row["label"],set()).add(row["paper_id"])
    measures=sorted(({"label":label,"distinct_article_frequency":len(pids),"paper_ids":sorted(pids),"scientifically_accepted":False} for label,pids in groups.items()),key=lambda x:(-x["distinct_article_frequency"],x["label"].casefold()))
    edges=[]
    for reference in refs:
        if reference.get("edge_status") != "reviewed_edge": continue
        target=reference["target_paper_id"]
        payload={"reference_id":reference["reference_id"],"source_paper_id":reference["source_paper_id"],"target_paper_id":target}
        if target not in identities or not isinstance(reference.get("edge_review"),Mapping) or not _receipt(reference["edge_review"],"source_reviewed_complete",next(a["source_pdf_sha256"] for a in assets if a["paper_id"]==reference["source_paper_id"]),payload):
            reference_queue.append({"paper_id":reference["source_paper_id"],"reason":"citation_edge_target_or_review_unresolved","reference_id":reference["reference_id"]}); continue
        edges.append({**payload,"status":"reviewed_edge","scientifically_accepted":False})
    apa_by_paper = {row["paper_id"]: row["citation_apa"] for row in bib}
    measure_versions = [{"id": "public-measure-" + hashlib.sha256(canonical_bytes([row["paper_id"], row["label"], row["witness"]])).hexdigest(), "name": row["label"], "construct": row["label"], "study_id": "not assigned", "coverage": "uncertain", "item_disclosure": "source_reviewed_verbatim_items" if row["items"] else "not_printed_in_supplied_evidence", "items": [item["text"] for item in row["items"]], "item_evidence": [{"quote": item["text"], "pages": [next(unit["page"] for unit in raw if unit["paper_id"] == row["paper_id"] and unit["unit_id"] == item["unit_id"])]} for item in row["items"]], "question_stems": [], "response_scale": [], "description_evidence": [{"quote": row["witness"]["text"], "pages": row["pages"]}], "name_evidence": [{"quote": row["witness"]["text"], "pages": row["pages"]}], "reported_item_count": None, "selected_item_count": len(row["items"]), "rights": {"statement": "Not established."}, "paper": {"paper_id": row["paper_id"], "title": identities[row["paper_id"]]["title"], "authors": identities[row["paper_id"]]["authors"], "journal": identities[row["paper_id"]]["journal"], "year": identities[row["paper_id"]]["year"], "citation": apa_by_paper.get(row["paper_id"]), "source_pdf_sha256": next(a["source_pdf_sha256"] for a in assets if a["paper_id"] == row["paper_id"])} } for row in reviewed_measures if row["paper_id"] in identities]
    try:
        study_manifest, study_files = build_from_public_handoffs(sources, identities={
            paper_id: {**identity, "citation": apa_by_paper.get(paper_id, "")}
            for paper_id, identity in identities.items()})
    except ValueError as exc:
        raise PublicServingRefusal(str(exc)) from exc
    index=build_index({"records":[],"catalog":catalog,"source_passages":source_passages},output/"search.sqlite",workspace=workspace); frontend=_copy_frontend(workspace,output,index["snapshot_id"],paper_count=len(seen),corpus_mode=True,measure_versions=measure_versions,study_manifest=study_manifest,study_files=study_files)
    paper_items=[{"paper_id":paper_id,"title":identity["title"],"authors":identity["authors"],"journal":identity["journal"],"year":identity["year"],"identity_review":{key:identity[key] for key in ("reviewer_type","receipt_sha256","review_version")}} for paper_id,identity in sorted(identities.items())]
    payloads={"candidate-report.json":{"version":VERSION,"reports":report_bodies,"scientifically_accepted":False},"candidate-reports.json":{"version":VERSION,"items":reports,"scientifically_accepted":False},"bibliography.json":{"version":VERSION,"items":bib,"paper_items":paper_items,"unresolved_count":len(identity_queue),"scientifically_accepted":False},"citations.json":{"version":VERSION,"items":edges,"literal_references":refs,"status":"reviewed_edges_only","scientifically_accepted":False},"measures.json":{"version":VERSION,"items":measures,"status":"reviewed_literal_measures_only","scientifically_accepted":False},"advisories.json":{"version":VERSION,"items":advisories,"status":"advisory_only","scientifically_accepted":False},"raw-passages.json":{"items":raw},"default-passages.json":{"items":default},"identity-review-queue.json":{"version":VERSION,"items":identity_queue},"reference-review-queue.json":{"version":VERSION,"items":reference_queue},"measure-review-queue.json":{"version":VERSION,"items":measure_queue},"source-assets.json":{"version":"mosaic-source-assets/v1","assets":assets,"scientifically_accepted":False,"public_distribution_authorized":False}}
    for name,value in payloads.items(): workspace.write_canonical_json(output/name,value)
    artifacts=["search.sqlite", *payloads, *copied_artifacts]
    bundle={"version":SERVING_VERSION,"paper_ids":sorted(seen),"source_pdf_sha256s":{x["paper_id"]:x["source_pdf_sha256"] for x in assets},"allowed_source_pages":pages,"artifacts":{name:_sha256(output/name) for name in artifacts},"frontend":frontend,"scientifically_accepted":False}; workspace.write_canonical_json(output/"bundle.json",bundle)
    return {"bundle_path":str(output/"bundle.json"),"bundle_sha256":_sha256(output/"bundle.json"),"paper_count":len(seen),"admitted_identity_count":len(identities),"apa_count":len(bib),"measure_count":len(measures),"identity_queue_count":len(identity_queue),"scientifically_accepted":False}
