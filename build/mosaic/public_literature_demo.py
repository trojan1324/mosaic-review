"""Two-paper fictional corpus for the public literature-review interface.

Unlike :mod:`mosaic.public_pipeline`, this companion fixture includes complete
printed metadata and controlled review receipts so the clean Papers, APA and
literature-assistant surfaces have something legitimate to demonstrate.  Its
candidate reports are manufactured fixtures, never model output or evidence of
scientific accuracy.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from mosaic.canonical import canonical_bytes
from mosaic.public_build import register_sources, relative_path
from mosaic.public_capture import capture_public
from mosaic.public_corpus import materialize_public_corpus
from mosaic.workspace import Workspace


VERSION = "mosaic-public-literature-demo/v1"
_ARTICLES = (
    {"filename": "coordination.pdf", "title": "Coordination Routines in Fictional Teams", "title_apa": "Coordination routines in fictional teams",
     "authors": "Hannah Price", "authors_apa": ["Price, H."], "journal": "Journal of Fictional Organization Studies",
     "year": 2024, "volume": "12", "pages": "101-114", "measure": "Team Coordination Scale",
     "abstract": "This fictional article describes coordination routines in temporary teams.",
     "body": "Participants completed the Team Coordination Scale during a fictional field exercise.",
     "items": ["Team members coordinate their timing.", "Team members clarify shared responsibilities."],
     "reference": "Singh, O. (2025). Shared reflection in fictional groups. Journal of Fictional Organization Studies, 13, 15-29."},
    {"filename": "reflection.pdf", "title_apa": "Shared reflection in fictional groups", "title": "Shared Reflection in Fictional Groups",
     "authors": "Omar Singh", "authors_apa": ["Singh, O."], "journal": "Journal of Fictional Organization Studies",
     "year": 2025, "volume": "13", "pages": "15-29", "measure": "Shared Reflection Inventory",
     "abstract": "This fictional article describes how teams record and discuss their work.",
     "body": "Participants completed the Shared Reflection Inventory after a fictional collaborative exercise.",
     "items": ["Team members discuss what they learned.", "Team members record improvements for later work."],
     "reference": "Price, H. (2024). Coordination routines in fictional teams. Journal of Fictional Organization Studies, 12, 101-114."},
)


def _pdf_bytes(article: dict[str, Any]) -> bytes:
    import pymupdf
    lines = (
        article["title"], article["authors"], article["journal"], str(article["year"]),
        article["volume"], article["pages"], "Abstract", article["abstract"],
        "Methods", article["body"], *article["items"], "References", article["reference"], "Appendix",
        "This fictional appendix is retained as a source-bound appendix.",
    )
    with pymupdf.open() as document:
        page = document.new_page(width=612, height=792)
        for index, text in enumerate(lines):
            page.insert_text((72, 58 + index * 43), text, fontsize=10)
        return document.tobytes(no_new_id=True)


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _frontend_sha() -> str:
    root = Path(__file__).resolve().parent.parent / "public_frontend"
    return sha256(canonical_bytes({path.relative_to(root).as_posix(): _sha(path)
                                   for path in sorted(root.rglob("*")) if path.is_file()})).hexdigest()


def _selection(reader: dict, source_sha: str, value: object, *, own_article: bool = True) -> dict:
    text = str(value)
    for unit in reader["units"]:
        if own_article and unit.get("lane") not in {"front_matter", "body"}:
            continue
        offset = unit["quote"].find(text)
        if offset >= 0:
            return {"unit_id": unit["unit_id"], "text": text, "start": unit["start"] + offset,
                    "end": unit["start"] + offset + len(text), "source_pdf_sha256": source_sha,
                    "quote_sha256": unit["quote_sha256"]}
    raise ValueError("public_literature_demo_literal_missing")


def _line_selection(reader: dict, source_sha: str, value: object) -> dict:
    """Select one controlled metadata line, never a coincidental substring."""
    text = str(value)
    for unit in reader["units"]:
        if unit.get("lane") in {"front_matter", "body"} and unit["quote"].strip() == text:
            return {"unit_id": unit["unit_id"], "text": text, "start": unit["start"],
                    "end": unit["start"] + len(text), "source_pdf_sha256": source_sha,
                    "quote_sha256": unit["quote_sha256"]}
    raise ValueError("public_literature_demo_metadata_line_missing")


def _receipt(status: str, payload: object, source_sha: str) -> dict:
    receipt = {"source_pdf_sha256": source_sha, "reviewer_type": "caller_attested_source_review",
               "version": "fictional-controlled-layout/v1",
               "reviewed_payload_sha256": sha256(canonical_bytes(payload)).hexdigest()}
    return {"status": status, "reviewer_type": receipt["reviewer_type"], "version": receipt["version"],
            "receipt": receipt, "receipt_sha256": sha256(canonical_bytes(receipt)).hexdigest()}


def _review(capture: dict, article: dict[str, Any], target_paper_id: str) -> dict:
    reader, source_sha = capture["native_reader"], capture["source_pdf_sha256"]
    fields = {}
    for name in ("title", "authors", "journal", "year"):
        value = article[name]
        fields[name] = {"value": value, "witness": _line_selection(reader, source_sha, value)}
    identity = {**_receipt("source_reviewed_complete", fields, source_sha), "fields": fields}
    supplements = {"volume": {"value": article["volume"], "witness": _line_selection(reader, source_sha, article["volume"])},
                   "pages_verbatim": {"value": article["pages"], "witness": _line_selection(reader, source_sha, article["pages"])}}
    apa_payload = {"authors_apa": article["authors_apa"], "title_apa": article["title_apa"], "byline_witness": fields["authors"]["witness"],
                   "supplements": supplements}
    apa = {**_receipt("source_reviewed_apa_rendering", apa_payload, source_sha), **apa_payload}
    ref_witness = _selection(reader, source_sha, article["reference"], own_article=False)
    measure_witness = _selection(reader, source_sha, article["measure"], own_article=False)
    item_witnesses = [_selection(reader, source_sha, item, own_article=False) for item in article["items"]]
    measure_payload = {"label": article["measure"], "witness": measure_witness, "items": item_witnesses}
    measure = {"label": article["measure"], "literal_evidence": article["measure"], "witness": measure_witness,
               "item_texts": item_witnesses,
               "review": _receipt("source_reviewed_meaningful_instrument", measure_payload, source_sha)}
    reference_id = "reference:" + sha256(canonical_bytes([capture["paper_id"], ref_witness])).hexdigest()
    edge_payload = {"reference_id": reference_id, "source_paper_id": capture["paper_id"], "target_paper_id": target_paper_id}
    return {"capture": capture, "identity_review": identity, "apa_review": apa,
            "references": [{"literal_reference": article["reference"], "witness": ref_witness,
                            "target_paper_id": target_paper_id, "edge_status": "reviewed_edge",
                            "edge_review": _receipt("source_reviewed_complete", edge_payload, source_sha)}],
            "measures": [measure]}


def _candidate_artifacts(root: Path, capture: dict, *, workspace: Workspace) -> dict:
    source = json.loads(Path(capture["source_packet_path"]).read_bytes())
    paper_id, source_sha = capture["paper_id"], capture["source_pdf_sha256"]
    original = {"version": "complete-extraction-sequence/v1", "paper": {"paper_id": paper_id,
        "title": "Fictional fixture; bibliographic view is separately source reviewed", "citation": "",
        "source_pdf_sha256": source_sha}, "findings": {"status": "not_started"}, "core": {"status": "not_started"},
        "measures": {"status": "not_started"}, "schema_omissions": [], "scientifically_accepted": False,
        "production_accepted": False, "complete_extraction_claimed": False}
    report = {"version": "complete-extraction-with-study-selections/v1", "original_report": original,
              "original_report_sha256": sha256(canonical_bytes(original)).hexdigest(),
              "study_fact_stage": {"status": "not_started"}, "declared_unavailable_stages": [],
              "scientifically_accepted": False, "production_accepted": False, "complete_extraction_claimed": False}
    report_path = workspace.write_canonical_json(root / "reports" / (paper_id + ".json"), report)
    snapshot = {"version": "mosaic-public-snapshot/v1", "paper": original["paper"],
                "source": {"units_sha256": sha256(canonical_bytes(source["units"])).hexdigest()},
                "source_packet_candidate": source, "source_packet_sha256": _sha(Path(capture["source_packet_path"])),
                "candidate_report": report, "validation_status": "candidate_only", "scientifically_accepted": False,
                "qualified": False}
    snapshot_path = workspace.write_canonical_json(root / "snapshots" / (paper_id + ".json"), snapshot)
    return {"report_path": report_path, "report_sha256": _sha(report_path), "snapshot_path": snapshot_path,
            "snapshot_sha256": _sha(snapshot_path)}


def build_literature_demo(project_root: Path, *, workspace: Workspace,
                          run_root: str = "run/literature-demo", port: int = 8798) -> dict:
    """Create a new two-source candidate-only bundle for assistant UI validation."""
    root = project_root / relative_path(run_root)
    report_path = root / "literature-demo-report.json"
    if report_path.is_file():
        from mosaic.public_serving import _verify_bundle
        stored = json.loads(report_path.read_bytes())
        _verify_bundle(Path(stored["bundle_path"]))
        if (stored.get("fixture_definition_sha256") != sha256(canonical_bytes(_ARTICLES)).hexdigest()
                or stored.get("implementation_sha256") != _sha(Path(__file__))
                or stored.get("frontend_sha256") != _frontend_sha()):
            raise ValueError("public_literature_demo_replay_changed")
        return stored
    pdf_root = project_root / "example-data" / "literature-demo-pdfs"
    for article in _ARTICLES:
        workspace.write_once(pdf_root / article["filename"], _pdf_bytes(article))
    config = {"version": "mosaic-public-build/v1", "profile": "demo",
              "sources": {"root": "example-data/literature-demo-pdfs", "files": [a["filename"] for a in _ARTICLES],
                          "permission": "fictional_fixture"}, "run": {"root": run_root},
              "runtime": {"kind": "recorded_replay"},
              "view": {"columns": ["title", "authors", "journal", "year"], "default_screen": "papers",
                       "sort": "distinct_article_count_desc"}, "serve": {"port": port},
              "jev": {"enabled": False, "replay_only": True}}
    manifest = register_sources(config, project_root=project_root, workspace=workspace)
    captures: list[dict] = []
    for row, article in zip(manifest["sources"], _ARTICLES):
        capture_root = root / "captured" / row["paper_id"]
        captured = capture_public(row, article["filename"], pdf_root, capture_root,
                                  title="Unreviewed fictional literature-demo source", workspace=workspace)
        receipt = json.loads((capture_root / "receipt.json").read_bytes())
        if receipt.get("status") != "source_packet_ready":
            raise ValueError("public_literature_demo_capture_refused")
        reader = json.loads((capture_root / "native-reader.json").read_bytes())
        captures.append({"paper_id": row["paper_id"], "source_pdf_path": capture_root / "source.pdf",
                         "source_pdf_sha256": row["source_pdf_sha256"], "native_reader": reader,
                         "capture_receipt_path": capture_root / "receipt.json", "capture_receipt_sha256": _sha(capture_root / "receipt.json"),
                         "source_packet_path": capture_root / "source.json", "source_packet_sha256": _sha(capture_root / "source.json")})
    sources = []
    for index, (capture, article) in enumerate(zip(captures, _ARTICLES)):
        source = _review(capture, article, captures[1 - index]["paper_id"])
        source["snapshot"] = _candidate_artifacts(root, capture, workspace=workspace)
        sources.append(source)
    materialized = materialize_public_corpus(sources, root / "display", workspace=workspace)
    result = {"version": VERSION, "bundle_path": materialized["bundle_path"], "bundle_sha256": _sha(Path(materialized["bundle_path"])),
              "paper_count": materialized["paper_count"], "admitted_identity_count": materialized["admitted_identity_count"],
              "apa_count": materialized["apa_count"], "fixture_review_provenance": "controlled fictional layout; caller-attested source review, never human review",
              "fixture_definition_sha256": sha256(canonical_bytes(_ARTICLES)).hexdigest(),
              "implementation_sha256": _sha(Path(__file__)), "frontend_sha256": _frontend_sha(),
              "candidate_reports_manufactured": True, "real_model_inference_run": False,
              "scientifically_accepted": False, "publication_approved": False,
              "limitations": ["Fictional controlled layout only.", "Candidate reports are manufactured and contain no model output.",
                              "Clean identities and APA records demonstrate source-bound display contracts, not external metadata accuracy."]}
    workspace.write_canonical_json(report_path, result)
    return result
