"""Actual fictional PDF -> native protocol -> reviewed views -> optional advice.

No private corpus or real provider/model is used. The identity witnesses come
from actual capture, unlike the intentionally hand-built negative controls.
"""
from hashlib import sha256
from http.server import ThreadingHTTPServer
import json
import re
from pathlib import Path
from threading import Thread
from decimal import Decimal
from urllib.request import urlopen

import pymupdf
import pytest

from mosaic.canonical import canonical_bytes
from mosaic.jev.ledger import Budget
from mosaic.public_corpus import materialize_public_corpus
from mosaic.public_jev import assess_public_task, load_public_task, make_public_client
from mosaic.public_serving import serve_public
from mosaic.public_workflow import run_public_corpus, source_handoff
from test_public_execution import _Loopback
from test_public_workflow import _inputs
from test_public_jev import Transport

pytestmark = pytest.mark.evidence_mock


def build_reviewed_fixture(tmp_path):
    config, kwargs = _inputs(tmp_path)
    ws = kwargs["workspace"]
    title = "Study 1: Coordination in a Fictional Team"
    byline, journal, year = "Ruth Keller", "Fictional Methods Journal", 2024
    lines = [title, byline, journal, str(year), "Volume 12, pages 10-19", "Abstract",
             "This fictional article illustrates the software workflow.", "Methods",
             "The Team Coordination Scale measures fictional team coordination.", "Results",
             "Participants in this fictional demonstration reported coordination.", "References",
             "Keller, R. (2024). A fictional reference. Fictional Methods Journal, 12, 1-9."]
    with pymupdf.open() as document:
        page = document.new_page()
        for index, line in enumerate(lines):
            page.insert_text((50, 50 + index * 35), line, fontsize=10)
        ws.write_once(tmp_path / "sources/reviewed.pdf", document.tobytes())
    config["sources"]["files"] = ["reviewed.pdf"]
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Loopback)
    worker = Thread(target=server.serve_forever)
    worker.start()
    config["runtime"]["base_url"] = f"http://127.0.0.1:{server.server_address[1]}/v1"
    try:
        completed = run_public_corpus(config, **kwargs, live=True)
    finally:
        server.shutdown(); worker.join(); server.server_close()
    capture = next((tmp_path / "run/corpus/stages/capture").rglob("native-reader.json")).parent
    snapshot = next((tmp_path / "run/corpus/native").rglob("public-snapshot.json"))
    receipt = json.loads((snapshot.parent / "execution-receipt.json").read_bytes())
    report = snapshot.parent / "driver" / receipt["paper_report_path"]
    source = source_handoff(capture, report, snapshot, workspace=ws)
    pdf_sha = source["capture"]["source_pdf_sha256"]
    units = source["capture"]["native_reader"]["units"]
    def witness(text):
        unit = next(u for u in units if text in u["quote"])
        start = unit["start"] + unit["quote"].index(text)
        return {"unit_id": unit["unit_id"], "source_pdf_sha256": pdf_sha,
                "quote_sha256": unit["quote_sha256"], "start": start, "end": start + len(text), "text": text}
    def reviewed(status, payload):
        receipt = {"source_pdf_sha256": pdf_sha, "reviewer_type": "caller_attested_source_review",
                   "version": "fictional-protocol-check/v1", "reviewed_payload_sha256": sha256(canonical_bytes(payload)).hexdigest()}
        return {"status": status, "reviewer_type": receipt["reviewer_type"], "version": receipt["version"],
                "receipt": receipt, "receipt_sha256": sha256(canonical_bytes(receipt)).hexdigest()}
    fields = {name: {"value": value, "witness": witness(str(value))}
              for name, value in (("title", title), ("authors", byline), ("journal", journal), ("year", year))}
    source["identity_review"] = {**reviewed("source_reviewed_complete", fields), "fields": fields}
    apa_payload = {"authors_apa": ["Keller, R."], "title_apa": "Study 1: Coordination in a fictional team",
                   "byline_witness": fields["authors"]["witness"],
                   "supplements": {"volume": {"value": "12", "witness": witness("12")},
                                   "pages_verbatim": {"value": "10-19", "witness": witness("10-19")}}}
    source["apa_review"] = {**reviewed("source_reviewed_apa_rendering", apa_payload), **apa_payload}
    measure_witness = witness("Team Coordination Scale")
    source["measures"] = [{"label": "Team Coordination Scale", "literal_evidence": "Team Coordination Scale",
        "witness": measure_witness, "review": reviewed("source_reviewed_meaningful_instrument",
                                                       {"label": "Team Coordination Scale", "witness": measure_witness})}]
    packet = json.loads(snapshot.read_bytes())
    selected = packet["source"]["units"][0]
    task = {"version": "mosaic-public-jev-task/v1", "task_id": "public-fictional-integrated",
            "snapshot": {"path": str(snapshot), "sha256": sha256(snapshot.read_bytes()).hexdigest()},
            "source": {"source_sha256": packet["source"]["source_sha256"], "pdf_sha256": pdf_sha,
                       "aliases": ["fictional-integrated"], "declared_protected_hashes": []},
            "caller_permission": {"authorized": True, "scope": "post_extraction_advisory", "declared_by": "fictional control"},
            "source_selections": [{"unit_id": selected["unit_id"], "start": 0, "end": len(selected["text"])}],
            "definitions": {"classification": "fictional software check only"},
            "questions": {"classification": {"type": "single_label", "labels": ["advisory"]}}, "pattern": "public-fictional-integrated-v1"}
    task_path = ws.write_canonical_json(tmp_path / "advisory-task.json", task)
    client = make_public_client(load_public_task(task_path), ledger_path=tmp_path / "installation-jev.jsonl",
                                budget=Budget(Decimal("1"), 2), workspace=ws, transport=Transport(), api_key="test-key")
    advisory = assess_public_task(task_path, tmp_path / "sidecars", client=client, workspace=ws)
    source["advisory_sidecars"] = [{"task_path": str(task_path), "sidecar_path": advisory["sidecar_path"],
                                    "sidecar_sha256": advisory["sidecar_sha256"]}]
    result = materialize_public_corpus([source], tmp_path / "reviewed-view", workspace=ws)
    return result, completed, source


def test_actual_capture_to_reviewed_interface_and_optional_advice(tmp_path):
    result, completed, source = build_reviewed_fixture(tmp_path)
    assert completed["status"] == "candidate_corpus_served" and not completed["real_model_inference_run"]
    assert result["admitted_identity_count"] == result["apa_count"] == result["measure_count"] == 1
    server = serve_public(result["bundle_path"], 0)
    worker = Thread(target=server.serve_forever); worker.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        get = lambda route: json.load(urlopen(base + route))
        assert get("/api/display/v1/bibliography")["total"] == 1
        assert get("/api/display/v1/apa")["total"] == 1
        assert get("/api/display/v1/measures")["items"][0]["distinct_article_frequency"] == 1
        advice = get("/api/display/v1/advisories")
        assert advice["total"] == 1
        assert not advice["items"][0]["scientifically_accepted"]
        assert "Fictional build demonstration" not in urlopen(base + "/workspace.html").read().decode()
    finally:
        server.shutdown(); worker.join(); server.server_close()


def test_actual_capture_studies_manifest_and_payload_serve_over_http(tmp_path):
    """Generated PDF -> corpus -> Studies page/data/API remains candidate-only."""
    result, completed, source = build_reviewed_fixture(tmp_path)
    assert completed["status"] == "candidate_corpus_served"
    paper_id = source["capture"]["paper_id"]
    expected_pdf_sha = source["capture"]["source_pdf_sha256"]
    server = serve_public(result["bundle_path"], 0)
    worker = Thread(target=server.serve_forever)
    worker.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        html = urlopen(base + f"/studies.html?paper={paper_id}").read().decode("utf-8")
        match = re.search(r'<script id="study-manifest" type="application/json">(.*?)</script>', html, re.S)
        assert match is not None
        manifest = json.loads(match.group(1))
        row = next(item for item in manifest["papers"] if item["paper_id"] == paper_id)
        payload = urlopen(base + "/" + row["data_path"]).read()
        assert sha256(payload).hexdigest() == row["data_sha256"]
        study = json.loads(payload)
        identifier = study["studies"][0]["identifier_evidence"][0]
        assert study["paper"]["title"] == "Study 1: Coordination in a Fictional Team"
        assert identifier["quote"] == "Study 1: Coordination in a Fictional Team"
        assert identifier["pages"] == [1]
        assert sha256(urlopen(base + f"/sources/{paper_id}.pdf").read()).hexdigest() == expected_pdf_sha
        assert row["study_fact_status"] == study["public_wrapper_status"]["study_fact"] == "invalid_proposal"
        assert row["statistic_annotation_status"] == study["public_wrapper_status"]["statistic_annotation"] == "not_supplied"
        assert "Candidate-stage availability" in html
        assert "invalid_proposal" in html and "not_supplied" in html
        report = json.load(urlopen(base + "/api/display/v1/candidate-report"))
        assert report["scientifically_accepted"] is False
    finally:
        server.shutdown(); worker.join(); server.server_close()
