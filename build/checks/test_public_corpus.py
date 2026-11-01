from __future__ import annotations

import json
import hashlib
import threading
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

import pytest

from mosaic.public_corpus import materialize_public_corpus
from mosaic.public_serving import PublicServingRefusal, serve_public
from mosaic.workspace import Workspace
from test_public_serving import fixture


def test_bibliographic_display_preserves_raw_fields_and_uses_reviewed_case() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed for the frontend behavior check")
    script = Path(__file__).resolve().parents[1] / "public_frontend/assets/research-display-v1.js"
    check = r"""
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const nodes=new Map();
const document={querySelector(key){if(!nodes.has(key))nodes.set(key,{textContent:key==='#manifest'?'{}':'',options:[],append(o){this.options.push(o)}});return nodes.get(key)},createElement(){return {}}};
const context=vm.createContext({document,localStorage:{getItem(){return null}}});
const source=fs.readFileSync(process.argv[1],'utf8');
const prefix=source.slice(0,source.indexOf("  document.querySelectorAll('[data-screen]').forEach(x=>"));
vm.runInContext(prefix+';globalThis.ui={S,paperRows,card,selectedFilter,title,byline,bibLine};})();',context);
const ui=context.ui;
const row={paper_id:'p',title_verbatim:'RAW GATE TITLE',title_display:'A GATE <example>',authors_verbatim:['RAW NAME'],authors_display:['Jane <Doe>'],journal_verbatim:'RAW JOURNAL',journal_display:'Journal & Society',publication_year:2024};
const raw=JSON.stringify(row),html=ui.paperRows([row]);
assert.ok(html.includes('A GATE &lt;example&gt;'));assert.ok(html.includes('Jane &lt;Doe&gt;'));assert.ok(html.includes('Journal &amp; Society'));
assert.ok(!html.includes('RAW GATE TITLE'));assert.ok(!html.includes('<example>'));
const passage={paper_id:'p',id:'u1',quote:'Original GATE text.',page:2};
const card=ui.card(passage);assert.ok(card.includes('A GATE &lt;example&gt;'));assert.ok(card.includes('Original GATE text.'));
ui.S.paper='p';ui.selectedFilter();assert.equal(nodes.get('#paper-filter').options[0].textContent,'A GATE <example>');
assert.equal(JSON.stringify(row),raw);assert.equal(passage.quote,'Original GATE text.');
const fallback={...row,paper_id:'q',title_display:' ',authors_display:[],journal_display:null};
const fallbackHtml=ui.paperRows([fallback]);assert.ok(fallbackHtml.includes('RAW GATE TITLE'));assert.ok(fallbackHtml.includes('RAW NAME'));assert.ok(fallbackHtml.includes('RAW JOURNAL'));
assert.equal(ui.title({paper_id:'absent'}),'Article awaiting bibliographic review');
assert.ok(source.includes('o.textContent=paperTitle(r)'));
"""
    result = subprocess.run([node, "-e", check, str(script)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_two_sources_materialize_literal_review_queues(tmp_path: Path) -> None:
    first = tmp_path / "first"; first.mkdir(); _, capture1, snapshot1 = fixture(first)
    second = tmp_path / "second"; second.mkdir(); _, capture2, snapshot2 = fixture(second)
    result = materialize_public_corpus([
        {"capture": capture1, "snapshot": snapshot1, "references": [{"literal_reference": "Fictional public source.", "witness": {"unit_id": "u001", "source_pdf_sha256": capture1["source_pdf_sha256"], "quote_sha256": capture1["native_reader"]["units"][1]["quote_sha256"]}}], "measures": [{"measure_id": "m", "literal_evidence": "Fictional public source.", "witness": {"unit_id": "u001", "source_pdf_sha256": capture1["source_pdf_sha256"], "quote_sha256": capture1["native_reader"]["units"][1]["quote_sha256"]}}]},
        {"capture": capture2, "snapshot": snapshot2},
    ], tmp_path / "corpus", workspace=Workspace(repo_root=tmp_path, temp_root=tmp_path / "temp"))
    assert result["paper_count"] == result["identity_queue_count"] == 2
    queue = json.loads((tmp_path / "corpus" / "reference-review-queue.json").read_bytes())
    assert queue["items"][0]["reason"] == "literal_reference_witness_invalid"
    assert len(json.loads((tmp_path / "corpus" / "raw-passages.json").read_bytes())["items"]) == 4


def _reviewed(capture: dict, *, title: str, year: int) -> dict:
    """Make an explicit synthetic source review with literal unit spans."""
    reader = capture["native_reader"]
    quote = f"{title} | Doe, J. | Journal of Fiction | {year} | 12 | 10-19 | Team Coordination Scale"
    unit = reader["units"][1]
    unit["quote"] = quote
    unit["quote_sha256"] = hashlib.sha256(quote.encode()).hexdigest()
    unit["start"], unit["end"] = 9, 9 + len(quote)
    fields = {}
    for name, value in (("title", title), ("authors", "Doe, J."), ("journal", "Journal of Fiction"), ("year", year)):
        text = str(value); offset = quote.index(text)
        fields[name] = {"value": value, "witness": {"unit_id": "u001", "text": text,
            "start": unit["start"] + offset, "end": unit["start"] + offset + len(text),
            "source_pdf_sha256": capture["source_pdf_sha256"], "quote_sha256": unit["quote_sha256"]}}
    receipt = json.loads(Path(capture["capture_receipt_path"]).read_bytes())
    receipt["native_reader"]["sha256"] = hashlib.sha256(json.dumps(reader, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path = Path(capture["capture_receipt_path"]); path.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    capture["capture_receipt_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    witness = fields["title"]["witness"]
    identity_receipt = {"source_pdf_sha256": capture["source_pdf_sha256"], "reviewer_type": "caller_attested_source_review", "version": "test/v1", "reviewed_payload_sha256": hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    measure_witness = {**witness, "text": "Team Coordination Scale", "start": unit["start"] + quote.index("Team Coordination Scale"), "end": unit["start"] + quote.index("Team Coordination Scale") + len("Team Coordination Scale"), "quote_sha256": unit["quote_sha256"]}
    measure_payload = {"label": "Team Coordination Scale", "witness": measure_witness}
    measure_receipt = {"source_pdf_sha256": capture["source_pdf_sha256"], "reviewer_type": "caller_attested_source_review", "version": "test/v1", "reviewed_payload_sha256": hashlib.sha256(json.dumps(measure_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    def literal(text: str) -> dict:
        offset = quote.index(text)
        return {"unit_id": "u001", "text": text, "start": unit["start"] + offset,
                "end": unit["start"] + offset + len(text), "source_pdf_sha256": capture["source_pdf_sha256"],
                "quote_sha256": unit["quote_sha256"]}
    supplements = {name: {"value": value, "witness": literal(value)}
                   for name, value in (("volume", "12"), ("pages_verbatim", "10-19"))}
    apa_payload = {"authors_apa": ["Doe, J."], "title_apa": title.lower(),
                   "byline_witness": literal("Doe, J."), "supplements": supplements}
    apa_receipt = {"source_pdf_sha256": capture["source_pdf_sha256"], "reviewer_type": "caller_attested_source_review",
                   "version": "test/v1", "reviewed_payload_sha256": hashlib.sha256(json.dumps(apa_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    return {"capture": capture, "identity_review": {"status": "source_reviewed_complete", "reviewer_type": "caller_attested_source_review", "receipt": identity_receipt, "receipt_sha256": hashlib.sha256(json.dumps(identity_receipt, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "version": "test/v1", "fields": fields}, "apa_review": {"status": "source_reviewed_apa_rendering", "reviewer_type": "caller_attested_source_review", "receipt": apa_receipt, "receipt_sha256": hashlib.sha256(json.dumps(apa_receipt, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "version": "test/v1", **apa_payload}, "references": [{"literal_reference": title, "witness": witness}], "measures": [{"label": "Team Coordination Scale", "literal_evidence": "Team Coordination Scale", "witness": measure_witness, "review": {"status": "source_reviewed_meaningful_instrument", "reviewer_type": "caller_attested_source_review", "receipt": measure_receipt, "receipt_sha256": hashlib.sha256(json.dumps(measure_receipt, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "version": "test/v1"}}]}


def test_two_reviewed_sources_serve_identity_apa_measure_and_images(tmp_path: Path) -> None:
    first = tmp_path / "first"; second = tmp_path / "second"; first.mkdir(); second.mkdir()
    ws, capture_one, snapshot_one = fixture(first); _, capture_two, snapshot_two = fixture(second)
    sources = [_reviewed(capture_one, title="Coordination in Fiction", year=2024), _reviewed(capture_two, title="Coordination Across Fiction", year=2025)]
    sources[0]["snapshot"] = snapshot_one; sources[1]["snapshot"] = snapshot_two
    result = materialize_public_corpus(sources, tmp_path / "corpus", workspace=Workspace(repo_root=tmp_path, temp_root=tmp_path / "temp"))
    assert result["admitted_identity_count"] == result["apa_count"] == 2
    assert result["measure_count"] == 1
    server = serve_public(result["bundle_path"], 0); thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        index_page = urllib.request.urlopen(base + "/index.html").read().decode("utf-8")
        citations_page = urllib.request.urlopen(base + "/citation-map.html").read().decode("utf-8")
        assert "Local candidate display" in index_page and "Synthetic demonstration" not in index_page
        assert "Local candidate display" in citations_page and "Synthetic demonstration" not in citations_page
        assert "public-apa.js" in citations_page and "assets/citation-display-client-v1.js" not in citations_page
        bibliography = json.loads(urllib.request.urlopen(base + "/api/display/v1/bibliography?q=Coordination").read())
        assert bibliography["total"] == 2
        measures = json.loads(urllib.request.urlopen(base + "/api/display/v1/measures").read())
        assert measures["items"][0]["distinct_article_frequency"] == 2
        search = json.loads(urllib.request.urlopen(base + "/api/display/v1/passages?q=Fiction&page=1&page_size=10").read())
        assert search["total"] == 2
        for paper in result and json.loads(Path(result["bundle_path"]).read_bytes())["paper_ids"]:
            assert urllib.request.urlopen(base + f"/sources/{paper}.pdf").read().startswith(b"%PDF")
            assert urllib.request.urlopen(base + f"/api/display/v1/source-page?paper={paper}&page=1").read().startswith(b"\x89PNG")
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=5)


def test_forged_or_duplicate_public_sources_refuse(tmp_path: Path) -> None:
    ws, capture, snapshot = fixture(tmp_path)
    forged = dict(capture); forged["source_pdf_sha256"] = "0" * 64
    with pytest.raises(PublicServingRefusal):
        materialize_public_corpus([{"capture": forged, "snapshot": snapshot}], tmp_path / "bad", workspace=ws)
    with pytest.raises(PublicServingRefusal):
        materialize_public_corpus([{"capture": capture, "snapshot": snapshot}, {"capture": capture, "snapshot": snapshot}], tmp_path / "dup", workspace=ws)


def test_forged_review_receipt_or_mutated_field_stays_in_queue(tmp_path: Path) -> None:
    first = tmp_path / "first"; second = tmp_path / "second"; first.mkdir(); second.mkdir()
    _, capture_one, snapshot_one = fixture(first); _, capture_two, snapshot_two = fixture(second)
    forged = _reviewed(capture_one, title="Receipt Fiction", year=2024); forged["snapshot"] = snapshot_one
    forged["identity_review"]["receipt_sha256"] = "0" * 64
    mutated = _reviewed(capture_two, title="Mutated Fiction", year=2025); mutated["snapshot"] = snapshot_two
    mutated["identity_review"]["fields"]["title"]["value"] = "Different title"
    result = materialize_public_corpus([forged, mutated], tmp_path / "corpus", workspace=Workspace(repo_root=tmp_path, temp_root=tmp_path / "temp"))
    assert result["admitted_identity_count"] == result["apa_count"] == 0
    queue = json.loads((tmp_path / "corpus" / "identity-review-queue.json").read_bytes())["items"]
    assert {item["reason"] for item in queue} >= {"identity_review_receipt_missing_or_invalid", "identity_value_not_literal"}


def test_clean_paper_identity_does_not_require_apa_author_rendering(tmp_path: Path) -> None:
    ws, capture, snapshot = fixture(tmp_path)
    source = _reviewed(capture, title="Identity Before APA", year=2024); source["snapshot"] = snapshot
    del source["apa_review"]
    result = materialize_public_corpus([source], tmp_path / "corpus", workspace=ws)
    assert result["admitted_identity_count"] == 1
    assert result["apa_count"] == 0


def test_missing_curated_apa_title_withholds_only_apa(tmp_path: Path) -> None:
    ws, capture, snapshot = fixture(tmp_path)
    source = _reviewed(capture, title="Printed Title With NASA", year=2024); source["snapshot"] = snapshot
    source["apa_review"].pop("title_apa")
    result = materialize_public_corpus([source], tmp_path / "missing-title-apa", workspace=ws)
    assert result["admitted_identity_count"] == 1 and result["apa_count"] == 0


def test_curated_apa_title_requires_receipt_and_case_only_source_wording(tmp_path: Path) -> None:
    ws, capture, snapshot = fixture(tmp_path)
    source = _reviewed(capture, title="MOSAIC and iPhone in Fiction", year=2024); source["snapshot"] = snapshot
    source["apa_review"]["title_apa"] = "MOSAIC and iPhone in fiction"
    result = materialize_public_corpus([source], tmp_path / "tampered-title-apa", workspace=ws)
    assert result["admitted_identity_count"] == 1 and result["apa_count"] == 0
    valid = _reviewed(capture, title="MOSAIC and iPhone in Fiction", year=2024); valid["snapshot"] = snapshot
    valid["apa_review"]["title_apa"] = "MOSAIC and iPhone in fiction"
    payload = {key: valid["apa_review"][key] for key in ("authors_apa", "title_apa", "byline_witness", "supplements")}
    valid["apa_review"]["receipt"]["reviewed_payload_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    valid["apa_review"]["receipt_sha256"] = hashlib.sha256(json.dumps(valid["apa_review"]["receipt"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    accepted = materialize_public_corpus([valid], tmp_path / "curated-proper-names", workspace=ws)
    assert accepted["apa_count"] == 1
    assert valid["identity_review"]["fields"]["title"]["value"] == "MOSAIC and iPhone in Fiction"
    bibliography = json.loads((tmp_path / "curated-proper-names" / "bibliography.json").read_bytes())
    assert "MOSAIC and iPhone in fiction." in bibliography["items"][0]["citation_apa"]
    forged = _reviewed(capture, title="MOSAIC and iPhone in Fiction", year=2024); forged["snapshot"] = snapshot
    forged["apa_review"]["title_apa"] = "Different wording"
    payload = {key: forged["apa_review"][key] for key in ("authors_apa", "title_apa", "byline_witness", "supplements")}
    forged["apa_review"]["receipt"]["reviewed_payload_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    forged["apa_review"]["receipt_sha256"] = hashlib.sha256(json.dumps(forged["apa_review"]["receipt"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    rejected = materialize_public_corpus([forged], tmp_path / "forged-wording", workspace=ws)
    assert rejected["admitted_identity_count"] == 1 and rejected["apa_count"] == 0


def test_corpus_static_ui_uses_real_products_and_never_uses_demo_banner(tmp_path: Path) -> None:
    ws, capture, snapshot = fixture(tmp_path)
    source = _reviewed(capture, title="UI Wiring Fiction", year=2024); source["snapshot"] = snapshot
    result = materialize_public_corpus([source], tmp_path / "corpus", workspace=ws)
    bundle = json.loads(Path(result["bundle_path"]).read_bytes())
    static = tmp_path / "corpus" / "static"
    assert "assets/research-display-v1.js" in bundle["frontend"]
    assert "Fictional build demonstration" not in (static / "workspace.html").read_text(encoding="utf-8")
    assert "Synthetic demonstration" not in (static / "index.html").read_text(encoding="utf-8")
    assert "Local candidate display" in (static / "index.html").read_text(encoding="utf-8")
    assert "Synthetic demonstration" not in (static / "citation-map.html").read_text(encoding="utf-8")
    assert "Local candidate display" in (static / "citation-map.html").read_text(encoding="utf-8")
    paper_script = (static / "assets/research-display-v1.js").read_text(encoding="utf-8")
    assert "/api/display/v1/bibliography" in paper_script and "Keep in Saved" in paper_script
    measure_page = (static / "measures.html").read_text(encoding="utf-8")
    assert "Team Coordination Scale" in measure_page and "Project" in measure_page
    assert "Fictional build demonstration" not in measure_page and "Synthetic demonstration" not in measure_page
    assert "UI Wiring Fiction" in measure_page
    data = json.loads(re.search(r'<script id="measure-data" type="application/json">(.*?)</script>', measure_page, re.S)[1])
    assert data["versions"][0]["item_evidence"] == []
    bibliography = json.loads(re.search(r'<script id="measure-bibliography-v1" type="application/json">(.*?)</script>', measure_page, re.S)[1])
    assert next(iter(bibliography.values()))["title_verbatim"] == "UI Wiring Fiction"
    citation_script = (static / "public-apa.js").read_text(encoding="utf-8")
    assert "/api/display/v1/apa" in citation_script and "/api/citations/health" in citation_script
    assert "CitationMapCore" in (static / "citation-map.js").read_text(encoding="utf-8")
    assert "Fictional build demonstration" not in (static / "reference-network-review.html").read_text(encoding="utf-8")


def test_corpus_refuses_changed_index_banner_template(tmp_path: Path, monkeypatch) -> None:
    import mosaic.public_serving as serving
    frontend = tmp_path / "frontend"
    shutil.copytree(serving.FRONTEND_ROOT, frontend)
    index = frontend / "index.html"
    index.write_text(index.read_text(encoding="utf-8").replace("Synthetic demonstration.", "Changed demonstration.", 1), encoding="utf-8")
    monkeypatch.setattr(serving, "FRONTEND_ROOT", frontend)
    input_root = tmp_path / "input"; input_root.mkdir()
    ws, capture, snapshot = fixture(input_root)
    source = _reviewed(capture, title="Seam Refusal Fiction", year=2024); source["snapshot"] = snapshot
    with pytest.raises(PublicServingRefusal, match="index_banner_seam_missing"):
        materialize_public_corpus([source], input_root / "corpus", workspace=ws)


def test_apa_rejects_identity_overrides_and_unwitnessed_supplements(tmp_path: Path) -> None:
    ws, capture, snapshot = fixture(tmp_path)
    source = _reviewed(capture, title="Safe APA Fiction", year=2024); source["snapshot"] = snapshot
    source["apa_review"]["supplements"]["title"] = {"value": "Injected", "witness": {}}
    result = materialize_public_corpus([source], tmp_path / "corpus", workspace=ws)
    assert result["admitted_identity_count"] == 1
    assert result["apa_count"] == 0


def test_apa_byline_must_match_the_admitted_identity_witness(tmp_path: Path) -> None:
    ws, capture, snapshot = fixture(tmp_path)
    source = _reviewed(capture, title="Bound Byline Fiction", year=2024); source["snapshot"] = snapshot
    witness = dict(source["apa_review"]["byline_witness"])
    witness["text"] = "Doe"
    witness["end"] = witness["start"] + len("Doe")
    source["apa_review"]["byline_witness"] = witness
    payload = {key: source["apa_review"][key] for key in ("authors_apa", "title_apa", "byline_witness", "supplements")}
    receipt = source["apa_review"]["receipt"]
    receipt["reviewed_payload_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    source["apa_review"]["receipt_sha256"] = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    result = materialize_public_corpus([source], tmp_path / "corpus", workspace=ws)
    assert result["admitted_identity_count"] == 1
    assert result["apa_count"] == 0
    queue = json.loads((tmp_path / "corpus" / "identity-review-queue.json").read_bytes())
    assert any(row["reason"] == "apa_review_byline_witness_mismatch" for row in queue["items"])


def _add_reference_lane(capture: dict, *, text: str) -> dict:
    """Add an otherwise valid raw reference unit and refresh its capture binding."""
    reader = capture["native_reader"]
    body = reader["units"][1]
    reference = {**body, "unit_id": "u-reference", "ordinal": len(reader["units"]), "quote": text,
                 "quote_sha256": hashlib.sha256(text.encode()).hexdigest(), "start": 1000,
                 "end": 1000 + len(text), "lane": "references", "default_visible": False}
    reader["units"].append(reference)
    receipt_path = Path(capture["capture_receipt_path"])
    receipt = json.loads(receipt_path.read_bytes())
    receipt["native_reader"]["sha256"] = hashlib.sha256(json.dumps(reader, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    receipt["native_reader"]["unit_count"] = len(reader["units"])
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    capture["capture_receipt_sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    return reference


def _reference_witness(capture: dict, unit: dict, text: str) -> dict:
    start = unit["start"] + unit["quote"].index(text)
    return {"unit_id": unit["unit_id"], "text": text, "start": start, "end": start + len(text),
            "source_pdf_sha256": capture["source_pdf_sha256"], "quote_sha256": unit["quote_sha256"]}


def test_materializer_withholds_identity_and_apa_supplement_from_reference_lane(tmp_path: Path) -> None:
    ws, capture, snapshot = fixture(tmp_path)
    source = _reviewed(capture, title="Lane Bound Fiction", year=2024); source["snapshot"] = snapshot
    reference = _add_reference_lane(capture, text="Doe, J. (2024). Lane Bound Fiction. Journal of Fiction. doi:10.9999/cited")

    # A hand-authored, receipt-rehashed identity that cites a reference must
    # remain in the queue even though each span is literal and source-bound.
    for field in source["identity_review"]["fields"].values():
        field["witness"] = _reference_witness(capture, reference, field["witness"]["text"])
    fields = source["identity_review"]["fields"]
    identity_receipt = source["identity_review"]["receipt"]
    identity_receipt["reviewed_payload_sha256"] = hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    source["identity_review"]["receipt_sha256"] = hashlib.sha256(json.dumps(identity_receipt, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    # Restore a valid own-article identity, then show that a cited DOI cannot
    # make the APA rendering visible either.
    valid = _reviewed(capture, title="Lane Bound Fiction", year=2024)
    valid["snapshot"] = snapshot
    valid["apa_review"]["supplements"]["doi"] = {"value": "doi:10.9999/cited", "witness": _reference_witness(capture, reference, "doi:10.9999/cited")}
    payload = {key: valid["apa_review"][key] for key in ("authors_apa", "title_apa", "byline_witness", "supplements")}
    apa_receipt = valid["apa_review"]["receipt"]
    apa_receipt["reviewed_payload_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    valid["apa_review"]["receipt_sha256"] = hashlib.sha256(json.dumps(apa_receipt, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    rejected = materialize_public_corpus([source], tmp_path / "reference-identity", workspace=ws)
    assert rejected["admitted_identity_count"] == 0
    allowed_identity = materialize_public_corpus([valid], tmp_path / "reference-doi", workspace=ws)
    assert allowed_identity["admitted_identity_count"] == 1 and allowed_identity["apa_count"] == 0


@pytest.mark.parametrize(("relative", "before", "after", "reason"), [
    ("index.html", "Synthetic demonstration.", "Changed demonstration.", "index_banner_seam_missing"),
    ("index.html", "Synthetic demonstration.", "Synthetic demonstration.Synthetic demonstration.", "index_banner_seam_missing"),
    ("citation-map.html", "assets/citation-display-client-v1.js", "", "citation_client_seam_missing"),
    ("citation-map.html", "assets/citation-display-client-v1.js", "assets/citation-display-client-v1.jsassets/citation-display-client-v1.js", "citation_client_seam_missing"),
    ("measures.html", '<script id="measure-bibliography-v1" type="application/json">{}</script>', "", "measure_bibliography_seam_missing"),
    ("measures.html", '<script id="measure-bibliography-v1" type="application/json">{}</script>', '<script id="measure-bibliography-v1" type="application/json">{}</script><script id="measure-bibliography-v1" type="application/json">{}</script>', "measure_bibliography_seam_missing"),
])
def test_corpus_refuses_zero_or_duplicate_reviewed_frontend_seams(tmp_path: Path, monkeypatch, relative: str, before: str, after: str, reason: str) -> None:
    import mosaic.public_serving as serving
    frontend = tmp_path / "frontend"
    shutil.copytree(serving.FRONTEND_ROOT, frontend)
    changed = frontend / relative
    changed.write_text(changed.read_text(encoding="utf-8").replace(before, after, 1), encoding="utf-8")
    monkeypatch.setattr(serving, "FRONTEND_ROOT", frontend)
    input_root = tmp_path / "input"; input_root.mkdir()
    ws, capture, snapshot = fixture(input_root)
    source = _reviewed(capture, title="Seam Variant Fiction", year=2024); source["snapshot"] = snapshot
    with pytest.raises(PublicServingRefusal, match=reason):
        materialize_public_corpus([source], input_root / "corpus", workspace=ws)
