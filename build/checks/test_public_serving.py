"""Synthetic controls for the public local serving boundary."""
from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pymupdf
import pytest

from mosaic.public_serving import PublicServingRefusal, materialize_public, serve_public
from mosaic.workspace import Workspace


pytestmark = pytest.mark.evidence_mock


def fixture(tmp_path: Path) -> tuple[Workspace, dict, dict]:
    ws = Workspace(repo_root=tmp_path, temp_root=tmp_path / "temp")
    doc = pymupdf.open(); page = doc.new_page(); page.insert_text((72, 72), "Fictional public source.")
    raw = doc.tobytes(garbage=4, deflate=True); pdf = ws.write_once(tmp_path / "input.pdf", raw); sha = hashlib.sha256(raw).hexdigest()
    heading = "Abstract\n"; quote = "Fictional public source.\n"
    units = [
        {"unit_id": "u000", "page": 1, "physical_page_id": "p001", "text_region_id": "r000", "start": 0, "end": len(heading), "quote": heading, "quote_sha256": hashlib.sha256(heading.encode()).hexdigest(), "source_pdf_sha256": sha, "lane": "abstract_heading", "default_visible": False, "ordinal": 0},
        {"unit_id": "u001", "page": 1, "physical_page_id": "p001", "text_region_id": "r001", "start": len(heading), "end": len(heading) + len(quote), "quote": quote, "quote_sha256": hashlib.sha256(quote.encode()).hexdigest(), "source_pdf_sha256": sha, "lane": "body", "default_visible": True, "ordinal": 1},
    ]
    reader = {"version": "public-native-reader/v1", "package_manifest_id": "manifest-fixture", "source_pdf_sha256": sha, "scope": "native_capture_full_package_candidate", "units": units, "default_mask": {"status": "candidate_literal_abstract_heading", "anchor_unit_id": "u000"}, "scientifically_accepted": False}
    paper = "public-" + sha
    source = {"work_id": paper, "source_pdf_sha256": sha, "source_policy_sha256": "0" * 64, "units": [{"unit_id": "U001", "text": quote, "page": 1}]}
    source_path = ws.write_canonical_json(tmp_path / "source.json", source); source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    capture_report = {"version": "public-capture-source-packet/v1", "work_id": paper, "status": "source_packet_ready", "source_pdf_sha256": sha, "source_packet_sha256": source_sha, "package_manifest_id": "manifest-fixture", "native_reader": {"path": "native-reader.json", "sha256": hashlib.sha256(json.dumps(reader, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "unit_count": len(units), "scope": reader["scope"]}, "scientifically_accepted": False}
    capture_path = ws.write_canonical_json(tmp_path / "capture-receipt.json", capture_report)
    report_data = {"version": "complete-extraction-with-study-selections/v1", "original_report": {"version": "complete-extraction-sequence/v1", "paper": {"paper_id": paper, "title": "Unreviewed", "citation": "", "source_pdf_sha256": sha}, "findings": {"status": "not_started"}, "core": {"status": "not_started"}, "measures": {"status": "not_started"}, "schema_omissions": [], "scientifically_accepted": False, "production_accepted": False, "complete_extraction_claimed": False}, "original_report_sha256": "", "study_fact_stage": {"status": "not_started"}, "declared_unavailable_stages": [], "scientifically_accepted": False, "production_accepted": False, "complete_extraction_claimed": False}
    report_data["original_report_sha256"] = hashlib.sha256(json.dumps(report_data["original_report"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    report = ws.write_canonical_json(tmp_path / "report.json", report_data)
    snapshot_data = {"version": "mosaic-public-snapshot/v1", "paper": {"paper_id": paper, "title": "Unreviewed", "citation": "", "source_pdf_sha256": sha}, "source": {"units_sha256": hashlib.sha256(json.dumps(source["units"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()}, "source_packet_candidate": source, "source_packet_sha256": source_sha, "candidate_report": report_data, "validation_status": "candidate_only", "scientifically_accepted": False, "qualified": False}
    snapshot_path = ws.write_canonical_json(tmp_path / "public-snapshot.json", snapshot_data)
    return ws, {"paper_id": paper, "source_pdf_path": pdf, "source_pdf_sha256": sha, "native_reader": reader, "capture_receipt_path": capture_path, "capture_receipt_sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest(), "source_packet_path": source_path, "source_packet_sha256": source_sha}, {"snapshot_path": snapshot_path, "snapshot_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(), "report_path": report, "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest()}


def test_materialize_and_serve_exact_source(tmp_path: Path) -> None:
    ws, capture, snapshot = fixture(tmp_path); result = materialize_public(capture, snapshot, tmp_path / "run", workspace=ws)
    server = serve_public(result["bundle_path"], 0); thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"; paper = result["paper_id"]
        for route in ("/index.html", "/workspace.html", "/measures.html", "/citation-map.html", "/collection.html", "/studies.html",
                      f"/source-viewer.html?paper={paper}&page=1"):
            response = urllib.request.urlopen(base + route)
            assert response.status == 200
        viewer = urllib.request.urlopen(base + f"/source-viewer.html?paper={paper}&page=1").read().decode("utf-8")
        expected_nav = ["Start", "Measures", "Papers", "Compare", "Project", "Citations"]
        assert all(viewer.index(label) < viewer.index(expected_nav[index + 1]) for index, label in enumerate(expected_nav[:-1]))
        assert "width:164px" in viewer and "width:126px" in viewer and "Fictional build demonstration" in viewer
        assert "index.html?paper=" in viewer and "Fictional build demonstration" in urllib.request.urlopen(base + "/workspace.html").read().decode("utf-8")
        measures_page = urllib.request.urlopen(base + "/measures.html").read().decode("utf-8")
        assert all(marker in measures_page for marker in ('id="measure-display-keys-v1" type="application/json">{}', 'id="measure-bibliography-v1" type="application/json">{}', 'id="measure-method-exclusions-v1" type="application/json">[]'))
        assert result["snapshot_id"] in measures_page and "public-unresolved" not in measures_page
        assert paper not in measures_page  # reader identity remains in the local API, not a false measure source.
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(base + "/citation-map-data.json")
        assert exc.value.code == 404
        citation_client = urllib.request.urlopen(base + "/citation-map.js").read().decode("utf-8")
        assert "Reference network is not built for this demonstration" in citation_client and "/api/citations/" not in citation_client
        image = urllib.request.urlopen(base + f"/api/display/v1/source-page?paper={paper}&page=1"); pdf = urllib.request.urlopen(base + f"/sources/{paper}.pdf")
        assert image.status == pdf.status == 200 and image.read().startswith(b"\x89PNG") and pdf.read().startswith(b"%PDF-")
        assert image.headers["X-Source-PDF-SHA256"] == capture["source_pdf_sha256"]
        raw = json.loads(urllib.request.urlopen(base + f"/api/display/v1/raw-passages?paper={paper}").read())
        assert raw["total"] == 2 and raw["passages"][1]["quote"] == "Fictional public source.\n"
        default = json.loads(urllib.request.urlopen(base + f"/api/display/v1/passages?paper={paper}").read())
        assert [item["unit_id"] for item in default["passages"]] == ["u001"]
        searched = json.loads(urllib.request.urlopen(base + f"/api/display/v1/passages?paper={paper}&q=Fictional&page=1&page_size=1").read())
        assert searched["total"] == 1 and len(searched["passages"]) == 1 and searched["page_size"] == 1
        with pytest.raises(urllib.error.HTTPError) as exc: urllib.request.urlopen(base + f"/api/display/v1/source-page?paper={paper}&page=2")
        assert exc.value.code == 404
        with pytest.raises(urllib.error.HTTPError) as exc: urllib.request.urlopen(base + "/api/display/v1/source-page?paper=../x&page=1")
        assert exc.value.code == 404
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=5)


def test_tampered_bundle_and_bad_reader_refuse(tmp_path: Path) -> None:
    ws, capture, snapshot = fixture(tmp_path); result = materialize_public(capture, snapshot, tmp_path / "run", workspace=ws)
    bundle = Path(result["bundle_path"]); data = json.loads(bundle.read_bytes()); data["artifacts"]["raw-passages.json"] = "0" * 64; bundle.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(PublicServingRefusal, match="artifact_mismatch"): serve_public(bundle)
    capture["paper_id"] = "public-wrong"
    with pytest.raises(PublicServingRefusal, match="paper_id_mismatch"): materialize_public(capture, snapshot, tmp_path / "other", workspace=ws)


def test_bundle_requires_every_hashed_artifact(tmp_path: Path) -> None:
    ws, capture, snapshot = fixture(tmp_path)
    result = materialize_public(capture, snapshot, tmp_path / "run", workspace=ws)
    bundle = Path(result["bundle_path"])
    data = json.loads(bundle.read_bytes())
    del data["artifacts"]["candidate-report.json"]
    bundle.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(PublicServingRefusal, match="artifact_set_invalid"):
        serve_public(bundle)


def test_tampered_source_refuses_before_server_start(tmp_path: Path) -> None:
    ws, capture, snapshot = fixture(tmp_path)
    result = materialize_public(capture, snapshot, tmp_path / "run", workspace=ws)
    packaged = tmp_path / "run" / "sources" / "source.pdf"
    packaged.write_bytes(b"%PDF-tampered")
    with pytest.raises(PublicServingRefusal, match="source_pdf_hash_mismatch"):
        serve_public(result["bundle_path"])


def test_rejects_nested_acceptance_and_mask_that_exposes_front_matter(tmp_path: Path) -> None:
    ws, capture, snapshot = fixture(tmp_path)
    report = Path(snapshot["report_path"])
    body = json.loads(report.read_bytes()); body["nested"] = {"scientifically_accepted": True}
    report.write_text(json.dumps(body), encoding="utf-8")
    snapshot["report_sha256"] = hashlib.sha256(report.read_bytes()).hexdigest()
    with pytest.raises(PublicServingRefusal, match="report_binding"):
        materialize_public(capture, snapshot, tmp_path / "nested", workspace=ws)
    second = tmp_path / "second"; second.mkdir()
    ws, capture, snapshot = fixture(second)
    capture["native_reader"]["units"][0]["lane"] = "front_matter"
    with pytest.raises(PublicServingRefusal, match="default_anchor"):
        materialize_public(capture, snapshot, second / "mask", workspace=ws)


def test_rejects_masked_or_unknown_lane_visibility(tmp_path: Path) -> None:
    ws, capture, snapshot = fixture(tmp_path)
    capture["native_reader"]["units"][1]["lane"] = "table_region_candidate"
    with pytest.raises(PublicServingRefusal, match="default_visibility_invalid"):
        materialize_public(capture, snapshot, tmp_path / "table", workspace=ws)
    second = tmp_path / "second"; second.mkdir()
    ws, capture, snapshot = fixture(second)
    capture["native_reader"]["units"][1]["lane"] = "unrecognized_lane"
    with pytest.raises(PublicServingRefusal, match="binding_invalid"):
        materialize_public(capture, snapshot, second / "unknown", workspace=ws)


def test_duplicate_units_report_binding_and_unlisted_static_refuse(tmp_path: Path) -> None:
    ws, capture, snapshot = fixture(tmp_path)
    capture["native_reader"]["units"].append(dict(capture["native_reader"]["units"][1], ordinal=2))
    with pytest.raises(PublicServingRefusal, match="units_invalid"):
        materialize_public(capture, snapshot, tmp_path / "duplicate", workspace=ws)
    bound = tmp_path / "bound"; bound.mkdir()
    ws, capture, snapshot = fixture(bound)
    snapshot_report = Path(snapshot["report_path"])
    report = json.loads(snapshot_report.read_bytes()); report["paper_id"] = "public-" + "0" * 64
    snapshot_report.write_text(json.dumps(report), encoding="utf-8")
    snapshot["report_sha256"] = hashlib.sha256(snapshot_report.read_bytes()).hexdigest()
    with pytest.raises(PublicServingRefusal, match="report_binding"):
        materialize_public(capture, snapshot, bound / "bound-run", workspace=ws)
    static = tmp_path / "static"; static.mkdir()
    ws, capture, snapshot = fixture(static)
    result = materialize_public(capture, snapshot, static / "run", workspace=ws)
    (static / "run" / "static" / "unlisted.html").write_text("not bundled", encoding="utf-8")
    with pytest.raises(PublicServingRefusal, match="member_set_mismatch"):
        serve_public(result["bundle_path"])


def test_actual_public_capture_runtime_snapshot_envelope_materializes(tmp_path: Path, monkeypatch) -> None:
    """Exercise the real fictional capture → driver → snapshot handoff."""
    import socket
    from mosaic.public_runtime import run_extraction
    from test_public_runtime import FixtureTransport, _packet, _profile
    monkeypatch.setattr(socket, "socket", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no socket")))
    ws, packet = _packet(tmp_path)
    transport = FixtureTransport()
    runtime = run_extraction(packet, tmp_path / "runtime", workspace=ws, profile=_profile(),
                             replay={"kind": "recorded_fictional_fixture", "fixture_transport": transport,
                                     "fixture_count": lambda text: len(text) // 3, "deadline_epoch": time.time() + 120})
    capture_root = tmp_path / "capture"; receipt = packet["capture_receipt"]
    result = materialize_public({"paper_id": receipt["work_id"], "source_pdf_path": tmp_path / "sources" / "fixture.pdf",
        "source_pdf_sha256": receipt["source_pdf_sha256"], "native_reader": capture_root / "native-reader.json",
        "capture_receipt_path": capture_root / "receipt.json", "capture_receipt_sha256": hashlib.sha256((capture_root / "receipt.json").read_bytes()).hexdigest(),
        "source_packet_path": capture_root / "source.json", "source_packet_sha256": receipt["source_packet_sha256"]},
        {"snapshot_path": runtime["snapshot_path"], "snapshot_sha256": runtime["receipt"]["snapshot_sha256"],
         "report_path": runtime["report_path"], "report_sha256": runtime["receipt"]["report_sha256"]},
        tmp_path / "served", workspace=ws)
    assert result["paper_id"] == receipt["work_id"] and result["scientifically_accepted"] is False


def test_refuses_missing_frontend_manifest_seam(tmp_path: Path, monkeypatch) -> None:
    import mosaic.public_serving as serving
    ws, capture, snapshot = fixture(tmp_path)
    frontend = tmp_path / "frontend"; frontend.mkdir()
    (frontend / "index.html").write_text("<html><body>missing seam</body></html>", encoding="utf-8")
    (frontend / "workspace.html").write_text("<html>__PUBLIC_PAPER_COUNT__</html>", encoding="utf-8")
    monkeypatch.setattr(serving, "FRONTEND_ROOT", frontend)
    with pytest.raises(PublicServingRefusal, match="manifest_seam_missing"):
        materialize_public(capture, snapshot, tmp_path / "seam", workspace=ws)


def test_frontend_link_refused_before_reading_target(tmp_path: Path, monkeypatch) -> None:
    import mosaic.public_serving as serving
    ws, capture, snapshot = fixture(tmp_path)
    frontend = tmp_path / "frontend"; frontend.mkdir()
    outside = tmp_path / "outside.html"; outside.write_text("outside", encoding="utf-8")
    try:
        (frontend / "index.html").symlink_to(outside)
    except OSError:
        pytest.skip("file symlinks unavailable on this Windows test host")
    monkeypatch.setattr(serving, "FRONTEND_ROOT", frontend)
    with pytest.raises(PublicServingRefusal, match="frontend_link_refused"):
        materialize_public(capture, snapshot, tmp_path / "linked", workspace=ws)


def test_linked_bundle_static_member_refused_at_startup(tmp_path: Path) -> None:
    ws, capture, snapshot = fixture(tmp_path)
    result = materialize_public(capture, snapshot, tmp_path / "run", workspace=ws)
    static = tmp_path / "run" / "static"
    original = static / "index.html"
    outside = tmp_path / "outside.html"; outside.write_bytes(original.read_bytes())
    original.unlink()
    try:
        original.symlink_to(outside)
    except OSError:
        pytest.skip("file symlinks unavailable on this Windows test host")
    with pytest.raises(PublicServingRefusal, match="frontend_link_refused"):
        serve_public(result["bundle_path"])


def test_serves_verified_static_snapshot_after_local_change(tmp_path: Path) -> None:
    ws, capture, snapshot = fixture(tmp_path)
    result = materialize_public(capture, snapshot, tmp_path / "run", workspace=ws)
    original = (tmp_path / "run" / "static" / "index.html").read_bytes()
    server = serve_public(result["bundle_path"], 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        (tmp_path / "run" / "static" / "index.html").write_bytes(b"tampered static bytes")
        actual = urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/index.html").read()
        assert actual == original
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=5)
