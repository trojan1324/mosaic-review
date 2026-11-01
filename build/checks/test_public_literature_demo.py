"""The literature assistant fixture needs clean identities without relaxing admission."""
from __future__ import annotations

import json
import shutil
import subprocess
import re
from pathlib import Path
import threading
from urllib.request import urlopen

import pytest

from mosaic.public_literature_demo import build_literature_demo
from mosaic.public_serving import serve_public
from mosaic.workspace import Workspace


pytestmark = pytest.mark.evidence_mock


def test_two_fictional_sources_have_reviewed_display_metadata_and_candidate_reports(tmp_path: Path):
    project = tmp_path / "public"; workspace = Workspace(repo_root=project, temp_root=tmp_path / "temp")
    result = build_literature_demo(project, workspace=workspace)
    assert result["paper_count"] == result["admitted_identity_count"] == result["apa_count"] == 2
    assert result["candidate_reports_manufactured"] and not result["real_model_inference_run"]
    html=(Path(result['bundle_path']).parent/'static/measures.html').read_text(encoding='utf-8')
    payload=json.loads(re.search(r'<script id="measure-data" type="application/json">(.*?)</script>',html,re.S)[1])
    assert len(payload['versions']) == 2
    assert all(len(row['items']) == row['selected_item_count'] == 2 and row['reported_item_count'] is None
               for row in payload['versions'])
    report = json.loads((project / "run/literature-demo/literature-demo-report.json").read_bytes())
    assert "never human review" in report["fixture_review_provenance"]
    server = serve_public(result["bundle_path"], port=0); thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        papers = json.loads(urlopen(base + "/api/display/v1/bibliography?q=Fictional").read())
        assert papers["total"] == 2
        assert {item["title_verbatim"] for item in papers["items"]} == {
            "Coordination Routines in Fictional Teams", "Shared Reflection in Fictional Groups"}
        apa = json.loads(urlopen(base + "/api/display/v1/apa").read())
        assert apa["total"] == 2
        citations = {item["citation_apa"] for item in apa["items"]}
        assert any("Coordination routines in fictional teams." in citation for citation in citations)
        assert any("Shared reflection in fictional groups." in citation for citation in citations)
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=5)


def test_literature_demo_is_reusable_without_rewriting_artifacts(tmp_path: Path):
    project = tmp_path / "public"; workspace = Workspace(repo_root=project, temp_root=tmp_path / "temp")
    first = build_literature_demo(project, workspace=workspace)
    assert build_literature_demo(project, workspace=workspace) == first

def test_public_measure_string_item_renderer_preserves_text_and_matching_evidence(tmp_path: Path):
    project = tmp_path / "public"
    workspace = Workspace(repo_root=project, temp_root=tmp_path / "temp")
    result = build_literature_demo(project, workspace=workspace)
    node = shutil.which("node")
    if node is None:
        import pytest
        pytest.skip("Node.js is required for the actual renderer contract check")
    page = Path(result["bundle_path"]).parent / "static" / "measures.html"
    runner = Path(__file__).with_name("public_measure_renderer_contract_test.js")
    completed = subprocess.run([node, str(runner), str(page)], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "passed": True, "fictional_items_visible": True, "locators_retained": True,
        "copy_download_text": True, "legacy_object_rows": True, "mismatched_evidence_unbound": True,
    }
