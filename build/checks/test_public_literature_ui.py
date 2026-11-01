from pathlib import Path
import shutil
import subprocess
import json

ROOT = Path(__file__).parents[1]
FRONTEND = ROOT / "public_frontend"

def test_public_literature_page_has_explicit_preview_and_generation_boundaries():
    page = (FRONTEND / "assistant.html").read_text(encoding="utf-8")
    assert 'id="literature-preview"' in page
    assert 'id="literature-generate" disabled' in page
    assert "Source preview stays local" in page
    assert "linked citations" in page

def test_public_literature_script_uses_evidence_contract_and_parses():
    script = (FRONTEND / "assets" / "literature-assistant.js")
    source = script.read_text(encoding="utf-8")
    for endpoint in ("/api/literature/status", "/api/literature/preview", "/api/literature/draft"):
        assert endpoint in source
    assert "enable_model:enableModel" in source
    assert "source_id===witness.source_id" in source
    assert "API key" in source
    assert "generation_configured" in source
    assert "previewFingerprint" in source
    assert "innerHTML" not in source
    node = shutil.which("node")
    if node:
        result = subprocess.run([node, "--check", str(script)], capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr


def test_public_literature_ui_consumes_a_fictional_response_without_markup_injection():
    node = shutil.which("node")
    if node is None:
        return
    runner = str(Path(__file__).with_name("public_literature_ui_runtime_test.js"))
    script = str((FRONTEND / "assets" / "literature-assistant.js"))
    for offline in (False, True):
        result = subprocess.run([node, runner, script, *( ["offline"] if offline else [])],
                                capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == {
            "passed": True, "preview_first": True, "unsafe_text_not_markup": True,
            "query_change_invalidates": True, "offline": offline,
        }


def test_public_principal_pages_offer_the_literature_workspace():
    import re
    for name in ('workspace.html', 'index.html', 'measures.html', 'studies.html', 'citation-map.html'):
        page = (FRONTEND / name).read_text(encoding='utf-8')
        nav = re.search(r'<nav\b[^>]*>(.*?)</nav>', page, re.S).group(1)
        assert re.search(r'<a\b[^>]*href="assistant.html"[^>]*>Ask MOSAIC</a>', nav), name
    home = (FRONTEND / 'workspace.html').read_text(encoding='utf-8')
    assert 'Preview evidence, then ask for a linked literature draft' in home


def test_public_literature_ui_deduplicates_visible_citations_by_paper_id(tmp_path):
    node = shutil.which("node")
    if node is None:
        return
    papers = [("S1", "p2", "Singh, 2025"), ("S2", "p1", "Price, 2024"), ("S3", "p1", "Price, 2024"),
              ("S4", "p2", "Singh, 2025"), ("S5", "p1", "Price, 2024"), ("S6", "p2", "Singh, 2025")]
    response = {"status": "ok", "retrieved_papers": 2, "model": "fixture", "draft": {"title": "Fixture", "gaps": [],
                "sections": [{"heading": "Comparison", "paragraphs": [{"text": "Comparison.", "kind": "source_summary",
                "citations": ["S5", "S6", "S1", "S2"], "support": [{"source_id": f"u{i}", "quote": f"support {i}"} for i in range(1, 5)]}]}]},
                "sources": [{"citation_id": sid, "source_id": f"u{i + 1}", "paper_id": paper, "parenthetical": label,
                "title": paper, "citation": label, "quote": f"quote {i}", "page": 1, "article_url": f"studies.html?paper={paper}"}
                for i, (sid, paper, label) in enumerate(papers)]}
    replay = tmp_path / "fictional-response.json"
    replay.write_text(json.dumps(response), encoding="utf-8")
    runner = Path(__file__).with_name("public_literature_ui_runtime_test.js")
    script = (FRONTEND / "assets" / "literature-assistant.js")
    result = subprocess.run([node, str(runner), str(script), "online", str(replay)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
