"""Deliberately fictional inputs for testing mechanics, never scientific accuracy."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from mosaic.workspace import Workspace


DEMO_VERSION = "mosaic-fictional-build-fixture/v1"
PAGES = (
    ((72, "Study 1"), (94, "Abstract"),
     (112, "Fictional abstract words are retained exactly for the local package fixture."),
     (148, "Methods"), (170, "Participants completed the printed measure in the first study."),
     (208, "Results"), (230, "The reported effect remained visible in the source prose.")),
    ((120, "Table 1. Descriptive statistics"), (140, "Measure        M        SD"),
     (158, "Outcome        3.2      1.1"), (250, "References"),
     (270, "Fictional reference words are retained exactly."), (330, "Appendix"),
     (350, "Fictional appendix words are retained exactly.")),
)


def demo_pdf_bytes() -> bytes:
    """Generate the same invented two-page source with the pinned PDF library."""
    import pymupdf
    with pymupdf.open() as doc:
        for lines in PAGES:
            page = doc.new_page(width=612, height=792)
            for y, text in lines:
                page.insert_text((72, y), text, fontsize=10)
        return doc.tobytes(no_new_id=True)


def prepare_demo(project_root: Path, *, workspace: Workspace) -> dict:
    """Create source bytes once, with no model outputs or bibliography invented."""
    raw = demo_pdf_bytes()
    target = project_root / "example-data/pdfs/fictional-build.pdf"
    workspace.write_once(target, raw)
    fixture = {"version": DEMO_VERSION, "fictional": True,
               "source_pdf_sha256": sha256(raw).hexdigest(), "path": "example-data/pdfs/fictional-build.pdf",
               "scientific_accuracy_evidence": False, "model_responses": "deliberate test fixtures, never model-generated"}
    workspace.write_canonical_json(project_root / "example-data/fixture.json", fixture)
    return fixture


def demo_config(*, run_root: str = "run/demo", port: int = 8798) -> dict:
    return {"version": "mosaic-public-build/v1", "profile": "demo",
            "sources": {"root": "example-data/pdfs", "files": ["fictional-build.pdf"], "permission": "fictional_fixture"},
            "run": {"root": run_root}, "runtime": {"kind": "recorded_replay"},
            "view": {"columns": ["title", "authors", "journal", "year"], "default_screen": "papers", "sort": "distinct_article_count_desc"},
            "serve": {"port": port}, "jev": {"enabled": False, "replay_only": True}}
