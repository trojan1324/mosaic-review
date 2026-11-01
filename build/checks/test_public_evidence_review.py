"""Explicit evidence review helpers preserve literal source selections."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mosaic.public_evidence_review import review_citation_edge, review_measure
from mosaic.public_literature_demo import build_literature_demo
from mosaic.workspace import Workspace


pytestmark = pytest.mark.evidence_mock


def _selector(reader: dict, text: str, *, lane: str | None = None) -> dict:
    for unit in reader["units"]:
        if lane is not None and unit["lane"] != lane:
            continue
        at = unit["quote"].find(text)
        if at >= 0:
            return {"value": text, "unit_id": unit["unit_id"], "start": unit["start"] + at,
                    "end": unit["start"] + at + len(text)}
    raise AssertionError(text)


def test_measure_and_reference_helpers_require_actual_literal_capture_selections(tmp_path: Path):
    project = tmp_path / "public"; workspace = Workspace(repo_root=project, temp_root=tmp_path / "temp")
    build_literature_demo(project, workspace=workspace)
    roots = sorted((project / "run/literature-demo/captured").iterdir())
    root = next(path for path in roots if "Team Coordination Scale" in (path / "native-reader.json").read_text(encoding="utf-8"))
    receipt = json.loads((root / "receipt.json").read_bytes())
    reader = json.loads((root / "native-reader.json").read_bytes())
    actor = {"reviewer_type": "caller_attested_source_review", "version": "test/v1"}
    measure = review_measure(str(root), label=_selector(reader, "Team Coordination Scale"),
                             items=[_selector(reader, "Team members coordinate their timing.")],
                             attestation=actor, workspace=workspace)
    assert measure["item_texts"][0]["text"] == "Team members coordinate their timing."
    target = next(path.name for path in roots if path != root)
    edge = review_citation_edge(str(root), literal_reference=_selector(reader, "Singh, O. (2025). Shared reflection in fictional groups. Journal of Fictional Organization Studies, 13, 15-29.", lane="references"),
                                source_paper_id=receipt["work_id"], target_paper_id=target,
                                attestation=actor, workspace=workspace)
    assert edge["edge_status"] == "reviewed_edge"
