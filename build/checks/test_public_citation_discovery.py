from __future__ import annotations

import importlib.util
from pathlib import Path


from mosaic import public_citation_api as module


def _routes():
    works = {"paper_items": [
        {"paper_id": "a", "title": "Alpha"}, {"paper_id": "b", "title": "Beta"},
        {"paper_id": "c", "title": "Gamma"}, {"paper_id": "d", "title": "Delta"},
        {"paper_id": "e", "title": "Epsilon"},
    ]}
    reviewed = [("a-d", "a", "d"), ("a-e", "a", "e"), ("b-d", "b", "d"),
                ("b-e", "b", "e"), ("c-d", "c", "d")]
    refs = [{"reference_id": rid, "source_paper_id": source,
             "literal_reference": f"{source} cites {target}", "witness": {}}
            for rid, source, target in reviewed]
    # This literal string names a local-looking target but does not carry a
    # reviewed edge, so it must not alter any graph score.
    refs.append({"reference_id": "a-unresolved", "source_paper_id": "a",
                 "literal_reference": "Unreviewed mention of Delta", "target_paper_id": "d", "witness": {}})
    edges = [{"reference_id": rid, "source_paper_id": source, "target_paper_id": target,
              "status": "reviewed_edge"} for rid, source, target in reviewed]
    return module.citation_routes(works, {"items": edges, "literal_references": refs})


def test_shared_references_are_ranked_and_explained_from_reviewed_edges_only():
    route = _routes()["/api/citations/work"]
    result = route({"work_id": "a"})
    peers = result["bibliographic_coupling"]
    assert [(row["work_id"], row["shared_resolved_reference_count"]) for row in peers] == [("b", 2), ("c", 1)]
    assert [row["work_id"] for row in peers[0]["shared_resolved_references"]] == ["d", "e"]
    assert peers[0]["shared_resolved_reference_total"] == 2
    assert all(row["work_id"] != "d" for row in peers)  # targets are not mistaken for citing peers
    assert result["references"]["total"] == 3  # unresolved raw row survives


def test_co_citation_uses_distinct_reviewed_citers_and_bridges_are_advisory():
    route = _routes()["/api/citations/work"]
    result = route({"work_id": "d"})
    assert result["co_citation"] == [{"work_id": "e", "title": "Epsilon", "authors": "", "journal": "", "year": None,
                                        "identity_status": "source_reviewed_complete", "shared_citing_work_count": 2,
                                        "shared_citing_work_ids": ["a", "b"], "count_is_lower_bound": False,
                                        "reason": "co_cited_by_same_reviewed_local_sources"}]
    bridges = route({"work_id": "a"})["bridge_candidates"]
    assert [(row["work_id"], row["recommendation_only"], row["reason"]) for row in bridges] == [
        ("b", True, "shared_reviewed_references"), ("c", True, "shared_reviewed_references")]
    assert {row["work_id"] for row in bridges}.isdisjoint({"d", "e"})


def test_no_reviewed_edges_means_no_discovery_even_when_literal_rows_name_targets():
    routes = module.citation_routes({"paper_items": [{"paper_id": "a", "title": "Alpha"}, {"paper_id": "d", "title": "Delta"}]},
        {"items": [], "literal_references": [{"reference_id": "raw", "source_paper_id": "a", "literal_reference": "Delta", "target_paper_id": "d"}]})
    result = routes["/api/citations/work"]({"work_id": "a"})
    assert result["references"]["items"][0]["resolution_status"] == "unresolved"
    assert result["bibliographic_coupling"] == result["co_citation"] == result["bridge_candidates"] == []
