from __future__ import annotations

import pytest

from mosaic.public_citation_api import PublicCitationApiRefusal, citation_routes


def _routes():
    bibliography = {"paper_items": [
        {"paper_id": "public-a", "title": "Coordination", "authors": "A. Able", "journal": "Journal A", "year": 2024},
        {"paper_id": "public-b", "title": "Teams", "authors": "B. Baker", "journal": "Journal B", "year": 2025},
    ]}
    citations = {"items": [{"reference_id": "r-resolved", "source_paper_id": "public-a", "target_paper_id": "public-b", "status": "reviewed_edge"}],
                 "literal_references": [
                     {"reference_id": "r-unresolved", "source_paper_id": "public-a", "literal_reference": "Unknown (2020).", "witness": {"unit_id": "u2", "source_pdf_sha256": "a" * 64}},
                     {"reference_id": "r-resolved", "source_paper_id": "public-a", "literal_reference": "Baker (2025).", "witness": {"unit_id": "u1", "source_pdf_sha256": "a" * 64}},
                 ]}
    return citation_routes(bibliography, citations)


def test_health_search_and_reference_pagination_preserve_unresolved_rows() -> None:
    routes = _routes()
    assert routes["/api/citations/health"]({}) == {"status": "ready", "coverage_scope": "admitted local public sources only", "source_record_count": 2, "resolved_edge_count": 1, "unresolved_reference_count": 1, "total_admitted_works": 2, "provenance_status": "bound_v2"}
    assert routes["/api/citations/search"]({"q": "coord"})["items"][0]["work_id"] == "public-a"
    unresolved = routes["/api/citations/references"]({"q": "unknown"})
    assert unresolved["total"] == 1
    assert unresolved["items"][0]["resolution_status"] == "unresolved"
    assert "target" not in unresolved["items"][0]


def test_reference_search_uses_stable_twenty_row_pages() -> None:
    citations = {"items": [], "literal_references": [
        {"reference_id": f"r-{number:02d}", "source_paper_id": "public-a",
         "literal_reference": f"Printed reference {number:02d}", "witness": {}}
        for number in range(21)
    ]}
    routes = citation_routes({"paper_items": [{"paper_id": "public-a", "title": "A"}]}, citations)
    first = routes["/api/citations/references"]({})
    assert len(first["items"]) == 20 and first["next_cursor"] == 20
    second = routes["/api/citations/references"]({"cursor": "20"})
    assert [row["reference_id"] for row in second["items"]] == ["r-20"]
    assert second["next_cursor"] is None


def test_work_exposes_only_reviewed_target_and_incoming_link() -> None:
    routes = _routes()
    source = routes["/api/citations/work"]({"work_id": "public-a"})
    refs = source["references"]
    assert refs["total"] == 2
    resolved = next(row for row in refs["items"] if row["reference_id"] == "r-resolved")
    assert resolved["target"]["work_id"] == "public-b"
    assert source["bibliographic_coupling"] == source["co_citation"] == source["bridge_candidates"] == []
    target = routes["/api/citations/work"]({"work_id": "public-b"})
    assert target["incoming"]["items"] == [{"work_id": "public-a", "title": "Coordination", "authors": "A. Able", "journal": "Journal A", "year": 2024, "identity_status": "source_reviewed_complete", "reference_id": "r-resolved"}]


def test_capture_only_source_retains_unresolved_references_with_generic_work_label() -> None:
    routes = citation_routes(
        {"paper_items": [{"paper_id": "public-reviewed", "title": "Reviewed paper"}]},
        {"items": [], "literal_references": [{"reference_id": "r-capture", "source_paper_id": "public-capture-only",
            "literal_reference": "A literal reference from an unreviewed packet.", "witness": {}}]},
        source_ids=["public-reviewed", "public-capture-only"],
    )
    work = routes["/api/citations/work"]({"work_id": "public-capture-only"})
    assert work["work"] == {"work_id": "public-capture-only", "title": "Unreviewed source packet",
                             "identity_status": "bibliographic_identity_unresolved"}
    assert work["references"]["items"][0]["resolution_status"] == "unresolved"
    assert routes["/api/citations/health"]({})["total_admitted_works"] == 2


def test_refuses_unreviewed_or_missing_edge_targets_and_bad_cursor() -> None:
    with pytest.raises(PublicCitationApiRefusal, match="reviewed_edge_invalid"):
        citation_routes({"paper_items": [{"paper_id": "a"}]}, {"items": [{"reference_id": "r", "source_paper_id": "a", "target_paper_id": "unknown", "status": "reviewed_edge"}], "literal_references": []})
    with pytest.raises(PublicCitationApiRefusal, match="edge_source_mismatch"):
        citation_routes({"paper_items": [{"paper_id": "a"}, {"paper_id": "b"}]}, {"items": [{"reference_id": "r", "source_paper_id": "a", "target_paper_id": "b", "status": "reviewed_edge"}], "literal_references": [{"reference_id": "r", "source_paper_id": "b", "literal_reference": "x"}]})
    with pytest.raises(PublicCitationApiRefusal, match="literal_reference_invalid"):
        citation_routes({"paper_items": []}, {"items": [], "literal_references": [{"reference_id": "r", "source_paper_id": "public-not-rostered", "literal_reference": "x"}]}, source_ids=["public-other"])
    with pytest.raises(PublicCitationApiRefusal, match="reviewed_edge_invalid"):
        citation_routes({"paper_items": [{"paper_id": "public-reviewed"}]}, {"items": [{"reference_id": "r", "source_paper_id": "public-reviewed", "target_paper_id": "public-capture-only", "status": "reviewed_edge"}], "literal_references": [{"reference_id": "r", "source_paper_id": "public-reviewed", "literal_reference": "x"}]}, source_ids=["public-capture-only"])
    routes = _routes()
    with pytest.raises(ValueError, match="invalid_citation_cursor"):
        routes["/api/citations/search"]({"cursor": "-1"})
    with pytest.raises(ValueError, match="unknown_citation_work"):
        routes["/api/citations/work"]({"work_id": "not-public"})
