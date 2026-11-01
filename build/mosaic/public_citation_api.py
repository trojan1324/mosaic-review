"""Public citation routes for reviewed public citation discovery.

The adapter intentionally builds its small graph only from the materialized
public bibliography and the explicitly reviewed edges.  Literal references
remain visible even when they have no reviewed target; no search index or
private corpus data is consulted.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Mapping


class PublicCitationApiRefusal(ValueError):
    """The supplied public citation products cannot safely form a route API."""


_PAGE_SIZE = 20


def _rows(value: object, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping) or not isinstance(value.get(field, []), list):
        raise PublicCitationApiRefusal("public_citation_payload_invalid")
    rows = value.get(field, [])
    if not all(isinstance(row, Mapping) for row in rows):
        raise PublicCitationApiRefusal("public_citation_row_invalid")
    return [dict(row) for row in rows]


def _cursor(params: Mapping[str, str]) -> int:
    raw = params.get("cursor", "0")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_citation_cursor") from exc
    if value < 0 or str(value) != str(raw):
        raise ValueError("invalid_citation_cursor")
    return value


def _page(rows: list[dict[str, Any]], params: Mapping[str, str]) -> dict[str, Any]:
    start = _cursor(params)
    end = start + _PAGE_SIZE
    return {"items": rows[start:end], "total": len(rows),
            "next_cursor": end if end < len(rows) else None}


def citation_routes(bibliography: Mapping[str, Any], citations: Mapping[str, Any], *,
                    source_ids: list[str] | None = None) -> dict[str, Callable[[Mapping[str, str]], dict[str, Any]]]:
    """Return pure callables for ``/api/citations/*`` legacy frontend routes.

    ``bibliography.paper_items`` supplies the only works that can be named as
    graph targets.  A target appears only where a matching ``citations.items``
    record identifies that reference as a reviewed edge.  The full
    ``literal_references`` list is retained independently.
    """
    paper_rows = _rows(bibliography, "paper_items")
    references = _rows(citations, "literal_references")
    edge_rows = _rows(citations, "items")
    works: dict[str, dict[str, Any]] = {}
    for row in paper_rows:
        work_id = row.get("paper_id")
        if not isinstance(work_id, str) or not work_id or work_id in works:
            raise PublicCitationApiRefusal("public_citation_work_id_invalid")
        works[work_id] = {"work_id": work_id, "title": row.get("title", ""),
                          "authors": row.get("authors", ""), "journal": row.get("journal", ""),
                          "year": row.get("year"), "identity_status": "source_reviewed_complete"}
    reviewed_identities = set(works)
    for source_id in source_ids or []:
        if not isinstance(source_id, str) or not source_id:
            raise PublicCitationApiRefusal("public_citation_source_id_invalid")
        works.setdefault(source_id, {"work_id": source_id, "title": "Unreviewed source packet",
                                    "identity_status": "bibliographic_identity_unresolved"})
    reviewed: dict[str, tuple[str, str]] = {}
    for edge in edge_rows:
        reference_id, source, target = edge.get("reference_id"), edge.get("source_paper_id"), edge.get("target_paper_id")
        if (edge.get("status") != "reviewed_edge" or not isinstance(reference_id, str)
                or not isinstance(source, str) or not isinstance(target, str)
                or source not in works or target not in reviewed_identities or reference_id in reviewed):
            raise PublicCitationApiRefusal("public_citation_reviewed_edge_invalid")
        reviewed[reference_id] = (source, target)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_references: list[dict[str, Any]] = []
    seen_references: set[str] = set()
    for raw in references:
        ref_id, source, literal = raw.get("reference_id"), raw.get("source_paper_id"), raw.get("literal_reference")
        if (not isinstance(ref_id, str) or not ref_id or ref_id in seen_references
                or not isinstance(source, str) or source not in works or not isinstance(literal, str)):
            raise PublicCitationApiRefusal("public_citation_literal_reference_invalid")
        seen_references.add(ref_id)
        edge = reviewed.get(ref_id)
        if edge is not None and edge[0] != source:
            raise PublicCitationApiRefusal("public_citation_edge_source_mismatch")
        target_id = edge[1] if edge is not None else None
        witness = raw.get("witness") if isinstance(raw.get("witness"), Mapping) else {}
        item = {"reference_id": ref_id, "literal_reference": literal,
                "resolution_status": "reviewed_edge" if target_id else "unresolved",
                "provenance": {"kind": "printed_source_literal", "unit_id": witness.get("unit_id"),
                               "file_sha256": witness.get("source_pdf_sha256")}}
        if target_id:
            item["target"] = dict(works[target_id])
            incoming[target_id].append({**works[source], "reference_id": ref_id})
        by_source[source].append(item)
        all_references.append({**item, "citing_work": dict(works[source])})
    if set(reviewed) - seen_references:
        raise PublicCitationApiRefusal("public_citation_edge_reference_missing")
    targets_by_source: dict[str, set[str]] = defaultdict(set)
    citers_by_target: dict[str, set[str]] = defaultdict(set)
    for source, target in reviewed.values():
        targets_by_source[source].add(target)
        citers_by_target[target].add(source)

    def peer_rows(work_id: str) -> list[dict[str, Any]]:
        mine = targets_by_source.get(work_id, set())
        scored = []
        for peer_id, targets in targets_by_source.items():
            if peer_id == work_id:
                continue
            shared = sorted(mine & targets)
            if shared:
                scored.append((peer_id, shared))
        scored.sort(key=lambda item: (-len(item[1]), item[0]))
        return [{**works[peer_id], "shared_resolved_reference_count": len(shared),
                 "shared_resolved_reference_total": len(shared),
                 "shared_resolved_references": [dict(works[target]) for target in shared[:50]],
                 "shared_resolved_reference_cap": 50}
                for peer_id, shared in scored[:20]]

    def co_citation_rows(work_id: str) -> list[dict[str, Any]]:
        scored: dict[str, set[str]] = defaultdict(set)
        for citer in citers_by_target.get(work_id, set()):
            for target in targets_by_source.get(citer, set()):
                if target != work_id:
                    scored[target].add(citer)
        ordered = sorted(scored.items(), key=lambda item: (-len(item[1]), item[0]))[:20]
        return [{**works[target], "shared_citing_work_count": len(citers),
                 "shared_citing_work_ids": sorted(citers), "count_is_lower_bound": False,
                 "reason": "co_cited_by_same_reviewed_local_sources"}
                for target, citers in ordered]

    for rows in by_source.values(): rows.sort(key=lambda row: row["reference_id"])
    for rows in incoming.values(): rows.sort(key=lambda row: row["reference_id"])
    all_references.sort(key=lambda row: row["reference_id"])

    def health(_: Mapping[str, str]) -> dict[str, Any]:
        resolved = len(reviewed)
        return {"status": "ready", "coverage_scope": "admitted local public sources only",
                "source_record_count": len(all_references), "resolved_edge_count": resolved,
                "unresolved_reference_count": len(all_references) - resolved,
                "total_admitted_works": len(works), "provenance_status": "bound_v2"}

    def search(params: Mapping[str, str]) -> dict[str, Any]:
        query = params.get("q", "").casefold().strip()
        rows = [work for work in works.values() if query in " ".join(str(work.get(key, "")) for key in ("title", "authors", "journal", "year")).casefold()]
        rows.sort(key=lambda row: (str(row.get("title", "")).casefold(), row["work_id"]))
        return _page(rows, params)

    def reference_search(params: Mapping[str, str]) -> dict[str, Any]:
        query = params.get("q", "").casefold().strip()
        return _page([row for row in all_references if query in row["literal_reference"].casefold()], params)

    def work(params: Mapping[str, str]) -> dict[str, Any]:
        work_id = params.get("work_id")
        if not isinstance(work_id, str) or work_id not in works:
            raise ValueError("unknown_citation_work")
        coupling = peer_rows(work_id)
        result = {"work": dict(works[work_id]), "references": _page(by_source.get(work_id, []), params),
                  "incoming": _page(incoming.get(work_id, []), params),
                  "bibliographic_coupling": coupling, "co_citation": co_citation_rows(work_id),
                  "bridge_candidates": [{**row, "recommendation_only": True,
                                         "reason": "shared_reviewed_references"} for row in coupling],
                  "co_citation_scan_truncated": False}
        return result

    return {"/api/citations/health": health, "/api/citations/search": search,
            "/api/citations/references": reference_search, "/api/citations/work": work}
