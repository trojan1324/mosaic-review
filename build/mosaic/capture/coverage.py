"""Coverage denominators and page reconciliation (FINAL v2 G8; audit M-01, M-02).

E's native capture never reconciled `page_count` against the pages present
in its blocks, so an image-only page vanished without a trace (M-02). Here the
page count is taken when the source is registered, independently of capture,
and every physical page must end up with exactly one page object and exactly
one page disposition. The manifest's coverage block uses the registered page
counts as its denominator, so a missing page shows up as a gap instead of a
smaller total.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

PDF = "application/pdf"


def reconcile_pages(objects: dict[str, dict[str, Any]]) -> list[str]:
    """Return a problem per PDF page that lacks a page object or a single disposition."""
    problems: list[str] = []
    indices: dict[str, list[int]] = {}
    page_ids: dict[str, str] = {}
    disposed: Counter[str] = Counter()
    for oid, body in objects.items():
        if body["type"] == "page":
            indices.setdefault(body["source_id"], []).append(body["physical_index"])
            page_ids[oid] = body["source_id"]
        elif body["type"] == "disposition" and body["subject_kind"] == "page":
            disposed[body["subject_ref"]] += 1
    for oid, body in objects.items():
        if body["type"] != "source" or body["artifact"]["media_type"] != PDF:
            continue
        count = body["artifact"].get("page_count")
        if count is None:
            problems.append(f"{oid}: PDF source registered without a page count")
            continue
        present = sorted(indices.get(oid, []))
        if present != list(range(count)):
            missing = sorted(set(range(count)) - set(present))
            extra = sorted(set(present) - set(range(count)))
            dupes = sorted(i for i, n in Counter(present).items() if n > 1)
            problems.append(f"{oid}: {count} registered pages; missing {missing}, unexpected {extra}, duplicated {dupes}")
    for pid in page_ids:
        if disposed[pid] != 1:
            problems.append(f"{pid}: {disposed[pid]} page dispositions, expected exactly 1")
    for ref in disposed:
        if ref not in page_ids:
            problems.append(f"page disposition refers to {ref}, which is not a page in this package")
    return problems


def coverage_summary(objects: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Manifest coverage block: denominators from registration, counts by subject kind and disposition."""
    pages_total = sum(body["artifact"].get("page_count") or 0 for body in objects.values()
                      if body["type"] == "source" and body["artifact"]["media_type"] == PDF)
    counts = Counter(f"{body['subject_kind']}:{body['disposition']}" for body in objects.values()
                     if body["type"] == "disposition")
    paragraphs = sum(len(body["blocks"]) for body in objects.values()
                     if body["type"] == "text-region" and body["derivation"]["kind"] == "docx_body")
    return {
        "pages_total": pages_total,
        "dispositions": dict(sorted(counts.items())),
        "docx_paragraphs_total": paragraphs,
        "denominator_note": ("pages_total is the sum of page counts taken at source registration, before capture; "
                             "every disposition, including failed, blocked, pending, stalled and quarantined ones, "
                             "is counted (keys are <subject_kind>:<disposition>); docx_paragraphs_total counts every "
                             "w:p in word/document.xml, empty ones included (G8)"),
    }
