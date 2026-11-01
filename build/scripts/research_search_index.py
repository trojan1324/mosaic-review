"""Immutable local full-text search over source passages and evidence records.

The index is a derived, read-only discovery aid.  It preserves source quotes and
provenance, keeps title metadata distinct from passage hits, and never promotes a
model-selected record to an accepted scientific assertion.
"""
from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import mimetypes
import os
import re
import sqlite3
import statistics
import tempfile
import time
import urllib.parse
from pathlib import Path
from typing import Any

from mosaic.canonical import canonical_bytes
from mosaic.workspace import ImmutableArtifactError, Workspace


VERSION = "research-search-index-v4"
MAX_PAGE_SIZE = 100
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_QUOTED = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"|([^\s]+)')


def normalize_search(text: str) -> str:
    """Normalize only the indexed representation, never displayed evidence."""
    if not isinstance(text, str):
        raise ValueError("search_text_must_be_string")
    text = re.sub(r"(?<=[A-Za-z])[-\u00ad]\s*\n\s*(?=[a-z])", "", text)
    return " ".join(text.split())


def safe_fts_query(query: str) -> str | None:
    """Convert words and quoted phrases into literal FTS terms, not FTS syntax."""
    if not isinstance(query, str):
        raise ValueError("query_must_be_string")
    clauses = []
    for quoted, raw in _QUOTED.findall(query):
        words = _WORD.findall(normalize_search(quoted or raw))
        if not words:
            continue
        # One user-quoted group becomes a phrase.  Every other group is an ANDed
        # literal term; no raw operator, prefix, column, or NEAR syntax survives.
        clauses.append('"' + " ".join(words).replace('"', '""') + '"')
    return " AND ".join(clauses) if clauses else None


def _require_list(value: Any, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(name + "_must_be_object_list")
    return value


def _paper_rows(data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    records = _require_list(data.get("records"), "records")
    sources = _require_list(data.get("source_passages"), "source_passages")
    papers: dict[str, dict[str, Any]] = {}
    passages: list[dict[str, Any]] = []
    for source in sources:
        paper_id = source.get("paper_id")
        if not isinstance(paper_id, str) or not paper_id:
            raise ValueError("source_paper_id_missing")
        if paper_id in papers:
            raise ValueError("duplicate_source_paper")
        paper = {key: source.get(key) for key in ("paper_id", "title", "citation",
                                                   "source_pdf_sha256", "source_units_sha256")}
        paper["source_inventory_status"] = "admitted_source_packet"
        if not isinstance(paper["title"], str) or not paper["title"]:
            raise ValueError("source_title_missing")
        papers[paper_id] = paper
        seen_units: set[str] = set()
        for ordinal, item in enumerate(_require_list(source.get("passages"), "source_passages.passages")):
            unit_id, quote, page = item.get("unit_id"), item.get("text"), item.get("page")
            if not isinstance(unit_id, str) or not isinstance(quote, str) or not isinstance(page, int):
                raise ValueError("invalid_source_passage")
            if unit_id in seen_units:
                raise ValueError("duplicate_source_unit")
            seen_units.add(unit_id)
            passages.append({"id": f"passage:{paper_id}:{unit_id}", "paper_id": paper_id,
                             "unit_id": unit_id, "page": page, "quote": quote,
                             "ordinal": ordinal, **paper})
    record_ids: set[str] = set()
    context_passages_seen: set[tuple[str, str, str, int]] = set()
    indexed_records = []
    for record_index, record in enumerate(records):
        if record.get("scientifically_accepted") is not False:
            raise ValueError("unverified_record_required")
        paper_id, finding_id = record.get("paper_id"), record.get("finding_id")
        if not isinstance(paper_id, str) or not isinstance(finding_id, str) or not finding_id:
            raise ValueError("record_identity_missing")
        if paper_id not in papers:
            title = record.get("title")
            if not isinstance(title, str) or not title:
                raise ValueError("record_only_paper_title_missing")
            papers[paper_id] = {"paper_id": paper_id, "title": title, "citation": record.get("citation"),
                                "source_pdf_sha256": record.get("source_pdf_sha256"),
                                "source_units_sha256": record.get("source_units_sha256"),
                                "source_inventory_status": "record_context_only"}
        record_id = f"record:{paper_id}:{finding_id}"
        if record_id in record_ids:
            raise ValueError("duplicate_record_identity")
        record_ids.add(record_id)
        indexed_records.append({"id": record_id, "record_index": record_index,
                                "paper_id": paper_id, "record": record})
        # Some supplied reader records precede their full admitted-source packet.
        # Index their already-present exact evidence as context-only rather than
        # inventing a whole-paper source or dropping the record from search.
        if papers[paper_id]["source_inventory_status"] == "record_context_only":
            pages = record.get("selected_evidence_pages")
            page = pages[0] if isinstance(pages, list) and len(pages) == 1 and isinstance(pages[0], int) else -1
            groups = [("selected", record.get("selected_evidence", [])),
                      ("context", record.get("source_passages_context_only", []))]
            for qualification_index, qualification in enumerate(record.get("paper_qualifications", [])):
                if isinstance(qualification, dict):
                    groups.append((f"qualification-{qualification_index}", qualification.get("evidence", [])))
            for group, items in groups:
                if not isinstance(items, list):
                    raise ValueError("record_evidence_must_be_list")
                for ordinal, item in enumerate(items):
                    if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                        raise ValueError("record_evidence_text_missing")
                    item_page = item.get("page")
                    if not isinstance(item_page, int) or item_page <= 0:
                        item_page = page if group == "selected" else -1
                    unit_id = str(item.get("unit_id", item.get("id", ordinal)))
                    identity = (paper_id, unit_id, item["text"], item_page)
                    if identity in context_passages_seen:
                        continue
                    context_passages_seen.add(identity)
                    passages.append({"id": f"passage:{paper_id}:record-{record_index}:{group}:{ordinal}",
                                     "paper_id": paper_id, "unit_id": unit_id,
                                     "page": item_page, "quote": item["text"], "ordinal": ordinal,
                                     **papers[paper_id]})
    catalog_ids = set()
    for row in _require_list(data.get("catalog", []), "catalog"):
        paper_id, title = row.get("paper_id"), row.get("title")
        if not isinstance(paper_id, str) or not paper_id or not isinstance(title, str) or not title.strip():
            raise ValueError("catalog_identity_missing")
        if paper_id in catalog_ids:
            raise ValueError("duplicate_catalog_identity")
        catalog_ids.add(paper_id)
        if paper_id in papers:
            if papers[paper_id]["title"] != title:
                raise ValueError("catalog_source_identity_conflict")
            continue
        papers[paper_id] = {"paper_id": paper_id, "title": title, "citation": row.get("citation"),
                            "source_pdf_sha256": None, "source_units_sha256": None,
                            "source_inventory_status": "metadata_only"}
    if not papers:
        raise ValueError("empty_collection")
    return list(papers.values()), passages, indexed_records


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        PRAGMA journal_mode=DELETE;
        PRAGMA synchronous=FULL;
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE papers (
            paper_id TEXT PRIMARY KEY, title TEXT NOT NULL, citation TEXT,
            source_pdf_sha256 TEXT, source_units_sha256 TEXT, source_inventory_status TEXT NOT NULL
        );
        CREATE TABLE passages (
            id TEXT PRIMARY KEY, paper_id TEXT NOT NULL, unit_id TEXT NOT NULL,
            page INTEGER NOT NULL, ordinal INTEGER NOT NULL, quote TEXT NOT NULL,
            source_pdf_sha256 TEXT, source_units_sha256 TEXT,
            FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
        );
        CREATE TABLE records (id TEXT PRIMARY KEY, record_index INTEGER NOT NULL UNIQUE, paper_id TEXT NOT NULL, body_json TEXT NOT NULL,
            FOREIGN KEY (paper_id) REFERENCES papers(paper_id));
        CREATE VIRTUAL TABLE search_fts USING fts5(
            stable_id UNINDEXED, paper_id UNINDEXED, hit_kind UNINDEXED, normalized_text,
            tokenize='unicode61'
        );
        CREATE VIRTUAL TABLE title_fts USING fts5(
            stable_id UNINDEXED, paper_id UNINDEXED, normalized_text,
            tokenize='unicode61'
        );
        CREATE INDEX passages_by_paper_page ON passages(paper_id, page, ordinal);
        CREATE INDEX records_by_paper ON records(paper_id);
        CREATE TABLE bibliography (
            paper_id TEXT PRIMARY KEY REFERENCES papers(paper_id),
            year INTEGER, journal TEXT, doi TEXT, authors_json TEXT NOT NULL
        );
        CREATE INDEX bibliography_year ON bibliography(year);
        CREATE INDEX bibliography_journal ON bibliography(journal);
    """)


def build_index(data: dict[str, Any], output: Path | str, *, workspace: Workspace | None = None) -> dict[str, Any]:
    """Build one immutable SQLite index and return its explicit coverage snapshot."""
    if not isinstance(data, dict):
        raise ValueError("input_must_be_object")
    ws = workspace or Workspace()
    target = ws.check(output)
    if target.exists():
        raise ImmutableArtifactError("derived_index_already_exists")
    papers, passages, records = _paper_rows(data)
    started = time.perf_counter()
    conn = ws.sqlite_connect(target)
    try:
        _create_schema(conn)
        conn.executemany("INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?)", [
            (p["paper_id"], p["title"], p.get("citation"), p.get("source_pdf_sha256"),
             p.get("source_units_sha256"), p["source_inventory_status"]) for p in papers])
        conn.executemany("INSERT INTO passages VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [
            (p["id"], p["paper_id"], p["unit_id"], p["page"], p["ordinal"], p["quote"],
             p.get("source_pdf_sha256"), p.get("source_units_sha256")) for p in passages])
        conn.executemany("INSERT INTO records VALUES (?, ?, ?, ?)", [
            (record["id"], record["record_index"], record["paper_id"],
             json.dumps(record["record"], ensure_ascii=False, separators=(",", ":"))) for record in records])
        metadata = {}
        for row in data.get("source_passages", []) + data.get("catalog", []):
            key = row["paper_id"]
            previous = metadata.get(key, {})
            if previous and previous["title"] != row["title"]:
                raise ValueError("catalog_source_identity_conflict")
            metadata[key] = {**previous, **row}
        bibliography = []
        for paper in papers:
            row = metadata.get(paper["paper_id"], {})
            year, journal, doi, authors = (row.get(k) for k in ("year", "journal", "doi", "authors"))
            authors = [] if authors is None else authors
            if year is not None and (type(year) is not int or not 1500 <= year <= 2200):
                raise ValueError("catalog_year_invalid")
            if any(v is not None and not isinstance(v, str) for v in (journal, doi)):
                raise ValueError("catalog_bibliography_invalid")
            if not isinstance(authors, list) or not all(isinstance(v, str) for v in authors):
                raise ValueError("catalog_authors_invalid")
            bibliography.append((paper["paper_id"], year, journal, doi, json.dumps(authors, ensure_ascii=False)))
        conn.executemany("INSERT INTO bibliography VALUES (?, ?, ?, ?, ?)", bibliography)
        fts_rows = []
        title_rows = []
        for paper in papers:
            row = metadata.get(paper["paper_id"], {})
            terms = " ".join([paper["title"], row.get("doi") or "", row.get("journal") or "",
                              *(row.get("authors") or [])])
            title_rows.append((f"title:{paper['paper_id']}", paper["paper_id"], normalize_search(terms)))
        fts_rows.extend((p["id"], p["paper_id"], "passage", normalize_search(p["quote"])) for p in passages)
        conn.executemany("INSERT INTO search_fts VALUES (?, ?, ?, ?)", fts_rows)
        conn.executemany("INSERT INTO title_fts VALUES (?, ?, ?)", title_rows)
        snapshot = {"version": VERSION, "input_sha256": hashlib.sha256(canonical_bytes(data)).hexdigest(),
                    "paper_count": len(papers), "source_passage_count": len(passages),
                    "record_count": len(records), "title_metadata_count": len(papers),
                    "admitted_source_paper_count": sum(p["source_inventory_status"] == "admitted_source_packet" for p in papers),
                    "record_context_only_paper_count": sum(p["source_inventory_status"] == "record_context_only" for p in papers),
                    "metadata_only_paper_count": sum(p["source_inventory_status"] == "metadata_only" for p in papers),
                    "papers_with_extraction_proposals": len({r["paper_id"] for r in records}),
                    "qualified_paper_count": 0,
                    "scientifically_accepted": False, "coverage_verified": False}
        snapshot["build_seconds"] = round(time.perf_counter() - started, 6)
        snapshot["snapshot_id"] = VERSION + ":" + snapshot["input_sha256"]
        conn.executemany("INSERT INTO metadata VALUES (?, ?)",
                         [(key, json.dumps(value, ensure_ascii=False, separators=(",", ":")))
                          for key, value in snapshot.items()])
        conn.commit()
    finally:
        conn.close()
    snapshot["sqlite_bytes"] = target.stat().st_size
    return snapshot


class ResearchSearchIndex:
    """Read-only API over an immutable built index.  It never creates WAL files."""
    def __init__(self, path: Path | str):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)

    def _connect(self) -> sqlite3.Connection:
        uri = self.path.resolve().as_uri() + "?mode=ro&immutable=1"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        return conn

    def snapshot(self) -> dict[str, Any]:
        with self._connect() as conn:
            result = {row["key"]: json.loads(row["value"])
                      for row in conn.execute("SELECT key, value FROM metadata")}
        result["sqlite_bytes"] = self.path.stat().st_size
        return result

    def papers(self, *, evidence_only: bool = False) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT papers.*, count(records.id) AS record_count FROM papers "
                                "LEFT JOIN records USING(paper_id) " +
                                ("WHERE source_inventory_status != 'metadata_only' " if evidence_only else "") +
                                "GROUP BY papers.paper_id ORDER BY title COLLATE NOCASE")
            return [dict(row) | {"scientifically_accepted": False} for row in rows]

    def catalog(self, query: str = "", *, page: int = 1, page_size: int = 30,
                year: str = "", journal: str = "", status: str = "") -> dict[str, Any]:
        """Bounded bibliography browsing; metadata never becomes extraction evidence."""
        from datetime import date
        from mosaic.catalog_quality import VERSION, metadata_issues
        review_year = date.today().year
        if type(page) is not int or type(page_size) is not int or page < 1 or not 1 <= page_size <= MAX_PAGE_SIZE:
            raise ValueError("invalid_pagination")
        if status not in ("", "metadata_only", "admitted_source_packet", "record_context_only", "proposals"):
            raise ValueError("invalid_processing_status")
        compiled = safe_fts_query(query)
        where, params = [], []
        if query.strip():
            if compiled:
                where.append("p.paper_id IN (SELECT paper_id FROM title_fts WHERE title_fts MATCH ?)")
                params.append(compiled)
            else:
                where.append("0")
        with self._connect() as conn:
            has_bib = conn.execute("SELECT 1 FROM sqlite_master WHERE name='bibliography'").fetchone()
            bib = "bibliography" if has_bib else "(SELECT paper_id, NULL year, NULL journal, NULL doi, '[]' authors_json FROM papers)"
            if year:
                if not re.fullmatch(r"\d{4}", year):
                    raise ValueError("invalid_year")
                where.append("b.year = ?"); params.append(int(year))
            if journal:
                where.append("b.journal = ?"); params.append(journal)
            if status == "proposals":
                where.append("EXISTS (SELECT 1 FROM records r WHERE r.paper_id=p.paper_id)")
            elif status:
                where.append("p.source_inventory_status = ?"); params.append(status)
            base = " FROM papers p LEFT JOIN " + bib + " b USING(paper_id)"
            clause = " WHERE " + " AND ".join(where) if where else ""
            total = conn.execute("SELECT count(*)" + base + clause, params).fetchone()[0]
            rows = conn.execute("SELECT p.*, b.year, b.journal, b.doi, b.authors_json, "
                                "(SELECT count(*) FROM passages s WHERE s.paper_id=p.paper_id) AS passage_count, "
                                "(SELECT count(*) FROM records r WHERE r.paper_id=p.paper_id) AS record_count" + base + clause +
                                " ORDER BY p.title COLLATE NOCASE, p.paper_id LIMIT ? OFFSET ?",
                                params + [page_size, (page - 1) * page_size]).fetchall()
            result = []
            for row in rows:
                item = dict(row); item["authors"] = json.loads(item.pop("authors_json"))
                item["metadata_issues"] = metadata_issues(item, as_of_year=review_year)
                item.update(scientifically_accepted=False, coverage_verified=False)
                result.append(item)
            years = [r[0] for r in conn.execute("SELECT DISTINCT year FROM " + bib + " WHERE year IS NOT NULL ORDER BY year DESC")]
            journals = [r[0] for r in conn.execute("SELECT DISTINCT journal FROM " + bib + " WHERE journal IS NOT NULL ORDER BY journal COLLATE NOCASE")]
        return {"papers": result, "total": total, "page": page, "page_size": page_size,
                "facets": {"years": years, "journals": journals}, "snapshot": self.snapshot(),
                "metadata_review_version": VERSION, "metadata_review_year": review_year,
                "metadata_source_verified": False}

    def record(self, stable_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            if stable_id.isdecimal():
                row = conn.execute("SELECT id, record_index, body_json FROM records WHERE record_index = ?",
                                   (int(stable_id),)).fetchone()
            else:
                row = conn.execute("SELECT id, record_index, body_json FROM records WHERE id = ?", (stable_id,)).fetchone()
        if row is None:
            return None
        return {"id": row["id"], "record_index": row["record_index"], "record": json.loads(row["body_json"]),
                "scientifically_accepted": False, "coverage_verified": False}

    def passage(self, stable_id: str) -> dict[str, Any] | None:
        """Fetch the immutable source passage used by a saved search result."""
        with self._connect() as conn:
            row = conn.execute("SELECT passages.*, papers.title, papers.citation, papers.source_inventory_status "
                               "FROM passages JOIN papers USING(paper_id) WHERE passages.id = ?", (stable_id,)).fetchone()
        return (self._passage_hit(row) | {"snapshot_id": self.snapshot()["snapshot_id"]}) if row else None

    def page_passages(self, paper: str, page: int, *, exclude: str = "",
                      limit: int = 5) -> dict[str, Any]:
        """Other indexed excerpts on a recorded page, never inferred neighbors.

        Packet ordinal is preserved for reproducibility. It does not establish
        reading order, continuous prose, or complete coverage of the PDF page.
        """
        if not isinstance(paper, str) or not paper or not isinstance(page, int) or page < 1:
            raise ValueError("recorded_paper_and_positive_page_required")
        if not isinstance(limit, int) or not 1 <= limit <= MAX_PAGE_SIZE:
            raise ValueError("invalid_page_excerpt_limit")
        with self._connect() as conn:
            if exclude:
                anchor = conn.execute("SELECT paper_id, page FROM passages WHERE id = ?", (exclude,)).fetchone()
                if not anchor or anchor["paper_id"] != paper or anchor["page"] != page:
                    raise ValueError("excluded_passage_not_on_requested_page")
            params = (paper, page, exclude)
            total = conn.execute("SELECT count(*) FROM passages WHERE paper_id = ? AND page = ? AND id != ?",
                                 params).fetchone()[0]
            rows = conn.execute(
                "SELECT passages.*, papers.title, papers.citation, papers.source_inventory_status "
                "FROM passages JOIN papers USING(paper_id) "
                "WHERE passages.paper_id = ? AND page = ? AND id != ? ORDER BY ordinal, id LIMIT ?",
                (*params, limit)).fetchall()
        return {"paper_id": paper, "page": page, "total": total,
                "passages": [self._passage_hit(row) for row in rows],
                "truncated": total > len(rows), "order": "source_packet_ordinal",
                "reading_order_verified": False, "coverage_verified": False,
                "snapshot_id": self.snapshot()["snapshot_id"]}

    @staticmethod
    def _passage_hit(passage) -> dict[str, Any]:
        return {"id": passage["id"], "kind": "source_passage", "paper_id": passage["paper_id"],
                "title": passage["title"], "citation": passage["citation"], "page": passage["page"],
                "unit_id": passage["unit_id"], "quote": passage["quote"],
                "source_pdf_sha256": passage["source_pdf_sha256"],
                "source_units_sha256": passage["source_units_sha256"],
                "source_inventory_status": passage["source_inventory_status"],
                "scientifically_accepted": False, "coverage_verified": False}

    def _matching_papers(self, conn: sqlite3.Connection, compiled: str, paper: str | None) -> tuple[int, list[dict[str, Any]]]:
        clause = " AND paper_id = ?" if paper else ""
        params = [compiled, paper, compiled, paper] if paper else [compiled, compiled]
        # A paper can match through its title OR a source passage. Restricting
        # this set to title matches silently loses passage-only paper results.
        ids = conn.execute(
            "WITH matched AS (SELECT paper_id FROM title_fts WHERE title_fts MATCH ?" + clause +
            " UNION SELECT paper_id FROM search_fts WHERE search_fts MATCH ?" + clause +
            ") SELECT paper_id, count(*) OVER () AS total FROM matched ORDER BY paper_id LIMIT ?",
            params + [MAX_PAGE_SIZE]).fetchall()
        output = []
        for item in ids[:MAX_PAGE_SIZE]:
            row = conn.execute("SELECT papers.*, count(records.id) AS record_count FROM papers "
                               "LEFT JOIN records USING(paper_id) WHERE papers.paper_id = ? GROUP BY papers.paper_id",
                               (item["paper_id"],)).fetchone()
            output.append(dict(row) | {"scientifically_accepted": False})
        return ids[0]["total"] if ids else 0, output

    def _records_for_papers(self, conn: sqlite3.Connection, paper_ids: list[str]) -> list[dict[str, Any]]:
        if not paper_ids:
            return []
        marks = ",".join("?" for _ in paper_ids)
        rows = conn.execute("SELECT id, record_index, body_json FROM records WHERE paper_id IN (" + marks +
                            ") ORDER BY record_index LIMIT ?", paper_ids + [MAX_PAGE_SIZE]).fetchall()
        return [{"id": row["id"], "record_index": row["record_index"],
                 "record": json.loads(row["body_json"]), "scientifically_accepted": False,
                 "coverage_verified": False} for row in rows]

    def _search_kind(self, conn: sqlite3.Connection, compiled: str | None, paper: str | None,
                     kind: str, page: int, page_size: int) -> tuple[int, list[dict[str, Any]]]:
        table = "title_fts" if kind == "title" else "search_fts"
        where = f"{table} MATCH ?" if compiled else ("1 = 1" if kind == "title" else "hit_kind = ?")
        params: list[Any] = [compiled] if compiled else ([] if kind == "title" else [kind])
        if paper:
            where += " AND paper_id = ?"
            params.append(paper)
        total = conn.execute("SELECT count(*) FROM " + table + " WHERE " + where, params).fetchone()[0]
        # FTS5's hidden ``rank`` column is the configured bm25 score, but lets
        # SQLite stop after the requested page rather than materializing and
        # sorting every matching row.  Within one immutable index its rowid
        # tie order is stable, so repeated page requests remain reproducible.
        ordering = "rank" if compiled else "paper_id, rowid"
        fields = "stable_id, paper_id" + (", hit_kind" if kind != "title" else "")
        rows = conn.execute("SELECT " + fields + " FROM " + table + " WHERE " + where +
                            " ORDER BY " + ordering + " LIMIT ? OFFSET ?",
                            params + [page_size, (page - 1) * page_size]).fetchall()
        hits = []
        for row in rows:
            if kind == "title":
                paper_row = conn.execute("SELECT * FROM papers WHERE paper_id = ?", (row["paper_id"],)).fetchone()
                hits.append({"id": row["stable_id"], "kind": "title_metadata", **dict(paper_row),
                             "scientifically_accepted": False})
            else:
                passage = conn.execute("SELECT passages.*, papers.title, papers.citation, papers.source_inventory_status FROM passages "
                                       "JOIN papers USING(paper_id) WHERE passages.id = ?", (row["stable_id"],)).fetchone()
                hits.append(self._passage_hit(passage))
        return total, hits

    def search(self, query: str, *, paper: str | None = None, page: int = 1,
               page_size: int = 20) -> dict[str, Any]:
        compiled = safe_fts_query(query)
        if page < 1 or not 1 <= page_size <= MAX_PAGE_SIZE:
            raise ValueError("invalid_pagination")
        if paper is not None and not isinstance(paper, str):
            raise ValueError("paper_must_be_string")
        if not compiled and query.strip():
            return {"query": query, "page": page, "page_size": page_size, "title_hits": [],
                    "passage_hits": [], "total_title_hits": 0, "total_passage_hits": 0,
                    "total": 0, "matching_papers": [], "total_matching_papers": 0,
                    "records": [], "snapshot_id": self.snapshot()["snapshot_id"], "snapshot": self.snapshot()}
        with self._connect() as conn:
            title_total, title_hits = self._search_kind(conn, compiled, paper, "title", page, page_size)
            passage_total, passage_hits = self._search_kind(conn, compiled, paper, "passage", page, page_size)
            if compiled:
                matching_total, matching_papers = self._matching_papers(conn, compiled, paper)
            else:
                matching_papers = [row for row in self.papers() if not paper or row["paper_id"] == paper]
                matching_total = len(matching_papers)
                matching_papers = matching_papers[:MAX_PAGE_SIZE]
            records = self._records_for_papers(conn, [row["paper_id"] for row in matching_papers])
        return {"query": query, "paper": paper, "page": page, "page_size": page_size,
                "title_hits": title_hits, "passage_hits": passage_hits,
                "total_title_hits": title_total, "total_passage_hits": passage_total,
                "total": title_total + passage_total, "matching_papers": matching_papers,
                "total_matching_papers": matching_total, "records": records,
                "records_are_related_context_not_search_hits": True,
                "snapshot_id": self.snapshot()["snapshot_id"], "snapshot": self.snapshot()}


def make_server(index: ResearchSearchIndex, port: int = 0, *, static_root: Path | str | None = None,
                workspace: Workspace | None = None, assistant=None, source_assets=None,
                citation_index=None) -> http.server.ThreadingHTTPServer:
    assets = source_assets or {}
    if assets:
        indexed = {p['paper_id']: p.get('source_pdf_sha256') for p in index.papers()}
        if any(indexed.get(pid) != row.get('source_pdf_sha256') for pid, row in assets.items()):
            raise ValueError('pdf_assets_do_not_match_index')
    static = None
    if static_root is not None:
        static = (workspace or Workspace()).check(static_root).resolve()
        if not static.is_dir():
            raise ValueError("static_root_not_directory")
    class Handler(http.server.BaseHTTPRequestHandler):
        def _json(self, status: int, body: Any) -> None:
            data = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if not self._local_request():
                self._json(403, {"error": "Local preview requests only."})
                return
            parsed = urllib.parse.urlparse(self.path)
            args = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
            try:
                if parsed.path.startswith('/api/citations/'):
                    if citation_index is None:
                        self._json(503, {'status':'unavailable','error':'citation_index_not_loaded'})
                    elif parsed.path == '/api/citations/health':
                        self._json(200, {**citation_index.health(),'scientifically_accepted':False})
                    elif parsed.path == '/api/citations/search':
                        value=citation_index.search(args.get('q',[''])[0],
                            limit=int(args.get('limit',['20'])[0]),cursor=int(args.get('cursor',['0'])[0]))
                        self._json(200,{**value,'scientifically_accepted':False})
                    elif parsed.path == '/api/citations/references':
                        value=citation_index.search_references(args.get('q',[''])[0],
                            limit=int(args.get('limit',['20'])[0]),cursor=int(args.get('cursor',['0'])[0]))
                        self._json(200,{**value,'scientifically_accepted':False})
                    elif parsed.path == '/api/citations/work':
                        try:
                            value=citation_index.work(args.get('work_id',[''])[0],
                                limit=int(args.get('limit',['20'])[0]),cursor=int(args.get('cursor',['0'])[0]))
                            self._json(200,{**value,'scientifically_accepted':False})
                        except KeyError:
                            self._json(404,{'error':'citation_work_not_found'})
                    else:
                        self._json(404,{'error':'citation_route_not_found'})
                elif parsed.path == "/api/assistant/status":
                    if not self._local_request():
                        self._json(403, {"error": "Local preview requests only."})
                    else:
                        self._json(200, assistant.status() if assistant else {"enabled": False})
                elif parsed.path == "/api/health":
                    self._json(200, {"status": "ok", "snapshot": index.snapshot()})
                elif parsed.path == "/api/papers":
                    rows=index.papers(evidence_only=args.get("scope", ["evidence"]) == ["evidence"])
                    if (self.headers.get('X-Mosaic-Feature')=='researcher' and assistant is not None
                            and assistant.allowed_source_sha256s is not None):
                        rows=[r for r in rows if r.get('source_pdf_sha256') in assistant.allowed_source_sha256s]
                    self._json(200, {"papers":rows, "snapshot":index.snapshot()})
                elif parsed.path == "/api/catalog":
                    self._json(200, index.catalog(args.get("q", [""])[0],
                        page=int(args.get("page", ["1"])[0]), page_size=int(args.get("page_size", ["30"])[0]),
                        year=args.get("year", [""])[0], journal=args.get("journal", [""])[0],
                        status=args.get("status", [""])[0]))
                elif parsed.path == "/api/search":
                    value = args.get("q", [""])[0]
                    paper = args.get("paper", [None])[0] or None
                    page = int(args.get("page", ["1"])[0])
                    page_size = int(args.get("page_size", ["20"])[0])
                    result = index.search(value, paper=paper, page=page, page_size=page_size)
                    for item in result.get('records', []):
                        record = item['record']
                        if record['paper_id'] in assets:
                            record['source_pdf_available'] = assets[record['paper_id']]['availability'] == 'local_file_present'
                    self._json(200, result)
                elif parsed.path == "/api/record":
                    stable_id = args.get("id", [""])[0]
                    found = index.record(stable_id)
                    if found and found.get('record', {}).get('paper_id') in assets:
                        pid = found['record']['paper_id']
                        found['record']['source_pdf_available'] = assets[pid]['availability'] == 'local_file_present'
                    self._json(200 if found else 404, found or {"error": "record_not_found"})
                elif parsed.path == "/api/passage":
                    found = index.passage(args.get("id", [""])[0])
                    self._json(200 if found else 404, found or {"error": "passage_not_found"})
                elif parsed.path == "/api/page-passages":
                    self._json(200, index.page_passages(
                        args.get("paper", [""])[0], int(args.get("page", ["0"])[0]),
                        exclude=args.get("exclude", [""])[0], limit=int(args.get("limit", ["5"])[0])))
                else:
                    match = re.fullmatch(r'/sources/([A-Za-z0-9_.:-]+)\.pdf', parsed.path)
                    if match and match[1] in assets:
                        from mosaic.source_assets import pdf_bytes
                        try:
                            raw = pdf_bytes(assets, match[1])
                        except (ValueError, OSError):
                            self._json(409, {'error': 'Source PDF unavailable or identity changed. Captured text remains available.'})
                            return
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/pdf')
                        self.send_header('Content-Length', str(len(raw)))
                        self.send_header('X-Content-Type-Options', 'nosniff')
                        self.send_header('Cache-Control', 'no-store')
                        self.end_headers()
                        self.wfile.write(raw)
                        return
                    self._static(parsed.path)
            except (ValueError, sqlite3.Error) as exc:
                self._json(400, {"error": str(exc)})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != '/api/assistant' or assistant is None:
                self._json(405, {"error": "read_only"})
                return
            import hmac
            if (not self._local_request() or not hmac.compare_digest(
                    self.headers.get('X-Mosaic-Request', ''), assistant.csrf_token)):
                self._json(403, {"error": "This request did not originate from the local MOSAIC preview."})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 50000 or self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                    raise ValueError('Invalid request size or content type.')
                self.connection.settimeout(10)
                body = json.loads(self.rfile.read(length))
                self._json(200, assistant.answer(body))
            except (ValueError, RuntimeError) as exc:
                self._json(400, {"error": str(exc)})
            except Exception:
                self._json(500, {"error": "The assistant could not complete this request. Search and saved work remain available."})

        def _local_request(self) -> bool:
            port = self.server.server_address[1]
            allowed = {f'127.0.0.1:{port}', f'localhost:{port}'}
            host = self.headers.get('Host', '')
            origin = self.headers.get('Origin')
            return host in allowed and (origin is None or origin == 'http://' + host)

        def _static(self, raw_path: str) -> None:
            if static is None:
                self._json(404, {"error": "not_found"})
                return
            relative = "index.html" if raw_path == "/" else urllib.parse.unquote(raw_path).lstrip("/")
            candidate = (static / relative).resolve()
            try:
                candidate.relative_to(static)
            except ValueError:
                self._json(404, {"error": "not_found"})
                return
            if not candidate.is_file():
                self._json(404, {"error": "not_found"})
                return
            data = candidate.read_bytes()
            mime = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt: str, *args: Any) -> None:
            return
    return http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)


def synthetic_benchmark(output: Path | str, *, papers: int = 30_000,
                        passages_per_paper: int = 3) -> dict[str, Any]:
    """Measure a synthetic index without holding its generated passages in memory.

    This deliberately bypasses ``build_index`` only for synthetic performance
    fixtures.  It writes the same immutable schema and deterministic metadata,
    but streams fixed synthetic rows in batches so a million-passage benchmark
    measures query behaviour rather than Python object retention.
    """
    output = Path(output).resolve()
    if papers < 1 or passages_per_paper < 1:
        raise ValueError("synthetic_dimensions_must_be_positive")
    root = Path(output).parent
    ws = Workspace(repo_root=root, temp_root=root)
    target = ws.check(output)
    if target.exists():
        raise ImmutableArtifactError("derived_index_already_exists")
    started = time.perf_counter()
    conn = ws.sqlite_connect(target)
    try:
        _create_schema(conn)
        paper_rows: list[tuple[Any, ...]] = []
        passage_rows: list[tuple[Any, ...]] = []
        fts_rows: list[tuple[Any, ...]] = []
        title_rows: list[tuple[Any, ...]] = []
        for n in range(papers):
            paper_id = f"synthetic-{n:06d}"
            title = f"Synthetic organization study {n}"
            paper_rows.append((paper_id, title, None, "0" * 64, "1" * 64, "synthetic_only"))
            title_rows.append((f"title:{paper_id}", paper_id, normalize_search(title)))
            for j in range(passages_per_paper):
                stable_id = f"passage:{paper_id}:u{j}"
                quote = f"Synthetic passage {j} for organizational study {n} and collaboration."
                passage_rows.append((stable_id, paper_id, f"u{j}", j + 1, j, quote, "0" * 64, "1" * 64))
                fts_rows.append((stable_id, paper_id, "passage", normalize_search(quote)))
            if len(paper_rows) >= 2_000:
                conn.executemany("INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?)", paper_rows)
                conn.executemany("INSERT INTO passages VALUES (?, ?, ?, ?, ?, ?, ?, ?)", passage_rows)
                conn.executemany("INSERT INTO search_fts VALUES (?, ?, ?, ?)", fts_rows)
                conn.executemany("INSERT INTO title_fts VALUES (?, ?, ?)", title_rows)
                paper_rows.clear(); passage_rows.clear(); fts_rows.clear(); title_rows.clear()
        if paper_rows:
            conn.executemany("INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?)", paper_rows)
            conn.executemany("INSERT INTO passages VALUES (?, ?, ?, ?, ?, ?, ?, ?)", passage_rows)
            conn.executemany("INSERT INTO search_fts VALUES (?, ?, ?, ?)", fts_rows)
            conn.executemany("INSERT INTO title_fts VALUES (?, ?, ?)", title_rows)
        fingerprint = hashlib.sha256(f"synthetic:{papers}:{passages_per_paper}".encode()).hexdigest()
        snapshot = {"version": VERSION, "input_sha256": fingerprint, "paper_count": papers,
                    "source_passage_count": papers * passages_per_paper, "record_count": 0,
                    "title_metadata_count": papers, "admitted_source_paper_count": 0,
                    "record_context_only_paper_count": 0, "scientifically_accepted": False,
                    "coverage_verified": False, "synthetic_only": True}
        snapshot["build_seconds"] = round(time.perf_counter() - started, 6)
        snapshot["snapshot_id"] = VERSION + ":" + fingerprint
        conn.executemany("INSERT INTO metadata VALUES (?, ?)",
                         [(key, json.dumps(value, separators=(",", ":"))) for key, value in snapshot.items()])
        conn.commit()
    finally:
        conn.close()
    snapshot["sqlite_bytes"] = target.stat().st_size
    index = ResearchSearchIndex(output)
    samples = []
    for query in ("organization", "collaboration", "synthetic study", "passage") * 5:
        started = time.perf_counter()
        index.search(query, page_size=20)
        samples.append(time.perf_counter() - started)
    samples.sort()
    return {**snapshot, "benchmark_kind": "synthetic_only", "query_samples": len(samples),
            "query_p50_seconds": samples[len(samples) // 2],
            "query_p95_seconds": samples[min(len(samples) - 1, int(len(samples) * .95))]}


def main() -> None:
    parser = argparse.ArgumentParser()
    command = parser.add_subparsers(dest="command", required=True)
    build = command.add_parser("build")
    build.add_argument("--input", required=True)
    build.add_argument("--output", required=True)
    serve = command.add_parser("serve")
    serve.add_argument("--index", required=True)
    serve.add_argument("--port", type=int, default=8767)
    serve.add_argument("--static-root")
    serve.add_argument('--source-assets', help='Private admitted-PDF lookup manifest; files are hash-checked on every open.')
    serve.add_argument('--citation-index', help='Optional immutable admitted citation SQLite snapshot.')
    serve.add_argument('--citation-manifest-sha256', help='Independent byte digest required for a provenance-bound v2 citation snapshot.')
    serve.add_argument("--research-assistant", action="store_true",
                       help="Enable user-authorized OpenAI researcher drafts within the separate USD20 test cap.")
    benchmark = command.add_parser("benchmark")
    benchmark.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "build":
        with open(args.input, encoding="utf-8") as handle:
            data = json.load(handle)
        print(json.dumps(build_index(data, args.output), ensure_ascii=False))
    elif args.command == "benchmark":
        print(json.dumps(synthetic_benchmark(args.output), ensure_ascii=False))
    else:
        index = ResearchSearchIndex(args.index)
        assistant = None
        if args.research_assistant:
            from scripts.research_assistant import ResearchAssistant
            assistant = ResearchAssistant(index)
        from mosaic.source_assets import load_assets
        citations=None
        if args.citation_index:
            from mosaic.citation_graph import CitationGraphIndex
            citations=CitationGraphIndex(args.citation_index,
                expected_manifest_sha256=args.citation_manifest_sha256)
        server = make_server(index, args.port, static_root=args.static_root, assistant=assistant,
                             source_assets=load_assets(args.source_assets) if args.source_assets else None,
                             citation_index=citations)
        print(f"research search API: http://127.0.0.1:{server.server_address[1]}/api/health")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
            if citations is not None: citations.close()


if __name__ == "__main__":
    main()
