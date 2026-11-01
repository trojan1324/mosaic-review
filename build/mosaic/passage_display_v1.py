"""Conservative, read-only article-body projection for source passages.

The search index deliberately preserves every captured text unit.  That is the
right archival representation, but it is a poor default reader: page-one
mastheads, bylines, affiliations and reference lists look like evidence when
they are put beside body paragraphs.  This module adds a *view*, never a
correction.  It returns the original passage identifiers and byte-for-byte
quotes and records every excluded unit's reason.

The projection is intentionally abstaining.  A paper is shown only after an
explicit Abstract marker and before an explicit References/Bibliography
heading.  It does not infer a beginning from a page number or try to repair
text.  Papers without those boundaries remain available through the archival
API but are not mixed into the reader's default article-body list.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import re
import sqlite3
from typing import Any

from mosaic.workspace import readonly_sqlite


VERSION = "passage-display/v1"
_ABSTRACT = re.compile(r"^\s*(?:abstract|summary)\s*[:.\-–—]?\s*$", re.I)
_ABSTRACT_SPACED = re.compile(r"^\s*a\s+b\s+s\s+t\s+r\s+a\s+c\s+t\s*$", re.I)
_ABSTRACT_INLINE = re.compile(r"^\s*(?:abstract|summary)\s*[:.\-–—]\s+\S", re.I)
_ABSTRACT_BLOCK = re.compile(r"^\s*(?:abstract|summary)\s*[:.\-–—]?\s*\r?\n\s*\S", re.I)
_BACK = re.compile(r"^\s*(?:references|bibliography|literature cited)\s*$", re.I)
_REFERENCE_BLOCK = re.compile(
    r"^\s*(?:references|bibliography|literature cited)\s*[:.\-–—]?\s*\r?\n\s*\S", re.I)
_LATE_BACK = re.compile(
    r"^\s*(?:disclosure\s+statement|acknowledg(?:e)?ments?|competing\s+interests?|conflict(?:s)?\s+of\s+interest|"
    r"supplementary\s+(?:material|information)|data\s+availability)\s*[:.\-–—]?\s*$", re.I)
_LATE_BACK_BLOCK = re.compile(
    r"^\s*(?:disclosure\s+statement|acknowledg(?:e)?ments?|competing\s+interests?|conflict(?:s)?\s+of\s+interest|"
    r"supplementary\s+(?:material|information)|data\s+availability)\s*[:.\-–—]?\s*\r?\n\s*\S", re.I)
_LATE_BACK_GENERIC = re.compile(r"^\s*(?:funding(?:\s+statement)?|author\s+contributions?)\s*[:.\-–—]?\s*$", re.I)
_LATE_BACK_GENERIC_BLOCK = re.compile(r"^\s*(?:funding(?:\s+statement)?|author\s+contributions?)\s*[:.\-–—]?\s*\r?\n\s*\S", re.I)
_NUMBERED_HEADING = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+[A-Z][A-Z \-:]{2,}\s*$")
_NUMBERED_TITLECASE_HEADING = re.compile(r"^\s*\d+(?:\.\d+)+\.?\s+(?:[A-Z][a-z]+)(?:[ \-:]+[A-Z][a-z]+){1,10}\s*$")
_KNOWN_NAVIGATION_HEADING = re.compile(r"^\s*(?:introduction|conclusion|future issues)\s*$", re.I)
_SENTENCE_END = re.compile(r"[.!?…][\"'”’\)\]\}]*\s*$")
_REFERENCE_ENTRY = re.compile(r"(?:\b(?:18|19|20)\d{2}[a-z]?\b|\bdoi\s*:|\bhttps?://)", re.I)
_FRONT = re.compile(
    r"^\s*(?:original\s+article|research\s+article|article\s+info|"
    r"a\s+r\s+t\s+i\s+c\s+l\s+e\s+i\s+n\s+f\s+o|"
    r"(?:doi\s*:\s*\S{4,}|https?://\S{4,})|(?:copyright\s+)?©\s*\d{4}\b|"
    r"(?:[⇑*]\s*)?corresponding\s+author\s*[:.]|all\s+rights\s+reserved\s*$)", re.I)
_PAGE_ONLY = re.compile(r"^\s*(?:page\s+)?\d{1,4}\s*$", re.I)
_ARTICLE_HISTORY = re.compile(
    r"^\s*article\s+(?:history|info)\s*:\s*(?=.*\b(?:received|accepted|available\s+online)\b)"
    r"(?=.*\b\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b)", re.I | re.S)
_KEYWORDS = re.compile(r"^\s*keywords?\s*:\s*\S", re.I | re.S)
_PUBLISHER_BANNER = re.compile(
    r"^\s*(?:contents\s+lists\s+available\s+at\s+(?:science(?:direct|verse))|"
    r"journal\s+homepage\s*:|e-?mail\s+addresses?\s*:|.*\bsee\s+front\s+matter\b)", re.I | re.S)
_CONTROL_COPYRIGHT = re.compile(r"(?:©|\x02)\s*\d{4}\b", re.I)
_AFFILIATION = re.compile(
    r"^\s*[a-z*†‡]\s*[A-Z][^\n]{0,220}\b(?:[A-Z]{2}\s*\d{5}(?:-\d{4})?|USA|United\s+States|Canada)\b", re.S)


@dataclass(frozen=True)
class Passage:
    id: str
    paper_id: str
    unit_id: str
    page: int
    ordinal: int
    quote: str
    source_pdf_sha256: str | None
    source_units_sha256: str | None


@dataclass(frozen=True)
class PassagePage:
    passages: list[dict[str, Any]]
    total: int
    page: int
    page_size: int
    excluded: dict[str, int]
    unresolved_papers: int
    version: str = VERSION


def _connect(index_path: Path | str) -> sqlite3.Connection:
    conn = readonly_sqlite(Path(index_path).resolve(), immutable=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def _validate_allowed(conn: sqlite3.Connection, allowed_sources: dict[str, str] | None) -> list[str] | None:
    """Validate identity metadata before any quotation text is selected."""
    if allowed_sources is None:
        return None
    if not allowed_sources or any(not isinstance(key, str) or not isinstance(value, str)
                                 for key, value in allowed_sources.items()):
        raise ValueError("invalid_allowed_sources")
    placeholders = ",".join("?" for _ in allowed_sources)
    rows = conn.execute(f"""SELECT paper_id, source_pdf_sha256, source_inventory_status
        FROM papers WHERE paper_id IN ({placeholders})""", tuple(allowed_sources)).fetchall()
    observed = {row["paper_id"]: row for row in rows}
    if set(observed) != set(allowed_sources):
        raise ValueError("allowed_source_missing_from_index")
    for paper_id, expected_sha in allowed_sources.items():
        row = observed[paper_id]
        if row["source_inventory_status"] != "admitted_source_packet":
            raise ValueError("allowed_source_not_admitted")
        if row["source_pdf_sha256"] != expected_sha:
            raise ValueError("allowed_source_hash_mismatch")
    return list(allowed_sources)


def _rows(conn: sqlite3.Connection, paper_id: str | None = None,
          allowed_sources: dict[str, str] | None = None) -> list[Passage]:
    where = "p.source_inventory_status = 'admitted_source_packet'"
    params: list[object] = []
    allowed = _validate_allowed(conn, allowed_sources)
    if allowed is not None:
        where += " AND x.paper_id IN (" + ",".join("?" for _ in allowed) + ")"
        params.extend(allowed)
    if paper_id is not None:
        where += " AND x.paper_id = ?"
        params.append(paper_id)
    sql = f"""
      SELECT x.id, x.paper_id, x.unit_id, x.page, x.ordinal, x.quote,
             x.source_pdf_sha256, x.source_units_sha256
      FROM passages x JOIN papers p ON p.paper_id=x.paper_id
      WHERE {where} ORDER BY x.paper_id, x.page, x.ordinal, x.id
    """
    return [Passage(**dict(row)) for row in conn.execute(sql, params)]


def _front_reason(text: str) -> str | None:
    compact = " ".join(text.split())
    if _ARTICLE_HISTORY.match(compact):
        return "article_history_metadata"
    if _KEYWORDS.match(compact):
        return "keyword_metadata"
    if _PUBLISHER_BANNER.match(compact):
        return "publisher_banner_metadata"
    if _CONTROL_COPYRIGHT.search(compact):
        return "copyright_metadata"
    if _AFFILIATION.match(compact):
        return "affiliation_metadata"
    if _FRONT.match(compact) or _PAGE_ONLY.match(compact):
        return "in_body_furniture_or_note"
    return None


def _complete_sentence(text: str) -> bool:
    """A deliberately narrow UI check; it does not repair or join passages."""
    compact = text.strip()
    first_letter = next((char for char in compact if char.isalpha()), None)
    if first_letter is None or first_letter.islower():
        return False
    return bool(_SENTENCE_END.search(compact))


def classify(passage_rows: list[Passage]) -> tuple[list[Passage], dict[str, int], bool]:
    """Return displayable passages, exclusion counts, and boundary certainty.

    Markers must be standalone units.  This avoids treating a body sentence
    that merely mentions "the abstract" or "references" as a document edge.
    """
    reasons: Counter[str] = Counter()
    abstract_at = next((i for i, row in enumerate(passage_rows)
                        if (_ABSTRACT.fullmatch(row.quote.strip()) or _ABSTRACT_INLINE.match(row.quote)
                            or _ABSTRACT_BLOCK.match(row.quote)
                            or _ABSTRACT_SPACED.fullmatch(row.quote.strip()))), None)
    if abstract_at is None:
        reasons["unresolved_no_standalone_abstract"] += len(passage_rows)
        return [], dict(reasons), False
    candidates = [i for i, row in enumerate(passage_rows[abstract_at + 1:], abstract_at + 1)
                  if _BACK.fullmatch(row.quote.strip()) or _REFERENCE_BLOCK.match(row.quote)]
    # A one-word heading can be discussed or rendered as an in-text artifact.
    # A terminal boundary needs intervening content and either a reference-entry
    # signal immediately after it or a position in the final third of the
    # captured units.  This admits short papers without using a bare word in
    # the middle of a paper as an end boundary.
    references_at = next((i for i in candidates
                          if i - abstract_at >= 2 and
                          (_REFERENCE_BLOCK.match(passage_rows[i].quote)
                           or any(_REFERENCE_ENTRY.search(row.quote) for row in passage_rows[i + 1:i + 9])
                           or i * 3 >= len(passage_rows) * 2)), None)
    back_matter_at = next((i for i, row in enumerate(passage_rows[abstract_at + 1:], abstract_at + 1)
                           if (_LATE_BACK.fullmatch(row.quote.strip()) or _LATE_BACK_BLOCK.match(row.quote)
                               or (references_at is not None and i < references_at and i * 3 >= len(passage_rows) * 2
                                   and (_LATE_BACK_GENERIC.fullmatch(row.quote.strip())
                                        or _LATE_BACK_GENERIC_BLOCK.match(row.quote))))), None)
    terminal_at = min((i for i in (references_at, back_matter_at) if i is not None), default=None)
    if terminal_at is None:
        reasons["unresolved_no_terminal_references"] += len(passage_rows)
        return [], dict(reasons), False
    output: list[Passage] = []
    inline_abstract = (_ABSTRACT_INLINE.match(passage_rows[abstract_at].quote) is not None
                       or _ABSTRACT_BLOCK.match(passage_rows[abstract_at].quote) is not None)
    for ordinal, row in enumerate(passage_rows):
        if ordinal < abstract_at or (ordinal == abstract_at and not inline_abstract):
            reasons["front_matter_before_abstract"] += 1
        elif ordinal >= terminal_at:
            reasons["references_or_back_matter" if terminal_at == references_at else "late_back_matter"] += 1
        elif (front_reason := _front_reason(row.quote)) is not None:
            reasons[front_reason] += 1
        elif (_NUMBERED_HEADING.fullmatch(row.quote.strip())
              or _NUMBERED_TITLECASE_HEADING.fullmatch(row.quote.strip())
              or _KNOWN_NAVIGATION_HEADING.fullmatch(row.quote.strip())):
            reasons["heading_context"] += 1
        else:
            output.append(row)
    return output, dict(reasons), True


def display_page(index_path: Path | str, *, paper_id: str | None = None,
                 query: str = "", page: int = 1, page_size: int = 25,
                 allowed_sources: dict[str, str] | None = None,
                 scope_paper_ids: set[str] | None = None,
                 complete_sentences_only: bool = False) -> dict[str, Any]:
    """Return a stable, conservative article-body page without changing index data."""
    if page < 1 or not 1 <= page_size <= 100:
        raise ValueError("invalid_page")
    needle = " ".join(query.casefold().split())
    with _connect(index_path) as conn:
        rows = _rows(conn, paper_id, allowed_sources)
    by_paper: dict[str, list[Passage]] = {}
    for row in rows:
        by_paper.setdefault(row.paper_id, []).append(row)
    eligible: list[Passage] = []
    excluded: Counter[str] = Counter()
    unresolved = 0
    for source_rows in by_paper.values():
        accepted, reasons, certain = classify(source_rows)
        excluded.update(reasons)
        if not certain:
            unresolved += 1
        eligible.extend(accepted)
    if scope_paper_ids is not None:
        if not all(isinstance(value, str) for value in scope_paper_ids):
            raise ValueError("invalid_scope_paper_ids")
        eligible = [row for row in eligible if row.paper_id in scope_paper_ids]
    if complete_sentences_only:
        before = len(eligible)
        eligible = [row for row in eligible if _complete_sentence(row.quote)]
        excluded["sentence_boundary_ambiguous"] += before - len(eligible)
    if needle:
        eligible = [row for row in eligible if needle in " ".join(row.quote.casefold().split())]
    total = len(eligible)
    start = (page - 1) * page_size
    result = PassagePage([asdict(row) for row in eligible[start:start + page_size]], total, page,
                         page_size, dict(sorted(excluded.items())), unresolved)
    return asdict(result)


class PassageDisplayIndex:
    """Cached projection for an HTTP server that must not rescan the corpus.

    `warm()` makes the one deliberate read-only pass at server startup.  Every
    subsequent request slices already classified immutable rows.  The caller
    can retain it for a server lifetime and discard it when the snapshot
    changes; it holds no SQLite handle or writable state.
    """

    def __init__(self, index_path: Path | str, *, allowed_sources: dict[str, str] | None = None) -> None:
        self.index_path = Path(index_path)
        self.allowed_sources = allowed_sources
        self._snapshot_sha256 = _file_sha256(self.index_path)
        self._loaded = False
        self._all: list[Passage] = []
        self._per_paper: dict[str, list[Passage]] = {}
        self._source_rows: dict[str, list[Passage]] = {}
        self._excluded: Counter[str] = Counter()
        self._unresolved = 0

    def warm(self) -> dict[str, Any]:
        if self._loaded:
            return self.status()
        with _connect(self.index_path) as conn:
            rows = _rows(conn, allowed_sources=self.allowed_sources)
        grouped: dict[str, list[Passage]] = {}
        for row in rows:
            grouped.setdefault(row.paper_id, []).append(row)
        for paper_id, source_rows in grouped.items():
            self._source_rows[paper_id] = source_rows
            accepted, reasons, certain = classify(source_rows)
            self._per_paper[paper_id] = accepted
            self._all.extend(accepted)
            self._excluded.update(reasons)
            if not certain:
                self._unresolved += 1
        self._loaded = True
        return self.status()

    def status(self) -> dict[str, Any]:
        return {"version": VERSION, "source_index_sha256": self._snapshot_sha256, "warmed": self._loaded,
                "displayable_passage_count": len(self._all),
                "displayable_paper_count": sum(bool(rows) for rows in self._per_paper.values()),
                "unresolved_papers": self._unresolved,
                "excluded": dict(sorted(self._excluded.items()))}

    def page(self, *, paper_id: str | None = None, query: str = "", page: int = 1,
             page_size: int = 25, scope_paper_ids: set[str] | None = None,
             complete_sentences_only: bool = False) -> dict[str, Any]:
        if page < 1 or not 1 <= page_size <= 100:
            raise ValueError("invalid_page")
        self.warm()
        if scope_paper_ids is not None and not all(isinstance(value, str) for value in scope_paper_ids):
            raise ValueError("invalid_scope_paper_ids")
        if paper_id and scope_paper_ids is not None and paper_id not in scope_paper_ids:
            return asdict(PassagePage([], 0, page, page_size, {"identity_pending": 1}, 1))
        rows = list(self._per_paper.get(paper_id, []) if paper_id else self._all)
        if scope_paper_ids is not None:
            rows = [row for row in rows if row.paper_id in scope_paper_ids]
        sentence_boundary_ambiguous = 0
        if complete_sentences_only:
            before = len(rows)
            rows = [row for row in rows if _complete_sentence(row.quote)]
            sentence_boundary_ambiguous = before - len(rows)
        needle = " ".join(query.casefold().split())
        if needle:
            rows = [row for row in rows if needle in " ".join(row.quote.casefold().split())]
        start = (page - 1) * page_size
        # Paper views must report only that paper's exclusions, not corpus-wide
        # totals that make a clean article look unresolved.
        if paper_id:
            source_rows = self._source_rows.get(paper_id, [])
            _, excluded, certain = classify(source_rows)
            excluded = {**excluded, **({"sentence_boundary_ambiguous": sentence_boundary_ambiguous}
                                        if complete_sentences_only else {})}
            result = PassagePage([asdict(row) for row in rows[start:start + page_size]], len(rows), page,
                                 page_size, excluded, 0 if certain else 1)
        else:
            excluded = dict(sorted(self._excluded.items()))
            if complete_sentences_only:
                excluded["sentence_boundary_ambiguous"] = sentence_boundary_ambiguous
            result = PassagePage([asdict(row) for row in rows[start:start + page_size]], len(rows), page,
                                 page_size, excluded, self._unresolved)
        return asdict(result)

    def display_page(self, paper_id: str | None = None, query: str = "", page: int = 1,
                     page_size: int = 25, scope_paper_ids: set[str] | None = None,
                     complete_sentences_only: bool = False) -> dict[str, Any]:
        """Compatibility name for the versioned display server."""
        return self.page(paper_id=paper_id, query=query, page=page, page_size=page_size,
                         scope_paper_ids=scope_paper_ids, complete_sentences_only=complete_sentences_only)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
