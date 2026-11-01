"""Boundary-pinned, read-only content projection for source passages.

This is deliberately a display layer.  The search index remains the archival
record and every returned ``quote`` is its original value.  A separately
pinned boundary manifest says where each article's abstract begins; rows before
that point are withheld from this view, while references and appendices remain
available after it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
from hashlib import sha256
import json
from pathlib import Path
import re
import math
from urllib.parse import urlencode
from typing import Any

from mosaic.passage_display_v1 import Passage, PassagePage, _connect, _file_sha256, _rows, _validate_allowed


VERSION = "passage-display/v2"
MANIFEST_VERSION = "mosaic-article-start-boundaries/v1"
_COPYRIGHT_PROPERTY = re.compile(
    r"^\s*copyright\s+of\s+.{3,180}?\s+is\s+the\s+property\s+of\b.*"
    r"(?:may\s+not\s+be\s+copied|copyright\s+holder|without\s+permission)", re.I | re.S)
_PAGE_ONLY = re.compile(r"^\s*(?:page\s+)?\d{1,4}\s*$", re.I)
_DOI_ONLY = re.compile(r"^\s*(?:doi\s*:\s*|https?://(?:dx\.)?doi\.org/)\S+\s*$", re.I)
_EMAIL_ONLY = re.compile(r"^\s*(?:e-?mail(?:\s+address(?:es)?)?\s*:\s*)?[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\s*$", re.I)
_NAMED_EMAIL = re.compile(r"^(?:[*†‡]\s*)?(?P<name>.+?)\s+[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\.?$")
_CORRESPONDENCE = re.compile(r"^\s*(?:[*†‡]\s*)?(?:correspondence|corresponding\s+author)\s*[:.]", re.I)
_DATE_TOKEN = r"(?:\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})"
_HISTORY = re.compile(r"^\s*article\s+(?:history|info)\s*:\s*(?=.*\b(?:received|accepted|available\s+online)\b)(?=.*" + _DATE_TOKEN + r")", re.I | re.S)
_DATED_HISTORY_LINE = re.compile(r"^\s*(?:received|accepted|available\s+online)\s*(?::|on)?\s*" + _DATE_TOKEN + r"\s*$", re.I)
_COPYRIGHT = re.compile(r"^\s*(?:copyright\s*)?(?:©|\(c\)|c\u20dd|\x02)\s*\d{4}\b.*(?:all\s+rights\s+reserved|publisher|elsevier|springer|wiley|annual\s+reviews)?\s*$", re.I | re.S)
_BANNER = re.compile(r"^\s*(?:contents\s+lists\s+available\s+at|journal\s+homepage\s*:)", re.I | re.S)
_AFFILIATION = re.compile(r"^\s*[a-z*†‡]\s+[A-Z][^\n]{0,220}\b(?:university|department|school)\b[^\n]{0,160}\b(?:USA|United\s+States|Canada|\d{5}(?:-\d{4})?)\b", re.I | re.S)
_ONLINE_PANEL = re.compile(r"^\s*(?:(?:the\s+)?annual\s+review\s+of\s+[A-Za-z &-]+|journal\s+of\s+[A-Za-z &-]+)\s+is\s+online\s+at\s*\r?\n\s*[a-z0-9.-]+\.[a-z]{2,}\s*$", re.I)
_ARTICLE_DOI_PANEL = re.compile(r"^\s*this\s+article(?:'|’)?s\s+doi\s*:\s*\r?\n\s*10\.\d{4,9}/\S+\s*$", re.I)
_KEYWORDS = re.compile(r"^\s*keywords?\s*:\s*\S", re.I | re.S)
_KEYWORD_PANEL = re.compile(r"^\s*keywords?(?:[ \t]*[:\u2002\u2003]|[ \t]*\r?\n)\s*\S", re.I)
_JEL_PANEL = re.compile(r"^JEL(?:\s+(?:classifications?|codes?))?\s*:?\s*[A-Z]\d{1,2}(?:\s*[·,;]\s*[A-Z]\d{1,2})*\.?$", re.I)
_VOLUME_ARTIFACT = re.compile(r"^Vol\.\s*:\s*\(0123456789\)\s*$", re.I)
_REFERENCES_HEADING = re.compile(r"^\s*(?:references|bibliography|literature\s+cited)\s*[:.\-–—]?\s*$", re.I)
_APPENDIX_HEADING = re.compile(r"^\s*appendix(?:\s+[A-Z0-9]+)?\s*[:.\-–—]?\s*$", re.I)
_PROTECTED_EDGE_HEADINGS = {
    "abstract", "introduction", "methods", "method", "results", "discussion", "references",
    "bibliography", "appendix", "participants",
}
_STUDY_HEADING = re.compile(r"^study\s+\d+[a-z]?$", re.I)


@dataclass(frozen=True)
class StartBoundary:
    paper_id: str
    source_pdf_sha256: str
    start_passage_id: str
    start_char: int
    method: str
    evidence: Any
    anchor: str | None = None


def _author_contact_only(text: str) -> bool:
    """Recognize a short marked or two-line byline/email contact block."""
    if "@" not in text:
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    compact = " ".join(text.split())
    if len(compact) > 160 or len(lines) > 3:
        return False
    if not compact.startswith(("*", "†", "‡")) and len(lines) < 2:
        return False
    match = _NAMED_EMAIL.fullmatch(compact)
    if match is None:
        return False
    tokens = match["name"].split()
    if not 2 <= len(tokens) <= 8:
        return False
    if tokens[0].casefold() in {"we", "our", "the", "this", "participants", "respondents", "please"}:
        return False
    particles = {"de", "del", "di", "da", "dos", "van", "von", "der", "den", "le", "la"}
    for token in tokens:
        letters = "".join(char for char in token if char.isalpha())
        if (not letters or any(not (char.isalpha() or char in ".-'’") for char in token)
                or not (letters[0].isupper() or token.casefold() in particles)):
            return False
    return True


def _numbered_contact_address(text: str) -> bool:
    """A short numbered postal/institution line; caller requires contact context."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not 2 <= len(lines) <= 6 or not re.fullmatch(r"\d{1,2}", lines[0]):
        return False
    address = " ".join(lines[1:])
    if not 5 <= len(address) <= 300 or "@" in address:
        return False
    if re.search(r"\b(?:participants?|respondents?|researchers?|study|sample|reported|found|were|was|is|are|measured|collected)\b", address, re.I):
        return False
    words = re.findall(r"[^\W\d_]+", address, re.UNICODE)
    particles = {"of", "and", "the", "de", "del", "di", "da", "van", "der", "den", "le", "la", "in", "upon"}
    if not words or any(not (word[0].isupper() or word.casefold() in particles) for word in words):
        return False
    return bool(re.search(r"\b(?:university|department|school|institute|college)\b", address, re.I)
                or re.search(r"(?:,\s*)(?:UK|USA|US|United Kingdom|United States|Canada|Italy|Germany|France|China|Japan|Australia|Netherlands|Sweden|Denmark|Norway|Switzerland|Belgium|Spain|Finland|Austria|Singapore|India)\.?$", address))


def _nuisance_reason(text: str) -> str | None:
    """Reject only standalone layout metadata, never body-word heuristics."""
    compact = " ".join(text.split())
    if _PAGE_ONLY.fullmatch(compact):
        return "page_furniture"
    if _DOI_ONLY.fullmatch(compact):
        return "standalone_doi_metadata"
    if _EMAIL_ONLY.fullmatch(compact):
        return "standalone_email_metadata"
    if _author_contact_only(text):
        return "author_contact_metadata"
    if _CORRESPONDENCE.match(compact):
        return "correspondence_metadata"
    if _HISTORY.match(compact):
        return "article_history_metadata"
    if _DATED_HISTORY_LINE.fullmatch(compact):
        return "article_history_metadata"
    if _COPYRIGHT.fullmatch(compact) or _COPYRIGHT_PROPERTY.search(compact):
        return "copyright_metadata"
    if _BANNER.match(compact):
        return "publisher_banner_metadata"
    if _ONLINE_PANEL.fullmatch(text):
        return "publisher_banner_metadata"
    if _ARTICLE_DOI_PANEL.fullmatch(text):
        return "standalone_doi_metadata"
    if _KEYWORDS.match(compact) or _KEYWORD_PANEL.match(text):
        return "keyword_metadata"
    if _JEL_PANEL.fullmatch(compact):
        return "classification_metadata"
    if _VOLUME_ARTIFACT.fullmatch(compact):
        return "publisher_volume_artifact"
    if _AFFILIATION.fullmatch(compact):
        return "affiliation_metadata"
    return None


def _edge_signature(text: str) -> str | None:
    """Return an intentionally exact, small candidate for header/footer matching."""
    if len(text) > 160 or len(text.splitlines()) > 2:
        return None
    signature = " ".join(text.casefold().split())
    if not signature or signature in _PROTECTED_EDGE_HEADINGS or _STUDY_HEADING.fullmatch(signature):
        return None
    return signature


def _repeated_page_edge_ids(rows: list[Passage]) -> set[str]:
    """Find exact repeated native page-edge strings on at least three pages."""
    page_edges: dict[int, list[Passage]] = {}
    for row in rows:
        if (row.unit_id or "").casefold().startswith("record-"):
            continue
        page_edges.setdefault(row.page, []).append(row)
    signatures: dict[str, set[int]] = {}
    candidates: list[tuple[str, int, str]] = []
    for page, page_rows in page_edges.items():
        ordered = sorted(page_rows, key=lambda row: (row.ordinal, row.id))
        for row in {ordered[0].id: ordered[0], ordered[-1].id: ordered[-1]}.values():
            signature = _edge_signature(row.quote)
            if signature is not None:
                signatures.setdefault(signature, set()).add(page)
                candidates.append((signature, page, row.id))
    repeated = {signature for signature, pages in signatures.items() if len(pages) >= 3}
    return {row_id for signature, _page, row_id in candidates if signature in repeated}


class ContentPassageDisplayIndex:
    """Cached content view whose article starts are manifest-pinned.

    Missing manifest items are *pending*, not rejected: their archival rows are
    intentionally not projected, while the raw index remains independently
    accessible through its existing API.
    """

    def __init__(self, index_path: Path | str, *, allowed_sources: dict[str, str],
                 boundaries_path: Path | str, expected_boundaries_sha256: str,
                 expected_source_assets_sha256: str) -> None:
        if not expected_boundaries_sha256 or not expected_source_assets_sha256:
            raise ValueError("missing_content_projection_pin")
        self.index_path = Path(index_path)
        self.allowed_sources = dict(allowed_sources)
        self.boundaries_path = Path(boundaries_path)
        self.expected_boundaries_sha256 = expected_boundaries_sha256
        self.expected_source_assets_sha256 = expected_source_assets_sha256
        self._snapshot_sha256 = _file_sha256(self.index_path)
        self._loaded = False
        self._boundaries: dict[str, StartBoundary] = {}
        self._pending: set[str] = set()
        self._all: list[dict[str, Any]] = []
        self._per_paper: dict[str, list[dict[str, Any]]] = {}
        self._source_rows: dict[str, list[Passage]] = {}
        self._excluded: Counter[str] = Counter()
        self._reviewed_masks: dict[str, dict[str, Any]] = {}
        self._seen_masks: set[str] = set()
        self.opening_images: dict[str, dict[str, Any]] = {}

    def _opening_image(self, rows: list[Passage], start_at: int, boundary: StartBoundary):
        spec = boundary.evidence.get("opening_image") if isinstance(boundary.evidence, dict) else None
        if spec is None:
            return None
        try:
            parts = spec["parts"]
            bbox = spec["source_bbox"]
            version = spec["version"]
            if version == "source-opening-image/v1":
                valid_parts = isinstance(parts, list) and len(parts) == 2
                valid_start = len(rows[start_at].quote.strip()) == 1 and rows[start_at].quote.strip().isalpha()
            elif version == "source-opening-image/v2":
                valid_parts = isinstance(parts, list) and 1 <= len(parts) <= 32
                valid_start = spec.get("purpose") == "source_opening_fidelity"
            else:
                raise ValueError("invalid_opening_image_version")
            if (spec["source_pdf_sha256"] != boundary.source_pdf_sha256
                    or spec["source_page"] != rows[start_at].page or boundary.start_char != 0
                    or not valid_parts or not valid_start
                    or not isinstance(bbox, list) or len(bbox) != 4
                    or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0 for v in bbox)
                    or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
                raise ValueError("invalid_opening_image")
            originals = rows[start_at:start_at + len(parts)]
            if (len(originals) != len(parts)
                    or (version == "source-opening-image/v1" and not originals[1].quote.lstrip()[:1].islower())):
                raise ValueError("invalid_opening_image_parts")
            for original, part in zip(originals, parts):
                if (part["id"] != original.id or part["quote_sha256"] != sha256(original.quote.encode()).hexdigest()
                        or original.page != spec["source_page"] or original.paper_id != boundary.paper_id
                        or original.source_pdf_sha256 != boundary.source_pdf_sha256):
                    raise ValueError("opening_image_source_parts_mismatch")
            relative = Path(spec["path"])
            root = self.boundaries_path.parent.resolve()
            path = (root / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(root) or path.suffix.lower() != ".png":
                raise ValueError("opening_image_path_escape")
            raw = path.read_bytes()
            if (len(raw) > 5 * 1024 * 1024 or not raw.startswith(b"\x89PNG\r\n\x1a\n")
                    or sha256(raw).hexdigest() != spec["sha256"]
                    or int.from_bytes(raw[16:20], "big") != spec["width"]
                    or int.from_bytes(raw[20:24], "big") != spec["height"]
                    or not 1 <= spec["width"] <= 10000 or not 1 <= spec["height"] <= 10000):
                raise ValueError("opening_image_artifact_mismatch")
        except (KeyError, TypeError, IndexError, OSError) as exc:
            raise ValueError("invalid_opening_image") from exc
        self.opening_images[boundary.paper_id] = {
            "path": path, "sha256": spec["sha256"], "source_pdf_sha256": boundary.source_pdf_sha256}
        return {"url": "/api/display/v1/opening-image?" + urlencode({"paper": boundary.paper_id, "sha256": spec["sha256"]}),
                "sha256": spec["sha256"], "page": spec["source_page"], "width": spec["width"], "height": spec["height"],
                "parts": [{**asdict(row), "quote_sha256": sha256(row.quote.encode()).hexdigest()} for row in originals]}

    def _load_manifest(self) -> None:
        if _file_sha256(self.boundaries_path) != self.expected_boundaries_sha256:
            raise ValueError("content_boundaries_hash_mismatch")
        try:
            payload = json.loads(self.boundaries_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("invalid_content_boundaries_manifest") from exc
        if not isinstance(payload, dict) or payload.get("version") != MANIFEST_VERSION:
            raise ValueError("invalid_content_boundaries_manifest")
        if payload.get("source_index_sha256") != self._snapshot_sha256:
            raise ValueError("content_boundaries_index_mismatch")
        if payload.get("source_assets_sha256") != self.expected_source_assets_sha256:
            raise ValueError("content_boundaries_assets_mismatch")
        items = payload.get("items")
        pending = payload.get("pending", [])
        if not isinstance(items, list) or not isinstance(pending, list):
            raise ValueError("invalid_content_boundaries_manifest")
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("invalid_content_boundary")
            paper_id, source_sha = item.get("paper_id"), item.get("source_pdf_sha256")
            start_id, start_char, method = item.get("start_passage_id"), item.get("start_char"), item.get("method")
            if (not isinstance(paper_id, str) or not isinstance(source_sha, str) or not isinstance(start_id, str)
                    or not isinstance(start_char, int) or isinstance(start_char, bool) or not isinstance(method, str)
                    or paper_id in self._boundaries or paper_id not in self.allowed_sources
                    or self.allowed_sources[paper_id] != source_sha):
                raise ValueError("invalid_content_boundary")
            anchor = item.get("anchor")
            if anchor is not None and not isinstance(anchor, str):
                raise ValueError("invalid_content_boundary")
            self._boundaries[paper_id] = StartBoundary(paper_id, source_sha, start_id, start_char,
                                                       method, item.get("evidence"), anchor)
        for item in pending:
            paper_id = item if isinstance(item, str) else item.get("paper_id") if isinstance(item, dict) else None
            if not isinstance(paper_id, str) or paper_id not in self.allowed_sources or paper_id in self._boundaries:
                raise ValueError("invalid_content_boundaries_pending")
            self._pending.add(paper_id)
        self._pending.update(set(self.allowed_sources) - set(self._boundaries))
        masks = payload.get("display_masks", [])
        if not isinstance(masks, list):
            raise ValueError("invalid_content_display_masks")
        for mask in masks:
            if not isinstance(mask, dict):
                raise ValueError("invalid_content_display_mask")
            identifier, paper = mask.get("id"), mask.get("paper_id")
            if (not isinstance(identifier, str) or not isinstance(paper, str)
                    or identifier in self._reviewed_masks
                    or paper not in self._boundaries
                    or mask.get("source_pdf_sha256") != self.allowed_sources.get(paper)
                    or mask.get("purpose") != "publication_metadata"
                    or any(not isinstance(mask.get(key), str)
                           or not re.fullmatch(r"[0-9a-f]{64}", mask[key])
                           for key in ("quote_sha256", "review_receipt_sha256"))):
                raise ValueError("invalid_content_display_mask")
            self._reviewed_masks[identifier] = mask

    def warm(self) -> dict[str, Any]:
        if self._loaded:
            return self.status()
        self._load_manifest()
        with _connect(self.index_path) as conn:
            _validate_allowed(conn, self.allowed_sources)
            # Check every stored row's source binding without selecting quote
            # text.  The papers table alone cannot detect a tampered later row.
            metadata_rows = conn.execute(
                "SELECT paper_id, source_pdf_sha256, COUNT(*) AS row_count "
                "FROM passages GROUP BY paper_id, source_pdf_sha256"
            ).fetchall()
            for metadata in metadata_rows:
                paper_id = metadata["paper_id"]
                if paper_id in self.allowed_sources and metadata["source_pdf_sha256"] != self.allowed_sources[paper_id]:
                    raise ValueError("allowed_passage_source_hash_mismatch")
            rows = _rows(conn, allowed_sources=self.allowed_sources)
        for row in rows:
            self._source_rows.setdefault(row.paper_id, []).append(row)
        for paper_id, source_rows in self._source_rows.items():
            boundary = self._boundaries.get(paper_id)
            if boundary is None:
                self._excluded["content_start_pending"] += len(source_rows)
                self._per_paper[paper_id] = []
                continue
            selected = self._project_paper(source_rows, boundary)
            self._per_paper[paper_id] = selected
            self._all.extend(selected)
        if self._seen_masks != set(self._reviewed_masks):
            raise ValueError("content_display_mask_source_row_missing")
        self._loaded = True
        return self.status()

    def _project_paper(self, rows: list[Passage], boundary: StartBoundary) -> list[dict[str, Any]]:
        start_at = next((i for i, row in enumerate(rows) if row.id == boundary.start_passage_id), None)
        if start_at is None:
            raise ValueError("content_boundary_start_missing")
        start = rows[start_at]
        if start.paper_id != boundary.paper_id or start.source_pdf_sha256 != boundary.source_pdf_sha256:
            raise ValueError("content_boundary_source_mismatch")
        if not 0 <= boundary.start_char < len(start.quote):
            raise ValueError("content_boundary_start_char_invalid")
        if (boundary.start_char > 0 and start.quote[boundary.start_char - 1].isalnum()
                and start.quote[boundary.start_char].isalnum()):
            raise ValueError("content_boundary_start_splits_word")
        if boundary.anchor is not None and not start.quote.startswith(boundary.anchor, boundary.start_char):
            raise ValueError("content_boundary_anchor_mismatch")
        opening_image = self._opening_image(rows, start_at, boundary)
        output: list[dict[str, Any]] = []
        repeated_edge_ids = _repeated_page_edge_ids(rows)
        contact_pages = {row.page for row in rows if row.page <= 3 and (
            _author_contact_only(row.quote) or _CORRESPONDENCE.match(" ".join(row.quote.split())))}
        self._excluded["before_content_start"] += start_at
        reference_or_appendix = False
        for index, row in enumerate(rows[start_at:], start_at):
            raw = row.quote
            mask = self._reviewed_masks.get(row.id)
            if mask is not None:
                if (mask["paper_id"] != row.paper_id
                        or mask["source_pdf_sha256"] != row.source_pdf_sha256
                        or mask["quote_sha256"] != sha256(raw.encode("utf-8")).hexdigest()):
                    raise ValueError("content_display_mask_source_mismatch")
                if (index == start_at or _REFERENCES_HEADING.fullmatch(raw.strip())
                        or _APPENDIX_HEADING.fullmatch(raw.strip())
                        or (opening_image is not None
                            and row.id in {part["id"] for part in opening_image["parts"]})):
                    raise ValueError("content_display_mask_protected_target")
                self._seen_masks.add(row.id)
                self._excluded["source_reviewed_publication_metadata"] += 1
                continue
            display_start = boundary.start_char if index == start_at else 0
            display_quote = raw[display_start:]
            if not display_quote:
                raise ValueError("content_boundary_empty_start")
            compact = " ".join(display_quote.split())
            if index == start_at and re.fullmatch(r"(?:abstract|summary)\s*[:.\-–—]?", compact, re.I):
                self._excluded["abstract_label_only"] += 1
                continue
            first_line = display_quote.splitlines()[0].strip()
            if (_REFERENCES_HEADING.fullmatch(compact) or _APPENDIX_HEADING.fullmatch(compact)
                    or _REFERENCES_HEADING.fullmatch(first_line) or _APPENDIX_HEADING.fullmatch(first_line)):
                reference_or_appendix = True
            reason = _nuisance_reason(display_quote)
            if reason is None and row.page in contact_pages and _numbered_contact_address(display_quote):
                reason = "numbered_author_address_metadata"
            # A bare DOI or address after an explicit References/Appendix heading
            # can be a wrapped citation or a scale/item continuation, rather than
            # page furniture.  Preserve it; the raw citation graph is reviewed
            # separately from this passage projection.
            if reference_or_appendix and reason in {
                "standalone_doi_metadata", "standalone_email_metadata", "page_furniture",
                "keyword_metadata", "author_contact_metadata", "numbered_author_address_metadata",
                "classification_metadata",
            }:
                reason = None
            if reason is None and row.id in repeated_edge_ids:
                reason = "repeated_page_edge_metadata"
            if reason:
                self._excluded[reason] += 1
                continue
            value = asdict(row)
            value.update({"display_quote": display_quote, "display_start": display_start,
                          "display_end": len(raw), "content_start_method": boundary.method})
            if opening_image is not None and row.id in {part["id"] for part in opening_image["parts"]}:
                value["opening_image"] = opening_image
            output.append(value)
        return output

    def status(self) -> dict[str, Any]:
        return {"version": VERSION, "source_index_sha256": self._snapshot_sha256, "warmed": self._loaded,
                "displayable_passage_count": len(self._all),
                "displayable_paper_count": sum(bool(rows) for rows in self._per_paper.values()),
                "pending_papers": len(self._pending), "unresolved_papers": len(self._pending),
                "excluded": dict(sorted(self._excluded.items()))}

    def page(self, *, paper_id: str | None = None, query: str = "", page: int = 1,
             page_size: int = 25, scope_paper_ids: set[str] | None = None,
             complete_sentences_only: bool = False) -> dict[str, Any]:
        if page < 1 or not 1 <= page_size <= 100:
            raise ValueError("invalid_page")
        # Kept for v1 call compatibility.  V2 deliberately never withholds a
        # lower-case continuation or clipped-looking unit on punctuation alone.
        self.warm()
        if scope_paper_ids is not None:
            if (not scope_paper_ids or not all(isinstance(value, str) for value in scope_paper_ids)
                    or not set(scope_paper_ids).issubset(self.allowed_sources)):
                raise ValueError("invalid_scope_paper_ids")
        if paper_id is not None and paper_id not in self.allowed_sources:
            raise ValueError("paper_outside_allowed_sources")
        if paper_id is not None and scope_paper_ids is not None and paper_id not in scope_paper_ids:
            raise ValueError("paper_outside_scope")
        rows = list(self._per_paper.get(paper_id, []) if paper_id else self._all)
        if scope_paper_ids is not None:
            rows = [row for row in rows if row["paper_id"] in scope_paper_ids]
        needle = " ".join(query.casefold().split())
        if needle:
            rows = [row for row in rows if needle in " ".join(row["display_quote"].casefold().split())]
        captured_unit_count = len(rows)
        # A source crop represents both immutable captures. Group before paging
        # so its two parts never create duplicate cards on adjacent pages.
        seen_images = set()
        grouped = []
        image_count = 0
        for row in rows:
            image = row.get("opening_image")
            if image:
                key = (row["paper_id"], image["sha256"])
                if key in seen_images:
                    continue
                seen_images.add(key)
                image_count += 1
            grouped.append(row)
        rows = grouped
        start = (page - 1) * page_size
        excluded = (dict(sorted(self._excluded.items())) if paper_id is None else
                    ({"content_start_pending": len(self._source_rows.get(paper_id, []))}
                     if paper_id in self._pending else {}))
        result = PassagePage(rows[start:start + page_size], len(rows), page, page_size, excluded,
                             1 if paper_id in self._pending else (len(self._pending) if paper_id is None else 0), VERSION)
        payload = asdict(result)
        if image_count:
            payload["source_image_openings"] = image_count
            payload["original_captured_unit_count"] = captured_unit_count
        return payload

    def display_page(self, paper_id: str | None = None, query: str = "", page: int = 1,
                     page_size: int = 25, scope_paper_ids: set[str] | None = None,
                     complete_sentences_only: bool = False) -> dict[str, Any]:
        return self.page(paper_id=paper_id, query=query, page=page, page_size=page_size,
                         scope_paper_ids=scope_paper_ids, complete_sentences_only=complete_sentences_only)
