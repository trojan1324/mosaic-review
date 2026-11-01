"""Fail-closed display projections for citations and measure discovery.

These functions are deliberately projections: they neither write nor alter the
immutable citation graph or structured-measure records supplied by a caller.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Any, Iterable


VERSION = "mosaic-discovery-display/v1"
MAX_PAGE_SIZE = 100
_YEAR = re.compile(r"^(?:1[5-9]\d{2}|20\d{2}|21\d{2})$")
_DOI = re.compile(r"^(?:https://doi\.org/)?10\.\d{4,9}/[-._;()/:A-Z0-9]+$", re.I)
_DOI_CONTAMINATION = re.compile(r"(?:copyright|published|thissuggests|supponseptember)", re.I)
_APA_AUTHOR = re.compile(r"^(?P<surname>[^,\d]+),\s*(?P<initials>[A-ZÀ-ÖØ-Þ.\-\s]+)(?P<suffix>\s+(?:Jr\.|III|IV))?$")
_PRINTED_REFERENCE = re.compile(r"(?P<authors>[A-Z].{1,300}?)\s*\((?P<year>(?:19|20)\d{2})\)\.\s*(?P<title>[^.]{3,500})\.\s*(?P<journal>[A-Z][^,]{2,300}),\s*(?P<volume>\d+)(?:\((?P<issue>[^)\s]{1,30})\))?,\s*(?P<pages>(?:\d+[-–]\d+|e\d+))\.?(?:\s*(?P<doi>https?://doi\.org/10\.\d{4,9}/\S+|10\.\d{4,9}/\S+))?", re.I)
_NUMERIC_OR_UNIT = re.compile(r"^\d+(?:[.,]\d+)?(?:\s*(?:%|ms|s|m|h|d|kg|g|cm|mm|px))?$", re.I)
_FILELIKE = re.compile(r"(?:^|[\\/])[^\\/]+\.(?:pdf|docx?|indd|txt)$", re.I)
_ARTIFACT = re.compile(r"^(?:n/?a|none|null|unknown|nan|page\s*\d+|figure\s*\d+|table\s*\d+)$", re.I)
_PAGE_RANGE = re.compile(r"^(?P<start>\d+)[–-](?P<end>\d+)$")
# Exact generic labels only: initial policy plus the136 two-agent agreements.
_GENERIC_METHOD_LABELS = frozenset({
    '3 item manipulation check',
    '62 semi struc tured interviews',
    'ar chival documents',
    'archival and interview data',
    'archival and live data',
    'archival data',
    'archival data and historical documents',
    'archival data and internal documents',
    'archival data collection',
    'archival document',
    'archival documentation',
    'archival documents',
    'archival documents and digital footprints',
    'archival information',
    'archival journals',
    'archival material',
    'archival materials',
    'archival records',
    'archival research',
    'archival texts',
    'attention check',
    'attention check questions',
    'attention checks',
    'cnbc interview',
    'company documentation',
    'comprehensive interviews',
    'court documents',
    'critical event recall interviews',
    'data collection time',
    'date and time of observation',
    'document',
    'document analysis',
    'documentary material',
    'documentary online material',
    'documentation',
    'documents',
    'documents and artifacts',
    'documents and social media',
    'ethnographic data',
    'ethnographic interviews',
    'ethnographic journal',
    'ethnographic observation',
    'ethnographic observations',
    'ethological observations',
    'expanded field notes',
    'exploratory pilot interviews',
    'face to face interviews',
    'field notes',
    'field notes and memos',
    'field observations',
    'fieldnotes',
    'first hand observations',
    'first round interviews',
    'focus group discussion',
    'focus group discussions (fgds)',
    'focus group interviews',
    'focus groups',
    'follow up interview protocol',
    'follow up interviews',
    'formal and informal observations',
    'formal interviews',
    'in depth interviews',
    'in depth, semistruc tured interviews',
    'in situ observations',
    'individual interviews',
    'infor mally interviewed members of the film crews',
    'informal interviews',
    'internal company documents',
    'internal documents',
    'internal documents, press articles and published case studies',
    'interview guide',
    'interview guidelines',
    'interview guides',
    'interview protocol',
    'interview protocols',
    'interview questions',
    'interview schedule',
    'interview transcripts',
    'interviews',
    'interviews and informal conversations',
    'key informant interviews',
    'manipulation and attention check',
    'manipulation and attention checks',
    'manipulation check',
    'manipulation check item',
    'manipulation check questions',
    'manipulation checks',
    'narrative interviews',
    'non participant observation',
    'non participant observations',
    'nonparticipant observation',
    'nonparticipant observations',
    'observation',
    'observation data',
    'observation forms',
    'observational data',
    'observational protocol',
    'observations',
    'observations at equilate',
    'observations of day to day operations',
    'observations of everyday activities, experiences and practices in the homes',
    'observations of talks and conferences',
    'observations of teachers interacting with various actors',
    'on site observations',
    'open ended interview protocol',
    'open ended semi structured interviews',
    'organiza tional documents and reports',
    'partici pant observations',
    'participant and non participant observation',
    'participant observation',
    'participant observations',
    'participatory observation data',
    'pilot interviews',
    'pre event interviews',
    'primary interviews',
    'publicly available documents',
    'qualitative data',
    'qualitative interviews',
    'rehearsal observation',
    'second round interviews',
    'second round of interviews',
    'secondary data',
    'secondary interviews',
    'self reported attention check',
    'sem istructured interviews',
    'semi struc tured follow up interviews',
    'semi structured depth interviews',
    'semi structured expert interviews',
    'semi structured in depth interviews',
    'semi structured interview',
    'semi structured interview data',
    'semi structured interview format',
    'semi structured interview guide',
    'semi structured interview protocol',
    'semi structured interview questions',
    'semi structured interview schedule',
    'semi structured interviews',
    'semi structured interviews and extended conversations',
    'semi structured, formal interviews',
    'semi structured, in depth interviews',
    'semi structured, open ended interviews',
    'semi structured, qualitative interviews',
    'semistruc tured interviews',
    'semistructured and narrative interviews',
    'semistructured individual interview',
    'semistructured interview',
    'semistructured interview guide',
    'semistructured interview protocol',
    'semistructured interviews',
    'semistructured telephone interviews',
    'semistructured, in depth interviews',
    'strategic planning documents',
    'structured interview',
    'structured interviews',
    'unstructured interviews',
    'virtual semi structured interviews',
    'walking interviews',
    'weekly follow up interviews',
    'written responses to the interview questions',
})


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _page(page: int, page_size: int) -> tuple[int, int]:
    if type(page) is not int or type(page_size) is not int or page < 1 or not 1 <= page_size <= MAX_PAGE_SIZE:
        raise ValueError("invalid_pagination")
    return page, page_size


def _qualified(record: dict[str, Any]) -> bool:
    """Require an explicit authority claim; no title/DOI inference is allowed."""
    if record.get("bibliographic_provenance") == "printed_source_verified":
        return True
    fields = record.get("field_provenance")
    required = {"authors", "year", "title_verbatim", "journal", "volume"}
    required.add("article_number" if record.get("record_type") == "journal_article_article_number" else "pages_verbatim")
    return isinstance(fields, dict) and required <= set(fields) and all(
        fields[field] == "printed_source_verified" for field in required
    )


def citation_authority_candidates(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Return source-only audit inputs without exposing or parsing raw literals.

    A caller may use the exact identifiers and controlled missing-field reasons
    to construct a separately reviewed ``printed_source_verified`` structured
    row.  This function does not infer any bibliographic value.
    """
    candidates: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        identifier = _text(record.get("reference_id")) or _text(record.get("work_id"))
        if identifier is None or _qualified(record):
            continue
        required = ("authors", "year", "title_verbatim", "journal", "volume",
                    "article_number" if record.get("record_type") == "journal_article_article_number" else "pages_verbatim")
        missing = [field for field in required if (
            _authors(record.get(field)) is None if field == "authors" else not _text(record.get(field))
        )]
        candidates.append({"id": identifier, "required_fields": list(required),
                           "missing_fields": missing,
                           "required_authority": "printed_source_verified"})
    return {"version": VERSION, "items": sorted(candidates, key=lambda item: item["id"]),
            "raw_graph_modified": False}


def _normalize_apa_author(value: object) -> str | None:
    """Mechanically punctuate recorded initials; never infer a given name."""
    if not isinstance(value, str) or "et al" in value.casefold():
        return None
    match = _APA_AUTHOR.fullmatch(value.strip())
    if match is None:
        return None
    initials = match["initials"].strip()
    compact = re.sub(r"\s+", "", initials)
    if not re.fullmatch(r"(?:[A-ZÀ-ÖØ-Þ]\.?)+(?:-(?:[A-ZÀ-ÖØ-Þ]\.?)+)*", compact):
        return None
    def group_render(group: str) -> str:
        return " ".join(letter + "." for letter in re.findall(r"[A-ZÀ-ÖØ-Þ]", group))
    rendered = "-".join(group_render(group) for group in compact.split("-"))
    return match["surname"].strip() + ", " + rendered + (match["suffix"] or "")


def parse_apa_authors(value: object) -> list[str] | None:
    """Parse a semicolon-delimited recorded author string without name inference."""
    if not isinstance(value, str) or not value.strip():
        return None
    authors = [_normalize_apa_author(part) for part in value.split(";")]
    return authors if authors and all(authors) else None


def _authors(value: object) -> list[str] | None:
    if isinstance(value, str):
        return parse_apa_authors(value)
    if not isinstance(value, list) or not value:
        return None
    authors = [_normalize_apa_author(item) for item in value]
    return authors if all(authors) else None


def _author_text(authors: list[str]) -> str:
    # Names must arrive already recorded in APA-ready form.  Only list joining
    # is performed here; no initials, casing, or name-order are inferred.
    if len(authors) == 1:
        return authors[0]
    if len(authors) == 2:
        return authors[0] + ", & " + authors[1]
    if len(authors) > 20:
        return ", ".join(authors[:19]) + ", … " + authors[-1]
    return ", ".join(authors[:-1]) + ", & " + authors[-1]


def apa_page_span(pages_verbatim: str) -> str | None:
    """Normalize a numeric range, refusing an ambiguous decreasing shorthand."""
    match = _PAGE_RANGE.fullmatch(pages_verbatim)
    if match is None:
        return pages_verbatim
    start, end = match["start"], match["end"]
    expanded_end = end if len(end) >= len(start) else start[:len(start) - len(end)] + end
    if int(expanded_end) < int(start):
        return None
    return start + "–" + expanded_end


def format_apa_citation(record: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return a complete APA-style citation or a controlled exclusion reason.

    The record must be an explicitly verified, structured journal article.
    Missing data is never filled from raw reference text, graph work metadata,
    filename-like text, or a DOI.
    """
    if not isinstance(record, dict):
        return None, "invalid_record"
    if not _qualified(record):
        return None, "unverified_bibliographic_provenance"
    record_type = record.get("record_type", "journal_article")
    if record_type not in {"journal_article", "journal_article_article_number"}:
        return None, "unsupported_record_type"
    raw_authors = record.get("authors")
    if isinstance(raw_authors, list) and any(isinstance(item, str) and "et al" in item.casefold() for item in raw_authors):
        return None, "incomplete_authors"
    authors = _authors(raw_authors)
    if authors is None:
        return None, "missing_or_non_apa_authors"
    raw_year = record.get("year")
    year = str(raw_year) if type(raw_year) is int else _text(raw_year)
    title_verbatim, title_apa, journal, volume = (_text(record.get(field)) for field in ("title_verbatim", "title_apa", "journal", "volume"))
    pages_verbatim, article_number = _text(record.get("pages_verbatim")), _text(record.get("article_number"))
    if year is None or not _YEAR.fullmatch(year):
        return None, "missing_or_invalid_year"
    for value, reason in ((title_verbatim, "missing_title_verbatim"), (title_apa, "missing_curated_apa_title"),
                          (journal, "missing_journal"), (volume, "missing_volume")):
        if value is None:
            return None, reason
    if record_type == "journal_article" and pages_verbatim is None:
        return None, "missing_pages"
    if record_type == "journal_article_article_number" and article_number is None:
        return None, "missing_article_number"
    issue = _text(record.get("issue"))
    doi = _text(record.get("doi"))
    if doi is not None and (not _DOI.fullmatch(doi) or _DOI_CONTAMINATION.search(doi)):
        return None, "invalid_doi"
    display_pages = apa_page_span(pages_verbatim) if pages_verbatim is not None else article_number
    if display_pages is None:
        return None, "ambiguous_or_decreasing_page_range"
    locator = ("Article " + display_pages if record_type == "journal_article_article_number" else display_pages)
    location = volume + ("(" + issue + ")" if issue else "") + ", " + locator + "."
    title_stop = "" if title_apa[-1] in ".?!" else "."
    citation = f"{_author_text(authors)} ({year}). {title_apa}{title_stop} {journal}, {location}"
    if doi:
        citation += " " + (doi if doi.casefold().startswith("https://doi.org/") else "https://doi.org/" + doi)
    return citation, None


def citation_render_parts(record: dict[str, Any]) -> dict[str, str] | None:
    """Return semantic rendering parts only for an already APA-ready record."""
    citation, _ = format_apa_citation(record)
    if citation is None:
        return None
    title = _text(record.get("title_apa")) or ""
    return {"citation_apa": citation, "authors_year": _author_text(_authors(record["authors"]) or []) + " (" + str(record["year"]) + ").",
            "title": title, "title_verbatim": _text(record.get("title_verbatim")) or "",
            "title_stop": "" if title[-1] in ".?!" else ".",
            "journal": _text(record.get("journal")) or "", "volume": _text(record.get("volume")) or "",
            "issue": _text(record.get("issue")) or "", "locator": ("Article " + _text(record.get("article_number")) if record.get("record_type") == "journal_article_article_number" else apa_page_span(_text(record.get("pages_verbatim")) or "")),
            "doi_url": _text(record.get("doi")) or ""}


def parse_printed_reference_candidate(text: str) -> dict[str, Any] | None:
    """Mechanically parse one printed-reference candidate; it remains unaudited."""
    if not isinstance(text, str):
        return None
    match = _PRINTED_REFERENCE.search(text)
    if match is None:
        return None
    values = match.groupdict(); doi = values.get("doi")
    if doi and (not _DOI.fullmatch(doi) or _DOI_CONTAMINATION.search(doi)):
        return None
    return {"span_start": match.start(), "span_end": match.end(), "authors_raw": values["authors"].strip(),
            "year": values["year"], "title": values["title"].strip(), "journal": values["journal"].strip(),
            "volume": values["volume"], "issue": values.get("issue"), "pages": values["pages"], "doi": doi,
            "needs_independent_audit": True, "bibliographic_provenance": "candidate_printed_span_only"}


def project_apa_citations(records: Iterable[dict[str, Any]], *, page: int = 1,
                          page_size: int = 25) -> dict[str, Any]:
    """Project only APA-ready citations; report exclusions without exposing them."""
    page, page_size = _page(page, page_size)
    items, deficits = [], Counter()
    for record in records:
        citation, reason = format_apa_citation(record)
        if citation is None:
            deficits[reason or "invalid_record"] += 1
            continue
        identifier = _text(record.get("reference_id")) or _text(record.get("work_id"))
        if identifier is None:
            deficits["missing_identity"] += 1
            continue
        items.append({"id": identifier, "citation_apa": citation,
                      "render_parts": citation_render_parts(record)})
    items.sort(key=lambda item: (item["citation_apa"].casefold(), item["id"]))
    start = (page - 1) * page_size
    return {"version": VERSION, "items": items[start:start + page_size], "total": len(items),
            "page": page, "page_size": page_size, "deficits": dict(sorted(deficits.items())),
            "raw_graph_modified": False}


def is_displayable_measure_label(value: object) -> bool:
    """Conservatively refuse obvious symbols, units, placeholders and filenames."""
    label = _text(value)
    if label is None:
        return False
    compact = " ".join(label.split())
    if len(compact) < 2 or not any(ch.isalpha() for ch in compact):
        return False
    return not (_NUMERIC_OR_UNIT.fullmatch(compact) or _FILELIKE.search(compact) or _ARTIFACT.fullmatch(compact))


def generic_method_display_labels() -> list[str]:
    """One exact exclusion vocabulary shared by the API and generated client."""
    return sorted(_GENERIC_METHOD_LABELS)


def is_generic_method_label(value: object) -> bool:
    """Reject exact generic method labels, never a substring of a measure name."""
    label = _text(value)
    return bool(label) and " ".join(re.sub(r"[-–—\u00ad]", " ", label).split()).casefold() in _GENERIC_METHOD_LABELS


def _measure_terms(version: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    for key in ("name", "construct"):
        value = version.get(key)
        if isinstance(value, str):
            terms.append(value)
        elif isinstance(value, list):
            terms.extend(item for item in value if isinstance(item, str))
    return terms


def project_measure_discovery(versions: Iterable[dict[str, Any]], *, page: int = 1,
                              page_size: int = 50) -> dict[str, Any]:
    """Group measure/construct names by distinct source article, never by rows."""
    page, page_size = _page(page, page_size)
    groups: dict[str, dict[str, Any]] = {}
    review = Counter()
    for version in versions:
        if not isinstance(version, dict):
            review["invalid_version"] += 1
            continue
        paper = version.get("paper")
        paper_id = paper.get("paper_id") if isinstance(paper, dict) else version.get("paper_id")
        if not isinstance(paper_id, str) or not paper_id:
            review["missing_paper_id"] += 1
            continue
        for term in _measure_terms(version):
            if is_generic_method_label(term):
                review["suppressed_generic_method_label"] += 1
                continue
            if not is_displayable_measure_label(term):
                review["suppressed_nonmeasure_label"] += 1
                continue
            label = " ".join(term.split())
            key = label.casefold()
            group = groups.setdefault(key, {"label": label, "paper_ids": set()})
            group["paper_ids"].add(paper_id)
    items = [{"label": group["label"], "distinct_article_count": len(group["paper_ids"]),
              "paper_ids": sorted(group["paper_ids"])} for group in groups.values()]
    items.sort(key=lambda item: (-item["distinct_article_count"], item["label"].casefold()))
    start = (page - 1) * page_size
    return {"version": VERSION, "items": items[start:start + page_size], "total": len(items),
            "page": page, "page_size": page_size, "review": dict(sorted(review.items()))}
