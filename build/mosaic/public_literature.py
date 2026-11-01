"""Portable, opt-in literature drafting over a reviewed public display bundle.

This is deliberately downstream of extraction and display review.  It accepts
only the default passage projection and complete bibliographic identities that
the public server has already hash-checked.  Preview is local and deterministic;
generation is a separately enabled, bounded OpenAI Responses request.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import secrets
from typing import Any, Callable, Mapping
import urllib.error
import urllib.request
import uuid

from mosaic.adapted.k.env_loader import require_secret
from mosaic.adapted.k.kb_writer_lock import KBWriterLockHeld, acquire, release
from mosaic.workspace import Workspace

VERSION = "mosaic-public-literature/v1"
MODEL = "gpt-6-sol"
MAX_OUTPUT_TOKENS = 3_000
MAX_SOURCES = 12
MAX_PROVIDER_REQUEST_BYTES = 240_000
SERVICE_TIER = "default"
# Conservative declared bounds for the selected model profile.  An operator may
# raise these when their account/model terms differ; lowering them is refused.
# Checked against the official OpenAI pricing page on 2026-10-31.
DEFAULT_INPUT_UPPER_USD_PER_MILLION = 6.0
DEFAULT_OUTPUT_UPPER_USD_PER_MILLION = 20.0


class PublicLiteratureRefusal(ValueError):
    """A safe refusal; callers can keep ordinary source browsing available."""


@dataclass(frozen=True)
class PublicLiteratureConfig:
    """Installation-owned generation controls.  Generation is off by default."""
    enabled: bool = False
    cap_usd: float = 20.0
    ledger_path: str = "public-ledgers/literature_assistant_calls.jsonl"
    model: str = MODEL
    input_upper_usd_per_million: float = DEFAULT_INPUT_UPPER_USD_PER_MILLION
    output_upper_usd_per_million: float = DEFAULT_OUTPUT_UPPER_USD_PER_MILLION

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool or self.model != MODEL:
            raise PublicLiteratureRefusal("public_literature_config_invalid")
        if not isinstance(self.cap_usd, (int, float)) or not 0 < float(self.cap_usd) <= 200:
            raise PublicLiteratureRefusal("public_literature_budget_invalid")
        for value, floor in ((self.input_upper_usd_per_million, DEFAULT_INPUT_UPPER_USD_PER_MILLION),
                             (self.output_upper_usd_per_million, DEFAULT_OUTPUT_UPPER_USD_PER_MILLION)):
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < floor:
                raise PublicLiteratureRefusal("public_literature_rate_bound_invalid")
        candidate = Path(self.ledger_path)
        if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
            raise PublicLiteratureRefusal("public_literature_ledger_path_invalid")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _request(body: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {"question", "search_terms", "paper", "limit", "output_type", "history", "model", "enable_model"}
    if not isinstance(body, Mapping) or set(body) - allowed:
        raise PublicLiteratureRefusal("public_literature_request_invalid")
    question = body.get("question", "")
    terms = body.get("search_terms", "")
    paper = body.get("paper", "")
    if not isinstance(question, str) or not question.strip() or len(question) > 4_000:
        raise PublicLiteratureRefusal("public_literature_question_invalid")
    if not isinstance(terms, str) or len(terms) > 400 or not isinstance(paper, str) or len(paper) > 180:
        raise PublicLiteratureRefusal("public_literature_request_invalid")
    limit = body.get("limit", MAX_SOURCES)
    if type(limit) is not int or not 1 <= limit <= MAX_SOURCES:
        raise PublicLiteratureRefusal("public_literature_limit_invalid")
    output_type = body.get("output_type", "literature_review")
    if output_type not in {"answer", "literature_review", "comparison", "research_outline"}:
        raise PublicLiteratureRefusal("public_literature_output_type_invalid")
    history = body.get("history", [])
    if not isinstance(history, list) or len(history) > 6 or any(
        not isinstance(item, Mapping) or set(item) != {"question", "answer"} or
        any(not isinstance(item[key], str) or len(item[key]) > 6_000 for key in item)
        for item in history
    ):
        raise PublicLiteratureRefusal("public_literature_history_invalid")
    if body.get("model", MODEL) != MODEL or body.get("enable_model", False) not in (True, False):
        raise PublicLiteratureRefusal("public_literature_model_invalid")
    return {"question": question.strip(), "search_terms": terms.strip(), "paper": paper,
            "limit": limit, "output_type": output_type, "history": list(history),
            "enable_model": body.get("enable_model", False)}


def _citation_maps(bibliography: Mapping[str, Any]) -> tuple[dict[str, dict], dict[str, str]]:
    paper_items = bibliography.get("paper_items", [])
    apa_items = bibliography.get("items", [])
    if not isinstance(paper_items, list) or not isinstance(apa_items, list):
        raise PublicLiteratureRefusal("public_literature_bibliography_invalid")
    papers = {}
    for row in paper_items:
        if not isinstance(row, Mapping) or not isinstance(row.get("paper_id"), str):
            continue
        title = row.get("title_verbatim", row.get("title"))
        authors = row.get("authors_verbatim", row.get("authors"))
        journal = row.get("journal_verbatim", row.get("journal"))
        year = row.get("publication_year", row.get("year"))
        if isinstance(authors, str):
            authors = [authors]
        if isinstance(title, str) and isinstance(authors, list) and isinstance(journal, str) and isinstance(year, int):
            papers[row["paper_id"]] = {"paper_id": row["paper_id"], "title_verbatim": title,
                                       "authors_verbatim": authors, "journal_verbatim": journal,
                                       "publication_year": year}
    apa = {row.get("paper_id"): row.get("citation_apa") for row in apa_items if isinstance(row, Mapping)
           and isinstance(row.get("paper_id"), str) and isinstance(row.get("citation_apa"), str)
           and row["citation_apa"].strip()}
    # A complete identity and an exact projected APA entry are both required.
    return papers, {key: value for key, value in apa.items() if key in papers}


def _parenthetical(identity: Mapping[str, Any], apa: str) -> str:
    """Render author-year from the admitted APA row, never a byline guess."""
    year = str(identity["publication_year"])
    match = re.match(r"\s*(.+?)\s*\(" + re.escape(year) + r"\)", apa)
    if not match:
        raise PublicLiteratureRefusal("public_literature_apa_invalid")
    author_part = match.group(1)
    families = re.findall(r"(?:^|&|,)\s*([^,&.]+),\s*(?:[A-Z][^,]*|\w)", author_part)
    if not families:
        raise PublicLiteratureRefusal("public_literature_apa_invalid")
    label = families[0].strip() if len(families) == 1 else (families[0].strip() + " & " + families[1].strip() if len(families) == 2 else families[0].strip() + " et al.")
    return f"{label}, {year}"


def _quote(row: Mapping[str, Any]) -> str:
    value = row.get("quote", row.get("text", ""))
    if not isinstance(value, str) or not value.strip() or len(value) > 20_000:
        raise PublicLiteratureRefusal("public_literature_passage_invalid")
    return value


def _intact_for_synthesis(text: str) -> bool:
    """Withhold a known damaged-statistic pattern instead of repairing it."""
    return re.search(r"\b(?:r|p|b|t|F|z)\s+5\s*[+−-]?(?:\.\d|\d)|\bp\s*[,\.]\s*\.\d", text) is None


def _passage_id(row: Mapping[str, Any]) -> str:
    value = row.get("id", row.get("passage_id", row.get("unit_id")))
    if not isinstance(value, str) or not value:
        raise PublicLiteratureRefusal("public_literature_passage_id_invalid")
    return value


def _schema(ids: list[str], source_ids: list[str] | None = None) -> dict[str, Any]:
    source_ids = source_ids if source_ids is not None else ids
    support = {"type": "object", "additionalProperties": False,
               "properties": {"source_id": {"type": "string", "enum": source_ids},
                              "quote": {"type": "string", "minLength": 20, "maxLength": 1000}},
               "required": ["source_id", "quote"]}
    paragraph = {"type": "object", "additionalProperties": False,
                 "properties": {"text": {"type": "string"},
                                "kind": {"type": "string", "enum": ["source_summary", "research_suggestion"]},
                                "citations": {"type": "array", "items": {"type": "string", "enum": ids}},
                                "support": {"type": "array", "items": support}},
                 "required": ["text", "kind", "citations", "support"]}
    return {"type": "object", "additionalProperties": False,
            "properties": {"title": {"type": "string"}, "sections": {"type": "array", "maxItems": 8,
                           "items": {"type": "object", "additionalProperties": False,
                                     "properties": {"heading": {"type": "string"}, "paragraphs": {"type": "array", "maxItems": 6, "items": paragraph}},
                                     "required": ["heading", "paragraphs"]}},
                           "gaps": {"type": "array", "maxItems": 8, "items": {"type": "string"}}},
            "required": ["title", "sections", "gaps"]}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001,D401
        return None


def _transport(request: Mapping[str, Any]) -> dict[str, Any]:
    """The only real egress path: fixed API origin, no redirects, no retries."""
    key = require_secret("OPENAI_API_KEY")
    payload = json.dumps(request, ensure_ascii=False).encode("utf-8")
    call = urllib.request.Request("https://api.openai.com/v1/responses", data=payload, method="POST",
                                  headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        with opener.open(call, timeout=55) as response:
            if response.geturl() != "https://api.openai.com/v1/responses":
                raise RuntimeError("provider_redirect_refused")
            return json.loads(response.read(2_000_000))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"OpenAI returned HTTP {exc.code}; no automatic retry was made.") from None
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError("The provider request did not finish; no automatic retry was made.") from None


class PublicLiteratureAssistant:
    """Retrieval and drafting facade intended for the public local server."""
    def __init__(self, *, passages: list[Mapping[str, Any]], bibliography: Mapping[str, Any], snapshot_id: str,
                 workspace: Workspace | None = None, config: PublicLiteratureConfig | Mapping[str, Any] | None = None,
                 transport: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None) -> None:
        if not isinstance(passages, list) or not isinstance(snapshot_id, str) or not snapshot_id:
            raise PublicLiteratureRefusal("public_literature_inputs_invalid")
        self.identities, self.apa = _citation_maps(bibliography)
        self.passages = []
        for row in passages:
            if not isinstance(row, Mapping) or row.get("paper_id") not in self.apa:
                continue
            lane = row.get("lane")
            if lane == "references":
                continue
            item = dict(row)
            item["id"] = _passage_id(item)
            item["quote"] = _quote(item)
            if not _intact_for_synthesis(item["quote"]):
                continue
            if not isinstance(item.get("page"), int) or item["page"] < 1:
                raise PublicLiteratureRefusal("public_literature_page_invalid")
            self.passages.append(item)
        self.snapshot_id = snapshot_id
        self.workspace = workspace or Workspace()
        if config is None:
            self.config = PublicLiteratureConfig()
        elif isinstance(config, PublicLiteratureConfig):
            self.config = config
        elif isinstance(config, Mapping):
            try:
                self.config = PublicLiteratureConfig(**dict(config))
            except TypeError as exc:
                raise PublicLiteratureRefusal("public_literature_config_invalid") from exc
        else:
            raise PublicLiteratureRefusal("public_literature_config_invalid")
        self.transport = transport or _transport
        self.csrf_token = secrets.token_urlsafe(32)

    def status(self) -> dict[str, Any]:
        return {"enabled": True, "generation_configured": self.config.enabled, "csrf_token": self.csrf_token,
                "models": [{"id": MODEL, "label": "GPT-6 Sol", "available": self.config.enabled}],
                "scope": "reviewed_display_passages_only", "snapshot_id": self.snapshot_id,
                "budget": self._budget()}

    def _budget(self) -> dict[str, Any]:
        path = self.workspace.check(self.config.ledger_path)
        amounts: dict[str, float] = {}
        terminal: dict[str, str] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if not isinstance(row, Mapping) or row.get("event") not in {"reserved", "completed", "failed"}:
                    raise PublicLiteratureRefusal("public_literature_ledger_invalid")
                attempt = row.get("attempt_id")
                if not isinstance(attempt, str):
                    raise PublicLiteratureRefusal("public_literature_ledger_invalid")
                if row["event"] == "reserved":
                    if attempt in amounts or not isinstance(row.get("reserved_usd"), (int, float)) or not math.isfinite(float(row["reserved_usd"])) or float(row["reserved_usd"]) < 0:
                        raise PublicLiteratureRefusal("public_literature_ledger_invalid")
                    amounts[attempt] = float(row["reserved_usd"])
                elif row["event"] == "completed":
                    if attempt in terminal or attempt not in amounts or not isinstance(row.get("cost_upper_usd"), (int, float)) or not math.isfinite(float(row["cost_upper_usd"])) or not 0 <= float(row["cost_upper_usd"]) <= amounts[attempt]:
                        raise PublicLiteratureRefusal("public_literature_ledger_invalid")
                    terminal[attempt] = "completed"
                    amounts[attempt] = float(row["cost_upper_usd"])
                elif row["event"] == "failed":
                    if attempt not in amounts:
                        raise PublicLiteratureRefusal("public_literature_ledger_invalid")
                    # Older DE-62 records can report a local parsing failure
                    # after the provider completed and its cost was recorded.
                    # Permit that one historical terminal suffix only; the
                    # completed cost stays authoritative.  Every other second
                    # terminal event, including failed-then-completed, refuses.
                    if attempt not in terminal:
                        terminal[attempt] = "failed"
                    elif terminal[attempt] == "completed":
                        terminal[attempt] = "completed_then_failed"
                    else:
                        raise PublicLiteratureRefusal("public_literature_ledger_invalid")
        used = sum(amounts.values())
        return {"cap_usd": self.config.cap_usd, "accounted_upper_usd": round(used, 8),
                "remaining_usd": round(max(0.0, self.config.cap_usd - used), 8), "attempts": len(amounts)}

    def preview(self, body: Mapping[str, Any]) -> dict[str, Any]:
        request = _request(body)
        stop = {"about", "also", "among", "answer", "article", "articles", "compare", "does", "find", "from", "how", "literature", "mosaic", "outline", "paper", "papers", "please", "reported", "research", "review", "summarize", "summary", "that", "their", "these", "this", "what", "which", "with", "would", "write"}
        terms = [term for term in re.findall(r"[\w-]{3,}", (request["search_terms"] or request["question"]).casefold()) if term not in stop]
        candidates = []
        for order, row in enumerate(self.passages):
            if request["paper"] and row["paper_id"] != request["paper"]:
                continue
            haystack = (str(row.get("title", "")) + " " + row["quote"]).casefold()
            score = sum(term in haystack for term in terms)
            if not terms or not score:
                continue
            candidates.append((-score, order, row))
        selected, seen, per_paper = [], set(), {}
        for _score, _order, row in sorted(candidates):
            key = (row["paper_id"], row["id"])
            if key in seen or per_paper.get(row["paper_id"], 0) >= 3:
                continue
            seen.add(key)
            per_paper[row["paper_id"]] = per_paper.get(row["paper_id"], 0) + 1
            identity, apa = self.identities[row["paper_id"]], self.apa[row["paper_id"]]
            selected.append({"citation_id": "S" + str(len(selected) + 1), "source_id": row["id"],
                             "paper_id": row["paper_id"], "title": identity["title_verbatim"],
                             "citation": apa, "parenthetical": _parenthetical(identity, apa),
                             "citation_complete_apa": True, "page": row["page"], "quote": row["quote"],
                             "quote_sha256": sha256(row["quote"].encode("utf-8")).hexdigest(),
                             "article_url": "studies.html?paper=" + row["paper_id"]})
            if len(selected) >= request["limit"]:
                break
        if not selected:
            return {"status": "no_sources", "message": "No reviewed source passages matched. No provider call was made.",
                    "sources": [], "retrieved_papers": 0, "snapshot_id": self.snapshot_id}
        return {"status": "ok", "sources": selected, "retrieved_papers": len({x["paper_id"] for x in selected}),
                "snapshot_id": self.snapshot_id, "scientifically_accepted": False}

    def _provider_request(self, request: Mapping[str, Any], sources: list[Mapping[str, Any]]) -> dict[str, Any]:
        ids = [item["citation_id"] for item in sources]
        instructions = ("You are MOSAIC's literature assistant. Use only the supplied source excerpts. "
            "Never use outside knowledge or invent citations, results, statistics, scale items, or quotations. "
            "Every source_summary needs cited source IDs and literal support quotes. A research_suggestion is not a finding. "
            "Do not write author-year citations in prose; the interface renders only the supplied citation IDs. "
            "Say what is missing rather than claim completeness. This is query-time assistance, never extraction or validation.")
        evidence = [{key: item[key] for key in ("citation_id", "source_id", "title", "citation", "page", "quote")}
                    for item in sources]
        return {"model": MODEL, "store": False, "service_tier": SERVICE_TIER, "max_output_tokens": MAX_OUTPUT_TOKENS,
                "reasoning": {"effort": "low"}, "instructions": instructions,
                "input": json.dumps({"question": request["question"], "requested_output": request["output_type"],
                                     "conversation": request["history"], "source_excerpts": evidence}, ensure_ascii=False),
                "text": {"format": {"type": "json_schema", "name": "mosaic_public_literature", "strict": True,
                                     "schema": _schema(ids, [item["source_id"] for item in sources])}}}

    def draft(self, body: Mapping[str, Any]) -> dict[str, Any]:
        request = _request(body)
        if not request["enable_model"] or not self.config.enabled:
            raise PublicLiteratureRefusal("public_literature_generation_not_enabled")
        preview = self.preview(request)
        if preview["status"] != "ok":
            return preview
        provider_request = self._provider_request(request, preview["sources"])
        raw = json.dumps(provider_request, ensure_ascii=False).encode("utf-8")
        if len(raw) > MAX_PROVIDER_REQUEST_BYTES:
            raise PublicLiteratureRefusal("public_literature_request_too_large")
        reservation = round(((len(raw) + 2048) * self.config.input_upper_usd_per_million + MAX_OUTPUT_TOKENS * self.config.output_upper_usd_per_million) / 1_000_000, 8)
        ledger = self.workspace.check(self.config.ledger_path)
        try:
            lock = acquire(ledger, self.workspace, "MOSAIC public literature assistant")
        except KBWriterLockHeld:
            raise PublicLiteratureRefusal("public_literature_request_in_progress") from None
        attempt = uuid.uuid4().hex
        try:
            if self._budget()["accounted_upper_usd"] + reservation > self.config.cap_usd:
                raise PublicLiteratureRefusal("public_literature_budget_exhausted")
            def event(kind: str, **fields: Any) -> None:
                self.workspace.append_line(self.config.ledger_path, json.dumps({"attempt_id": attempt, "event": kind, "time": _now(), **fields}, separators=(",", ":")))
            event("reserved", reserved_usd=reservation, model=MODEL, request_sha256=sha256(raw).hexdigest(),
                  source_snapshot_id=self.snapshot_id, source_count=len(preview["sources"]))
            try:
                response = self.transport(provider_request)
                usage = response.get("usage", {})
                input_tokens, output_tokens = usage.get("input_tokens"), usage.get("output_tokens")
                if type(input_tokens) is not int or type(output_tokens) is not int or min(input_tokens, output_tokens) < 0:
                    raise PublicLiteratureRefusal("public_literature_usage_invalid")
                cost = round((input_tokens * self.config.input_upper_usd_per_million + output_tokens * self.config.output_upper_usd_per_million) / 1_000_000, 8)
                if cost > reservation:
                    raise PublicLiteratureRefusal("public_literature_usage_exceeds_reservation")
                draft = self._parse(response, preview["sources"])
                event("completed", cost_upper_usd=cost, input_tokens=input_tokens, output_tokens=output_tokens,
                      response_id=response.get("id"), model=response.get("model", MODEL))
            except Exception as exc:
                event("failed", error_class=type(exc).__name__)
                raise
            return {"status": "ok", "draft": draft, "sources": preview["sources"], "retrieved_papers": preview["retrieved_papers"],
                    "snapshot_id": self.snapshot_id, "model": response.get("model", MODEL), "generated_at": _now(),
                    "attempt_id": attempt, "cost_upper_usd": cost, "budget": self._budget(),
                    "span_binding_verified": True, "scientifically_accepted": False}
        finally:
            release(lock)

    @staticmethod
    def _parse(response: Mapping[str, Any], sources: list[Mapping[str, Any]]) -> dict[str, Any]:
        chunks = [item["text"] for message in response.get("output", []) if message.get("type") == "message"
                  for item in message.get("content", []) if item.get("type") == "output_text" and isinstance(item.get("text"), str)]
        if response.get("status") != "completed" or len(chunks) != 1:
            raise PublicLiteratureRefusal("public_literature_provider_response_invalid")
        draft = json.loads(chunks[0])
        from jsonschema import Draft202012Validator
        ids = [item["citation_id"] for item in sources]
        Draft202012Validator(_schema(ids, [item["source_id"] for item in sources])).validate(draft)
        by_id = {item["citation_id"]: item for item in sources}
        by_source_id = {item["source_id"]: item for item in sources}
        for section in draft["sections"]:
            for paragraph in section["paragraphs"]:
                if paragraph["kind"] == "source_summary" and (not paragraph["citations"] or not paragraph["support"]):
                    raise PublicLiteratureRefusal("public_literature_uncited_summary")
                if re.search(r"\([A-Za-z][^)]{0,80}(?:19|20)\d{2}[^)]{0,80}\)", paragraph["text"]):
                    raise PublicLiteratureRefusal("public_literature_freeform_citation")
                supported_citations = set()
                for support in paragraph["support"]:
                    matched = by_source_id.get(support["source_id"])
                    if (matched is None or len(support["quote"]) < 20 or support["quote"] not in matched["quote"]
                            or support["quote"] != support["quote"].strip()):
                        raise PublicLiteratureRefusal("public_literature_support_not_exact")
                    supported_citations.add(matched["citation_id"])
                if any(item not in by_id for item in paragraph["citations"]):
                    raise PublicLiteratureRefusal("public_literature_citation_invalid")
                if not supported_citations.issubset(set(paragraph["citations"])):
                    raise PublicLiteratureRefusal("public_literature_support_uncited")
                if not set(paragraph["citations"]).issubset(supported_citations):
                    raise PublicLiteratureRefusal("public_literature_citation_unsupported")
        return draft
