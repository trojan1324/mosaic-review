"""HTTP client for an OpenAI-compatible vLLM server (MODEL_SELECTION_PLAN.md S0; VS1-VS4, VS6-VS8;
VS5 packet_hash_mismatch is not implemented, D-047).

Mirrors `mosaic/jev/client.py`'s pattern: intent row before send, result row
after receipt, both required for the call to proceed; totals recomputed from
the ledger file on every check, never a process counter; startup
`reconcile()` closes orphaned intents as `unknown_outcome`; idempotency key
from canonical request bytes; one retry only on a fixed retryable-status set.

Nothing here opens a socket -- to a mock server or otherwise -- unless
`live=True` is set (VS6). Eric declined a live-switch acknowledgement token
(2026-09-24); `--live` is the only switch. A reachable server changes
nothing about that gate.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import time
import urllib.parse
from typing import Any, Protocol

from mosaic.canonical import canonical_bytes
from mosaic.vllm.ledger import VLLMBudget, VLLMHardStop, VLLMLedger
from mosaic.vllm.scope_contract import VLLMScopeContract
from mosaic.vllm.streaming import StreamDecodeError, decode_chat_stream
from mosaic.vllm.wire import chat_request_bytes

RETRYABLE = {429, 500, 502, 503, 504}
MAX_RETRY_WAIT_S = 30.0


class TransportError(RuntimeError):
    def __init__(self, message: str, sent: bool):
        super().__init__(message)
        self.sent = sent


class UnknownOutcome(RuntimeError):
    """The request may have reached the server; it is never retried (VS4)."""


class VLLMHTTPError(RuntimeError):
    def __init__(self, status: int, body: Any):
        super().__init__(f"vLLM server returned HTTP {status}: {str(body)[:300]}")
        self.status, self.body = status, body


class Transport(Protocol):
    def request(self, method: str, url: str, headers: dict[str, str], body: bytes | None,
                timeout: float) -> tuple[int, dict[str, str], bytes]: ...


class HttpTransport:
    """A failure while connecting sent nothing (`sent=False`); any failure after the request
    started is an unknown outcome (`sent=True`)."""

    def request(self, method, url, headers, body, timeout):
        parts = urllib.parse.urlsplit(url)
        cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
        conn = cls(parts.hostname, parts.port, timeout=timeout)
        try:
            try:
                conn.connect()
            except (OSError, http.client.HTTPException) as e:
                raise TransportError(f"connection not established: {type(e).__name__}", sent=False) from None
            try:
                conn.request(method, parts.path + (f"?{parts.query}" if parts.query else ""), body=body, headers=headers)
                resp = conn.getresponse()
                data = resp.read()
                return resp.status, {k.lower(): v for k, v in resp.getheaders()}, data
            except (OSError, http.client.HTTPException) as e:
                raise TransportError(f"failed after connecting: {type(e).__name__}", sent=True) from None
        finally:
            conn.close()


def packet_sha256(packet: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(packet)).hexdigest()


class VLLMClient:
    def __init__(self, *, candidate: str = "unknown", base_url: str | None = None, base_url_env: str | None = None,
                 ledger: VLLMLedger | None = None, scope: VLLMScopeContract | None = None, budget: VLLMBudget | None = None,
                 transport: Transport | None = None, timeout: float = 60.0, live: bool = False,
                 api_key: str | None = None, sleep=time.sleep, stream: bool = False):
        self.candidate = candidate
        self.ledger = ledger or VLLMLedger(candidate)
        if scope is None:
            # The private default refuses non-screen sources through the
            # sealed-set registry.  Public callers must supply their own
            # explicit scope and therefore never load that registry.
            from mosaic.vllm.scope import VLLMScope
            scope = VLLMScope()
        self.scope = scope
        self.budget = budget or VLLMBudget.load()
        resolved = base_url or (os.environ.get(base_url_env) if base_url_env else None)
        self.base_url = (resolved or "http://127.0.0.1:0").rstrip("/")
        # OpenAI chat routes live under /v1; vLLM utility routes do not.
        self.server_url = self.base_url.removesuffix("/v1")
        self.transport = transport or HttpTransport()
        self.timeout = timeout
        self.live = live
        self._api_key = api_key
        self.sleep = sleep
        self.stream = stream
        self.reconciled = self.ledger.reconcile()

    def __repr__(self) -> str:  # never shows the key
        return f"VLLMClient(candidate={self.candidate!r}, base_url={self.base_url!r}, live={self.live})"

    # ---------------------------------------------------------------------------------- VS6
    def _check_live(self) -> None:
        """VS6: `--live` is the only switch; without it, no socket opens, even against a
        reachable server. Eric declined a live-switch acknowledgement token (2026-09-24)."""
        if not self.live:
            raise VLLMHardStop("not_covered_without_live", "socket refused: --live was not set")

    @staticmethod
    def idempotency_key(route: str, work_id: str, batch_id: str | None, request_sha256: str) -> str:
        seed = canonical_bytes({"route": route, "work_id": work_id, "batch_id": batch_id, "request": request_sha256})
        return "idem-" + hashlib.sha256(seed).hexdigest()

    def _key_header(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    # --------------------------------------------------------------------------------- calls
    def health(self) -> dict[str, Any]:
        self._check_live()
        request_sha = hashlib.sha256(b"GET /health").hexdigest()
        key = self.idempotency_key("health", "(none)", None, request_sha)
        meta = {"route": "health", "method": "GET", "request_sha256": request_sha, "packet_sha256": request_sha,
                "target_work_id": "(none)", "text_provenance": "none"}
        return self._execute(key, f"{self.server_url}/health", "GET", None, meta)

    def tokenize(self, text: str) -> int:
        self._check_live()
        body = canonical_bytes({"prompt": text})
        request_sha = hashlib.sha256(body).hexdigest()
        key = self.idempotency_key("tokenize", "(none)", None, request_sha)
        meta = {"route": "tokenize", "method": "POST", "request_sha256": request_sha, "packet_sha256": request_sha,
                "target_work_id": "(none)", "text_provenance": "none"}
        out = self._execute(key, f"{self.server_url}/tokenize", "POST", body, meta)
        return int(out.get("count") or len(out.get("tokens") or []))

    def chat_completion(self, *, work_id: str, batch_id: str, packet: dict[str, Any], request_body: dict[str, Any],
                        source_sha256s: list[str], text_provenance: str, remote: bool, rights: list[str] | None,
                        candidate_ids_offered: list[str] | None = None) -> dict[str, Any]:
        return self._chat(work_id=work_id, batch_id=batch_id, packet=packet, request_body=request_body,
                          source_sha256s=source_sha256s, text_provenance=text_provenance, remote=remote, rights=rights,
                          route="chat_completions", is_repair=False, repair_of=None,
                          candidate_ids_offered=candidate_ids_offered)

    def repair(self, *, work_id: str, batch_id: str, packet: dict[str, Any], request_body: dict[str, Any],
              source_sha256s: list[str], text_provenance: str, remote: bool, rights: list[str] | None,
              repair_of: str, candidate_ids_offered: list[str] | None = None) -> dict[str, Any]:
        return self._chat(work_id=work_id, batch_id=batch_id, packet=packet, request_body=request_body,
                          source_sha256s=source_sha256s, text_provenance=text_provenance, remote=remote, rights=rights,
                          route="repair", is_repair=True, repair_of=repair_of,
                          candidate_ids_offered=candidate_ids_offered)

    def _chat(self, *, work_id, batch_id, packet, request_body, source_sha256s, text_provenance, remote, rights,
             route, is_repair, repair_of, candidate_ids_offered) -> dict[str, Any]:
        self.scope.check(work_id, source_sha256s, text_provenance, remote=remote, rights=rights)  # VS7
        self._check_live()  # VS6
        # Add streaming transport controls to a copy, before the wire bytes
        # become this call's immutable idempotency identity.  The caller may
        # reuse its original body for a repair or an audit comparison.
        body = dict(request_body)
        stream_enabled = self.stream or body.get("stream") is True
        if stream_enabled:
            options = body.get("stream_options", {})
            if not isinstance(options, dict):
                raise VLLMHardStop("invalid_stream_options", "stream_options must be an object")
            body["stream"] = True
            body["stream_options"] = {**options, "include_usage": True}
        data = chat_request_bytes(body)
        request_sha = hashlib.sha256(data).hexdigest()
        pkt_sha = packet_sha256(packet)  # logged for audit only; VS5 (compare against a caller-reviewed
                                         # hash) is not implemented -- no caller holds one yet (D-047)
        key = self.idempotency_key(route, work_id, batch_id, request_sha)
        meta = {"route": route, "method": "POST", "request_sha256": request_sha, "packet_sha256": pkt_sha,
                "target_work_id": work_id, "batch_id": batch_id, "is_repair": is_repair, "repair_of": repair_of,
                "candidate_ids_offered": sorted(candidate_ids_offered or []), "text_provenance": text_provenance,
                "source_sha256s": sorted(set(source_sha256s))}
        return self._execute(key, f"{self.server_url}/v1/chat/completions", "POST", data, meta, stream=stream_enabled)

    # ------------------------------------------------------------------------------- transport
    def _preserve_body(self, key: str, attempt: int, name: str, body: bytes) -> str:
        """Keep exact wire bodies privately; credentials/HTTP headers are never artifacts."""
        if self._api_key and self._api_key.encode() in body:
            raise VLLMHardStop("credential_in_body", "body contains the server credential; persistence refused")
        root = self.ledger.path.parent / f"{self.ledger.path.stem}.private.responses"
        self.ledger.workspace.write_once(root / key / str(attempt) / name, body)
        return hashlib.sha256(body).hexdigest()

    def _execute(self, key: str, url: str, method: str, data: bytes | None, meta: dict[str, Any], *,
                 stream: bool = False) -> dict[str, Any]:
        prior = self.ledger.attempts(key)
        attempts = (1, 2)
        if prior:  # VS1
            results = [r for r in prior if r["row_kind"] == "result"]
            intents = [r for r in prior if r["row_kind"] == "intent"]
            retryable_first = (len(intents) == 1 and len(results) == 1 and results[0].get("attempt", 1) == 1
                               and (results[0].get("outcome") == "transport_error"
                                    or (results[0].get("outcome") == "http_error" and results[0].get("http_status") in RETRYABLE)))
            if not retryable_first:
                raise VLLMHardStop("idempotency_key_already_sent", f"{key[:17]}... is already in the ledger; not resent")
            attempts = (2,)
        for attempt in attempts:
            totals = self.ledger.totals()  # VS3
            if totals.requests + 1 > self.budget.request_cap:
                raise VLLMHardStop("request_cap", f"{totals.requests} requests recorded; cap {self.budget.request_cap}")
            if totals.output_tokens > self.budget.output_token_cap_total:
                raise VLLMHardStop("output_token_cap", f"{totals.output_tokens} output tokens recorded; cap {self.budget.output_token_cap_total}")
            if totals.wall_clock_seconds > self.budget.wall_clock_cap_seconds:
                raise VLLMHardStop("wall_clock_cap", f"{totals.wall_clock_seconds}s recorded; cap {self.budget.wall_clock_cap_seconds}s")
            target = meta.get("target_work_id")
            if target and target != "(none)":  # VS8
                article = self.ledger.totals_for_article(target)
                article_limit = self.budget.article_limit(self.candidate, target)
                if article.requests + 1 > article_limit:
                    raise VLLMHardStop("article_request_cap",
                                       f"{target}: {article.requests} requests recorded; cap {article_limit} per article")
            intent = {**meta, "row_kind": "intent", "idempotency_key": key, "attempt": attempt, "outcome": "pending"}
            intent["request_body_sha256"] = self._preserve_body(key, attempt, "request.bin", data or b"")
            self.ledger.append(intent)  # VS2 (before)
            headers = {"Content-Type": "application/json", "Accept": "text/event-stream" if stream else "application/json",
                       **self._key_header()}
            started = time.monotonic()
            base = {k: meta[k] for k in ("route", "request_sha256", "packet_sha256", "target_work_id", "batch_id",
                                         "is_repair", "repair_of", "candidate_ids_offered", "text_provenance",
                                         "source_sha256s") if k in meta}
            base |= {"row_kind": "result", "idempotency_key": key, "attempt": attempt}
            try:
                status, _rheaders, raw = self.transport.request(method, url, headers, data, self.timeout)
            except TransportError as exc:
                latency = int((time.monotonic() - started) * 1000)
                if exc.sent:  # VS4: never retried
                    self.ledger.append(base | {"outcome": "unknown_outcome", "latency_ms": latency, "detail": str(exc)})
                    raise UnknownOutcome(f"{key[:17]}...: {exc}; not retried") from None
                self.ledger.append(base | {"outcome": "transport_error", "latency_ms": latency, "detail": str(exc)})
                if attempt == 2:
                    raise
                continue
            latency = int((time.monotonic() - started) * 1000)
            # Preserve before parsing or adapting: invalid/truncated answers are diagnostic evidence too.
            base["response_sha256"] = self._preserve_body(key, attempt, "response.bin", raw)
            base["response_bytes"] = len(raw)
            try:
                parsed = json.loads(raw.decode("utf-8")) if raw else {}
            except (UnicodeDecodeError, json.JSONDecodeError):
                parsed = {"unparsed_body_sha256": hashlib.sha256(raw).hexdigest()}
            if status == 200:
                if stream:
                    try:
                        parsed = decode_chat_stream(raw)
                    except StreamDecodeError as exc:
                        self.ledger.append(base | {"outcome": "unknown_outcome", "latency_ms": latency,
                                                   "detail": f"stream_decode_error:{exc}"})
                        raise UnknownOutcome(f"{key[:17]}...: stream response was incomplete; not retried") from None
                usage = parsed.get("usage") or {}
                row = base | {"outcome": "success", "http_status": status, "latency_ms": latency,
                              "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
                              "output_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
                              "model": parsed.get("model")}
                # candidate_ids_used is deliberately not written here: this transport layer never
                # parses the response body, so it cannot know which offered ids the model actually
                # used (finding: candidate_ids_used always equalled candidate_ids_offered).
                self.ledger.append({k: v for k, v in row.items() if v is not None})  # VS2 (after)
                return parsed
            self.ledger.append(base | {"outcome": "http_error", "http_status": status, "latency_ms": latency,
                                       "detail": json.dumps(parsed)[:500]})
            if status in RETRYABLE and attempt == 1:  # one retry
                self.sleep(1.0)
                continue
            raise VLLMHTTPError(status, parsed)
        raise AssertionError("unreachable")
