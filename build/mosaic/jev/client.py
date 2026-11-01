"""TypeSafe-compatible Jev client over the Vercel AI Gateway (HANDOFF §6.6; FINAL v2 §7, §12.5).

Route: POST <base>/v1/systemone with TypeSafe's body {model, state, questions}
plus `providerOptions.gateway.zeroDataRetention: true` on every request; the
models probe is GET <base>/v1/models. The base URL comes from
JEV_GATEWAY_BASE_URL (default https://ai-gateway.vercel.sh/typesafe) and the
key from AI_GATEWAY_API_KEY in this repository's git-ignored .env, loaded only
at send time and never written anywhere.

Every call is recorded in ledgers/jev_calls.jsonl (route, request and state
hashes, generationId, canonicalSlug, model, tokens, gatewayCost, timestamps,
idempotency key, target work, outcome, answers). The gateway echoes no
TypeSafe model version, so `generationId` and routing metadata are the only
version evidence; the alias is a moving target (FINAL v2 §7.1, Appendix C).
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import time
import urllib.parse
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from mosaic import REPO_ROOT
from mosaic.adapted.k.env_loader import load_mosaic_env, require_secret
from mosaic.canonical import canonical_bytes
from mosaic.jev.ledger import Budget, HardStop, JevLedger
from mosaic.jev.scope_contract import JevScope

DEFAULT_BASE = "https://ai-gateway.vercel.sh/typesafe"
MODEL = "typesafe-ai/jev"
ROUTE_SYSTEMONE = "typesafe_compatible_systemone"
ROUTE_MODELS = "typesafe_models_probe"
INPUT_PRICE_PER_TOKEN = Decimal("0.042") / Decimal(1_000_000)  # published price; output is free
ACK_FILE = REPO_ROOT / "config" / "eric_ack.json"
RETRYABLE = {429, 500, 502, 503, 504, 529}
MAX_RETRY_WAIT_S = 65.0  # the gateway's Retry-After for Jev has been 60 s


class TransportError(RuntimeError):
    def __init__(self, message: str, sent: bool):
        super().__init__(message)
        self.sent = sent


class UnknownOutcome(RuntimeError):
    """The request may have reached the gateway; it is never retried (hard stop 4)."""


class JevHTTPError(RuntimeError):
    def __init__(self, status: int, body: dict[str, Any] | str):
        super().__init__(f"gateway returned HTTP {status}: {str(body)[:300]}")
        self.status, self.body = status, body


class Transport(Protocol):
    def request(self, method: str, url: str, headers: dict[str, str], body: bytes | None,
                timeout: float) -> tuple[int, dict[str, str], bytes]: ...


class HttpTransport:
    """HTTP(S) via http.client with the connect phase separated from the exchange: a failure while
    connecting (refused, unresolved, TLS or connect timeout) sent nothing (`sent=False`); any failure
    after the request started is an unknown outcome (`sent=True`)."""

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


def state_sha256(state: Any, questions: dict[str, Any]) -> str:
    """The hash a caller reviews and asserts: canonical JSON of the state and the questions."""
    return hashlib.sha256(canonical_bytes({"state": state, "questions": questions})).hexdigest()


class JevClient:
    def __init__(self, *, ledger: JevLedger | None = None, scope: JevScope | None = None, budget: Budget | None = None,
                 base_url: str | None = None, api_key: str | None = None, transport: Transport | None = None,
                 timeout: float = 60.0, real: bool = False, eric_ack: str | None = None, ack_file: Path = ACK_FILE,
                 sleep=time.sleep):
        self.ledger = ledger or JevLedger()
        if scope is None:
            # The private default scope depends on protected-registry roots.
            # Public clients always provide an explicit independent scope and
            # must not import or discover the private registry merely to use
            # the shared transport and cumulative ledger.
            from mosaic.jev.scope import DocumentScope
            scope = DocumentScope()
        self.scope = scope
        self.budget = budget or Budget.load()
        self.base_url = (base_url or os.environ.get("JEV_GATEWAY_BASE_URL") or DEFAULT_BASE).rstrip("/")
        self._api_key = api_key
        self.transport = transport or HttpTransport()
        self.timeout = timeout
        self.real, self.eric_ack, self.ack_file = real, eric_ack, ack_file
        self.sleep = sleep
        self.clock = time.monotonic
        self._not_before = 0.0  # pacing from the gateway's own rate-limit headers (M-21)
        self.reconciled = self.ledger.reconcile()

    def __repr__(self) -> str:  # never shows the key
        return f"JevClient(base_url={self.base_url!r}, ledger={str(self.ledger.path)!r})"

    # ---------------------------------------------------------------------------------- stops
    def _authorize_beyond_auth1(self, model: str, purpose: str) -> None:
        """Hard stop 7: anything AUTH-1 does not cover needs --real and Eric's recorded token."""
        if model == MODEL and purpose in ("advisory", "probe"):
            return
        if not self.real or not self.eric_ack:
            raise HardStop(7, "not_covered_by_auth1", f"model {model!r} / purpose {purpose!r} needs --real --eric-ack")
        recorded = json.loads(self.ack_file.read_text(encoding="utf-8")).get("jev_beyond_auth1_token") if self.ack_file.exists() else None
        if not recorded or recorded != self.eric_ack:
            raise HardStop(7, "eric_ack_mismatch", "no matching acknowledgement recorded by Eric")

    def _key(self) -> str:
        if self._api_key is None:
            load_mosaic_env()
            self._api_key = require_secret("AI_GATEWAY_API_KEY")
        return self._api_key

    @staticmethod
    def idempotency_key(route: str, work_id: str, purpose: str, pattern: str, request_sha256: str) -> str:
        seed = canonical_bytes({"route": route, "work_id": work_id, "purpose": purpose, "pattern": pattern, "request": request_sha256})
        return "idem-" + hashlib.sha256(seed).hexdigest()

    # --------------------------------------------------------------------------------- calls
    def system_one(self, *, work_id: str, state: Any, questions: dict[str, Any], expected_state_sha256: str,
                   source_sha256s: list[str], text_provenance: str, pattern: str, purpose: str = "advisory",
                   fixture_replay: bool = False, model: str = MODEL, reuse_recorded: bool = False) -> dict[str, Any]:
        self.scope.check(work_id, source_sha256s, text_provenance)                          # stop 8
        self._authorize_beyond_auth1(model, purpose)                                       # stop 7
        asserted = state_sha256(state, questions)
        if asserted != expected_state_sha256:                                              # stop 5
            raise HardStop(5, "prompt_hash_mismatch", "the state/questions differ from the reviewed version")
        body = {"model": model, "state": state, "questions": questions,
                "providerOptions": {"gateway": {"zeroDataRetention": True}}}
        data = canonical_bytes(body)
        request_sha = hashlib.sha256(data).hexdigest()
        key = self.idempotency_key(ROUTE_SYSTEMONE, work_id, purpose, pattern, request_sha)
        meta = {"route": ROUTE_SYSTEMONE, "method": "POST", "request_sha256": request_sha, "prompt_state_sha256": asserted,
                "target_work_id": work_id, "fixture_replay": fixture_replay, "zero_data_retention": True, "purpose": purpose,
                "pattern": pattern, "question_ids": sorted(questions), "text_provenance": text_provenance,
                "source_sha256s": sorted(set(source_sha256s))}
        return self._execute(key, f"{self.base_url}/v1/systemone", "POST", data, meta, reuse_recorded)

    def probe_models(self, *, label: str) -> dict[str, Any]:
        """GET /v1/models: the first live use of the key, behind the same stops (1-4, 7)."""
        self._authorize_beyond_auth1(MODEL, "probe")
        request_sha = hashlib.sha256(f"GET /v1/models {label}".encode()).hexdigest()
        key = self.idempotency_key(ROUTE_MODELS, "(none)", "probe", label, request_sha)
        meta = {"route": ROUTE_MODELS, "method": "GET", "request_sha256": request_sha, "prompt_state_sha256": request_sha,
                "target_work_id": "(none)", "purpose": "probe", "pattern": label, "text_provenance": "none"}
        return self._execute(key, f"{self.base_url}/v1/models", "GET", None, meta, False)

    def _pace(self, limits: dict[str, str]) -> None:
        """When the gateway reports no remaining requests, wait for its stated reset before the next call."""
        remaining = limits.get("x-ratelimit-remaining-requests")
        reset = _seconds(limits.get("x-ratelimit-reset-requests"))
        if remaining is not None and remaining.strip() == "0" and reset is not None:
            self._not_before = max(self._not_before, self.clock() + reset)

    def _wait_turn(self) -> None:
        delay = self._not_before - self.clock()
        if delay > 0:
            self.sleep(delay)

    def _execute(self, key: str, url: str, method: str, data: bytes | None, meta: dict[str, Any], reuse: bool) -> dict[str, Any]:
        prior = self.ledger.attempts(key)
        attempts = (1, 2)
        if prior:                                                                           # stop 1
            ok = [r for r in prior if r["row_kind"] == "result" and r.get("outcome") == "success"]
            if reuse and ok:
                return {"answers": ok[-1].get("answers", {}), "recorded": True, "generation_id": ok[-1].get("generation_id")}
            results = [r for r in prior if r["row_kind"] == "result"]
            intents = [r for r in prior if r["row_kind"] == "intent"]
            # The one permitted retry of a call whose first attempt ended with a retryable failure and was never
            # retried (for example because Retry-After exceeded the wait allowed then) is still the same call.
            retryable_first = (len(intents) == 1 and len(results) == 1 and results[0].get("attempt", 1) == 1
                               and (results[0].get("outcome") == "transport_error"
                                    or (results[0].get("outcome") == "http_error" and results[0].get("http_status") in RETRYABLE)))
            if not retryable_first:
                raise HardStop(1, "idempotency_key_already_sent", f"{key[:17]}... is already in the ledger; not resent")
            attempts = (2,)
        for attempt in attempts:
            self._wait_turn()
            totals = self.ledger.totals()                                                   # stop 3
            estimate = (Decimal(len(data or b"")) / 2 * INPUT_PRICE_PER_TOKEN).quantize(Decimal("0.000000001"))
            if totals.requests + 1 > self.budget.request_cap:
                raise HardStop(3, "request_cap", f"{totals.requests} requests recorded; cap {self.budget.request_cap}")
            if totals.spent_usd + estimate > self.budget.usd_cap:
                raise HardStop(3, "usd_cap", f"${totals.spent_usd} recorded + ${estimate} estimate exceeds ${self.budget.usd_cap}")
            intent = {**meta, "row_kind": "intent", "idempotency_key": key, "attempt": attempt, "outcome": "pending",
                      "estimated_cost_usd": format(estimate, "f")}
            api_key = self._key()                                                          # load/validate before the intent is written
            self.ledger.append(intent)                                                      # stop 2 (before)
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                       "Accept": "application/json"}
            started = time.monotonic()
            base = {k: meta[k] for k in ("route", "method", "request_sha256", "prompt_state_sha256", "target_work_id")}
            base |= {"row_kind": "result", "idempotency_key": key, "attempt": attempt}
            if meta["route"] == ROUTE_SYSTEMONE:
                base["zero_data_retention"] = True
            try:
                status, rheaders, raw = self.transport.request(method, url, headers, data, self.timeout)
            except TransportError as exc:
                latency = int((time.monotonic() - started) * 1000)
                if exc.sent:                                                                # stop 4: never retried
                    self.ledger.append(base | {"outcome": "unknown_outcome", "latency_ms": latency, "detail": str(exc)})
                    raise UnknownOutcome(f"{key[:17]}...: {exc}; not retried") from None
                self.ledger.append(base | {"outcome": "transport_error", "latency_ms": latency, "detail": str(exc)})
                if attempt == 2:
                    raise
                continue
            latency = int((time.monotonic() - started) * 1000)
            limits = {k: v for k, v in rheaders.items() if "ratelimit" in k or k == "retry-after"}
            self._pace(limits)
            try:
                parsed = json.loads(raw.decode("utf-8")) if raw else {}
            except (UnicodeDecodeError, json.JSONDecodeError):
                parsed = {"unparsed_body_sha256": hashlib.sha256(raw).hexdigest()}
            if status == 200:
                gw = (parsed.get("provider_metadata") or {}).get("gateway") or {}
                routing = gw.get("routing") or {}
                usage = parsed.get("usage") or {}
                row = base | {"outcome": "success", "http_status": status, "latency_ms": latency,
                              "generation_id": gw.get("generationId"), "canonical_slug": routing.get("canonicalSlug"),
                              "model": parsed.get("model"), "input_tokens": usage.get("input_tokens"),
                              "output_tokens": usage.get("output_tokens"),
                              "gateway_cost_usd": _decimal_or_none(gw.get("gatewayCost", gw.get("cost"))),
                              "market_cost_usd": _decimal_or_none(gw.get("marketCost")),
                              "zdr_evidence": routing.get("planningReasoning"), "rate_limit_headers": limits}
                if meta["route"] == ROUTE_SYSTEMONE:
                    row["answers"] = parsed.get("answers", {})
                else:
                    row["response"] = parsed
                self.ledger.append({k: v for k, v in row.items() if v is not None or k in ("generation_id", "model")})  # stop 2 (after)
                return parsed
            self.ledger.append(base | {"outcome": "http_error", "http_status": status, "latency_ms": latency,
                                       "rate_limit_headers": limits, "detail": json.dumps(parsed)[:500]})
            if status in RETRYABLE and attempt == 1:                                        # stop 4: one retry
                wait = _retry_after(limits.get("retry-after"))
                if wait is None or wait > MAX_RETRY_WAIT_S:
                    raise JevHTTPError(status, parsed)
                self.sleep(wait)
                continue
            raise JevHTTPError(status, parsed)
        raise AssertionError("unreachable")


def _decimal_or_none(value: Any) -> str | None:
    if value is None:
        return None
    try:
        d = Decimal(str(value))
    except Exception:
        return None
    return format(d, "f") if d >= 0 else None


def _seconds(value: str | None) -> float | None:
    """'60s', '1m30s', '0.5s' or '60' to seconds."""
    if not value:
        return None
    import re

    total, found = 0.0, False
    for amount, unit in re.findall(r"([0-9.]+)\s*(ms|s|m|h)?", value.strip()):
        found = True
        total += float(amount) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600, "": 1}[unit or ""]
    return total if found else None


def _retry_after(value: str | None) -> float | None:
    if value is None:
        return 1.0
    try:
        return float(value)
    except ValueError:
        return None
