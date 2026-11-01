"""Opt-in, single-process reservation for one finite set of exact chat requests.

Construct every client before sealing. Each intent consumes a frozen key/attempt
under one lock. Existing cumulative ledger usage is retained. This is not a
multi-process coordinator, a source/rental authorization, or a transport timeout;
the caller must supply its separately bounded transport and stop on stop_reason.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
import math
import re
from threading import RLock

from mosaic.canonical import canonical_bytes
from mosaic.vllm.client import VLLMClient, packet_sha256
from mosaic.vllm.ledger import VLLMHardStop, VLLMLedger
from mosaic.vllm.wire import chat_request_bytes

_HASH = re.compile(r'[0-9a-f]{64}\Z')
_IDENTITY = ('idempotency_key', 'route', 'request_sha256', 'packet_sha256',
             'target_work_id', 'batch_id', 'source_sha256s', 'text_provenance')


def require(condition, code):
    if not condition:
        raise VLLMHardStop(code, 'frozen request reservation refused')


def freeze_request(*, work_id, batch_id, packet, body, source_sha256s,
                   max_request_seconds, max_attempts=2, stream=True,
                   text_provenance='native_article_text'):
    require(isinstance(work_id, str) and work_id and isinstance(batch_id, str) and batch_id,
            'reservation_identity_invalid')
    require(type(body.get('max_tokens')) is int and body['max_tokens'] > 0,
            'reservation_output_bound_required')
    require(type(max_attempts) is int and max_attempts in (1, 2)
            and type(max_request_seconds) in (int, float)
            and math.isfinite(max_request_seconds) and max_request_seconds > 0,
            'reservation_attempt_bound_invalid')
    require(source_sha256s and all(isinstance(h, str) and _HASH.fullmatch(h) for h in source_sha256s),
            'reservation_source_hash_required')
    require(text_provenance in ('native_article_text', 'fixture_replay'), 'reservation_provenance_invalid')
    wire = deepcopy(body)
    if stream or wire.get('stream') is True:
        options = wire.get('stream_options', {})
        require(isinstance(options, dict), 'reservation_stream_options_invalid')
        wire.update(stream=True, stream_options={**options, 'include_usage': True})
    request_hash = sha256(chat_request_bytes(wire)).hexdigest()
    return {'idempotency_key': VLLMClient.idempotency_key('chat_completions', work_id, batch_id, request_hash),
        'route': 'chat_completions', 'request_sha256': request_hash, 'packet_sha256': packet_sha256(packet),
        'target_work_id': work_id, 'batch_id': batch_id, 'source_sha256s': sorted(set(source_sha256s)),
        'text_provenance': text_provenance, 'max_output_tokens': body['max_tokens'],
        'max_request_seconds': max_request_seconds, 'max_attempts': max_attempts}


class FrozenRequestLedger(VLLMLedger):
    def __init__(self, *args, **kwargs):
        self._lock = RLock()
        self._permits = None
        self._expected_sha256 = None
        self.stop_reason = None
        super().__init__(*args, **kwargs)

    def fingerprint(self):
        with self._lock:
            return sha256(self.path.read_bytes() if self.path.exists() else b'').hexdigest()

    def rows(self):
        with self._lock:
            require(self._expected_sha256 is None or self.fingerprint() == self._expected_sha256,
                    'reservation_ledger_changed')
            return super().rows()

    def reconcile(self):
        with self._lock:
            require(self._permits is None, 'reservation_clients_already_sealed')
            require(not self.totals().unresolved, 'reservation_prior_unresolved_intent')
            return []

    def seal(self, permits, *, budget, expected_ledger_sha256, receipt_path):
        with self._lock:
            require(self._permits is None and permits, 'reservation_already_sealed_or_empty')
            require(self.fingerprint() == expected_ledger_sha256, 'reservation_start_changed')
            totals = self.totals()
            require(not totals.unresolved, 'reservation_prior_unresolved_intent')
            frozen = deepcopy(permits)
            required = set(_IDENTITY) | {'max_output_tokens', 'max_request_seconds', 'max_attempts'}
            for p in frozen:
                require(set(p) == required and p['route'] == 'chat_completions', 'reservation_permit_invalid')
                require(all(isinstance(p[k], str) and _HASH.fullmatch(p[k])
                            for k in ('request_sha256', 'packet_sha256')), 'reservation_permit_invalid')
                require(all(isinstance(p[k], str) and p[k] for k in ('target_work_id', 'batch_id'))
                        and isinstance(p['source_sha256s'], list) and p['source_sha256s']
                        and all(isinstance(h, str) and _HASH.fullmatch(h) for h in p['source_sha256s'])
                        and p['source_sha256s'] == sorted(set(p['source_sha256s']))
                        and p['text_provenance'] in ('native_article_text', 'fixture_replay'),
                        'reservation_permit_invalid')
                require(p['idempotency_key'] == VLLMClient.idempotency_key(
                    p['route'], p['target_work_id'], p['batch_id'], p['request_sha256']),
                    'reservation_permit_invalid')
                require(type(p['max_attempts']) is int and p['max_attempts'] in (1, 2)
                        and type(p['max_output_tokens']) is int and p['max_output_tokens'] > 0
                        and type(p['max_request_seconds']) in (int, float)
                        and math.isfinite(p['max_request_seconds']) and p['max_request_seconds'] > 0,
                        'reservation_permit_invalid')
                require(not self.attempts(p['idempotency_key']), 'reservation_already_attempted')
            by_key = {p['idempotency_key']: p for p in frozen}
            require(len(by_key) == len(frozen), 'reservation_duplicate_key')
            requests = sum(p['max_attempts'] for p in frozen)
            outputs = sum(p['max_attempts'] * p['max_output_tokens'] for p in frozen)
            seconds = sum(p['max_attempts'] * p['max_request_seconds'] for p in frozen)
            require(totals.requests + requests <= budget.request_cap
                    and totals.output_tokens + outputs <= budget.output_token_cap_total
                    and totals.wall_clock_seconds + seconds <= budget.wall_clock_cap_seconds,
                    'reservation_cumulative_budget')
            counts = Counter()
            for p in frozen:
                counts[p['target_work_id']] += p['max_attempts']
            require(all(self.totals_for_article(work).requests + count <= budget.article_limit(self.candidate, work)
                        for work, count in counts.items()), 'reservation_article_budget')
            receipt = {'version': 'frozen-request-reservation/v1', 'candidate': self.candidate,
                'ledger_before_sha256': expected_ledger_sha256, 'budget': asdict(budget), 'permits': frozen,
                'reserved_attempts': requests, 'reserved_output_tokens': outputs, 'reserved_request_seconds': seconds,
                'single_process_only': True, 'authorization_granted': False}
            self.workspace.write_once(receipt_path, canonical_bytes(receipt))
            self._permits = by_key
            self._expected_sha256 = expected_ledger_sha256
            return deepcopy(receipt)

    def append(self, row):
        with self._lock:
            require(self._permits is not None, 'reservation_not_sealed')
            rows = self.rows()
            permit = self._permits.get(row.get('idempotency_key'))
            require(permit is not None and all(row.get(k) == permit[k] for k in _IDENTITY),
                    'reservation_request_not_frozen')
            attempt = row.get('attempt')
            require(type(attempt) is int and 1 <= attempt <= permit['max_attempts'], 'reservation_attempt_exhausted')
            same = [r for r in rows if r['idempotency_key'] == row['idempotency_key']]
            pair = [r for r in same if r.get('attempt', 1) == attempt]
            if row.get('row_kind') == 'intent':
                require(self.stop_reason is None, 'reservation_stopped')
                require(not pair, 'reservation_duplicate_intent')
                require(row.get('request_body_sha256') == permit['request_sha256'], 'reservation_wire_changed')
                wire_path = self.path.parent / f'{self.path.stem}.private.responses' / row['idempotency_key'] / str(attempt) / 'request.bin'
                require(wire_path.is_file(), 'reservation_wire_missing')
                wire = wire_path.read_bytes()
                require(sha256(wire).hexdigest() == permit['request_sha256']
                        and json.loads(wire)['max_tokens'] == permit['max_output_tokens'], 'reservation_wire_changed')
                if attempt == 2:
                    first = [r for r in same if r.get('attempt', 1) == 1 and r['row_kind'] == 'result']
                    require(len(first) == 1 and (first[0]['outcome'] == 'transport_error'
                        or first[0]['outcome'] == 'http_error' and first[0].get('http_status') == 429),
                        'reservation_retry_not_known_safe')
            else:
                require(row.get('row_kind') == 'result' and len(pair) == 1 and pair[0]['row_kind'] == 'intent',
                        'reservation_result_without_unique_intent')
            super().append(row)
            self._expected_sha256 = self.fingerprint()
            if row['row_kind'] == 'result':
                if row['outcome'] == 'unknown_outcome' or (row['outcome'] == 'http_error' and row.get('http_status') != 429):
                    self.stop_reason = 'outcome_not_bounded'
                elif row['outcome'] == 'success' and type(row.get('output_tokens')) is not int:
                    self.stop_reason = 'output_usage_missing'
                elif isinstance(row.get('output_tokens'), int) and row['output_tokens'] > permit['max_output_tokens']:
                    self.stop_reason = 'output_reservation_exceeded'
                elif type(row.get('latency_ms')) is not int:
                    self.stop_reason = 'time_usage_missing'
                elif row['latency_ms'] > permit['max_request_seconds'] * 1000:
                    self.stop_reason = 'time_reservation_exceeded'
