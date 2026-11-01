"""Finite, checkpointed phases over the opt-in exact-request ledger.

Only this coordinator owns its candidate lock. Unfinished phases never resume
sends automatically. Completed phases replay their checked artifacts, including
stops; an empty unresolved list alone does not authorize the next phase. This
adds no source, model, rental, or scientific qualification authority.
"""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
import math
import re
import time

from jsonschema import Draft202012Validator, ValidationError

from mosaic.adapted.k.kb_writer_lock import acquire, release
from mosaic.canonical import canonical_bytes
from mosaic.vllm.client import VLLMClient
from mosaic.vllm.cache import VerifiedCachedResponse, verify_artifacts
from mosaic.vllm.deadline import DeadlineTransport
from mosaic.vllm.ledger import VLLMLedger, VLLMHardStop
from mosaic.vllm.reservation import FrozenRequestLedger, freeze_request, require
from mosaic.vllm.response import answer_outcome
from mosaic.vllm.unavailable import unavailable_outcome


def digest(value):
    return sha256(canonical_bytes(value)).hexdigest()


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _stop(rows, permits):
    """Reconstruct financial uncertainty from persisted rows, never counters."""
    by_key = {p['idempotency_key']: p for p in permits}
    for row in rows:
        require(row['idempotency_key'] in by_key, 'phase_unlisted_ledger_row')
        if row['row_kind'] != 'result':
            continue
        permit = by_key[row['idempotency_key']]
        if row['outcome'] == 'unknown_outcome' or (row['outcome'] == 'http_error' and row.get('http_status') != 429):
            return 'outcome_not_bounded'
        if row['outcome'] == 'success' and type(row.get('output_tokens')) is not int:
            return 'output_usage_missing'
        if type(row.get('output_tokens')) is int and row['output_tokens'] > permit['max_output_tokens']:
            return 'output_reservation_exceeded'
        if type(row.get('latency_ms')) is not int:
            return 'time_usage_missing'
        if row['latency_ms'] > permit['max_request_seconds'] * 1000:
            return 'time_reservation_exceeded'
    return None


class _PhaseLedger(FrozenRequestLedger):
    """One phase's send authority; retirement is irreversible, results can settle."""
    def __init__(self, *args, budget, starting_hash, deadline, cleanup_seconds, **kwargs):
        super().__init__(*args, **kwargs)
        self.approved_budget = digest(asdict(budget))
        self.starting_hash = starting_hash
        self.deadline = deadline
        self.monotonic_deadline = time.monotonic() + max(0, deadline - time.time())
        self.cleanup_seconds = cleanup_seconds
        self.retired = False

    def seal(self, permits, *, budget, expected_ledger_sha256, receipt_path):
        with self._lock:
            require(not self.retired, 'phase_intents_retired')
            require(digest(asdict(budget)) == self.approved_budget, 'phase_budget_changed')
            require(expected_ledger_sha256 == self.starting_hash, 'phase_start_changed')
            return super().seal(permits, budget=budget, expected_ledger_sha256=expected_ledger_sha256,
                                receipt_path=receipt_path)

    def retire(self):
        with self._lock:
            self.retired = True

    def append(self, row):
        with self._lock:
            if row.get('row_kind') == 'intent':
                require(not self.retired, 'phase_intents_retired')
                permit = (self._permits or {}).get(row.get('idempotency_key'))
                require(permit is not None, 'reservation_request_not_frozen')
                remaining = min(self.deadline - time.time(), self.monotonic_deadline - time.monotonic())
                if remaining <= permit['max_request_seconds'] + self.cleanup_seconds:
                    self.stop_reason = self.stop_reason or 'phase_deadline'
                    raise VLLMHardStop('phase_deadline', 'no bounded request fits before the cleanup deadline')
            super().append(row)


class PhasedRun:
    """Context-managed, finite synchronous coordinator with bounded concurrency.

    Caller freezes the complete program as plan_sha256, including source,
    recipe/tokenizer, builder versions, stage bounds and concurrency. Each phase
    then freezes exact tasks before sending. The default transport bounds an
    entire exchange. Test transports are injected explicitly. The candidate lock
    is cooperative: older plain-ledger callers still require external exclusion.
    """
    def __init__(self, *, candidate, ledger_root, receipt_root, workspace, budget,
                 phase_ids, expected_ledger_sha256, plan_sha256, deadline_epoch,
                 max_workers=1, cleanup_seconds=60, resume=False):
        require(isinstance(phase_ids, list) and phase_ids and all(isinstance(p, str)
                and re.fullmatch(r'[a-z][a-z0-9-]{0,63}', p) for p in phase_ids)
                and len(set(phase_ids)) == len(phase_ids), 'phase_roster_invalid')
        require(_hash(expected_ledger_sha256) and _hash(plan_sha256), 'phase_hash_required')
        require(type(max_workers) is int and 1 <= max_workers <= 32, 'phase_workers_invalid')
        require(type(deadline_epoch) in (int, float) and math.isfinite(deadline_epoch)
                and type(cleanup_seconds) in (int, float) and math.isfinite(cleanup_seconds)
                and cleanup_seconds >= 0, 'phase_deadline_invalid')
        require(all(type(v) is int and v > 0 for v in (budget.request_cap, budget.output_token_cap_total,
                budget.max_requests_per_article)) and type(budget.wall_clock_cap_seconds) in (int, float)
                and math.isfinite(budget.wall_clock_cap_seconds) and budget.wall_clock_cap_seconds > 0,
                'phase_budget_invalid')
        self.workspace = workspace
        self.root = workspace.check(receipt_root)
        self.ledger_root = workspace.check(ledger_root)
        self.candidate = candidate
        self.budget = deepcopy(budget)
        self.phase_ids = list(phase_ids)
        self.deadline = deadline_epoch
        self.workers = max_workers
        self.cleanup_seconds = cleanup_seconds
        self.closed = []
        self.active = None
        self.stop_reason = None
        self._owner = None
        ledger = VLLMLedger(candidate, root=self.ledger_root, workspace=workspace)
        # The reused portable lock is acquired once, without waiting or stale-lock deletion.
        self._owner = acquire(ledger.path, workspace, 'finite extraction phases')
        try:
            raw = ledger.path.read_bytes() if ledger.path.exists() else b''
            self.stop_path = ledger.path.with_name(ledger.path.stem + '.phased-stop.json')
            persisted_stop = json.loads(self.stop_path.read_bytes()) if self.stop_path.exists() else None
            if persisted_stop is not None:
                require(resume and persisted_stop['receipt_root'] == self.root.as_posix(), 'phase_candidate_stopped')
                end = persisted_stop['ledger_bytes']
                require(type(end) is int and 0 <= end <= len(raw)
                        and sha256(raw[:end]).hexdigest() == persisted_stop['ledger_sha256'], 'phase_stop_ledger_changed')
            manifest_path = self.root / 'manifest.json'
            if resume:
                require(manifest_path.is_file(), 'phase_manifest_missing')
                previous = json.loads(manifest_path.read_bytes())
                start_bytes = previous['starting_ledger_bytes']
            else:
                require(not manifest_path.exists(), 'phase_run_already_started')
                start_bytes = len(raw)
            require(type(start_bytes) is int and 0 <= start_bytes <= len(raw)
                    and sha256(raw[:start_bytes]).hexdigest() == expected_ledger_sha256,
                    'phase_start_changed')
            self.manifest = {'version': 'finite-phased-run/v1', 'candidate': candidate,
                'ledger_path': ledger.path.as_posix(), 'phase_ids': self.phase_ids, 'budget': asdict(budget),
                'plan_sha256': plan_sha256, 'starting_ledger_sha256': expected_ledger_sha256,
                'starting_ledger_bytes': start_bytes, 'deadline_epoch': deadline_epoch,
                'max_workers': max_workers, 'cleanup_seconds': cleanup_seconds,
                'authorization_granted': False}
            if resume:
                require(digest(previous) == digest(self.manifest), 'phase_manifest_changed')
                require(persisted_stop is None or persisted_stop['manifest_sha256'] == digest(self.manifest),
                        'phase_stop_manifest_changed')
            else:
                require(not ledger.totals().unresolved, 'phase_prior_unresolved')
                workspace.write_once(manifest_path, canonical_bytes(self.manifest))
            self.last_bytes = start_bytes
            self.last_hash = expected_ledger_sha256
            self._restore(ledger, raw)
            require(persisted_stop is None or self.stop_reason == persisted_stop['stop_reason'], 'phase_stop_changed')
            require(persisted_stop is None or self.closed
                    and digest(self.closed[-1]) == persisted_stop['closure_sha256'], 'phase_stop_closure_changed')
        except BaseException:
            release(self._owner)
            self._owner = None
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.active is not None:
            self.active.retire()
        # run_phase joins all its workers before returning or unwinding.
        if self._owner is not None:
            release(self._owner)
            self._owner = None

    def _restore(self, ledger, raw):
        for phase_id in self.phase_ids:
            root = self.root / phase_id
            if not (root / 'closed.json').exists():
                require(not (root / 'started.json').exists(), 'phase_interrupted_requires_audit')
                break
            require(self.stop_reason is None, 'phase_after_persisted_stop')
            closure = json.loads((root / 'closed.json').read_bytes())
            require({'started.json', 'requests.json', 'outcomes.json'} <= set(closure['artifacts']),
                    'phase_artifact_manifest_incomplete')
            for name, expected in closure['artifacts'].items():
                require(isinstance(name, str) and re.fullmatch(r'[a-z0-9-]+\.json', name)
                        and _hash(expected), 'phase_artifact_name_invalid')
                require(sha256((root / name).read_bytes()).hexdigest() == expected, 'phase_artifact_changed')
            start = json.loads((root / 'started.json').read_bytes())
            require(start['manifest_sha256'] == digest(self.manifest)
                    and start['phase_id'] == phase_id and start['ledger_before_sha256'] == self.last_hash
                    and start['ledger_before_bytes'] == self.last_bytes
                    and start['previous_closure_sha256'] == (digest(self.closed[-1]) if self.closed else None),
                    'phase_chain_changed')
            end = closure['ledger_after_bytes']
            require(type(end) is int and self.last_bytes <= end <= len(raw), 'phase_ledger_size_changed')
            part = raw[self.last_bytes:end]
            rows = [json.loads(line) for line in part.splitlines() if line.strip()]
            outputs = json.loads((root / 'outcomes.json').read_bytes())
            for output in outputs:
                if output.get('origin') == 'historical_cache':
                    verify_artifacts(output['cache_receipt'], self.workspace)
                elif output.get('origin') == 'declared_unavailable':
                    verify_artifacts(output['unavailable_receipt'], self.workspace)
            permits = start['permits']
            require(not permits or 'reservation.json' in closure['artifacts'], 'phase_artifact_manifest_incomplete')
            requests = json.loads((root / 'requests.json').read_bytes())
            require(digest(requests) == start['requests_sha256'] and len(requests) == len(outputs)
                    and all(o['id'] == r['id'] and o['request_sha256'] == digest(r)
                            for o, r in zip(outputs, requests)), 'phase_outcome_binding_changed')
            keys = {p['idempotency_key'] for p in permits}
            intents = {(r['idempotency_key'], r['attempt']) for r in rows if r['row_kind'] == 'intent'}
            results = {(r['idempotency_key'], r['attempt']) for r in rows if r['row_kind'] == 'result'}
            require(intents == results, 'phase_unresolved_intent')
            self._check_bodies(ledger, rows)
            unsent = sorted(keys - {k for k, _ in intents})
            reason = _stop(rows, permits) or ('phase_unsent_requests' if unsent else None)
            # A deadline or an unexpected local exception is an additional
            # conservative stop; every such reason must survive replay.
            local_stops = [o.get('hard_stop') for o in outputs if o.get('hard_stop')]
            reason = reason or (local_stops[0] if local_stops else None)
            expected = self._closure(start, rows, raw[:end], outputs, closure['artifacts'], reason, unsent)
            require(digest(expected) == digest(closure), 'phase_closure_changed')
            self.closed.append(closure)
            self.stop_reason = closure['stop_reason']
            self.last_bytes = end
            self.last_hash = sha256(raw[:end]).hexdigest()
        require(len(raw) == self.last_bytes and sha256(raw).hexdigest() == self.last_hash, 'phase_ledger_drift')
        # Parse/validate all old rows with the existing schema, too.
        require(not ledger.totals().unresolved, 'phase_unresolved_intent')
        for phase_id in self.phase_ids[len(self.closed):]:
            require(not (self.root / phase_id / 'closed.json').exists(), 'phase_closure_out_of_order')

    @staticmethod
    def _check_bodies(ledger, rows):
        root = ledger.path.parent / (ledger.path.stem + '.private.responses')
        for row in rows:
            for field, filename in (('request_body_sha256', 'request.bin'), ('response_sha256', 'response.bin')):
                if field in row:
                    path = root / row['idempotency_key'] / str(row['attempt']) / filename
                    require(path.is_file() and sha256(path.read_bytes()).hexdigest() == row[field],
                            'phase_wire_artifact_changed')

    def _closure(self, start, rows, raw, outputs, artifacts, reason, unsent):
        return {'version': 'finite-phase-closure/v1', 'phase_id': start['phase_id'],
            'manifest_sha256': digest(self.manifest), 'previous_closure_sha256': start['previous_closure_sha256'],
            'ledger_before_sha256': start['ledger_before_sha256'], 'ledger_before_bytes': start['ledger_before_bytes'],
            'ledger_after_sha256': sha256(raw).hexdigest(), 'ledger_after_bytes': len(raw),
            'phase_rows_sha256': digest(rows), 'outcomes_sha256': digest(outputs),
            'artifacts': artifacts, 'stop_reason': reason, 'unattempted_request_keys': unsent,
            'intents_retired': True, 'authorization_granted': False}

    def run_phase(self, phase_id, requests, *, scope, base_url=None, api_key=None, live=False, transport=None,
                  cached_responses=None):
        require(self._owner is not None and self.active is None, 'phase_owner_unavailable')
        require(digest(asdict(self.budget)) == digest(self.manifest['budget']), 'phase_budget_changed')
        require(phase_id in self.phase_ids, 'phase_unknown')
        frozen = deepcopy(requests)
        cached = dict(cached_responses or {})
        unavailable = {}
        require(set(cached) <= {r['id'] for r in frozen}, 'cache_unlisted_response')
        for request in frozen:
            saved = cached.get(request['id'])
            require(('cache_entry_sha256' in request) == (saved is not None), 'cache_response_missing')
            if saved is not None:
                require(isinstance(saved, VerifiedCachedResponse) and saved.request_sha256 == digest(request)
                        and digest(saved.outcome) == saved.outcome_sha256
                        and saved.outcome['cache_receipt']['entry_sha256'] == request['cache_entry_sha256'],
                        'cache_verified_response_changed')
                verify_artifacts(saved.outcome['cache_receipt'], self.workspace)
            omitted = unavailable_outcome(request, workspace=self.workspace, scope=scope)
            if omitted is not None:
                unavailable[request['id']] = omitted
        if self.phase_ids.index(phase_id) < len(self.closed):
            closure = self.closed[self.phase_ids.index(phase_id)]
            checked = {}
            for name in ('started.json', 'outcomes.json'):
                raw = (self.root / phase_id / name).read_bytes()
                require(sha256(raw).hexdigest() == closure['artifacts'][name], 'phase_artifact_changed')
                checked[name] = json.loads(raw)
            start = checked['started.json']
            require(digest(frozen) == start['requests_sha256'], 'phase_replay_requests_changed')
            return checked['outcomes.json']
        require(self.stop_reason is None, 'phase_previous_stop')
        require(len(self.closed) < len(self.phase_ids) and phase_id == self.phase_ids[len(self.closed)],
                'phase_out_of_order')
        require(isinstance(frozen, list) and len({r['id'] for r in frozen}) == len(frozen), 'phase_requests_invalid')
        permits = []
        for r in frozen:
            require(isinstance(r['id'], str) and r['id'] and _hash(r['binding_sha256']), 'phase_request_binding_invalid')
            task = r['task']
            Draft202012Validator.check_schema(task['schema'])
            require(type(r['prompt_tokens']) is int and r['prompt_tokens'] >= 0
                    and type(r['context_limit']) is int
                    and r['prompt_tokens'] + task['body']['max_tokens'] < r['context_limit'], 'phase_context_limit')
            # Check the entire roster before any call; the client checks again at send.
            scope.check(r['work_id'], r['source_sha256s'], r['text_provenance'], remote=r['remote'], rights=r['rights'])
            if r['id'] not in cached and r['id'] not in unavailable:
                permits.append(freeze_request(work_id=r['work_id'], batch_id=r['batch_id'], packet=task['payload'],
                    body=task['body'], source_sha256s=r['source_sha256s'], max_request_seconds=r['max_request_seconds'],
                    max_attempts=r['max_attempts'], stream=r['stream'], text_provenance=r['text_provenance']))
        ledger = _PhaseLedger(self.candidate, root=self.ledger_root, workspace=self.workspace,
            budget=self.budget, starting_hash=self.last_hash, deadline=self.deadline, cleanup_seconds=self.cleanup_seconds)
        require(ledger.fingerprint() == self.last_hash, 'phase_ledger_drift')
        require(not ledger.totals().unresolved, 'phase_unresolved_intent')
        root = self.root / phase_id
        require(not (root / 'started.json').exists(), 'phase_interrupted_requires_audit')
        start = {'version': 'finite-phase-start/v1', 'phase_id': phase_id, 'manifest_sha256': digest(self.manifest),
            'previous_closure_sha256': digest(self.closed[-1]) if self.closed else None,
            'ledger_before_sha256': self.last_hash, 'ledger_before_bytes': self.last_bytes,
            'requests_sha256': digest(frozen), 'permits': permits}
        artifacts = {}

        def save(name, value):
            data = canonical_bytes(value)
            self.workspace.write_once(root / name, data)
            artifacts[name] = sha256(data).hexdigest()

        # All clients are constructed before sealing. No implicit startup reconciliation.
        clients = [None if r['id'] in cached or r['id'] in unavailable else VLLMClient(candidate=self.candidate, base_url=base_url, ledger=ledger, scope=scope,
            budget=deepcopy(self.budget), transport=transport or DeadlineTransport(),
            timeout=r['max_request_seconds'], live=live, api_key=api_key, stream=r['stream']) for r in frozen]
        save('requests.json', frozen)
        save('started.json', start)
        self.active = ledger
        outputs = [deepcopy(cached[r['id']].outcome) if r['id'] in cached
                   else deepcopy(unavailable.get(r['id'])) for r in frozen]
        for index, output in enumerate(outputs):
            if output is not None:
                save(f'result-{index:05d}.json', output)

        def execute(index):
            r = frozen[index]
            began = time.monotonic()
            out = {'id': r['id'], 'request_sha256': digest(r), 'status': 'unavailable', 'answer': None}
            try:
                response = clients[index].chat_completion(work_id=r['work_id'], batch_id=r['batch_id'],
                    packet=r['task']['payload'], request_body=r['task']['body'], source_sha256s=r['source_sha256s'],
                    text_provenance=r['text_provenance'], remote=r['remote'], rights=r['rights'])
                # Private raw wire response is already preserved by the client.
                out['response'] = response
                choices = response.get('choices')
                if response.get('model') != r['task']['body']['model']:
                    out.update(status='model_identity_refused', hard_stop='phase_model_identity_changed')
                    ledger.retire()
                elif not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                    out['status'] = 'invalid_response'
                elif choices[0].get('finish_reason') != 'stop':
                    out['status'] = 'incomplete_response'
                else:
                    try:
                        answer = json.loads(choices[0]['message']['content'])
                        out.update(answer_outcome(r['task']['schema'], answer))
                    except (ValueError, KeyError, TypeError, ValidationError):
                        out['status'] = 'invalid_response'
            except Exception as exc:
                # Details can contain response text. Keep the type/code only here.
                out.update(status='request_failed', error_type=type(exc).__name__)
                if isinstance(exc, VLLMHardStop):
                    out['hard_stop'] = exc.code
                else:
                    # Known terminal failures still stop this run conservatively.
                    out['hard_stop'] = 'phase_request_exception'
                ledger.retire()
            out['seconds'] = time.monotonic() - began
            return out

        try:
            if permits:
                reservation = ledger.seal(permits, budget=self.budget, expected_ledger_sha256=self.last_hash,
                                          receipt_path=root / 'reservation.json')
                artifacts['reservation.json'] = digest(reservation)
                # Rolling submission: never queue the whole roster behind workers.
                with ThreadPoolExecutor(max_workers=self.workers) as pool:
                    active = {}
                    next_index = 0
                    while next_index < len(frozen) or active:
                        while next_index < len(frozen) and len(active) < self.workers and not ledger.stop_reason and not ledger.retired:
                            if outputs[next_index] is None:
                                active[pool.submit(execute, next_index)] = next_index
                            next_index += 1
                        if not active:
                            break
                        done, _ = wait(active, return_when=FIRST_COMPLETED)
                        for future in done:
                            index = active.pop(future)
                            outputs[index] = future.result()
                            save(f'result-{index:05d}.json', outputs[index])
            for index, out in enumerate(outputs):
                if out is None:
                    outputs[index] = {'id': frozen[index]['id'], 'request_sha256': digest(frozen[index]),
                        'status': 'unsent', 'answer': None, 'reason': ledger.stop_reason or 'phase_stopped'}
            ledger.retire()
            # Even an unresolved intent retains every returned/unsent outcome.
            # It gets no closure and therefore cannot authorize a next phase.
            save('outcomes.json', outputs)
            require(not ledger.totals().unresolved, 'phase_unresolved_intent')
            raw = ledger.path.read_bytes() if ledger.path.exists() else b''
            rows = [json.loads(line) for line in raw[self.last_bytes:].splitlines() if line.strip()]
            self._check_bodies(ledger, rows)
            keys = {r['idempotency_key'] for r in rows if r['row_kind'] == 'intent'}
            unsent = sorted({p['idempotency_key'] for p in permits} - keys)
            local_stops = [o['hard_stop'] for o in outputs if o.get('hard_stop')]
            reason = _stop(rows, permits) or ('phase_unsent_requests' if unsent else None)
            reason = reason or (local_stops[0] if local_stops else None)
            closure = self._closure(start, rows, raw, outputs, artifacts, reason, unsent)
            if reason:
                # A different receipt directory must not erase this candidate's
                # uncertainty. Clearing it requires a separately recorded audit;
                # this API intentionally supplies no reset/delete operation.
                self.workspace.write_once(self.stop_path, canonical_bytes({'version': 'finite-candidate-stop/v1',
                    'receipt_root': self.root.as_posix(), 'manifest_sha256': digest(self.manifest),
                    'ledger_bytes': len(raw), 'ledger_sha256': sha256(raw).hexdigest(),
                    'stop_reason': reason, 'closure_sha256': digest(closure)}))
            self.workspace.write_once(root / 'closed.json', canonical_bytes(closure))
            self.last_hash = closure['ledger_after_sha256']
            self.last_bytes = len(raw)
            self.closed.append(closure)
            self.stop_reason = reason
            return deepcopy(outputs)
        finally:
            ledger.retire()
            self.active = None
