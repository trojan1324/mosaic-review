"""Read-only, explicit historical response admission; no search or best-answer pick.

The caller freezes one ledger attempt, task, recipe receipt and context proof.
Context verification belongs to the extraction contract supplied by the caller.
Every admitted response retains its actual raw origin and historical usage.
"""
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from mosaic.canonical import canonical_bytes
from mosaic import REPO_ROOT
from mosaic.validation import registry
from mosaic.vllm.client import VLLMClient
from mosaic.vllm.ledger import ledger_path
from mosaic.vllm.reservation import require
from mosaic.vllm.recipe_compatibility import verify_schema_extension
from mosaic.vllm.response import answer_outcome
from mosaic.vllm.streaming import decode_chat_stream
from mosaic.vllm.wire import chat_request_bytes


def digest(value):
    return sha256(canonical_bytes(value)).hexdigest()


def _validation_fingerprint():
    """Detect hot edits that the registry's own process caches cannot reload."""
    value = sha256(registry.schemas_digest().encode('ascii'))
    for name in ('registry.py', 'objects.py'):
        value.update((REPO_ROOT / 'mosaic' / 'validation' / name).read_bytes())
    return value.hexdigest()


_PROCESS_VALIDATION_FINGERPRINT = _validation_fingerprint()


def _loaded_schema_fingerprint():
    return digest({name: registry.schema_validator(name).schema for name in registry.schema_names()})


_PROCESS_SCHEMA_FINGERPRINT = _loaded_schema_fingerprint()


@lru_cache(maxsize=8)
def _validate_ledger_prefix(raw, validation_key):
    # Cache successful validation only, never mutable rows or accounting. Exact
    # bytes and the active validation environment are both part of the key.
    validate = validation_key[0]
    require(all(not validate('vllm-call', json.loads(line))
                for line in raw.splitlines() if line.strip()), 'cache_ledger_row_invalid')


def _ledger_rows(raw):
    require(_validation_fingerprint() == _PROCESS_VALIDATION_FINGERPRINT,
            'cache_validation_environment_changed')
    require(not raw or raw.endswith(b'\n'), 'cache_ledger_prefix_incomplete')
    # Include referenced schemas and callable identities: test/runtime changes
    # to loaded validators cannot borrow a prior successful validation.
    require(_loaded_schema_fingerprint() == _PROCESS_SCHEMA_FINGERPRINT,
            'cache_validation_environment_changed')
    key = (registry.validate, registry.schema_validator, registry.code_validator,
           registry.code_validator('vllm-call'))
    _validate_ledger_prefix(raw, key)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def artifact(path, workspace):
    path = _read_path(path, workspace)
    return {'path': path.as_posix(), 'sha256': sha256(path.read_bytes()).hexdigest()}


def _read_path(path, workspace):
    # Historical raw responses may live in another MOSAIC checkout. These
    # frozen absolute references are read-only; every new write still uses
    # Workspace. Relative references belong to the current workspace.
    path = Path(path)
    return path if path.is_absolute() else workspace.check(path)


def read_artifact(ref, workspace):
    data = _read_path(ref['path'], workspace).read_bytes()
    if 'bytes' in ref:
        require(type(ref['bytes']) is int and 0 <= ref['bytes'] <= len(data), 'cache_prefix_size_invalid')
        data = data[:ref['bytes']]
    require(sha256(data).hexdigest() == ref['sha256'], 'cache_artifact_changed')
    return data


def verify_artifacts(receipt, workspace):
    for ref in receipt['artifacts']:
        read_artifact(ref, workspace)


@dataclass(frozen=True)
class VerifiedCachedResponse:
    request_sha256: str
    outcome: dict
    outcome_sha256: str = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'outcome_sha256', digest(self.outcome))


def load_cached_response(entry, request, *, candidate, recipe_id, workspace, scope, verify_context,
                         retain_schema_failure=None):
    """Validate one frozen origin. Any failure refuses; it never falls back live."""
    scope.check(request['work_id'], request['source_sha256s'], request['text_provenance'],
                remote=request['remote'], rights=request['rights'])
    require(entry.get('version') == 'vllm-historical-cache/v1'
            and request.get('cache_entry_sha256') == digest(entry), 'cache_entry_changed')
    require(entry['candidate'] == candidate and entry['recipe_id'] == recipe_id
            and entry['request_id'] == request['id']
            and entry['binding_sha256'] == request['binding_sha256'], 'cache_identity_changed')
    origin = entry['origin']
    recipe_receipt = json.loads(read_artifact(origin['recipe_receipt'], workspace))
    compatibility = None
    recipe_artifacts = []
    if recipe_receipt.get('recipe_id') != recipe_id:
        proof_ref = entry.get('recipe_compatibility')
        require(isinstance(proof_ref, dict), 'cache_recipe_changed')
        require(set(proof_ref) == {'path', 'sha256'}, 'cache_recipe_full_artifact_required')
        proof = json.loads(read_artifact(proof_ref, workspace))
        compatibility = verify_schema_extension(proof, historical_id=recipe_receipt.get('recipe_id'),
            current_id=recipe_id, candidate=candidate, read=lambda ref: read_artifact(ref, workspace))
        recipe_artifacts = [proof_ref, *compatibility['artifacts']]
    else:
        require('recipe_compatibility' not in entry, 'cache_unneeded_recipe_transition')
    ledger_ref = origin['ledger']
    path = _read_path(ledger_ref['path'], workspace)
    require(path.name == ledger_path(candidate).name, 'cache_candidate_ledger_changed')
    ledger_raw = read_artifact(ledger_ref, workspace)
    rows = _ledger_rows(ledger_raw)
    selected = [r for r in rows if r['idempotency_key'] == origin['idempotency_key']
                and r.get('attempt', 1) == origin['attempt']]
    intents = [r for r in selected if r['row_kind'] == 'intent']
    results = [r for r in selected if r['row_kind'] == 'result']
    require(len(selected) == 2 and len(intents) == len(results) == 1, 'cache_unique_attempt_required')
    intent, result = intents[0], results[0]
    for row in (intent, result):
        require(row['route'] == 'chat_completions' and row['target_work_id'] == request['work_id']
                and row.get('source_sha256s') == sorted(set(request['source_sha256s']))
                and row.get('text_provenance') == request['text_provenance'], 'cache_source_changed')
    require(result['outcome'] == 'success' and result.get('http_status') == 200
            and type(result.get('output_tokens')) is int and result['output_tokens'] >= 0
            and type(result.get('latency_ms')) is int and result['latency_ms'] >= 0, 'cache_bounded_success_required')
    require(all(intent[k] == result[k] for k in ('request_sha256', 'packet_sha256', 'batch_id')), 'cache_result_identity_changed')
    require(intent.get('request_body_sha256') == intent['request_sha256'], 'cache_request_body_changed')
    require(origin['idempotency_key'] == VLLMClient.idempotency_key('chat_completions', request['work_id'],
                intent['batch_id'], intent['request_sha256']), 'cache_idempotency_changed')
    raw_root = path.parent / (path.stem + '.private.responses') / origin['idempotency_key'] / str(origin['attempt'])
    request_ref = {'path': (raw_root / 'request.bin').as_posix(), 'sha256': intent['request_sha256']}
    response_ref = {'path': (raw_root / 'response.bin').as_posix(), 'sha256': result['response_sha256']}
    raw_request = read_artifact(request_ref, workspace)
    raw_response = read_artifact(response_ref, workspace)
    task = json.loads(read_artifact(origin['task'], workspace))
    old_context = json.loads(read_artifact(origin['context'], workspace))
    provenance = origin.get('provenance_artifacts', [])
    for ref in provenance:
        read_artifact(ref, workspace)
    body = deepcopy(request['task']['body'])
    if request['stream'] or body.get('stream') is True:
        body.update(stream=True, stream_options={**body.get('stream_options', {}), 'include_usage': True})
    require(raw_request == chat_request_bytes(body), 'cache_wire_changed')
    historical_body = deepcopy(task['body'])
    if body.get('stream') is True:
        historical_body.update(stream=True, stream_options={**historical_body.get('stream_options', {}), 'include_usage': True})
    require(chat_request_bytes(historical_body) == raw_request, 'cache_historical_task_wire_changed')
    require(digest(task['payload']) == intent['packet_sha256']
            and digest(task['schema']) == digest(request['task']['schema']), 'cache_historical_task_changed')
    context_receipt = verify_context(old_context, request['cache_context'], task, request['task'])
    response = decode_chat_stream(raw_response) if body.get('stream') is True else json.loads(raw_response)
    require(response.get('model') == body['model'] == result.get('model'), 'cache_model_changed')
    choices = response.get('choices')
    require(isinstance(choices, list) and len(choices) == 1 and choices[0].get('finish_reason') == 'stop',
            'cache_complete_response_required')
    answer = json.loads(choices[0]['message']['content'])
    fields = answer_outcome(request['task']['schema'], answer)
    retention = None
    if fields['status'] == 'invalid_response':
        if retain_schema_failure is None:
            Draft202012Validator(request['task']['schema']).validate(answer)
        else:
            # A caller may retain a complete failed proposal under a frozen,
            # bounded per-fact contract. It never becomes a schema-valid answer.
            retention = retain_schema_failure(request, answer)
    usage = response.get('usage', {})
    output_tokens = usage.get('completion_tokens', usage.get('output_tokens'))
    input_tokens = usage.get('prompt_tokens', usage.get('input_tokens'))
    require(type(output_tokens) is int and output_tokens == result['output_tokens']
            and output_tokens <= body['max_tokens'] and type(input_tokens) is int and input_tokens >= 0
            and input_tokens == result.get('input_tokens'), 'cache_usage_changed')
    receipt = {'version': 'verified-historical-cache/v1', 'entry_sha256': digest(entry),
        'candidate': candidate, 'recipe_id': recipe_id, 'idempotency_key': origin['idempotency_key'],
        'attempt': origin['attempt'], 'intent_sha256': digest(intent), 'result_sha256': digest(result),
        'original_packet_sha256': intent['packet_sha256'], 'current_packet_sha256': digest(request['task']['payload']),
        'context': context_receipt, 'historical_output_tokens': result['output_tokens'],
        'historical_request_seconds': result['latency_ms'] / 1000, 'new_provider_calls': 0,
        'artifacts': [ledger_ref, origin['task'], origin['context'], origin['recipe_receipt'], request_ref, response_ref,
                      *provenance, *recipe_artifacts],
        'scientifically_accepted': False}
    if compatibility is not None:
        receipt['recipe_compatibility'] = compatibility
    if retention is not None:
        receipt['schema_failure_retention'] = retention
    return VerifiedCachedResponse(digest(request), {'id': request['id'], 'request_sha256': digest(request),
        **fields, 'response': response, 'seconds': 0,
        'origin': 'historical_cache', 'cache_receipt': receipt})
