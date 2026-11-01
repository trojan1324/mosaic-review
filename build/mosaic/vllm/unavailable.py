"""Frozen no-send declarations retain an unanswered stage, never an answer.

Declarations bind to the actual request and cite immutable audit evidence.
They confer no provider, source, budget, or scientific acceptance authority.
"""
from copy import deepcopy
import re

from mosaic.vllm.cache import digest, read_artifact
from mosaic.vllm.reservation import require


VERSION = 'declared-unavailable-stage/v1'


def declare_unavailable(request, *, reason, evidence):
    """Build a declaration; execution still verifies its request and evidence."""
    return {'version': VERSION, 'request_id': request['id'],
        'binding_sha256': request['binding_sha256'], 'task_sha256': digest(request['task']),
        'reason': reason, 'artifacts': deepcopy(evidence)}


def unavailable_outcome(request, *, workspace, scope):
    """Validate a declaration before granting zero-send budget treatment."""
    if 'unavailable' not in request:
        return None
    scope.check(request['work_id'], request['source_sha256s'], request['text_provenance'],
                remote=request['remote'], rights=request['rights'])
    entry = request['unavailable']
    require(isinstance(entry, dict) and set(entry) == {
        'version', 'request_id', 'binding_sha256', 'task_sha256', 'reason', 'artifacts'},
        'unavailable_declaration_invalid')
    require(entry['version'] == VERSION and entry['request_id'] == request['id']
        and entry['binding_sha256'] == request['binding_sha256']
        and entry['task_sha256'] == digest(request['task']), 'unavailable_identity_changed')
    require('cache_entry_sha256' not in request, 'unavailable_cache_overlap')
    require(isinstance(entry['reason'], str)
        and re.fullmatch(r'[a-z][a-z0-9_]{0,127}', entry['reason']) is not None, 'unavailable_reason_invalid')
    refs = entry['artifacts']
    require(isinstance(refs, list) and 1 <= len(refs) <= 16
        and all(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}
            and isinstance(ref['path'], str) and ref['path']
            and isinstance(ref['sha256'], str) and re.fullmatch(r'[0-9a-f]{64}', ref['sha256']) is not None
            for ref in refs), 'unavailable_evidence_invalid')
    require(len({ref['path'] for ref in refs}) == len(refs), 'unavailable_evidence_invalid')
    for ref in refs:
        read_artifact(ref, workspace)
    return {'id': request['id'], 'request_sha256': digest(request), 'status': 'unavailable',
        'answer': None, 'origin': 'declared_unavailable', 'reason': entry['reason'], 'seconds': 0,
        'unavailable_receipt': {'version': VERSION, 'declaration_sha256': digest(entry),
            'artifacts': deepcopy(refs), 'new_provider_calls': 0, 'scientifically_accepted': False}}
