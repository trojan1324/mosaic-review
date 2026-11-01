"""One explicit recipe transition; no general recipe-hash bypass.

The model, parser, tokenizer, prompt and all other resolved recipe fields are
unchanged. Only the source-policy schema accepts one additional version. Actual
source/parent/task/wire equivalence still belongs to the cache's ordinary gates.
"""
from copy import deepcopy
from hashlib import sha256
import json
import re

from mosaic.canonical import canonical_bytes
from mosaic.vllm.reservation import require

PROFILE = 'source-policy-v5-recipe-compatibility/v1'
RECIPE_PAIR = (
    'recipe-ffb2edc8a45041c181b18ca4deb24d0802f336f810db7e1b0d5cb07e13faa492',
    'recipe-bf3dcfd1bd9935d594ad366b7279dedfc576fa81b6c1aaadb1b18d3c2a33c704')
POLICY_FILE = 'model-source-policy.schema.json'
POLICY_HASHES = (
    'd575f4ed5157a0f5d191269d89080da3d50e1c6c321090684d9c4541ae391856',
    'b50ab159e220eb3e42d05da5f4e94f5b460c23785d51772e242a48400201afec')


def verify_schema_extension(proof, *, historical_id, current_id, candidate, read):
    """Verify full original bytes and retain both identities, never replace one."""
    require(proof.get('version') == PROFILE and (historical_id, current_id) == RECIPE_PAIR,
            'cache_recipe_transition_not_allowed')
    refs = []

    def read_full(ref):
        require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'cache_recipe_full_artifact_required')
        refs.append(ref)
        return read(ref)

    old = json.loads(read_full(proof['historical_recipe']))
    new = json.loads(read_full(proof['current_recipe']))
    require(tuple('recipe-' + sha256(canonical_bytes(r)).hexdigest() for r in (old, new)) == RECIPE_PAIR,
            'cache_recipe_definition_changed')
    require(old.get('name') == new.get('name') == candidate
            and old.get('candidate_backend') == new.get('candidate_backend') == 'vllm',
            'cache_recipe_candidate_changed')
    require(canonical_bytes({k: v for k, v in old.items() if k != 'schemas_sha256'})
            == canonical_bytes({k: v for k, v in new.items() if k != 'schemas_sha256'}),
            'cache_recipe_non_schema_change')
    schemas = proof['current_schemas']
    require(isinstance(schemas, dict) and POLICY_FILE in schemas and all(
        re.fullmatch(r'[a-zA-Z0-9_-]+\.schema\.json', name) for name in schemas),
        'cache_recipe_schema_roster_invalid')
    old_policy = read_full(proof['historical_source_policy_schema'])
    new_policy = None
    old_hash, new_hash = sha256(), sha256()
    for name in sorted(schemas):
        raw = read_full(schemas[name])
        old_hash.update(name.encode('utf-8'))
        old_hash.update(old_policy if name == POLICY_FILE else raw)
        new_hash.update(name.encode('utf-8'))
        new_hash.update(raw)
        if name == POLICY_FILE:
            new_policy = raw
    require((sha256(old_policy).hexdigest(), sha256(new_policy).hexdigest()) == POLICY_HASHES,
            'cache_recipe_policy_bytes_changed')
    require(old_hash.hexdigest() == old['schemas_sha256'] and new_hash.hexdigest() == new['schemas_sha256'],
            'cache_recipe_schema_bundle_changed')
    reverted = deepcopy(json.loads(new_policy))
    for enums in (reverted['properties']['version']['enum'], reverted['if']['properties']['version']['enum']):
        require(enums.count('mosaic.model-source-policy/5') == 1, 'cache_recipe_policy_extension_changed')
        enums.remove('mosaic.model-source-policy/5')
    require(canonical_bytes(reverted) == canonical_bytes(json.loads(old_policy)),
            'cache_recipe_policy_extension_changed')
    return {'profile': PROFILE, 'historical_recipe_id': historical_id, 'current_recipe_id': current_id,
        'historical_schemas_sha256': old_hash.hexdigest(), 'current_schemas_sha256': new_hash.hexdigest(),
        'schema_count': len(schemas), 'all_other_recipe_fields_identical': True,
        'only_change': 'Source-policy version5 added to two schema enums.',
        'source_parent_task_and_wire_checks_still_required': True,
        'scientifically_accepted': False, 'artifacts': refs}
