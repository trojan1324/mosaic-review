"""Lossless prompt representation, independent of the unchanged decoder grammar.

Opt-in experiment: schema equivalence does not prove equal model accuracy or speed.
Only schema positions are factored; enum/const/example data are never rewritten.
"""
from collections import Counter
from copy import deepcopy
import json


MARKER = '\nRESPONSE_SCHEMA:\n'
MAPS = {'properties', 'patternProperties', '$defs', 'dependentSchemas'}
SINGLE = {'items', 'additionalProperties', 'contains', 'not', 'if', 'then', 'else',
          'propertyNames', 'unevaluatedProperties', 'unevaluatedItems'}
ARRAYS = {'allOf', 'anyOf', 'oneOf', 'prefixItems'}


def encode(value):
    # Preserve member order, including the order shown to the native decoder.
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def _children(node, visit):
    result = deepcopy(node)
    for key, value in node.items():
        if key in MAPS:
            result[key] = {name: visit(child) for name, child in value.items()}
        elif key in SINGLE and isinstance(value, (dict, bool)):
            result[key] = visit(value)
        elif key in ARRAYS:
            result[key] = [visit(child) for child in value]
    return result


def factor_schema(schema):
    """Share repeated schema nodes and prove exact expansion, including order.

    Resource IDs, dynamic scope and pointers into movable schema interiors are
    deliberately unsupported. Callers can still use their original prompt.
    """
    frequencies = Counter()

    def collect(node, root=False):
        if not isinstance(node, dict):
            return node
        if any(k in node for k in ('$id', '$anchor', '$dynamicAnchor', '$dynamicRef', '$recursiveRef')):
            raise ValueError('schema_reference_scope_not_supported')
        ref = node.get('$ref')
        if ref is not None and (not ref.startswith('#/$defs/') or '/' in ref[len('#/$defs/'):]):
            raise ValueError('schema_reference_scope_not_supported')
        if not root and '$defs' in node:
            raise ValueError('nested_schema_definitions_not_supported')
        key = encode(node)
        if len(key) > 120 and not root:
            frequencies[key] += 1
        return _children(node, lambda child: collect(child))

    collect(schema, root=True)
    names = {}
    used = set(schema.get('$defs', {}))
    for key, count in frequencies.items():
        if count < 2:
            continue
        index = len(names) + 1
        name = 'compact_' + str(index)
        while name in used:
            index += 1
            name = 'compact_' + str(index)
        names[key] = name
        used.add(name)
    definitions = {}

    def replace(node, preserve=False):
        if not isinstance(node, dict):
            return node
        key = encode(node)
        if not preserve and key in names:
            name = names[key]
            if name not in definitions:
                definitions[name] = replace(node, preserve=True)
            return {'$ref': '#/$defs/' + name}
        return _children(node, replace)

    candidate = replace(schema, preserve=True)
    generated = deepcopy(definitions)
    if generated:
        candidate.setdefault('$defs', {}).update(generated)

    def expand(node):
        if not isinstance(node, dict):
            return node
        name = node.get('$ref', '').removeprefix('#/$defs/')
        if set(node) == {'$ref'} and name in generated:
            return expand(generated[name])
        return _children(node, expand)

    restored = deepcopy(candidate)
    if generated:
        for name in generated:
            del restored['$defs'][name]
        if '$defs' not in schema:
            del restored['$defs']
    if encode(expand(restored)) != encode(schema):
        raise ValueError('schema_roundtrip_failed')
    return candidate


def compact_task(task, count):
    """Copy a prepared task; alter only schema/payload formatting in messages."""
    from scripts.integrated_extraction_trial import prompt_estimate
    body = task['body']
    before = prompt_estimate(body, count)
    instruction, schema_text = body['messages'][0]['content'].split(MARKER, 1)
    shown = json.loads(schema_text)
    if encode(shown) != encode(task['schema']):
        raise ValueError('prompt_schema_mismatch')
    if encode(json.loads(body['messages'][1]['content'])) != encode(task['payload']):
        raise ValueError('prompt_payload_mismatch')
    variants = [('compact_json', shown)]
    try:
        variants.append(('shared_schema', factor_schema(shown)))
    except ValueError as exc:
        if str(exc) not in ('schema_reference_scope_not_supported', 'nested_schema_definitions_not_supported'):
            raise
    best, best_size, mode = deepcopy(task), before, 'unchanged'
    for label, schema in variants:
        candidate = deepcopy(task)
        candidate['body']['messages'][0]['content'] = instruction + MARKER + encode(schema)
        candidate['body']['messages'][1]['content'] = encode(task['payload'])
        size = prompt_estimate(candidate['body'], count)
        if size < best_size:
            best, best_size, mode = candidate, size, label
    best['prompt_compaction'] = {'version': 'lossless-prompt-v1', 'mode': mode,
        'before_tokens': before, 'after_tokens': best_size,
        'source_payload_unchanged': True, 'decoder_unchanged': True,
        'model_equivalence_measured': False}
    return best
