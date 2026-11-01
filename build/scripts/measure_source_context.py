"""Restore full original units already identified by a measure's citations.

No neighboring blocks are inferred. Existing unit hashes bind every added
character; the expansion does not establish measure ownership or acceptance.
"""
from copy import deepcopy
from hashlib import sha256

from scripts.measure_catalog import _rights
from scripts.measure_field_trial import _digest, _require, _sources
from scripts.source_decoding import unit_slice


VERSION = "mosaic-measure-source-context/v1"


def _key(unit):
    return (unit['unit_id'], unit['source_locator'], unit['page'])


def source_context(version, source):
    rows = _sources(version)
    keys = {_key(row['segment']) for row in rows}
    lookup = {_key(unit): unit for unit in source['units']}
    _require(len(lookup) == len(source['units']), 'measure_context_duplicate_source_unit')
    _require(keys <= set(lookup), 'measure_context_source_unit_missing')
    context = {
        'version': VERSION, 'base_version_sha256': _digest(version),
        'source_packet_sha256': _digest(source),
        'units': [{key: deepcopy(unit[key]) for key in ('unit_id', 'source_locator', 'page', 'text', 'text_derivation') if key in unit}
                  for unit in source['units'] if _key(unit) in keys],
    }
    restore_context(version, context)
    return context


def restore_context(version, context):
    _require(not version.get('field_context_evidence'), 'measure_context_already_expanded')
    _require(context.get('version') == VERSION
             and context.get('base_version_sha256') == _digest(version), 'measure_context_base_changed')
    original = _sources(version)
    lookup = {_key(unit): unit for unit in context['units']}
    _require(len(lookup) == len(context['units']), 'measure_context_duplicate_source_unit')
    _require(set(lookup) == {_key(row['segment']) for row in original}, 'measure_context_unrelated_unit')
    for row in original:
        segment = row['segment']; unit = lookup[_key(segment)]
        _require(sha256(unit['text'].encode('utf-8')).hexdigest() == segment['unit_text_sha256'],
                 'measure_context_unit_changed')
        _require(unit['text'][segment['start']:segment['end']] == segment['text'],
                 'measure_context_citation_changed')
        _require(unit_slice(unit, segment['start'], segment['end']).get('text_derivation')
                 == segment.get('text_derivation'), 'measure_context_derivation_changed')
    _require(sum(len(unit['text']) for unit in context['units']) <= 24000, 'measure_context_too_large')
    out = deepcopy(version)
    out['field_context_evidence'] = [
        {'quote': unit['text'], 'pages': [unit['page']],
         'segments': [{**unit_slice(unit, 0, len(unit['text'])), 'page': unit['page'], 'source_locator': unit['source_locator'],
                       'unit_text_sha256': sha256(unit['text'].encode('utf-8')).hexdigest()}]}
        for unit in context['units']]
    out['field_context_receipt'] = {key: context[key] for key in
        ('version', 'base_version_sha256', 'source_packet_sha256')}
    out['rights'] = _rights(out['name'], [], out['name_evidence'],
                            out['description_evidence'] + out['field_context_evidence'], out['item_evidence'])
    return out
