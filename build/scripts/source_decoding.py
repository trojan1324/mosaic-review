"""Opt-in, raw-preserving evidence for separately proved source decoding.

Frozen proofs and diagnostic views are inputs, never rewritten source.
Specialized adapters validate exact font or linked-footnote witnesses; this
module preserves their raw evidence through each derived excerpt. No textual
guess or scientific interpretation authorizes a source substitution.
"""
from copy import deepcopy
from hashlib import sha256
import re

from mosaic.canonical import canonical_bytes

VERSION = 'bound-font-decoding/v1'
SLICE_VERSION = 'bound-font-decoding-slice/v1'
COMBINED_VERSION = 'bound-source-decoding/v2'
COMBINED_SLICE_VERSION = 'bound-source-decoding-slice/v2'
FONT_SHA256 = '4c4bc913479c4c4e677dc5232184568bd575ac27326dc72915c02e3f30e3a0e1'
_HASH = re.compile(r'[0-9a-f]{64}\Z')


def digest(value):
    return sha256(canonical_bytes(value)).hexdigest()


def _require(ok, reason):
    if not ok:
        raise ValueError(reason)


def _edit(edit):
    if edit.get('kind') == 'reviewed_font_glyph':
        from scripts.source_font_program import validate_edit
        validate_edit(edit)
        return
    if edit.get('kind') == 'linked_endnote_marker':
        from scripts.source_footnotes import validate_edit
        validate_edit(edit)
        return
    _require('kind' not in edit, 'decoding_unproved_mapping')
    _require(type(edit.get('start')) is int and type(edit.get('end')) is int
             and 0 <= edit['start'] < edit['end'] == edit['start'] + 1,
             'decoding_edit_offsets')
    _require(edit.get('raw') == '\x88' and edit.get('decoded') == '='
             and edit.get('embedded_encoding') == '136 /equals'
             and edit.get('embedded_font_sha256') == FONT_SHA256,
             'decoding_unproved_mapping')


def _apply(raw, edits, offset=0):
    chars = list(raw); previous = -1
    for edit in edits:
        _edit(edit)
        start = edit['start'] - offset
        _require(previous < start < len(chars), 'decoding_edit_order_or_range')
        _require(chars[start] == edit['raw'], 'decoding_raw_character_changed')
        chars[start] = edit['decoded']; previous = start
    return ''.join(chars)


def prepare(raw, view, proof, *, slot, raw_sha256, view_sha256, proof_sha256, pdf_sha256):
    """Bind separately frozen source, view and proof identities before use."""
    for value, expected in ((raw, raw_sha256), (view, view_sha256), (proof, proof_sha256)):
        _require(isinstance(expected, str) and _HASH.fullmatch(expected)
                 and digest(value) == expected, 'decoding_frozen_input_changed')
    _require(proof.get('version') == 'body-font-offsets/v1', 'decoding_proof_version')
    _require(not proof.get('refusals'), 'decoding_proof_has_refusals')
    _require(proof['inputs']['pdf_sha256'].get(str(slot)) == pdf_sha256, 'decoding_pdf_changed')
    receipt = view.get('font_decoding_receipt', {})
    _require(receipt.get('version') == 'reviewed-font-decoding/v1'
             and receipt.get('parent_source_sha256') == raw_sha256
             and receipt.get('offset_proof_sha256') == proof_sha256
             and receipt.get('pdf_sha256') == pdf_sha256
             and receipt.get('production_accepted') is False
             and receipt.get('extraction_consumption_authorized') is False,
             'decoding_diagnostic_receipt_changed')
    _require({k: v for k, v in view.items() if k not in ('units', 'font_decoding_receipt')}
             == {k: v for k, v in raw.items() if k != 'units'}, 'decoding_source_metadata_changed')
    _require([u['unit_id'] for u in view['units']] == [u['unit_id'] for u in raw['units']]
             and len({u['unit_id'] for u in raw['units']}) == len(raw['units']), 'decoding_source_roster_changed')
    points = [r for r in proof['proved_offsets'] if r['slot'] == slot]
    lookup = {(r['unit_id'], r['unit_offset']): r for r in points}
    _require(len(lookup) == len(points), 'decoding_duplicate_proof')
    bound = {'version': VERSION, 'raw_source_sha256': raw_sha256, 'diagnostic_view_sha256': view_sha256,
             'proof_sha256': proof_sha256, 'pdf_sha256': pdf_sha256,
             'raw_source_unchanged': True, 'scientifically_accepted': False, 'production_accepted': False}
    result = deepcopy(view); covered = set()
    for original, derived in zip(raw['units'], result['units']):
        if original == derived:
            _require('text_derivation' not in original, 'decoding_already_derived')
            continue
        _require('text_derivation' not in original
                 and {k: v for k, v in derived.items() if k not in ('text', 'text_derivation')}
                 == {k: v for k, v in original.items() if k != 'text'}, 'decoding_unit_metadata_changed')
        record = derived.get('text_derivation', {})
        _require(record.get('version') == 'reviewed-font-decoding/v1'
                 and record.get('raw_text') == original['text']
                 and record.get('raw_unit_sha256') == sha256(original['text'].encode('utf-8')).hexdigest()
                 and record.get('length_preserved') is True and record.get('scientifically_accepted') is False,
                 'decoding_raw_unit_changed')
        edits = record.get('changes', [])
        _require(edits and _apply(original['text'], edits) == derived['text'], 'decoding_view_changed')
        for edit in edits:
            key = (original['unit_id'], edit['start']); point = lookup.get(key)
            _require(point is not None and key not in covered, 'decoding_unproved_offset')
            _require(point['replacement'] == '=' and point['glyph_code'] == 136
                     and point['font'] == 'informMathSymbols' and point['rawdict_line_equals_native'] is True
                     and point['page'] == original['page'] == edit['physical_page_index'] + 1
                     and point['bbox'] == edit['bbox'] and point['rawdict_origin'] == edit['origin'],
                     'decoding_geometry_or_font_changed')
            covered.add(key)
        record['bound_receipt'] = deepcopy(bound)
    _require(covered == set(lookup) and len(covered) == receipt['decoded_character_count'], 'decoding_proof_coverage_changed')
    result['decoding_validation_receipt'] = {**bound, 'decoded_character_count': len(covered)}
    return result


def validate_source(source, *, pdf_sha256):
    """Reconstruct both frozen parents before a bound source enters a task."""
    if 'footnote_validation_receipt' in source:
        from scripts.source_footnotes import validate_source as validate_footnotes
        validate_footnotes(source, pdf_sha256=pdf_sha256)
        return
    if 'font_program_validation_receipt' in source:
        from scripts.source_font_program import validate_source as validate_font_program
        validate_font_program(source, pdf_sha256=pdf_sha256)
        return
    has_derivation = any('text_derivation' in u for u in source['units'])
    if not has_derivation and 'font_decoding_receipt' not in source and 'decoding_validation_receipt' not in source:
        return
    receipt = source.get('decoding_validation_receipt', {})
    _require(receipt.get('version') == VERSION and receipt.get('pdf_sha256') == pdf_sha256
             and receipt.get('raw_source_unchanged') is True
             and receipt.get('scientifically_accepted') is False
             and receipt.get('production_accepted') is False, 'decoding_bound_source_required')
    bound = {k: v for k, v in receipt.items() if k != 'decoded_character_count'}
    view = deepcopy(source); view.pop('decoding_validation_receipt')
    raw = deepcopy(view); raw.pop('font_decoding_receipt', None); count = 0
    for derived, original in zip(view['units'], raw['units']):
        if 'text_derivation' not in derived:
            continue
        record = derived['text_derivation']
        _require(record.pop('bound_receipt', None) == bound, 'decoding_unit_receipt_changed')
        _require(_apply(record['raw_text'], record['changes']) == derived['text']
                 and sha256(record['raw_text'].encode('utf-8')).hexdigest() == record['raw_unit_sha256'],
                 'decoding_unit_changed')
        original['text'] = record['raw_text']; original.pop('text_derivation')
        count += len(record['changes'])
    _require(count == receipt['decoded_character_count']
             and digest(view) == receipt['diagnostic_view_sha256']
             and digest(raw) == receipt['raw_source_sha256'], 'decoding_parent_reconstruction_changed')
    diagnostic = view.get('font_decoding_receipt', {})
    _require(diagnostic.get('offset_proof_sha256') == receipt['proof_sha256']
             and diagnostic.get('parent_source_sha256') == receipt['raw_source_sha256']
             and diagnostic.get('pdf_sha256') == pdf_sha256, 'decoding_proof_identity_changed')


def unit_slice(unit, start, end):
    """Return the existing exact segment shape, adding explicit raw provenance."""
    _require(type(start) is int and type(end) is int and 0 <= start < end <= len(unit['text']),
             'decoding_segment_offsets')
    result = {'unit_id': unit['unit_id'], 'start': start, 'end': end, 'text': unit['text'][start:end]}
    record = unit.get('text_derivation')
    if record is None:
        return result
    bound = record.get('bound_receipt', {})
    _require(bound.get('version') in (VERSION, COMBINED_VERSION) and bound.get('scientifically_accepted') is False
             and bound.get('production_accepted') is False, 'decoding_bound_receipt_required')
    _require(record.get('version') == ('reviewed-source-decoding/v2' if bound['version'] == COMBINED_VERSION
                                      else 'reviewed-font-decoding/v1'), 'decoding_unit_version_changed')
    raw = record['raw_text']; edits = record['changes']
    _require(sha256(raw.encode('utf-8')).hexdigest() == record['raw_unit_sha256']
             and _apply(raw, edits) == unit['text'], 'decoding_unit_changed')
    result['text_derivation'] = {
        'version': COMBINED_SLICE_VERSION if bound['version'] == COMBINED_VERSION else SLICE_VERSION,
        'raw_text': raw[start:end], 'raw_unit_sha256': record['raw_unit_sha256'],
        'view_unit_sha256': sha256(unit['text'].encode('utf-8')).hexdigest(),
        'proof_sha256': bound['proof_sha256'], 'pdf_sha256': bound['pdf_sha256'],
        'changes': [deepcopy(e) for e in edits if start <= e['start'] < end]}
    if bound['version'] == COMBINED_VERSION:
        result['text_derivation']['proof_chain'] = deepcopy(bound['proof_chain'])
    validate_slice(result)
    return result


def validate_slice(segment):
    record = segment.get('text_derivation')
    if record is None:
        return
    _require(record.get('version') in (SLICE_VERSION, COMBINED_SLICE_VERSION), 'decoding_slice_version')
    if record['version'] == COMBINED_SLICE_VERSION:
        chain = record.get('proof_chain')
        _require(isinstance(chain, list) and chain and len(set(chain)) == len(chain)
                 and all(isinstance(h, str) and _HASH.fullmatch(h) for h in chain)
                 and chain[-1] == record['proof_sha256'], 'decoding_proof_chain_changed')
    else:
        _require(all('kind' not in e for e in record['changes']), 'decoding_slice_version')
    for key in ('raw_unit_sha256', 'view_unit_sha256', 'proof_sha256', 'pdf_sha256'):
        _require(isinstance(record.get(key), str) and _HASH.fullmatch(record[key]), 'decoding_slice_hash')
    _require(segment.get('unit_text_sha256', record['view_unit_sha256']) == record['view_unit_sha256'],
             'decoding_slice_unit_changed')
    _require(len(record['raw_text']) == segment['end'] - segment['start']
             and _apply(record['raw_text'], record['changes'], segment['start']) == segment['text'],
             'decoding_slice_text_changed')


def subspan(segment, start, end):
    """Clip raw and decoded text together; arguments are relative offsets."""
    validate_slice(segment)
    _require(type(start) is int and type(end) is int and 0 <= start < end <= len(segment['text']),
             'decoding_subspan_offsets')
    result = deepcopy(segment)
    result.update(start=segment['start'] + start, end=segment['start'] + end, text=segment['text'][start:end])
    if 'text_derivation' in result:
        record = result['text_derivation']; record['raw_text'] = record['raw_text'][start:end]
        record['changes'] = [e for e in record['changes'] if result['start'] <= e['start'] < result['end']]
        validate_slice(result)
    return result


def model_view(value):
    """Keep raw proof metadata locally without duplicating it in model input."""
    if isinstance(value, dict):
        return {key: model_view(item) for key, item in value.items()
                if key not in ('text_derivation', 'statement_continuation')}
    if isinstance(value, list):
        return [model_view(item) for item in value]
    return value


def decoding_status(segments):
    versions = [s.get('text_derivation', {}).get('version') for s in segments]
    if COMBINED_SLICE_VERSION in versions:
        return 'reviewed_source_decoding'
    return 'reviewed_font_decoding' if SLICE_VERSION in versions else 'indexed_text_only'
