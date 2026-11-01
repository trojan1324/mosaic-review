"""Decode only frozen, visually reviewed font-program/glyph combinations.

These subset fonts give mathematical glyphs misleading ASCII names. The exact
program hash, glyph ID and raw character bind a reviewed rendering; a font name
or textual pattern alone never permits replacement. All source digits survive
in raw provenance, and this layer makes no statistical interpretation.
"""
from copy import deepcopy
from hashlib import sha256
import math
import re

from mosaic.canonical import canonical_bytes

VERSION = 'bound-font-program-decoding/v1'
BOUND_VERSION = 'bound-source-decoding/v2'
FIRST_FONT = '6e107dfa2fa9ce7a1b71362b5bbc289feacce3c3457b1d0e4e0e29642e927256'
SECOND_FONT = 'c1591d7b60edd75230735dd4913fe2dff77b81f6661e3de1d02e92e948c7b995'
MAPPINGS = {(FIRST_FONT, 9, 'a'): 'α', (FIRST_FONT, 6, '5'): '=',
            (SECOND_FONT, 10, 'a'): 'α', (SECOND_FONT, 6, '5'): '=', (SECOND_FONT, 2, '.'): '>'}
_HASH = re.compile(r'[0-9a-f]{64}\Z')


def digest(value):
    return sha256(canonical_bytes(value)).hexdigest()


def _require(ok, code):
    if not ok:
        raise ValueError(code)


def _hash(value):
    return isinstance(value, str) and _HASH.fullmatch(value)


def _coords(value, size):
    return isinstance(value, (list, tuple)) and len(value) == size and all(
        type(x) in (int, float) and math.isfinite(x) for x in value)


def validate_edit(edit):
    key = (edit.get('embedded_font_sha256'), edit.get('glyph_id'), edit.get('raw'))
    _require(edit.get('kind') == 'reviewed_font_glyph' and key in MAPPINGS
             and edit.get('decoded') == MAPPINGS[key], 'font_program_unreviewed_mapping')
    _require(type(edit.get('start')) is int and type(edit.get('end')) is int
             and 0 <= edit['start'] < edit['end'] == edit['start']+1, 'font_program_offset_invalid')
    _require(edit.get('font') == 'AdvOT8817665d' and edit.get('font_extension') == 'cff'
             and edit.get('font_kind') == 'Type1' and edit.get('trace_unicode') == ord(edit['raw'])
             and edit.get('trace_direction') == [1.0, 0.0] and edit.get('writing_mode') == 0,
             'font_program_identity_or_orientation')
    _require(_coords(edit.get('origin'), 2) and _coords(edit.get('trace_origin'), 2)
             and all(abs(a-b) <= .03 for a,b in zip(edit['origin'], edit['trace_origin']))
             and _coords(edit.get('bbox'), 4) and edit['bbox'][0] < edit['bbox'][2]
             and edit['bbox'][1] < edit['bbox'][3]
             and type(edit.get('font_size')) in (int, float) and math.isfinite(edit['font_size'])
             and edit['font_size'] > 0, 'font_program_geometry_changed')
    _require(type(edit.get('physical_page_index')) is int and edit['physical_page_index'] >= 0
             and edit.get('rawdict_line_equals_native') is True
             and _hash(edit.get('raw_unit_sha256')) and _hash(edit.get('native_line_sha256'))
             and _hash(edit.get('review_sha256')) and _hash(edit.get('review_render_sha256')),
             'font_program_source_witness_required')


def prepare(raw, proof, *, raw_sha256, proof_sha256, pdf_sha256):
    from scripts.source_decoding import _apply, validate_source
    _require(all(_hash(h) for h in (raw_sha256, proof_sha256, pdf_sha256)), 'font_program_input_hash_required')
    _require(digest(raw) == raw_sha256 and digest(proof) == proof_sha256, 'font_program_frozen_input_changed')
    _require(proof.get('version') == 'reviewed-font-program-offsets/v1'
             and proof.get('source_sha256') == raw_sha256 and proof.get('pdf_sha256') == pdf_sha256
             and proof.get('source_unchanged') is True and proof.get('scientifically_accepted') is False
             and proof.get('production_accepted') is False and proof.get('refusals') == [],
             'font_program_proof_identity_changed')
    _require('font_program_validation_receipt' not in raw
             and all('text_derivation' not in u for u in raw['units']), 'font_program_raw_parent_required')
    validate_source(raw, pdf_sha256=pdf_sha256)
    review = proof['review']
    _require(review.get('version') == 'reviewed-font-program-rules/v1'
             and review.get('glyph_names_are_not_authoritative') is True
             and review.get('scientifically_accepted') is False and review.get('production_accepted') is False
             and digest(review) == proof.get('review_sha256'), 'font_program_review_changed')
    rules = {r['rule']: r for r in review['rows']}
    _require(len(rules) == len(review['rows']) and rules, 'font_program_review_rules_required')
    for rule in rules.values():
        key = (rule['embedded_font_sha256'], rule['glyph_id'], rule['raw'])
        _require(key in MAPPINGS and rule['proposed_decoded'] == MAPPINGS[key]
                 and rule.get('review_complete') is True and _hash(rule.get('render_sha256')),
                 'font_program_rule_not_reviewed')
    units = {u['unit_id']: u for u in raw['units']}
    _require(len(units) == len(raw['units']), 'font_program_duplicate_unit')
    _require(isinstance(proof['points'], list) and proof['points'], 'font_program_points_required')
    grouped = {}; seen = set()
    for point in proof['points']:
        validate_edit(point)
        rule = rules.get(point['review_rule'])
        _require(rule is not None and rule['slot'] == proof['slot']
                 and all(point[k] == rule[k] for k in ('embedded_font_sha256', 'glyph_id', 'raw'))
                 and point['decoded'] == rule['proposed_decoded']
                 and point['review_sha256'] == proof['review_sha256']
                 and point['review_render_sha256'] == rule['render_sha256'], 'font_program_point_review_changed')
        unit = units.get(point['unit_id'])
        _require(unit is not None and unit['page'] == point['physical_page_index']+1
                 and sha256(unit['text'].encode('utf-8')).hexdigest() == point['raw_unit_sha256']
                 and unit['text'][point['start']:point['end']] == point['raw'], 'font_program_source_offset_changed')
        key = (point['unit_id'], point['start'])
        _require(key not in seen, 'font_program_duplicate_point'); seen.add(key)
        grouped.setdefault(point['unit_id'], []).append(deepcopy(point))
    bound = {'version': BOUND_VERSION, 'raw_source_sha256': raw_sha256, 'parent_source_sha256': raw_sha256,
        'proof_sha256': proof_sha256, 'proof_chain': [proof_sha256], 'pdf_sha256': pdf_sha256,
        'raw_source_unchanged': True, 'scientifically_accepted': False, 'production_accepted': False}
    result = deepcopy(raw)
    for unit in result['units']:
        if unit['unit_id'] not in grouped:
            continue
        text = unit['text']; edits = sorted(grouped[unit['unit_id']], key=lambda p: p['start'])
        unit['text'] = _apply(text, edits)
        unit['text_derivation'] = {'version': 'reviewed-source-decoding/v2', 'raw_text': text,
            'raw_unit_sha256': sha256(text.encode('utf-8')).hexdigest(), 'changes': edits,
            'length_preserved': True, 'scientifically_accepted': False, 'bound_receipt': deepcopy(bound)}
    result['font_program_validation_receipt'] = {'version': VERSION, 'raw_source_sha256': raw_sha256,
        'proof_sha256': proof_sha256, 'pdf_sha256': pdf_sha256, 'proof': deepcopy(proof),
        'decoded_character_count': len(seen), 'raw_source_unchanged': True,
        'scientifically_accepted': False, 'production_accepted': False}
    return result


def validate_source(source, *, pdf_sha256):
    receipt = source.get('font_program_validation_receipt', {})
    _require(receipt.get('version') == VERSION and receipt.get('pdf_sha256') == pdf_sha256,
             'font_program_receipt_required')
    raw = deepcopy(source); raw.pop('font_program_validation_receipt')
    for unit in raw['units']:
        if 'text_derivation' in unit:
            unit['text'] = unit.pop('text_derivation')['raw_text']
    expected = prepare(raw, receipt['proof'], raw_sha256=receipt['raw_source_sha256'],
                       proof_sha256=receipt['proof_sha256'], pdf_sha256=pdf_sha256)
    _require(expected == source, 'font_program_view_changed')
