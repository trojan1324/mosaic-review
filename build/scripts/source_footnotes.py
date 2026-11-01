"""Separate only frozen, linked endnote markers in an explicit source view.

Source digits remain in raw evidence. Raised text alone is never enough:
the smaller marker must link to its same-numbered Endnotes paragraph.
"""
from copy import deepcopy
from hashlib import sha256
import math
import re

from mosaic.canonical import canonical_bytes

VERSION = 'bound-linked-footnotes/v1'
BOUND_VERSION = 'bound-source-decoding/v2'
_HASH = re.compile(r'[0-9a-f]{64}\Z')


def digest(value):
    return sha256(canonical_bytes(value)).hexdigest()


def _require(ok, code):
    if not ok:
        raise ValueError(code)


def _coords(value, size):
    return isinstance(value, (list, tuple)) and len(value) == size and all(
        type(x) in (int, float) and math.isfinite(x) for x in value)


def validate_edit(edit):
    """Check one already frozen geometric/link witness; never discover markers."""
    raw = edit.get('raw')
    _require(edit.get('kind') == 'linked_endnote_marker' and isinstance(raw, str)
             and re.fullmatch(r'[0-9]', raw) and edit.get('decoded') == ' ', 'footnote_mapping_invalid')
    _require(type(edit.get('start')) is int and type(edit.get('end')) is int
             and 0 <= edit['start'] < edit['end'] == edit['start'] + 1, 'footnote_offset_invalid')
    marker = edit['source_glyph']; baseline = edit['preceding_baseline_characters']
    _require(marker['c'] == raw and not marker.get('synthetic') and marker['flags'] & 1
             and _coords(marker['origin'], 2) and _coords(marker['bbox'], 4), 'footnote_marker_geometry')
    _require(baseline and all(c['c'] in '0123456789' and len(c['c']) == 1
             and not c.get('synthetic') and _coords(c['origin'], 2)
             and c['font'] == marker['font'] for c in baseline), 'footnote_baseline_required')
    _require(0 < marker['size'] < min(c['size'] for c in baseline) * .85
             and min(c['origin'][1] for c in baseline) - marker['origin'][1] > 1,
             'footnote_not_smaller_and_raised')
    link = edit['internal_link']; target = edit['target']
    _require(link.get('kind') == 4 and link.get('nameddest') == 'WEBEN' + raw
             and type(link.get('page')) is int and link['page'] >= 0
             and link['page'] == target['physical_page_index'] and _coords(link['from'], 4),
             'footnote_internal_destination_required')
    a, b = marker['bbox'], link['from']
    _require(max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3]),
             'footnote_link_misses_marker')
    _require(target['heading'] == 'Endnotes' and target['label'].strip() == raw
             and target['paragraph_text'].lstrip().startswith(raw)
             and sha256(target['paragraph_text'].encode('utf-8')).hexdigest() == target['paragraph_sha256'],
             'footnote_matching_endnote_required')
    _require(_coords(target['label_bbox'], 4) and _coords(link['to'], 2)
             and target['named_destination_pdf_y'] == link['to'][1]
             and abs(target['named_destination_top_y'] - (target['page_height'] - link['to'][1])) < .001
             and abs(target['label_bbox'][1] - target['named_destination_top_y']) < 6,
             'footnote_destination_geometry')


def prepare(parent, proof, *, parent_sha256, proof_sha256, pdf_sha256):
    from scripts.source_decoding import _apply, validate_source
    _require(all(isinstance(h, str) and _HASH.fullmatch(h) for h in (parent_sha256, proof_sha256, pdf_sha256)),
             'footnote_input_hash_required')
    _require('footnote_validation_receipt' not in parent, 'footnote_already_prepared')
    _require(digest(parent) == parent_sha256 and digest(proof) == proof_sha256,
             'footnote_frozen_input_changed')
    _require(proof.get('version') == 'linked-footnote-boundary-proof/v1'
             and proof.get('source_pdf_sha256') == pdf_sha256
             and proof.get('parent_source_sha256') == parent_sha256, 'footnote_parent_identity_changed')
    _require(proof.get('scientifically_accepted') is False
             and proof.get('review', {}).get('source_crop_reviewed') is True
             and proof.get('review', {}).get('full_target_page_reviewed') is True,
             'footnote_layout_review_required')
    validate_source(parent, pdf_sha256=pdf_sha256)
    parent_receipt = parent.get('font_program_validation_receipt', parent.get('decoding_validation_receipt', {}))
    raw_source_hash = parent_receipt.get('raw_source_sha256', parent_sha256)
    _require(proof['raw_source_sha256'] == raw_source_hash
             and proof.get('parent_font_proof_sha256') == parent_receipt.get('proof_sha256'),
             'footnote_original_source_changed')
    originals = {u['unit_id']: u for u in parent['units']}
    _require(len(originals) == len(parent['units']), 'footnote_duplicate_unit')
    grouped = {}; seen = set()
    _require(isinstance(proof['points'], list) and proof['points'], 'footnote_points_required')
    for point in proof['points']:
        validate_edit(point)
        uid = point['unit_id']; unit = originals.get(uid)
        _require(unit is not None, 'footnote_unit_missing')
        raw = unit.get('text_derivation', {}).get('raw_text', unit['text'])
        _require(sha256(raw.encode('utf-8')).hexdigest() == point['raw_unit_sha256']
                 and 0 <= point['start'] < point['end'] <= len(raw)
                 and raw[point['start']:point['end']] == point['raw']
                 and unit['text'][point['start']:point['end']] == point['raw']
                 and unit['page'] == point['physical_page_index'] + 1, 'footnote_source_offset_changed')
        baseline = ''.join(c['c'] for c in point['preceding_baseline_characters'])
        _require(point['start'] >= len(baseline)
                 and raw[point['start']-len(baseline):point['start']] == baseline,
                 'footnote_baseline_text_changed')
        key = (uid, point['start']); _require(key not in seen, 'footnote_duplicate_point'); seen.add(key)
        grouped.setdefault(uid, []).append(deepcopy(point))
    result = deepcopy(parent); parents = {}
    for unit in result['units']:
        uid = unit['unit_id']
        if uid not in grouped:
            continue
        parents[uid] = deepcopy(originals[uid]); previous = unit.get('text_derivation')
        raw = previous['raw_text'] if previous else unit['text']
        previous_changes = deepcopy(previous['changes']) if previous else []
        changes = sorted(previous_changes + grouped[uid], key=lambda e: e['start'])
        chain = deepcopy(previous.get('bound_receipt', {}).get('proof_chain', [])) if previous else []
        if not chain and parent_receipt:
            chain = [parent_receipt['proof_sha256']]
        chain.append(proof_sha256)
        record = {'version': 'reviewed-source-decoding/v2', 'raw_text': raw,
            'raw_unit_sha256': sha256(raw.encode('utf-8')).hexdigest(), 'changes': changes,
            'length_preserved': True, 'scientifically_accepted': False,
            'bound_receipt': {'version': BOUND_VERSION, 'raw_source_sha256': raw_source_hash,
                'parent_source_sha256': parent_sha256, 'proof_sha256': proof_sha256, 'proof_chain': chain,
                'pdf_sha256': pdf_sha256, 'raw_source_unchanged': True,
                'scientifically_accepted': False, 'production_accepted': False}}
        unit['text'] = _apply(raw, changes); unit['text_derivation'] = record
    result['footnote_validation_receipt'] = {'version': VERSION, 'parent_source_sha256': parent_sha256,
        'proof_sha256': proof_sha256, 'pdf_sha256': pdf_sha256, 'proof': deepcopy(proof),
        'parent_units': parents, 'marker_count': len(seen), 'raw_source_unchanged': True,
        'scientifically_accepted': False, 'production_accepted': False}
    return result


def validate_source(source, *, pdf_sha256):
    receipt = source.get('footnote_validation_receipt', {})
    _require(receipt.get('version') == VERSION and receipt.get('pdf_sha256') == pdf_sha256,
             'footnote_receipt_required')
    parent = deepcopy(source); parent.pop('footnote_validation_receipt')
    existing = {u['unit_id'] for u in parent['units']}
    _require(set(receipt['parent_units']) <= existing, 'footnote_parent_unit_missing')
    parent['units'] = [deepcopy(receipt['parent_units'].get(u['unit_id'], u)) for u in parent['units']]
    expected = prepare(parent, receipt['proof'], parent_sha256=receipt['parent_source_sha256'],
                       proof_sha256=receipt['proof_sha256'], pdf_sha256=pdf_sha256)
    _require(expected == source, 'footnote_view_changed')
