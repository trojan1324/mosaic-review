"""Validate frozen native-layout witnesses for new measure continuations.

The proof is prepared from captured PDF blocks and reviewed renders. It binds
only physical continuation, never a study, measure owner or complete scale.
"""
from hashlib import sha256

from scripts.core_rendered_names import _coordinates, _hash
from scripts.measure_field_trial import _digest, _require

VERSION = 'checked-measure-continuations/v1'
PROOF_VERSION = 'measure-boundary-proof/v1'


def validate_boundary_proof(proof, source, *, pdf_sha256, proof_sha256):
    _require(isinstance(proof, dict) and _hash(proof_sha256)
             and _digest(proof) == proof_sha256, 'measure_boundary_frozen_proof_changed')
    _require(proof.get('version') == PROOF_VERSION and _hash(pdf_sha256)
             and proof.get('source_pdf_sha256') == pdf_sha256
             and proof.get('source_packet_sha256') == _digest(source),
             'measure_boundary_source_changed')
    _require(proof.get('scientifically_accepted') is False
             and proof.get('source_unchanged') is True, 'measure_boundary_unverified_proof_required')
    units = {u['unit_id']: u for u in source['units']}
    positions = {u['unit_id']: i for i, u in enumerate(source['units'])}
    _require(len(units) == len(source['units']), 'measure_boundary_duplicate_unit')
    rows = proof.get('rows')
    _require(isinstance(rows, list), 'measure_boundary_rows_required')
    allowed = {}
    for row in rows:
        for key in ('native_objects_sha256', 'source_bindings_sha256', 'recorded_source_sha256'):
            _require(_hash(row.get(key)), 'measure_boundary_capture_identity_required')
        for side in ('left', 'right'):
            endpoint = row[side]
            unit = units.get(endpoint.get('unit_id'))
            _require(unit is not None and endpoint.get('source_locator') == unit['source_locator']
                     and endpoint.get('page') == unit['page'], 'measure_boundary_unit_changed')
            raw = unit.get('text_derivation', {}).get('raw_text', unit['text'])
            start, end = endpoint.get('start'), endpoint.get('end')
            _require(type(start) is int and type(end) is int and 0 <= start < end
                     and end - start == len(raw)
                     and endpoint.get('raw_unit_sha256') == sha256(raw.encode()).hexdigest(),
                     'measure_boundary_raw_span_changed')
            _require(_coordinates(endpoint.get('bbox'), 4)
                     and endpoint['bbox'][0] < endpoint['bbox'][2]
                     and endpoint['bbox'][1] < endpoint['bbox'][3]
                     and _coordinates([endpoint.get('font_size')], 1) and endpoint['font_size'] > 0,
                     'measure_boundary_geometry_invalid')
            _require(all(type(endpoint.get(k)) is int and endpoint[k] >= 0
                         for k in ('block_index', 'order_index', 'admitted_ordinal'))
                     and all(isinstance(endpoint.get(k), str) and endpoint[k]
                             for k in ('text_region_id', 'source_id')),
                     'measure_boundary_order_required')
            paths = endpoint.get('section_paths')
            _require(isinstance(paths, list) and bool(paths)
                     and all(path and all(node.get('node_id') for node in path) for path in paths),
                     'measure_boundary_section_context_required')
            if unit.get('section_paths'):
                _require(unit['section_paths'] == paths, 'measure_boundary_section_changed')
            context = unit.get('source_context')
            if context:
                _require(all(context.get(k) == endpoint[k] for k in
                             ('source_id', 'text_region_id', 'admitted_ordinal'))
                         and context.get('physical_page_index') == unit['page'] - 1,
                         'measure_boundary_recorded_context_changed')
        left, right = row['left'], row['right']
        pair = left['unit_id'], right['unit_id']
        _require(pair not in allowed and positions[pair[1]] == positions[pair[0]] + 1
                 and right['admitted_ordinal'] == left['admitted_ordinal'] + 1
                 and left['source_id'] == right['source_id']
                 and left['section_paths'] == right['section_paths']
                 and right['page'] in (left['page'], left['page'] + 1),
                 'measure_boundary_not_adjacent')
        same_page = left['page'] == right['page']
        if same_page:
            _require(left['text_region_id'] == right['text_region_id']
                     and right['order_index'] == left['order_index'] + 1,
                     'measure_boundary_recorded_blocks_not_adjacent')
        if row.get('basis') == 'consecutive_same_page_native_blocks':
            gap = row.get('intervening_text')
            size = max(left['font_size'], right['font_size'])
            _require(same_page and isinstance(gap, str) and not gap.strip()
                     and right['start'] - left['end'] == len(gap)
                     and abs(left['bbox'][0] - right['bbox'][0]) <= 2 * size
                     and -1 <= right['bbox'][1] - left['bbox'][3] <= 2 * size,
                     'measure_boundary_native_continuation_unproved')
        elif row.get('basis') == 'reviewed_native_boundary':
            review = row.get('review', {})
            pages = list(dict.fromkeys([left['page'], right['page']]))
            _require(review.get('complete') is True and review.get('intervening_body_text') is False
                     and review.get('boundary_sha256') == _digest({'left': left, 'right': right})
                     and _hash(review.get('evidence_sha256'))
                     and review.get('pages') == pages
                     and len(review.get('render_sha256s', [])) == len(pages)
                     and all(_hash(value) for value in review['render_sha256s']),
                     'measure_boundary_visual_review_required')
        else:
            raise ValueError('measure_boundary_unknown_basis')
        allowed[pair] = _digest(row)
    return allowed
