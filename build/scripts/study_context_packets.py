"""Build experimental study packets from an already-admitted source union.

No source is added and no boundary is adjudicated. This repairs presentation
order and exposes the existing section hierarchy for a later model experiment.
It is not connected to production extraction.
"""
from __future__ import annotations

import hashlib
from collections import Counter


def build_packets(units, bindings, objects, section_paths=None):
    """Retain each supplied unit once, in recorded physical/reading order.

    section_paths is optional structural metadata keyed by short unit ID. It
    must come from the same package; it never supplies additional source text.
    """
    section_paths = section_paths or {}
    mapped = {row['unit_id']: row for row in bindings}
    if len(mapped) != len(bindings) or len({u['unit_id'] for u in units}) != len(units):
        raise ValueError('duplicate_unit_identity')
    if set(mapped) != {u['unit_id'] for u in units}:
        raise ValueError('source_binding_roster_mismatch')
    studies = {o['local_id']: o for o in objects.values() if o.get('type') == 'study'}
    ordered = []
    for unit in units:
        binding = mapped[unit['unit_id']]
        region = objects[binding['text_region_id']]
        start, end = binding['start'], binding['end']
        if not 0 <= start < end <= len(region['text']):
            raise ValueError('source_range_invalid')
        if (region['text'][start:end] != unit['text'] or
                hashlib.sha256(unit['text'].encode()).hexdigest() != binding['text_sha256']):
            raise ValueError('source_binding_changed')
        blocks = [b for b in region['blocks'] if b['start'] <= start and end <= b['end']]
        if len(blocks) != 1:
            raise ValueError('source_block_ambiguous')
        block = blocks[0]
        reading = region.get('reading_order')
        if not reading or len(set(reading)) != len(reading) or block['block_index'] not in reading:
            raise ValueError('reading_order_unavailable')
        source = objects[region['source_id']]
        membership = source['membership'][0]
        page = objects.get(region.get('page_id'))
        rank = (membership['role'] != 'main_article', membership['source_label'],
                page['physical_index'] if page else -1, reading.index(block['block_index']), start)
        hint_ids = {h['id'] for h in unit['study_hints']}
        if not hint_ids <= set(studies):
            raise ValueError('unknown_study_hint')
        empirical = sorted(sid for sid in hint_ids if studies[sid]['kind'] in ('study', 'substudy'))
        row = {**unit, 'source_binding': binding,
               'section_paths': section_paths.get(unit['unit_id'], []),
               'reading_order_status': (page.get('signals', {}).get('reading_order_confidence', 'unknown')
                                        if page else 'recorded_document_order')}
        ordered.append((rank, empirical, row))
    ordered.sort(key=lambda item: item[0])
    groups, remainder = {}, []
    for _, empirical, row in ordered:
        if len(empirical) != 1:
            remainder.append({**row, 'routing_status': 'unassigned' if not empirical else 'multiple_study_proposals'})
            continue
        sid = empirical[0]
        study = studies[sid]
        if sid not in groups:
            groups[sid] = {'study_id': sid, 'label': study['label'],
                           'inventory_status': study['inventory_status'],
                           'ambiguity': study.get('ambiguity'), 'units': []}
        groups[sid]['units'].append(row)
    emitted = [u['unit_id'] for s in groups.values() for u in s['units']] + [u['unit_id'] for u in remainder]
    assert Counter(emitted) == Counter(u['unit_id'] for u in units)
    return {'purpose': 'ordered_study_context_development_only',
            'boundary_accuracy_unassessed': True, 'semantic_accuracy_unassessed': True,
            'studies': list(groups.values()), 'unassigned_source_units': remainder,
            'coverage': {'input_units': len(units), 'emitted_units': len(emitted),
                         'omitted_units': 0, 'duplicated_units': 0},
            'limitations': ['Existing study membership remains a proposal, including its recorded ambiguity.',
                           'Unassigned introduction, discussion and supplement units require explicit source-backed routing.',
                           'No model call, source expansion, scientific acceptance or student task is created.']}
