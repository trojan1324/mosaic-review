"""Additional measure choices across already-cited, source-bound continuations.

Reuse the existing field-boundary and adjacent-block rules. A joined choice is
only an offered boundary; its role, owner and completeness remain unverified.
"""
from copy import deepcopy
from hashlib import sha256
import re

from scripts.measure_catalog import _statement
from scripts.measure_field_choices import _ENCODING
from scripts.measure_field_trial import _digest, _require, _sources
from scripts.measure_wording_trial import measure_groups
from scripts.result_block_groups import _recorded_boundary
from scripts.source_decoding import decoding_status, subspan, unit_slice, validate_source
from scripts.measure_continuation_boundaries import VERSION as CHECKED_VERSION, validate_boundary_proof

VERSION = 'source-bound-measure-continuations/v1'
FIELDS = ('name_evidence', 'description_evidence', 'item_evidence', 'field_context_evidence')
_FORMAT_LEFT = re.compile(r'\bscale\b[^.!?]{0,500}\b[0-9]{1,2}\s*\([^()]+\)\s+to\s*$', re.I)
_FORMAT_RIGHT = re.compile(r'^\s*[0-9]{1,2}\s*\([^()\n]+\)')


def _key(segment):
    return tuple(segment[k] for k in ('unit_id', 'page', 'source_locator', 'unit_text_sha256', 'start', 'end'))


def _span_key(choice):
    return tuple(_key(s) for s in choice['segments'])


def _groups(source, ids):
    groups = []
    positions = {u['unit_id']: i for i,u in enumerate(source['units'])}
    for group in measure_groups(source['units'], ids):
        if groups:
            left, right = groups[-1]['original_units'][-1], group['original_units'][0]
            # The older prose helper intentionally refuses a numeric start.
            # Recognize only a printed two-endpoint format at this boundary.
            if (positions[right['unit_id']] == positions[left['unit_id']] + 1
                and left.get('section_paths') == right.get('section_paths')
                and right['page'] in (left['page'], left['page'] + 1)
                and _recorded_boundary(left, right)
                and _FORMAT_LEFT.search(left['text']) and _FORMAT_RIGHT.match(right['text'])):
                groups[-1]['original_units'].extend(group['original_units'])
                groups[-1]['text'] += '\n' + group['text']
                groups[-1]['joins'].append({'previous_unit': left['unit_id'], 'next_unit': right['unit_id'],
                    'reason': 'adjacent_same_section_printed_scale_endpoints'})
                groups[-1]['joins'].extend(group['joins'])
                continue
        groups.append(deepcopy(group))
    return groups


def _checked_groups(source, ids, boundaries):
    """Use the old prose recognizer only inside separately proven runs."""
    runs, run = [], []
    for unit in source['units']:
        if unit['unit_id'] not in ids:
            if run:
                runs.append(run); run = []
            continue
        if run and (run[-1]['unit_id'], unit['unit_id']) not in boundaries:
            runs.append(run); run = []
        run.append(unit)
    if run:
        runs.append(run)
    result = []
    for run in runs:
        original = {u['unit_id']: u for u in run}
        recognition = deepcopy(run)
        for unit in recognition:
            # The proof supplies order only for this temporary recognizer.
            # Original page-wide uncertainty and source metadata stay intact.
            unit.pop('source_context', None)
        for group in _groups({'units': recognition}, ids):
            group['original_units'] = [original[u['unit_id']] for u in group['original_units']]
            result.append(group)
    return result


def _parents(version, source, boundaries=None):
    _require(isinstance(source, dict)
        and _digest(source) == version.get('field_context_receipt', {}).get('source_packet_sha256'),
        'measure_continuation_source_changed')
    validate_source(source, pdf_sha256=version['paper']['source_pdf_sha256'])
    lookup = {u['unit_id']: u for u in source['units']}
    _require(len(lookup) == len(source['units']), 'measure_continuation_duplicate_unit')
    rows = _sources(version)
    for row in rows:
        segment = row['segment']
        unit = lookup.get(segment['unit_id'])
        _require(unit is not None and segment['page'] == unit['page']
            and segment['source_locator'] == unit['source_locator']
            and segment['unit_text_sha256'] == sha256(unit['text'].encode()).hexdigest(),
            'measure_continuation_witness_changed')
        expected = unit_slice(unit, segment['start'], segment['end'])
        _require(expected['text'] == segment['text']
            and expected.get('text_derivation') == segment.get('text_derivation'),
            'measure_continuation_witness_changed')
    parents = []
    positions = {u['unit_id']: i for i,u in enumerate(source['units'])}
    for field in FIELDS:
        for index, value in enumerate(version.get(field, [])):
            if len(value.get('segments', [])) > 1:
                statement = _statement(value)
                for left, right in zip(statement['segments'], statement['segments'][1:]):
                    if left['unit_id'] == right['unit_id']:
                        adjacent = left['end'] == right['start']
                    else:
                        adjacent = (positions[right['unit_id']] == positions[left['unit_id']] + 1
                            and left['end'] == len(lookup[left['unit_id']]['text']) and right['start'] == 0)
                    _require(adjacent, 'measure_continuation_noncontiguous_parent')
                parents.append((field, index, statement, []))
    # Only full units already supplied as this target's context may participate.
    # measure_groups also sees omitted source units, so it cannot bridge them.
    ids = set()
    for statement in version.get('field_context_evidence', []):
        for segment in statement['segments']:
            unit = lookup[segment['unit_id']]
            _require(segment['start'] == 0 and segment['end'] == len(unit['text']),
                'measure_continuation_full_context_required')
            ids.add(segment['unit_id'])
    groups = _groups(source, ids)
    if boundaries is not None:
        # Keep historical recognition identities; every resulting selection is
        # still checked below. Add newly provable runs after those old groups.
        known = {tuple(u['unit_id'] for u in g['original_units']) for g in groups}
        groups.extend(g for g in _checked_groups(source, ids, boundaries)
                      if tuple(u['unit_id'] for u in g['original_units']) not in known)
    for index, group in enumerate(groups):
        if len(group['original_units']) < 2:
            continue
        parts = [{**unit_slice(u, 0, len(u['text'])), 'page': u['page'],
            'source_locator': u['source_locator'], 'unit_text_sha256': sha256(u['text'].encode()).hexdigest()}
            for u in group['original_units']]
        statement = {'quote': group['text'], 'segments': parts,
                     'pages': list(dict.fromkeys(s['page'] for s in parts))}
        parents.append(('recorded_unit_continuation', index, _statement(statement), group['joins']))
    return parents


def extend_choices(version, previous, source, *, formats=False, boundary_proof=None,
                   proof_sha256=None, audit=None):
    # Import at call time: the grounded builder invokes this optional extension.
    from scripts.measure_field_grounded import choices as boundary_choices
    from scripts.measure_format_choices import refine_choices

    offered = deepcopy(previous)
    seen = {_span_key(c) for c in offered}
    boundaries = (validate_boundary_proof(boundary_proof, source,
        pdf_sha256=version['paper']['source_pdf_sha256'], proof_sha256=proof_sha256)
        if boundary_proof is not None else None)
    refused = set()
    for field, index, statement, joins in _parents(version, source, boundaries):
        parts = statement['segments']
        separator = '' if ''.join(s['text'] for s in parts) == statement['quote'] else '\n'
        spans, offset = [], 0
        for i, segment in enumerate(parts):
            if i:
                offset += len(separator)
            spans.append((offset, offset + len(segment['text'])))
            offset += len(segment['text'])
        text = statement['quote']
        _require(offset == len(text), 'measure_continuation_parent_length_changed')
        virtual = {'unit_id': 'recognition-only', 'page': parts[0]['page'],
            'source_locator': 'recognition-only', 'unit_text_sha256': sha256(text.encode()).hexdigest(),
            'text': text, 'start': 0, 'end': len(text)}
        recognition = deepcopy(version)
        for old_field in FIELDS:
            recognition[old_field] = []
        recognition_field = field if field in FIELDS else 'field_context_evidence'
        recognition[recognition_field] = [{'quote': text, 'pages': [virtual['page']], 'segments': [virtual]}]
        candidates = boundary_choices(recognition, witnesses=True)
        if formats:
            candidates = refine_choices(recognition, candidates)
        for candidate in candidates:
            start, end = candidate['segments'][0]['start'], candidate['segments'][0]['end']
            mapped, indexes = [], []
            for i, (left, right) in enumerate(spans):
                a, b = max(left, start), min(right, end)
                if a < b:
                    mapped.append(subspan(parts[i], a - left, b - left))
                    indexes.append(i)
            if len(mapped) < 2:
                continue
            _require(separator.join(s['text'] for s in mapped) == candidate['quote'],
                     'measure_continuation_slice_changed')
            choice = deepcopy(candidate)
            choice.update(segments=mapped, pages=list(dict.fromkeys(s['page'] for s in mapped)),
                source_field=field, source_statement_index=index, source_segment_indexes=indexes)
            choice.pop('source_segment_index', None)
            key = _span_key(choice)
            if key in seen:
                continue
            pairs = [(a['unit_id'], b['unit_id']) for a,b in zip(mapped, mapped[1:])
                     if a['unit_id'] != b['unit_id']]
            if boundaries is not None and any(pair not in boundaries for pair in pairs):
                if audit is not None and key not in refused:
                    audit.append({'reason': 'continuation_boundary_unverified',
                        'quote': choice['quote'], 'segments': deepcopy(mapped),
                        'missing_boundaries': [list(pair) for pair in pairs if pair not in boundaries],
                        'scientifically_accepted': False})
                    refused.add(key)
                continue
            seen.add(key)
            choice['choice_id'] = f'measure-choice-{len(offered) + 1:03d}'
            choice['statement_continuation'] = {
                'version': VERSION if boundaries is None else CHECKED_VERSION, 'statement_sha256': _digest(statement),
                'source_packet_sha256': _digest(source), 'separator': separator,
                'start': start, 'end': end, 'joins': deepcopy(joins),
                'new_source_units_added': False, 'scientifically_accepted': False}
            if boundaries is not None:
                choice['statement_continuation'].update(boundary_proof_sha256=proof_sha256,
                    boundary_row_sha256s=[boundaries[pair] for pair in pairs],
                    joins=[j for j in joins if (j['previous_unit'], j['next_unit']) in pairs])
            choice['source_text_status'] = ('visual_check_required' if _ENCODING.search(choice['quote'])
                                             else decoding_status(mapped))
            offered.append(choice)
    return offered
