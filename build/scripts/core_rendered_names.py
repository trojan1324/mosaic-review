"""Recover name spans using frozen, reviewed PDF glyph and spacing evidence.

The projected spelling exists only while locating a name already proposed by
the model. Returned names/evidence keep native text and offsets. This does not
split unknown words, repair article text, or verify scientific labels.
"""
from copy import deepcopy
from hashlib import sha256
import math
import re

from mosaic.canonical import canonical_bytes
from scripts.core_declared_names import recover_declared_names
from scripts.core_evidence_recovery import source_name_mapping
from scripts.core_evidence_trial import validate
from scripts.core_linebreak_names import linebreak_mapping

VERSION = 'core-name-witnesses-v5'
PROOF_VERSION = 'reviewed-name-layout-proof/v1'
ACUTE_FONT = 'a3ab5b07ae8d14d78936659e886b6ed028c4efb695dd7a29937f95f013283f95'
BASE_FONTS = {
    ('bef439113828f880fe2ff8186681cd6e799671b84fa003faed6e6dee4d12dfa9', 60),
    ('53d76337158a7dd9389852a60225f7d0e7186b632472c67b7716616f9baf42f5', 51),
}
SPACE_FONT = 'b991e1a6b7dc9f571f44f7ae848700a66345a46c949240ced2dc8655b383943e'
digest = lambda value: sha256(canonical_bytes(value)).hexdigest()
_HASH = re.compile(r'[0-9a-f]{64}\Z')


def _require(condition, code):
    if not condition:
        raise ValueError(code)


def _hash(value):
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


def _coordinates(value, size):
    return isinstance(value, (tuple, list)) and len(value) == size and all(
        type(v) in (int, float) and math.isfinite(v) for v in value)


def validate_layout_proof(proof, *, units, pdf_sha256, proof_sha256):
    """Bind reviewed layout edits to an immutable source and exact font programs."""
    _require(isinstance(proof, dict) and _hash(proof_sha256) and digest(proof) == proof_sha256,
             'name_layout_frozen_proof_changed')
    _require(_hash(pdf_sha256) and proof.get('source_pdf_sha256') == pdf_sha256
             and proof.get('source_units_sha256') == digest(units), 'name_layout_source_identity_changed')
    _require(proof.get('version') == PROOF_VERSION and proof.get('source_unchanged') is True
             and proof.get('scientifically_accepted') is False and proof.get('production_accepted') is False,
             'name_layout_unverified_source_proof_required')
    byid = {u['unit_id']: u for u in units}
    _require(len(byid) == len(units), 'name_layout_duplicate_unit')
    rows = proof.get('rows')
    _require(isinstance(rows, list) and bool(rows), 'name_layout_review_rows_required')
    result = {}
    for row in rows:
        unit = byid.get(row.get('unit_id'))
        _require(unit is not None and row.get('page') == unit['page'], 'name_layout_unit_changed')
        raw = unit.get('text_derivation', {}).get('raw_text', unit['text'])
        _require(sha256(raw.encode()).hexdigest() == row.get('raw_unit_sha256'), 'name_layout_raw_unit_changed')
        start, end = row.get('line_start'), row.get('line_end')
        _require(type(start) is int and type(end) is int and 0 <= start < end <= len(raw)
                 and sha256(raw[start:end].encode()).hexdigest() == row.get('raw_line_sha256'),
                 'name_layout_line_changed')
        _require(row.get('review_complete') is True and _hash(row.get('render_sha256'))
                 and _coordinates(row.get('bbox'), 4) and row['bbox'][0] < row['bbox'][2]
                 and row['bbox'][1] < row['bbox'][3], 'name_layout_render_review_required')
        edits = row.get('edits')
        _require(isinstance(edits, list) and bool(edits), 'name_layout_edits_required')
        for edit in edits:
            a,b = edit.get('start'),edit.get('end')
            _require(type(a) is int and type(b) is int and start <= a <= b <= end
                     and raw[a:b] == edit.get('raw') and unit['text'][a:b] == edit.get('raw'),
                     'name_layout_edit_source_changed')
            glyphs = edit.get('glyphs')
            _require(isinstance(glyphs, list) and len(glyphs) == 2, 'name_layout_glyph_pair_required')
            for glyph in glyphs:
                pos = glyph.get('unit_offset')
                _require(isinstance(glyph.get('raw'), str) and len(glyph['raw']) == 1,
                         'name_layout_glyph_character_required')
                # This exact damaged accent is CID1 in rawdict and U+FFFD in
                # texttrace. Its frozen program/glyph/render supplies identity.
                trace_code = 65533 if glyph['raw'] == '\x01' else ord(glyph['raw'])
                _require(type(pos) is int and start <= pos < end and raw[pos] == glyph.get('raw')
                         and glyph.get('trace_matches') == 1 and glyph.get('trace_unicode') == trace_code
                         and glyph.get('direction') == [1.0, 0.0] and glyph.get('writing_mode') == 0
                         and type(glyph.get('glyph_id')) is int and glyph['glyph_id'] > 0
                         and _hash(glyph.get('font_sha256')), 'name_layout_glyph_identity_changed')
                _require(_coordinates(glyph.get('origin'),2) and _coordinates(glyph.get('trace_origin'),2)
                         and _coordinates(glyph.get('bbox'),4)
                         and all(abs(x-y) <= .03 for x,y in zip(glyph['origin'],glyph['trace_origin']))
                         and type(glyph.get('size')) in (int,float) and math.isfinite(glyph['size']) and glyph['size'] > 0
                         and glyph['bbox'][0] < glyph['bbox'][2] and glyph['bbox'][1] < glyph['bbox'][3],
                         'name_layout_glyph_geometry_changed')
            left,right = glyphs
            _require(abs(left['size']-right['size']) <= .03, 'name_layout_glyph_size_changed')
            if edit.get('kind') == 'reviewed_acute_overlay':
                _require(edit.get('raw') == '\x01e' and edit.get('search_text') == 'é' and b == a+2
                         and [left['unit_offset'],right['unit_offset']] == [a,a+1]
                         and left['font_sha256'] == ACUTE_FONT and left['glyph_id'] == 5
                         and (right['font_sha256'],right['glyph_id']) in BASE_FONTS
                         and abs(left['origin'][0]-right['origin'][0]) <= .03
                         and 0 <= right['origin'][1]-left['origin'][1] <= .06*left['size'],
                         'name_layout_unreviewed_overlay')
            elif edit.get('kind') == 'reviewed_word_boundary':
                gap = right['origin'][0]-left['bbox'][2]
                _require(a == b and edit.get('search_text') == ' '
                         and [left['unit_offset'],right['unit_offset']] == [a-1,a]
                         and all(g['raw'].isascii() and g['raw'].isalpha() and g['font_sha256'] == SPACE_FONT for g in glyphs)
                         and abs(left['origin'][1]-right['origin'][1]) <= .03
                         and 0 < gap <= .5*left['size'], 'name_layout_unreviewed_word_boundary')
            else:
                raise ValueError('name_layout_unknown_edit_kind')
            result.setdefault(unit['unit_id'], []).append(deepcopy(edit))
    for edits in result.values():
        edits.sort(key=lambda e:(e['start'],e['end']))
        _require(len({e['start'] for e in edits}) == len(edits)
                 and all(left['end'] <= right['start'] for left,right in zip(edits,edits[1:])),
                 'name_layout_overlapping_edits')
    return result


def project_statement(statement, edits_by_unit):
    """Build temporary search characters with maps back to unchanged offsets."""
    text = statement['text']; edits = []; cursor = 0
    for segment in statement['segments']:
        located = text.find(segment['text'], cursor)
        _require(located >= cursor and not text[cursor:located].strip(), 'name_layout_statement_segments_changed')
        for edit in edits_by_unit.get(segment['unit_id'], []):
            if segment['start'] <= edit['start'] and edit['end'] <= segment['end']:
                # Both boundary glyphs must be inside the cited segment.
                if edit['kind'] == 'reviewed_word_boundary' and not segment['start'] < edit['start'] < segment['end']:
                    continue
                start = located+edit['start']-segment['start']; end = located+edit['end']-segment['start']
                edits.append({**edit, 'start':start,'end':end, 'unit_id':segment['unit_id'],
                              'unit_start':edit['start'],'unit_end':edit['end']})
        cursor = located+len(segment['text'])
    _require(not text[cursor:].strip(), 'name_layout_statement_tail_changed')
    edits.sort(key=lambda e:(e['start'],e['end']))
    characters = []; positions = []; changed = []; cursor = 0
    for edit in edits:
        a,b = edit['start'],edit['end']
        _require(a >= cursor and text[a:b] == edit['raw'], 'name_layout_statement_edit_changed')
        characters.extend(text[cursor:a]); positions.extend((p,p+1) for p in range(cursor,a))
        changed.append({'search_start':len(characters),'search_end':len(characters)+len(edit['search_text']),
                        'unit_id':edit['unit_id'],'unit_start':edit['unit_start'],'unit_end':edit['unit_end'],'kind':edit['kind']})
        characters.extend(edit['search_text']); positions.extend((a,b) for _ in edit['search_text'])
        cursor = b
    characters.extend(text[cursor:]); positions.extend((p,p+1) for p in range(cursor,len(text)))
    return ''.join(characters),positions,changed


def projected_name_mapping(name, statement, edits_by_unit):
    search, positions, edits = project_statement(statement, edits_by_unit)
    match = source_name_mapping(name, search) or linebreak_mapping(name, search)
    if match is None or len(match.get('all_exact_source_spans', [match['source_span']])) != 1:
        return None
    span = match['source_span']; a,b = span['start'],span['end']
    witnesses = [e for e in edits if e['search_start'] <= b and e['search_end'] >= a]
    if not witnesses:
        return None
    start,end = positions[a][0],positions[b-1][1]
    original = statement['text'][start:end]
    return {'proposal_name':name,'source_name':original,'search_name':search[a:b],
        'match_type':'unique_reviewed_pdf_layout','source_span':{'start':start,'end':end,'text':original},
        'layout_edits':witnesses,'source_text_unchanged':True,'semantic_identity_verified':False}


def validate_bound_layout_evidence(evidence, *, name, name_evidence, source_identity, pdf_sha256):
    """Replay a catalog name projection against its copied source excerpts.

    Full font/geometry/source validation occurs when the inventory is built.
    Its independent proof hash is retained in inventory identity; a downstream
    catalog may not strip, replace or silently reinterpret that evidence.
    """
    proof = evidence.get('proof'); proof_hash = evidence.get('proof_sha256')
    _require(evidence.get('version') == 'bound-name-layout-evidence/v1'
             and evidence.get('scientifically_accepted') is False and evidence.get('source_text_unchanged') is True
             and isinstance(proof, dict) and _hash(proof_hash) and digest(proof) == proof_hash
             and source_identity.get('name_layout_proof_sha256') == proof_hash
             and proof.get('source_pdf_sha256') == pdf_sha256
             and proof.get('source_units_sha256') == source_identity.get('eligible_source_units_sha256'),
             'name_layout_bound_evidence_changed')
    mapping = evidence.get('mapping', {})
    _require(mapping.get('source_name') == name and mapping.get('layout_proof_sha256') == proof_hash,
             'name_layout_bound_name_changed')
    byunit = {}
    for row in proof['rows']:
        byunit.setdefault(row['unit_id'], []).extend(row['edits'])
    matches = []
    for statement in name_evidence:
        candidate = projected_name_mapping(mapping.get('proposal_name'),
            {'text':statement['quote'],'segments':statement['segments']}, byunit)
        if candidate is not None and {**candidate,'layout_proof_sha256':proof_hash} == mapping:
            matches.append(candidate)
    _require(len(matches) == 1, 'name_layout_bound_mapping_mismatch')


def recover_rendered_names(proposal, task, *, eligible_units, source_policy_sha256, pdf_sha256, proof, proof_sha256):
    """Add only previously refused names whose existing citations have proof."""
    baseline = recover_declared_names(proposal, task, eligible_units=eligible_units, source_policy_sha256=source_policy_sha256)
    edits = validate_layout_proof(proof, units=eligible_units, pdf_sha256=pdf_sha256, proof_sha256=proof_sha256)
    table = validate(task); result = deepcopy(baseline); repairs = []; fixed_paths = set()
    for refusal in baseline['quarantined_fields']:
        if refusal['reason'] != 'name_not_exact_or_uniquely_normalisable_in_cited_sentence':
            continue
        keys = refusal['path'].split('/')[1:]; kind,index,children,child_index,name_key = keys
        index,child_index = int(index),int(child_index)
        field = proposal[kind][index][children][child_index]
        evidence_key = 'description_sentence_ids' if name_key == 'name' else 'role_sentence_ids'
        alternatives = []
        for sid in dict.fromkeys([field['name_sentence_id']]+field[evidence_key]):
            mapping = projected_name_mapping(field[name_key],table[sid],edits)
            if mapping is not None:
                alternatives.append({'sentence_id':sid,'mapping':mapping})
        if len(alternatives) != 1:
            continue
        selected = alternatives[0]; mapping = selected['mapping']; sid = selected['sentence_id']
        identifier = f'study-{index}-measure-{child_index}' if name_key == 'name' else f'analysis-{index}-role-{child_index}'
        common = {'id':identifier,'name_mapping':{**mapping,'layout_proof_sha256':proof_sha256},
                  'name_evidence':[deepcopy(table[sid])]}
        if name_key == 'name':
            # Collective placeholders remain refused even if spacing is proved.
            if re.fullmatch(r'control\s+variables?',mapping['search_name'],re.I):
                continue
            recovered = {**common,'name':mapping['source_name'],
                'description_evidence':[deepcopy(table[s]) for s in field['description_sentence_ids']],
                'item_evidence':[deepcopy(table[s]) for s in field['item_sentence_ids']],
                'status':'candidate_source_backed','semantic_label_verified':False}
        else:
            recovered = {**common,'variable_name':mapping['source_name'],'role':field['role'],
                'role_evidence':[deepcopy(table[s]) for s in field['role_sentence_ids']],
                'status':'candidate_role_unverified','semantic_role_verified':False,
                'name_and_role_evidence_are_separate':sid not in field['role_sentence_ids']}
        result[kind][index][children].append(recovered)
        result[kind][index][children].sort(key=lambda r:int(r['id'].rsplit('-',1)[-1]))
        fixed_paths.add(refusal['path'])
        repairs.append({'field_path':'/'+('/'.join(keys[:-1])), 'original_name':field[name_key],
            'original_name_sentence_id':field['name_sentence_id'],'replacement_name_sentence_id':sid,
            'mapping':mapping,'semantic_identity_verified':False})
    result['quarantined_fields'] = [r for r in baseline['quarantined_fields'] if r['path'] not in fixed_paths]
    for kind,children in [('studies','measures'),('analyses','roles')]:
        for old,new in zip(baseline[kind],result[kind],strict=True):
            indexed = {r['id']:r for r in new[children]}
            _require(all(indexed[r['id']] == r for r in old[children]), 'name_layout_prior_record_changed')
            _require({k:v for k,v in old.items() if k != children} == {k:v for k,v in new.items() if k != children},
                     'name_layout_parent_record_changed')
    result.update(version=VERSION,rendered_name_base_record_sha256=digest(baseline),
        rendered_name_original_quarantined_fields=deepcopy(baseline['quarantined_fields']),
        rendered_name_repairs=repairs,rendered_name_proof=deepcopy(proof),rendered_name_proof_sha256=proof_sha256,
        rendered_name_pdf_sha256=pdf_sha256,all_prior_fields_unchanged=True,
        repair_status='reviewed_source_layout_name_locator_only_not_semantic_acceptance')
    return result
