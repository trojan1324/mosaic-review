"""Read statistic operators using a separately evidenced PDF font projection.

Original text, spans and numeric digits are preserved. Only operators already
identified from the PDF glyph may differ in the temporary parser projection.
"""
from copy import deepcopy
from scripts.named_effect_choices import build_named_effect_choices
from scripts.pdf_operator_candidates import _numeric_sign_position


def build_choices(groups,operator_candidates):
    source={u['unit_id']:u for g in groups for u in g['original_units']}
    evidence={}
    for candidate in operator_candidates:
        uid=candidate['unit_id'];start=candidate['unit_char_offset'];end=candidate['unit_char_end']
        if uid not in source:raise ValueError('operator_outside_source_groups')
        if end!=start+1 or source[uid]['text'][start:end]!=candidate['raw_character']:
            raise ValueError('operator_source_character_changed')
        if candidate['proposed_operator'] not in ('=','<','>','-'):raise ValueError('unsupported_operator')
        if candidate['proposed_operator'] == '-':
            if candidate.get('basis') != 'cff_geometry':raise ValueError('minus_requires_geometry_evidence')
            if not _numeric_sign_position(source[uid]['text'],start):raise ValueError('minus_not_numeric_token_start')
        if (uid,start) in evidence:raise ValueError('duplicate_operator_position')
        evidence[uid,start]=deepcopy(candidate)
    additions=[]
    for group in groups:
        original=group['text'];shadow=list(original);local=[]
        for segment in group['segments']:
            uid=segment['unit_id']
            for (eid,offset),item in evidence.items():
                if eid!=uid:continue
                position=segment['start']+offset
                if original[position]!=item['raw_character']:raise ValueError('group_operator_alignment_changed')
                shadow[position]=item['proposed_operator']
                local.append({**deepcopy(item),'group_offset':position})
        if not local:continue
        projected=''.join(shadow)
        for choice in build_named_effect_choices([{'unit_id':group['unit_id'],'text':projected}]):
            relevant=[item for item in local if choice['start']<=item['group_offset']<choice['end']]
            if not relevant:continue
            result=deepcopy(choice)
            result['font_recovered_reading']=choice['text']
            result['text']=original[choice['start']:choice['end']]
            result['operator_evidence']=relevant
            for component in result['components']:
                component['text']=original[component['start']:component['end']]
                ops=[item for item in relevant if component['start']<=item['group_offset']<component['end']]
                if ops:component['operator_evidence']=ops
            result['choice_id']=f'font{len(additions)+1:03d}'
            additions.append(result)
    return additions


FONT_INSTRUCTION = '''FONT RECOVERY: A numeric choice may preserve damaged printed_expression together with font_recovered_reading (and font_recovered_component for one list item). The reading changes only operator glyphs independently established from the embedded PDF font at the original offsets. Use that supplied reading; never infer another correction from statistical expectations. Original source characters and source ownership remain unchanged.'''


def with_font_readings(atoms, groups, operator_candidates, result_ids):
    """Retain existing catalogue IDs/spans; add only glyph-proved readings."""
    result = deepcopy(atoms)
    fields = ('unit_id', 'start', 'end', 'text', 'statistic_type')
    for choice in build_choices(groups, operator_candidates):
        if choice['unit_id'] not in result_ids:
            continue
        matches = [a for a in result if all(a['expression'][k] == choice[k] for k in fields)]
        if matches:
            for atom in matches:
                atom['expression']['font_recovered_reading'] = choice['font_recovered_reading']
                atom['expression']['operator_evidence'] = deepcopy(choice['operator_evidence'])
        else:
            values = [c for c in choice['components'] if c['name'].startswith('value_')]
            if len(values) > 1 and choice['statistic_type'] != 'confidence_interval':
                result.extend({'id': choice['choice_id'] + '/' + c['name'],
                               'expression': deepcopy(choice), 'selected_component': deepcopy(c)} for c in values)
            else:
                result.append({'id': choice['choice_id'], 'expression': deepcopy(choice), 'selected_component': None})
    if len({a['id'] for a in result}) != len(result):
        raise ValueError('font_statistic_id_collision')
    return result


def font_menu_fields(atom):
    expression = atom['expression']
    if 'font_recovered_reading' not in expression:
        return {}
    reading = expression['font_recovered_reading']
    result = {'font_recovered_reading': reading}
    component = atom['selected_component']
    if component:
        result['font_recovered_component'] = reading[component['start']-expression['start']:component['end']-expression['start']]
    return result


def source_atom_matches(previous, expected):
    """An older raw expression may lack new proof; existing proof must match."""
    if previous == expected:
        return True
    if not isinstance(previous, dict) or 'font_recovered_reading' in previous or 'operator_evidence' in previous:
        return False
    raw = deepcopy(expected)
    raw.pop('font_recovered_reading', None)
    raw.pop('operator_evidence', None)
    return previous == raw
