"""Conservative source sentence choices; cues constrain proposals, not truth.

Fragments at PDF block boundaries remain fragments and retain their original
paragraph context. No inferred sample count or control role is emitted here.
"""
import re
from copy import deepcopy
from scripts.result_block_groups import group_result_blocks
from scripts.source_decoding import unit_slice

_CONTROL = re.compile(r'\bcovariates?\b|\bcontrol\s+variables?\b|\bcontroll(?:ing|ed)\s+for\b|\badjust(?:ing|ed)\s+for\b',re.I)
_SAMPLE = re.compile(r'\b[nN]\s*=\s*\d|\b\d[\d,]*\s+(?:\([^)]*\)\s*)?(?:participants|respondents|employees|subjects|teams|firms|organizations|observations)\b',re.I)
_ABBREVIATION = re.compile(r'(?:\bvs|\be\.g|\bi\.e|\bet\s+al|\bfig|\beq|\bdr|\bmr|\bms)\.$',re.I)
_REPEATED = re.compile(r'\b([\w-]+)\s+vs\.?\s+\1\b',re.I)


def cue_text(text):
    """Normalize alphabetic PDF wraps for recognition only, never source copying."""
    return re.sub(r'(?<=[A-Za-z])[-\u00ad]\s*\n\s*(?=[a-z])','',text)


def coherent_units(units):
    groups=[]
    for group in group_result_blocks(units):
        original=deepcopy(group['units']);text='';segments=[]
        for unit in original:
            if segments:text+='\n'
            start=len(text);text+=unit['text']
            segments.append({'unit_id':unit['unit_id'],'start':start,'end':len(text)})
        groups.append({'unit_id':original[0]['unit_id'] if len(original)==1 else 'group:'+original[0]['unit_id'],
            'text':text,'original_units':original,'segments':segments,'joins':deepcopy(group['joins'])})
    if len({g['unit_id'] for g in groups})!=len(groups):raise ValueError('duplicate_group_id')
    return groups


def original_segments(group,start,end):
    units={u['unit_id']:u for u in group['original_units']};parts=[]
    for segment in group['segments']:
        left=max(start,segment['start']);right=min(end,segment['end'])
        if left<right:
            a,b=left-segment['start'],right-segment['start']
            parts.append(unit_slice(units[segment['unit_id']],a,b))
    return parts


def sentence_choices(units):
    choices=[];seen=set()
    for unit in units:
        uid,text=unit['unit_id'],unit['text']
        if uid in seen:raise ValueError('duplicate_unit_id')
        seen.add(uid)
        # Split only clear prose boundaries. Keeping too much is safer than
        # clipping a decimal, abbreviation, qualifier or coordinated negation.
        stops=[0]
        for match in re.finditer(r'(?<=[.!?])\s+(?=[A-Z])',text):
            if not _ABBREVIATION.search(text[:match.start()]):stops.append(match.end())
        stops.append(len(text))
        for start,end in zip(stops,stops[1:]):
            while start<end and text[start].isspace():start+=1
            while end>start and text[end-1].isspace():end-=1
            if start==end:continue
            value=text[start:end]
            recognized=cue_text(value)
            choices.append({'choice_id':f's{len(choices)+1:03d}','unit_id':uid,
                'start':start,'end':end,'text':value,
                'control_cue':bool(_CONTROL.search(recognized)),
                'sample_cue':bool(_SAMPLE.search(recognized)),
                'warnings':['repeated_comparison_label'] if _REPEATED.search(recognized) else []})
    return choices


def choice_schema(choices):
    if not choices:raise ValueError('empty_sentence_choices')
    return {'type':'object','properties':{'choice_id':{'enum':[c['choice_id'] for c in choices]}},
        'required':['choice_id'],'additionalProperties':False}
