"""Additional printed syntax, without general bare-number role guessing."""
import re
from scripts.reported_statistic_choices import build_statistic_choices, _chi_with_details
from scripts.typed_statistic_choices import _NUMBER, _label_choice

_CHI = re.compile(r'(?P<label>χ[2²]|chi2)\s*\.\s*\((?P<inside>[^)]{1,80})\)\s*(?P<comparator><=|>=|<|>|=|≤|≥)\s*(?P<value>'+_NUMBER+r')(?!\w)', re.I)
_R2 = re.compile(r'(?<!\w)(?P<label>R[2²])\s+of\s+(?P<value>'+_NUMBER+r')(?!\w)', re.I)
_PERCENT = re.compile(r'(?<![\w.])'+_NUMBER+r'\s*%')
_MEANS = re.compile(r'(?<!\w)(?P<label>Ms)\s*(?P<comparator>=)\s*(?P<first>'+_NUMBER+r')\s*\(\s*SD\s*=\s*'+_NUMBER+r'\s*\)\s*(?:vs\.?|versus)\s*(?P<second>'+_NUMBER+r')\s*\(\s*SD\s*=\s*'+_NUMBER+r'\s*\)', re.I)
_DF = r'\d+(?:\.\d+)?'
_BRACKET_F = re.compile(r'(?<!\w)(?P<label>F)\s*\[\s*(?P<df>'+_DF+r'\s*,\s*'+_DF+r')\s*\]\s*(?P<comparator><=|>=|<|>|=|≤|≥)\s*(?P<value>'+_NUMBER+r')(?!\w)', re.I)


def component(name, text, start, end):
    return {'name':name,'start':start,'end':end,'text':text[start:end]}


def build_additional_choices(units):
    choices=build_statistic_choices(units)
    for unit in units:
        uid,text=unit['unit_id'],unit['text']
        # The authors may print F's two degrees of freedom in square brackets.
        # Preserve this syntax and its spans; do not infer a result or ownership.
        for match in _BRACKET_F.finditer(text):
            choices.append(_label_choice(uid,text,match))
        for match in _CHI.finditer(text):
            choices=[c for c in choices if not (c['unit_id']==uid and match.start()<=c['start'] and c['end']<=match.end())]
            choices.append(_chi_with_details(uid,text,match))
        for match in _MEANS.finditer(text):
            # Preserve the paired expression while selecting each mean's exact
            # component; existing SD expressions remain separately available.
            choices.append({'unit_id':uid,'start':match.start(),'end':match.end(),'text':match.group(),
                'statistic_type':'mean','printed_label':match.group('label'),
                'components':[component('comparator',text,*match.span('comparator')),
                    component('value_1',text,*match.span('first')),component('value_2',text,*match.span('second'))]})
        for match in _R2.finditer(text):
            choices.append({'unit_id':uid,'start':match.start(),'end':match.end(),'text':match.group(),
                'statistic_type':'r_squared','printed_label':match.group('label'),
                'components':[component('value',text,*match.span('value'))]})
        for match in _PERCENT.finditer(text):
            if any(c['unit_id']==uid and c['start']<=match.start() and match.end()<=c['end'] for c in choices):continue
            choices.append({'unit_id':uid,'start':match.start(),'end':match.end(),'text':match.group(),
                'statistic_type':'percentage','printed_label':'%',
                'components':[component('value',text,match.start(),text.rfind('%',match.start(),match.end()))]})
    order={u['unit_id']:i for i,u in enumerate(units)}
    choices.sort(key=lambda c:(order[c['unit_id']],c['start'],c['end'],c['statistic_type']))
    for i,c in enumerate(choices,1):c['choice_id']=f'stat{i:03d}'
    return choices
