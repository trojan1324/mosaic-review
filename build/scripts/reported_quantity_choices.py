"""Opt-in printed-unit quantities; never infer effect type or convert units."""
import re
from scripts.additional_statistic_choices import build_additional_choices, component
from scripts.typed_statistic_choices import _NUMBER

_UNIT = r'(?:percentage\s+points?|percent\b|standard\s+deviations?\b|rupees?\b|dollars?\b)'
_QUANTITY = re.compile(r'(?<![\w.])(?P<first>'+_NUMBER+r')(?:\s*(?:-|–|to)\s*(?P<last>'+_NUMBER+r'))?\s*(?P<unit>'+_UNIT+r')',re.I)
_CURRENCY = re.compile(r'(?<!\w)(?P<unit>Rs\.?|USD|GBP|EUR|\$|£|€)\s*(?P<first>'+_NUMBER+r')(?!\w)',re.I)
_PROSE_P = re.compile(r'(?<!\w)(?P<label>p\s+value(?:\s+of\s+(?:the\s+)?difference)?)\s*(?P<comparator><=|>=|<|>|=|≤|≥)\s*(?P<value>'+_NUMBER+r')(?!\w)',re.I)


def build_quantity_choices(units):
    base=build_additional_choices(units)
    additions=[]
    for row in units:
        uid,text=row['unit_id'],row['text']
        for pattern in (_QUANTITY,_CURRENCY):
            for m in pattern.finditer(text):
                if any(c['unit_id']==uid and c['start']<=m.start() and m.end()<=c['end'] for c in base+additions):continue
                # A written-out confidence level is not a result percentage.
                if m.group('unit').lower()=='percent' and re.match(r'\s*(?:confidence|CI\b)',text[m.end():],re.I):continue
                ranged=pattern is _QUANTITY and m.group('last') is not None
                parts=[component('range_start' if ranged else 'value',text,*m.span('first'))]
                if ranged:parts.append(component('range_end',text,*m.span('last')))
                parts.append(component('printed_unit',text,*m.span('unit')))
                additions.append({'unit_id':uid,'start':m.start(),'end':m.end(),'text':m.group(),
                    'statistic_type':'reported_quantity_range' if ranged else 'reported_quantity',
                    'printed_label':m.group('unit'),'components':parts})
    prose=[]
    for row in units:
        for m in _PROSE_P.finditer(row['text']):
            prose.append({'unit_id':row['unit_id'],'start':m.start(),'end':m.end(),'text':m.group(),
                'statistic_type':'p_value','printed_label':m.group('label'),
                'components':[component(k,row['text'],*m.span(k)) for k in ('comparator','value')]})
    order={u['unit_id']:i for i,u in enumerate(units)}
    additions.sort(key=lambda c:(order[c['unit_id']],c['start'],c['end']))
    for i,c in enumerate(additions,1):c['choice_id']=f'quantity{i:03d}'
    for i,c in enumerate(prose,1):c['choice_id']=f'prose{i:03d}'
    # Existing statistical expressions and their IDs are byte-for-byte retained.
    return sorted(base+additions+prose,key=lambda c:(order[c['unit_id']],c['start'],c['end']))
