"""Interpret Results labels within an existing study; never move source units.

Study ownership and section role are separate dimensions. A heading can name
both, and a Results/Discussion section still contains Results evidence.
"""
from copy import deepcopy
import re
from mosaic.adapted.e.headings import split_section_number, study_label

RESULTS = re.compile(r'^(?:results?|findings?)(?:\s+(?:and|&)\s+discussion)?(?:\s*[:\u2013\u2014-]\s*[^.!?]{1,80})?$', re.I)
NAMED = re.compile(r'^((?:study|experiment)\s+(?:\d+[a-z]?|[ivxlcdm]+))\s*[:\u2013\u2014-]?\s+(.*)$',re.I)


def is_results_node(node, study_id):
    _,title=split_section_number(node.get('title',''))
    named=NAMED.fullmatch(title)
    if named:
        label=study_label(named.group(1))
        if not label or 'study:'+label[1]!=study_id:return False
        title=named.group(2)
    return node.get('role')=='results' or bool(RESULTS.fullmatch(title.strip()))


def is_results_unit(unit, study_id):
    return any(is_results_node(n,study_id) for p in unit.get('section_paths',[]) for n in p)


def main_results_payload(prepared):
    study=prepared['study'];units=study['units'];sid=study['study_id']
    results=[];inline_path=None
    for u in units:
        path=tuple(tuple(n.get('node_id') for n in p) for p in u.get('section_paths',[]))
        heading=u.get('inline_heading',{})
        if heading:
            inline_path=path if (heading.get('role')=='results' and path
                and all(all(n is not None for n in p) for p in path)) else None
        elif inline_path!=path:inline_path=None
        if u['source_locator'].startswith('main_') and (is_results_unit(u,sid) or inline_path is not None):
            results.append(u)
    if not results:raise ValueError('explicit_main_results_required')
    first=next(i for i,u in enumerate(units) if u['unit_id']==results[0]['unit_id'])
    return deepcopy({'physical_study_id':sid,'source':results,'study_method_context':{},
        'source_context':[u for u in units[:first] if u['source_locator'].startswith('main_')]})
