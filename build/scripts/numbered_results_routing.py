"""Opt-in routing of explicit, immediately following numbered Results siblings."""
from copy import deepcopy
from collections import Counter
from mosaic.adapted.e.headings import split_section_number
from scripts.results_heading_context import is_results_node


def numbers(title):
    number,_=split_section_number(title)
    if not number or not all(p.isdigit() for p in number.split('.')):return None
    return tuple(int(p) for p in number.split('.'))


def route_numbered_results(packet):
    result=deepcopy(packet);links=[];remaining=[]
    anchors={s['study_id']:numbers(s['label']) for s in result['studies']}
    for unit in result['unassigned_source_units']:
        matches=[]
        if unit.get('source_locator','').startswith('main_'):
            for study in result['studies']:
                anchor=anchors[study['study_id']]
                if not anchor:continue
                expected=anchor[:-1]+(anchor[-1]+1,)
                nodes=[n for path in unit.get('section_paths',[]) for n in path
                       if numbers(n.get('title',''))==expected and is_results_node(n,study['study_id'])]
                # Numbering alone cannot resolve conflicting study labels.
                if nodes:matches.append((study,nodes[0]))
        if len(matches)!=1:
            remaining.append(unit);continue
        study,node=matches[0]
        study['units'].append(unit)
        links.append({'unit_id':unit['unit_id'],'study_id':study['study_id'],
                      'section_id':node['node_id'],'basis':'immediate_numbered_results_sibling'})
    # Existing units retain their relative order; new siblings follow the
    # existing study extent, in the unassigned packet's physical reading order.
    result['unassigned_source_units']=remaining
    def ids(p):return Counter(u['unit_id'] for s in p['studies'] for u in s['units'])+Counter(u['unit_id'] for u in p['unassigned_source_units'])
    if ids(result)!=ids(packet):raise ValueError('routing_source_loss')
    result['numbered_results_routing']={'links':links,'semantic_ownership_unassessed':True}
    return result
