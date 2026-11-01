"""Quarantine competing additions without rewriting prior evidence.

A collision is a reason to withhold a new assertion, not proof that either
value is scientifically wrong. All candidates and their sources are retained.
"""
from copy import deepcopy
import re


def slot_key(statistic, applies_to):
    expression=statistic['expression']
    if statistic.get('selected_component'):return None
    kind=expression['statistic_type']
    if kind not in ('p_value','coefficient','estimate_adjacent_to_confidence_interval',
                    'confidence_interval','confidence_interval_unknown_level'):return None
    level=tuple(c['text'] for c in expression.get('components',[]) if c['name']=='confidence_level')
    text=re.sub(r'-\s*\n\s*','',expression['text']).casefold()
    methods=tuple(name for name,pattern in (
        ('bca',r'\bbca\b|bias[- ]corrected and accelerated'),
        ('bias_corrected',r'bias[- ]corrected'),('percentile',r'percentile'),
        ('bootstrap',r'bootstrap'),('wald',r'\bwald\b'),('profile',r'profile likelihood'))
        if re.search(pattern,text)) if kind.startswith('confidence_interval') else ()
    if 'bca' in methods:methods=tuple(m for m in methods if m not in ('bias_corrected','bootstrap'))
    else:methods=tuple(m for m in methods if m!='bootstrap')
    if kind.startswith('confidence_interval'):
        # An omitted confidence level/method does not prove a distinct slot.
        kind='confidence_interval'
        return kind,None,level,methods,applies_to
    return kind,expression.get('printed_label'),level,methods,applies_to


def slots_compete(a,b):
    if a is None or b is None:return False
    if a[0]!=b[0] or a[1]!=b[1]:return False
    # An unspecified group does not establish a distinct scientific owner.
    # Explicit different arms remain separate; unknown ownership competes.
    if a[4]!=b[4] and 'unspecified_group' not in (a[4],b[4]):return False
    return (not a[2] or not b[2] or a[2]==b[2]) and (not a[3] or not b[3] or a[3]==b[3])


def filter_delta(delta):
    """Keep distinct parameters/arms/intervals; expose occupied-slot candidates.

    Test IDs fix the proposed condition, outcome and specification. Unknown
    interval methods remain unknown: competing same-level intervals may need
    later machine resolution. No collision creates a student assignment.
    """
    result=deepcopy(delta);occupied={};kept=[];quarantined=[]
    for attachment in delta['baseline']['attachments']:
        for statistic in attachment['statistics']:
            scope=statistic['binding_proposal']['applies_to'];slot=slot_key(statistic,scope)
            if slot is not None:occupied.setdefault((attachment['test_id'],slot),[]).append(statistic)
    # Check all new proposals together so array order cannot select a winner.
    new_slots={}
    for binding in delta['new_bindings']:
        slot=slot_key(binding['statistic'],binding['applies_to'])
        if slot is not None:new_slots.setdefault((binding['test_id'],slot),[]).append(binding)
    for binding in result['new_bindings']:
        slot=slot_key(binding['statistic'],binding['applies_to'])
        peers=[b for (tid,s),rows in new_slots.items() if tid==binding['test_id'] and slots_compete(slot,s) for b in rows]
        previous=[b for (tid,s),rows in occupied.items() if tid==binding['test_id'] and slots_compete(slot,s) for b in rows]
        if previous or len(peers)>1:
            quarantined.append({'binding':binding,'reason':'competing_statistic_for_fixed_test_and_role',
                'previous_candidates':deepcopy(previous),'new_candidate_ids':[b['statistic']['id'] for b in peers],
                'scientific_owner_resolved':False,'human_assignment_created':False})
        else:kept.append(binding)
    result['new_bindings']=kept;result['numeric_ownership_quarantine']=quarantined
    result['original_before_ownership_guard']=deepcopy(delta)
    result['interpretation']='Prior proposals and all raw candidates are preserved. Competing additions are withheld pending source-ownership resolution, not declared wrong or assigned to a human.'
    return result
