"""Source-exact, opt-in annotations for shared printed statistic expressions.

The original validated stage stays intact. Syntax annotations add no measures,
sample assignments or estimates, and do not verify the model's interpretation.
"""
from copy import deepcopy
import re

from scripts.source_sentence_choices import coherent_units, original_segments
from scripts.study_fact_completion import digest
from scripts.study_fact_selections import materialize
from scripts.typed_statistic_choices import build_statistic_choices

VERSION = 'study-statistic-annotations/v1'
_NUMBER = r'[+\-\u2212\u2013]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][+\-]?\d+)?'
_SPACE = r'[ \t\r\n]*'
_PLURAL = re.compile(r'(?<!\w)(?:(?P<quantifier>all)(?P<separator>[ \t\r\n]*))?'
    r'(?P<label>αs|alphas|ρs|rhos|ωs|omegas)' + _SPACE
    + r'(?P<comparator><=|>=|<|>|=|≤|≥)' + _SPACE + r'(?P<value>'+_NUMBER+r')(?!\w|\.\d)',re.I)


def plural_choices(units):
    """Add plural expressions while preserving every existing choice and ID."""
    baseline = build_statistic_choices(units)
    result = deepcopy(baseline)
    for unit in units:
        text = unit['text']
        for match in _PLURAL.finditer(text):
            # Only a Greek mathematical symbol may adjoin "all". Do not split
            # fused ASCII words such as "allalphas" or guess missing spaces.
            if match.group('quantifier') and not match.group('separator') and match.group('label')[0].isascii():
                continue
            # A range, percentage or fraction needs a complete contract of its
            # own; do not truncate it to the first numeric component.
            if re.match(_SPACE+r'(?:%|[-–—−/÷]'+_SPACE+r'[.\d]|\bto\b)',text[match.end():],re.I):
                continue
            components = [{'name':name,'start':match.start(name),'end':match.end(name),'text':match.group(name)}
                for name in ('value','comparator','label','quantifier') if match.group(name) is not None]
            result.append({'unit_id':unit['unit_id'],'start':match.start(),'end':match.end(),'text':match.group(),
                'statistic_type':'reliability','printed_label':match.group('label'),'components':components,
                'choice_id':f"plural-statistic:{unit['unit_id']}:{match.start()}:{match.end()}",
                'reported_scope':'plural_expression','reported_quantifier':match.group('quantifier'),
                'member_count':None,'members_identified':False,'semantic_ownership_verified':False})
    byid = {u['unit_id']:i for i,u in enumerate(units)}
    return sorted(result,key=lambda c:(byid[c['unit_id']],c['start'],c['end']))


def annotate(task, answer, plan, core_proposal, original_stage):
    """Rebuild the frozen stage before adding independently linked expressions.

    Repeated targets can point to the same source expression, but those links
    neither establish membership in its plural scope nor create point estimates.
    Quarantines, coverage and all prior asserted labels remain visible unchanged.
    """
    rebuilt = materialize(task,answer,plan,core_proposal)
    if digest(rebuilt) != digest(original_stage):
        raise ValueError('statistic_annotation_original_stage_changed')
    table = {s['id']:s for s in plan['core_task']['statement_table']}
    choices = plural_choices([{'unit_id':s,'text':r['text']} for s,r in table.items()])
    groups = {g['unit_id']:g for g in coherent_units(plan['source']['units'])}
    expressions = {}; bindings = []
    for fact in rebuilt['facts']:
        value = fact['value_source']; matches = []
        if value is not None:
            matches = [c for c in choices if c['unit_id'] == fact['source_sentence_id'] and any(
                p['name'] == 'value' and (p['start'],p['end']) == (value['start'],value['end'])
                for p in c['components'])]
        binding = {'proposal_index':fact['proposal_index'],'original_fact_sha256':digest(fact),
            'source_expression_id':None,'semantic_ownership_verified':False,
            'status':'no_numeric_value' if value is None else 'untyped_numeric_value'}
        if len(matches) > 1:
            binding['status'] = 'ambiguous_printed_expression'
        elif len(matches) == 1:
            choice = matches[0]; source = table[choice['unit_id']]
            parts = original_segments(groups[source['group_id']],
                source['start']+choice['start'],source['start']+choice['end'])
            identity = {'source_identity':plan['source_identity'],'statement_id':source['id'],
                'start':choice['start'],'end':choice['end'],'text':choice['text']}
            eid = 'printed-expression-'+digest(identity)
            expression = {'id':eid,'statement_id':source['id'],'syntax':deepcopy(choice),
                'original_segments':parts,'shared_plural_report':choice.get('reported_scope') == 'plural_expression',
                'semantic_ownership_verified':False,'separate_estimates_established':False}
            if eid in expressions and expressions[eid] != expression:
                raise ValueError('statistic_annotation_expression_collision')
            expressions[eid] = expression
            binding.update(status='bound_printed_expression',source_expression_id=eid)
        bindings.append(binding)
    return {'version':VERSION,'original_stage':deepcopy(original_stage),
        'original_stage_sha256':digest(original_stage),'source_expressions':list(expressions.values()),
        'fact_bindings':bindings,'new_model_inference':False,'model_request_unchanged':True,
        'scientifically_accepted':False,'production_accepted':False,
        'limits':['Printed type/operator annotations do not establish measure, sample or exclusion scope.',
            'Repeated links to one plural expression are not separate point estimates.',
            'Original selections, refusals, evidence and semantic errors remain unchanged.']}
