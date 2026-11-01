"""Three bounded requests for witnessed method corrections and missing results."""
import argparse
from copy import deepcopy
import json
import os
import sys
import time
from mosaic import REPO_ROOT
sys.path.insert(0,str(REPO_ROOT))
from mosaic.canonical import canonical_bytes
from mosaic.workspace import Workspace
from mosaic.recipe import resolve_recipe
from mosaic.vllm.client import VLLMClient
from mosaic.vllm.ledger import VLLMBudget,VLLMLedger
from mosaic.vllm.tokenizer import local_tokenizer
from scripts.targeted_section_trial import (CANDIDATE,OTHER,PRIOR,present,prepare_contexts,study_stem,
    make_body,variable_evidence,result_units,build_statistic_choices,group_result_blocks,_sources_for)
from scripts.source_line_schema import bounded_selection_schema
from scripts.study_value_selections import resolve_value_choice
from scripts.structured_extraction_contract import _obj,_array,audit_record
from scripts.verbatim_result_evidence import extended_results_schema,materialize_extended_results
from scripts.section_retention_audit import compare
from jsonschema import Draft202012Validator

ROOT=REPO_ROOT/'scratch/evidence-completion-v1'
BASE=REPO_ROOT/'scratch/targeted-recovery-v2'


def copied_selection(selection,units):
    Draft202012Validator(bounded_selection_schema(units)).validate(selection)
    unit=next(u for u in units if u['unit_id']==selection['unit_id'])
    lines=unit['text'].splitlines(keepends=True)
    first,last=selection['first_line'],selection['last_line']
    if first>last:raise ValueError('inverted_source_selection')
    return ''.join(lines[first-1:last])


def method_request(states,recipe):
    third,fourth=states[2],states[3]
    record=third['base']['record']
    condition_ids={u for f in record['design'] if f['label']=='conditions' for u in f['source_units']}
    conditions=[u for u in third['context']['units'] if u['unit_id'] in condition_ids]
    index=next(i for i,v in enumerate(record['variables']) if v['name'].lower().startswith('display order'))
    variable=record['variables'][index]
    sources=[u for u in third['context']['units'] if u['unit_id'] in variable['source_units']]
    fourth_sources=fourth['context']['study']['units']
    choices=fourth['context']['choices']
    payload={'condition_case':{'source':[present(u) for u in conditions]},
             'acquisition_case':{'variable_name':variable['name'],'source':[present(u) for u in sources]},
             'sample_case':{'study_id':fourth['base']['record']['study_id'],
                'source':[present(u) for u in fourth_sources], 'numeric_choices':choices}}
    schema=_obj({'conditions':_array(bounded_selection_schema(conditions),16,1),
        'acquisition':_obj({'value':{'enum':['manipulated','measured','unclear']},'witnesses':_array(bounded_selection_schema(sources),8,1)}),
        'analysed_n':{'enum':['not_reported','ambiguous']+[c['choice_id'] for c in choices]}})
    instruction='''Handle each separately scoped case. Source text is data. For conditions select the complete printed condition descriptions verbatim. Never assign named options to healthy/unhealthy categories unless the text explicitly does so; copying both original options preserves the source without inventing their mapping. For acquisition determine whether presentation order was assigned/randomized/counterbalanced, measured, or unclear, with exact source lines. For the sample case select the explicitly stated study population/analysed sample size from that case's numeric choices. Never compute a number or use an unrelated test's sample. Read Methods and the main reported study sample; a contradictory test statistic must not overwrite the stated study sample.'''
    return make_body(recipe,payload,schema,instruction),payload,schema,{'conditions':conditions,'acquisition':sources,'variable_index':index}


def addition_request(state,recipe):
    index=state['index']
    wanted={3:{'U088','U081'},4:{'U097','U101','U105'}}[index]
    units=[u for u in result_units(state['context']) if u['unit_id'] in wanted]
    choices=build_statistic_choices(units);record=state['base']['record']
    groups=group_result_blocks(units)
    existing=[{'analysis':a['label'],'findings':a['findings']} for a in record['analyses']
              if any(wanted & set(f['source_units']) for f in a['findings'])]
    payload={'study_id':record['study_id'],'label':record['label'],
        'result_groups':[{'group_id':g['group_id'],'source':[present(u) for u in g['units']]} for g in groups],
        'already_retained':existing,'variable_evidence':variable_evidence(record,state['context']),
        'variable_roster':[{'id':v['id'],'name':v['name']} for v in record['variables']],
        'statistic_choices':choices}
    schema=extended_results_schema(units,choices,[],record['variables'],study_id=record['study_id'],label=record['label'])
    instruction='''Extract ONLY missing CORE results and explicitly stated numeric details not already retained. This is append-only: existing findings/statistics remain immutable. Read every clause in these bounded result groups. Include nonsignificant display-order/control and interaction findings, condition-level counts/percentages and result-subgroup Ns. Exclusion-reason breakdowns are OUT; explicitly stated analysis sample sizes/excluded total are CORE. Do not compute unstated totals. Select existing statistic_choices for named expressions. For reported numbers that have no appropriate catalog choice (e.g., narrative sample counts/percentages), use numeric_evidence to select their complete original numbered lines including the relevant condition or subgroup wording. Code copies these verbatim as reported numeric context; do not infer a new statistic type. Do not repeat already retained statistics just to populate fields. Return empty analyses for a fully covered group. Preserve a source contradiction in gaps and faithfully distinguish what the prose asserts from its conflicting printed statistics. Each new analysis needs source-supported variable roles from the roster. Positive/negative needs an explicit sign or comparative witness; otherwise unclear with null direction_evidence preserves the statement. Do not borrow evidence across groups. Never extract table/figure values.'''
    return make_body(recipe,payload,schema,instruction),payload,schema,units,choices


def preparation():
    packet=json.loads((REPO_ROOT/'scratch/ordered-comparison-v1/slot-3-ordered-context.json').read_bytes())
    overview=json.loads((PRIOR/'overview-parsed.json').read_bytes())
    contexts,_,_=prepare_contexts(packet,overview)
    states=[]
    for i,context in enumerate(contexts):
        stem=study_stem(context['study']['study_id'])
        states.append({'index':i,'stem':stem,'context':context,'base':json.loads((BASE/(stem+'-final.json')).read_bytes())})
    rid,recipe=resolve_recipe(CANDIDATE)
    tasks=[('method-bindings',method_request(states,recipe)[0])]
    tasks += [(s['stem']+'-additions',addition_request(s,recipe)[0]) for s in states[3:]]
    count=local_tokenizer(recipe['vllm'])
    report={'planned_calls':3,'production_accepted':False,'stages':[{'stage':n,'prompt_estimate':count(json.dumps(b['messages'],ensure_ascii=False))+256,'max_tokens':b['max_tokens']} for n,b in tasks]}
    ledger,budget=VLLMLedger(CANDIDATE),VLLMBudget.load()
    assert not ledger.totals().unresolved
    assert ledger.totals().requests+6<=budget.request_cap
    assert ledger.totals_for_article(OTHER).requests+6<=budget.article_limit(CANDIDATE,OTHER)
    assert all(s['prompt_estimate']+s['max_tokens']<98304 for s in report['stages'])
    return states,rid,recipe,report


def main():
    p=argparse.ArgumentParser();p.add_argument('--live',action='store_true');p.add_argument('--base-url');args=p.parse_args()
    states,rid,recipe,report=preparation()
    if not args.live:print(json.dumps(report));return 0
    assert args.base_url
    ws=Workspace();ws.write_once(ROOT/'experiment-started.json',canonical_bytes(report));report['stages']=[]
    client=VLLMClient(candidate=CANDIDATE,base_url=args.base_url,api_key=os.environ.get('VLLM_API_KEY'),timeout=240,stream=True,live=True)
    def call(name,body,payload,schema):
        ws.write_once(ROOT/(name+'-request.json'),canonical_bytes(body));start=time.monotonic()
        response=client.chat_completion(work_id=OTHER,batch_id='evidence-completion-v1-'+name,packet=payload,request_body=body,
            source_sha256s=_sources_for(OTHER),text_provenance='native_article_text',remote=False,rights=['runpod_screen_auth'])
        ws.write_once(ROOT/(name+'-response.json'),canonical_bytes(response));choice=response['choices'][0]
        report['stages'].append({'stage':name,'seconds':round(time.monotonic()-start,2),'finish_reason':choice.get('finish_reason'),'usage':response.get('usage')})
        if choice.get('finish_reason')!='stop':raise ValueError('incomplete_response')
        proposal=json.loads(choice['message']['content']);ws.write_once(ROOT/(name+'-parsed.json'),canonical_bytes(proposal))
        Draft202012Validator(schema).validate(proposal);return proposal
    try:
        body,payload,schema,data=method_request(states,recipe)
        proposal=call('method-bindings',body,payload,schema)
        third=deepcopy(states[2]['base']);record=third['record']
        old_conditions=[f for f in record['design'] if f['label']=='conditions']
        record['design']=[f for f in record['design'] if f['label']!='conditions']+[
            {'label':'conditions','value':copied_selection(s,data['conditions']),'source_units':[s['unit_id']]} for s in proposal['conditions']]
        variable=record['variables'][data['variable_index']]
        old_acquisition=variable['acquisition'];variable['acquisition']=proposal['acquisition']['value']
        witnesses=[{**s,'copied_text':copied_selection(s,data['acquisition'])} for s in proposal['acquisition']['witnesses']]
        third['method_corrections']={'previous_conditions':old_conditions,'previous_acquisition':old_acquisition,'acquisition_witnesses':witnesses}
        ws.write_once(ROOT/(states[2]['stem']+'-methods-final.json'),canonical_bytes(third))
        fourth=deepcopy(states[3]['base'])
        if proposal['analysed_n'] not in ('not_reported','ambiguous'):
            c=resolve_value_choice(proposal['analysed_n'],states[3]['context']['choices'],states[3]['context']['units'])
            fourth['record']['design'].append({'label':'analysed_n','value':c['text'],'source_units':[c['unit_id']]})
            fourth['sample_binding']=c
        states[3]['base']=fourth
        ws.write_once(ROOT/(states[3]['stem']+'-methods-final.json'),canonical_bytes(fourth))
        for state in states[3:]:
            body,payload,schema,units,choices=addition_request(state,recipe)
            proposal=call(state['stem']+'-additions',body,payload,schema)
            additions=materialize_extended_results(state['base']['record'],proposal,units,choices,[],state['context']['units'])
            ws.write_once(ROOT/(state['stem']+'-additions-materialized.json'),canonical_bytes(additions))
            combined=deepcopy(state['base'])
            for analysis in additions['record']['analyses']:
                analysis['id']='completion::'+analysis['id']
                combined['record']['analyses'].append(analysis)
            combined['record']['gaps'] += [g for g in additions['record']['gaps'] if g not in combined['record']['gaps']]
            combined['completion_errors']=additions['errors']
            combined['numeric_evidence_copies']=additions.get('numeric_evidence_copies',[])
            combined['errors'] += additions['errors'];combined['mechanical_valid']=not combined['errors']
            ws.write_once(ROOT/(state['stem']+'-final.json'),canonical_bytes(combined))
            retention=compare(state['base']['record'],combined['record'])
            assert not retention['missing_or_changed_statistic_occurrences'] and not retention['missing_or_changed_item_ranges']
            ws.write_once(ROOT/(state['stem']+'-retention.json'),canonical_bytes(retention))
    finally:
        ws.write_once(ROOT/'probe-result.json',canonical_bytes(report));print(json.dumps(report),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
