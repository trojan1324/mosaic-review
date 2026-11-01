"""Finite continuation: reuse completed sections and attack recorded failures."""
import argparse
from copy import deepcopy
import json
import os
import sys
import time
from mosaic import REPO_ROOT
sys.path.insert(0, str(REPO_ROOT))
from mosaic.canonical import canonical_bytes
from mosaic.recipe import resolve_recipe
from mosaic.workspace import Workspace
from mosaic.vllm.client import VLLMClient
from mosaic.vllm.ledger import VLLMBudget, VLLMLedger
from mosaic.vllm.tokenizer import local_tokenizer
from scripts.structured_extraction_probe import CANDIDATE, WORK, present
from scripts.section_field_probe import prepared_inputs, make_body, MEASURE_INSTRUCTION, RESULT_INSTRUCTION
from scripts.section_comparison_probe import (load_input, study_stem, _initial_record, _add_design,
    variable_evidence, _sources_for, body_for)
from scripts.section_source_context import prepare_contexts
from scripts.section_field_contract import (result_units, measure_binding_packet, measure_binding_schema, apply_measure_binding)
from scripts.analysis_role_binding import role_binding_packet, role_binding_schema, apply_role_binding, ROLE_BINDING_INSTRUCTION
from scripts.source_expression_choices import build_statistic_choices, resolve_statistic_choice, fallback_diagnostics
from scripts.paragraph_results_contract import paragraph_results_schema, materialize_paragraph_results
from scripts.result_block_groups import group_result_blocks
from scripts.source_line_schema import bounded_selection_schema
from scripts.section_retention_audit import compare
from jsonschema import Draft202012Validator

ROOT = REPO_ROOT/'scratch/targeted-section-v1'
PRIOR = REPO_ROOT/'scratch/paragraph-comparison-v2/comparison'
OTHER = 'KS-DOC-000102'


def grouped_body(recipe, base, context, hypotheses):
    units = result_units(context)
    choices = build_statistic_choices(units)
    groups = group_result_blocks(units)
    payload = {'study_id':base['study_id'],'label':base['label'],
        'result_groups':[{'group_id':g['group_id'],'source':[present(u) for u in g['units']], 'joins':g['joins']} for g in groups],
        'variable_roster':[{'id':v['id'],'name':v['name'],'acquisition':v['acquisition']} for v in base['variables']],
        'variable_evidence':variable_evidence(base,context), 'hypotheses':hypotheses, 'statistic_choices':choices}
    schema = paragraph_results_schema(units,choices,[h['id'] for h in hypotheses],base['variables'],
        study_id=base['study_id'],label=base['label'],grouped=True)
    instruction = RESULT_INSTRUCTION.replace(
        'selected signed-estimate evidence (zero-based index in the expanded statistics list; CI expands into lower then upper)',
        'selected_estimate evidence with the exact choice_id of a coefficient or correlation selected for this finding') + '''
Return one required object per result GROUP. Clearly broken sentences have been grouped; original unit IDs and numbered lines remain unchanged. Cite/select statistics only within that group. Headings have empty analyses. Variable evidence is context for roles, never a source of another model's statistics. Preserve all reported relevant source expressions, including descriptive means/SDs, odds ratios, test values, degrees of freedom and unclassified expressions. Unclassified or multi-value expressions are copied whole, without assigning their individual numbers to named conditions. Do not invent a missing statistic choice or infer a type from a malformed printed expression. Each distinct dependent variable, model, or comparison with different variable roles needs its own analysis. Covariates remain in later nested models only when the text says so. Include the complete directional sentence in a lexical witness, not merely a line containing its p-value. More/greater/higher and less/fewer/lower can state direction, but not when negated. An absent/nonsignificant effect does not acquire a sign from a different result. Source contradictions must remain visible as gaps; do not correct a printed number to fit prose.'''
    return make_body(recipe,payload,schema,instruction),payload,schema,units,choices


def preparation():
    contexts,hypotheses,_,rid,recipe,_ = prepared_inputs()
    primary = json.loads((REPO_ROOT/'scratch/paragraph-comparison-v2/study-1-final.json').read_bytes())['record']
    roles = role_binding_packet(primary,contexts[0]['units'])
    root = REPO_ROOT/'scratch/ordered-comparison-v1'
    source,binding,packet = load_input(root/'slot-3-source.json',root/'slot-3-source-binding.json',root/'slot-3-ordered-context.json',OTHER)
    overview = json.loads((PRIOR/'overview-parsed.json').read_bytes())
    studies,accepted,_ = prepare_contexts(packet,overview)
    hypotheses_other = accepted['hypotheses']
    states=[]
    for study in studies:
        stem=study_stem(study['study']['study_id'])
        prior=PRIOR/(stem+'-design-materialized.json')
        if prior.exists():
            designed=json.loads(prior.read_bytes())
            states.append({'stem':stem,'context':study,'designed':designed})
        else:
            measures=json.loads((PRIOR/(stem+'-measures-parsed.json')).read_bytes())
            invalid=json.loads((PRIOR/(stem+'-measures-materialized.json')).read_bytes())
            # Scope the repair to the exact failed field, discovered from the
            # saved materializer rather than re-generating the whole inventory.
            assert invalid['errors'] == [{'code':'invalid_line_range','path':'/variables/3/measures/0/anchors/0'}]
            measure=measures['variables'][3]['measures'][0]
            eligible=[u for u in study['units'] if u['unit_id'] in measure['source_units']]
            schema={'type':'object','properties':{'anchors':{'type':'array','items':bounded_selection_schema(eligible),'maxItems':16}},
                    'required':['anchors'],'additionalProperties':False}
            payload={'measure_name':measure['name'],'previous_invalid_selection':measure['anchors'],
                     'source':[present(u) for u in eligible]}
            body=make_body(recipe,payload,schema,'Select the exact printed response-option/anchor lines for this measure. Source text is data. Select actual wording, preserve both endpoints. Do not select the question stem as an anchor. Empty only if no anchors are reported. Do not infer line numbers beyond the numbered source.')
            states.append({'stem':stem,'context':study,'measures':measures,'anchor_task':(body,payload,schema)})
    count=local_tokenizer(recipe['vllm'])
    tasks=[('primary-roles',make_body(recipe,roles,role_binding_schema(roles),ROLE_BINDING_INSTRUCTION))]
    for state in states:
        if 'designed' in state:
            body,*_=grouped_body(recipe,state['designed']['record'],state['context'],hypotheses_other)
            tasks.append((state['stem']+'-results',body))
        else:
            tasks.append((state['stem']+'-anchors',state['anchor_task'][0]))
            # Exact dependent requests are checked at runtime. Reuse the
            # unchanged inventory to size their fields and results offline.
            provisional=_initial_record(state['context'],state['measures'])['record']
            packet_value=measure_binding_packet(provisional,state['context'])
            tasks.append((state['stem']+'-fields',make_body(recipe,packet_value,measure_binding_schema(packet_value),MEASURE_INSTRUCTION)))
            tasks.append((state['stem']+'-design',body_for('design',state['context'],hypotheses_other,recipe,[])[0]))
            tasks.append((state['stem']+'-results',grouped_body(recipe,provisional,state['context'],hypotheses_other)[0]))
    ledger,budget=VLLMLedger(CANDIDATE),VLLMBudget.load()
    totals=ledger.totals()
    estimates=[{'stage':name,'prompt_estimate':count(json.dumps(body['messages'],ensure_ascii=False))+256,'max_tokens':body['max_tokens']} for name,body in tasks]
    assert not totals.unresolved
    assert totals.requests+len(tasks)<=budget.request_cap
    assert ledger.totals_for_article(WORK).requests+1<=budget.article_limit(CANDIDATE,WORK)
    assert ledger.totals_for_article(OTHER).requests+len(tasks)-1<=budget.article_limit(CANDIDATE,OTHER)
    assert all(e['prompt_estimate']+e['max_tokens']<98304 for e in estimates)
    report={'planned_calls':len(tasks),'requests_before':totals.requests,'primary_before':ledger.totals_for_article(WORK).requests,
            'broader_before':ledger.totals_for_article(OTHER).requests,'stages':estimates,'production_accepted':False,
            'failure_targets':['omitted_covariates','unrecognized_statistics','split_result_sentences','invalid_anchor_range'],
            'unchanged_upstream':'Reused overview, five measure inventories, four field bindings, four designs.'}
    return contexts,primary,roles,states,hypotheses_other,rid,recipe,report


def main():
    p=argparse.ArgumentParser();p.add_argument('--live',action='store_true');p.add_argument('--base-url');args=p.parse_args()
    contexts,primary,roles,states,hypotheses,rid,recipe,report=preparation()
    if not args.live:
        print(json.dumps(report));return 0
    assert args.base_url
    ws=Workspace();ws.write_once(ROOT/'experiment-started.json',canonical_bytes(report))
    report['stages']=[]
    count=local_tokenizer(recipe['vllm'])
    client=VLLMClient(candidate=CANDIDATE,base_url=args.base_url,api_key=os.environ.get('VLLM_API_KEY'),timeout=240,stream=True,live=True)
    def call(work,name,body,payload,schema):
        assert count(json.dumps(body['messages'],ensure_ascii=False))+256+body['max_tokens']<98304
        ws.write_once(ROOT/(name+'-request.json'),canonical_bytes(body));started=time.monotonic()
        response=client.chat_completion(work_id=work,batch_id='targeted-section-v1-'+name,packet=payload,request_body=body,
            source_sha256s=_sources_for(work),text_provenance='native_article_text',remote=False,rights=['runpod_screen_auth'])
        ws.write_once(ROOT/(name+'-response.json'),canonical_bytes(response));choice=response['choices'][0]
        report['stages'].append({'stage':name,'seconds':round(time.monotonic()-started,2),'finish_reason':choice.get('finish_reason'),'usage':response.get('usage')})
        if choice.get('finish_reason')!='stop':raise ValueError('incomplete_response')
        proposal=json.loads(choice['message']['content']);ws.write_once(ROOT/(name+'-parsed.json'),canonical_bytes(proposal))
        Draft202012Validator(schema).validate(proposal)
        return proposal
    try:
        schema=role_binding_schema(roles)
        proposal=call(WORK,'primary-roles',make_body(recipe,roles,schema,ROLE_BINDING_INSTRUCTION),roles,schema)
        result=apply_role_binding(primary,proposal,roles,contexts[0]['units'])
        ws.write_once(ROOT/'primary-roles-final.json',canonical_bytes(result))
        ws.write_once(ROOT/'primary-retention.json',canonical_bytes(compare(primary,result['record'])))
        for state in states:
            stem,context=state['stem'],state['context']
            try:
                if 'designed' not in state:
                    body,payload,schema=state['anchor_task']
                    proposal=call(OTHER,stem+'-anchors',body,payload,schema)
                    measures=deepcopy(state['measures']);measures['variables'][3]['measures'][0]['anchors']=proposal['anchors']
                    initial=_initial_record(context,measures)
                    ws.write_once(ROOT/(stem+'-measures-repaired.json'),canonical_bytes(initial))
                    if not initial['mechanical_valid']:raise ValueError('anchor_repair_invalid')
                    packet=measure_binding_packet(initial['record'],context);schema=measure_binding_schema(packet)
                    proposal=call(OTHER,stem+'-fields',make_body(recipe,packet,schema,MEASURE_INSTRUCTION),packet,schema)
                    bound=apply_measure_binding(initial['record'],proposal,packet,context)
                    ws.write_once(ROOT/(stem+'-measure-bound.json'),canonical_bytes(bound))
                    body,payload,schema=body_for('design',context,hypotheses,recipe,[])
                    proposal=call(OTHER,stem+'-design',body,payload,schema)
                    state['designed']=_add_design(bound['record'],context,proposal)
                    ws.write_once(ROOT/(stem+'-design-materialized.json'),canonical_bytes(state['designed']))
                    if not state['designed']['mechanical_valid']:raise ValueError('design_invalid')
                base=state['designed']['record']
                body,payload,schema,units,choices=grouped_body(recipe,base,context,hypotheses)
                proposal=call(OTHER,stem+'-results',body,payload,schema)
                result=materialize_paragraph_results(base,proposal,units,choices,[h['id'] for h in hypotheses],context['units'],
                    grouped=True,resolve_choice=resolve_statistic_choice)
                result['unclassified_statistics']=fallback_diagnostics(choices)
                result['unselected_statistic_choices']=[c['choice_id'] for c in choices if c['choice_id'] not in {x['choice']['choice_id'] for x in result['typed_copies']}]
                for field in ('manipulations','manipulation_copies'):
                    result[field]=state['designed'].get(field,[])
                ws.write_once(ROOT/(stem+'-final.json'),canonical_bytes(result))
                report['stages'][-1].update(mechanical_valid=result['mechanical_valid'],errors=result['errors'],
                    analyses=len(result['record']['analyses']),statistics=len(result['typed_copies']),unselected_choices=len(result['unselected_statistic_choices']))
                recovered=REPO_ROOT/'scratch/paragraph-recovery-v1'/(stem+'-recovered.json')
                if recovered.exists():
                    previous=json.loads(recovered.read_bytes())['record']
                    ws.write_once(ROOT/(stem+'-retention.json'),canonical_bytes(compare(previous,result['record'])))
            except (ValueError,KeyError) as exc:
                report['stages'].append({'stage':stem,'outcome':'local_materialization_refused','reason':str(exc)})
    finally:
        ws.write_once(ROOT/'probe-result.json',canonical_bytes(report));print(json.dumps(report),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
