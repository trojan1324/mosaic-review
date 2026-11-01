"""Finite generalization test on two version-isolated main-article studies.

The source remains immutable. This is not a full article extraction: supplements,
front-end claims and cross-study robustness remain explicitly outside this trial.
"""
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
from mosaic.vllm.ledger import VLLMLedger,VLLMBudget
from mosaic.vllm.tokenizer import local_tokenizer
from scripts.section_comparison_probe import load_input,body_for,_initial_record,_add_design,_sources_for
from scripts.section_source_context import prepare_contexts
from scripts.section_field_contract import measure_binding_packet,measure_binding_schema,apply_measure_binding
from scripts.section_field_probe import make_body,MEASURE_INSTRUCTION
from scripts.targeted_section_trial import grouped_body,CANDIDATE,materialize_paragraph_results,resolve_statistic_choice
from jsonschema import Draft202012Validator

ROOT=REPO_ROOT/'scratch/fresh-section-v2'
WORK='KS-DOC-000129'
INPUT=REPO_ROOT/'scratch/ordered-comparison-v2/revised'


def inputs():
    source,binding,packet=load_input(INPUT/'slot-4-source.json',INPUT/'slot-4-source-binding.json',INPUT/'slot-4-ordered-context.json',WORK)
    # Explicit frozen development scope from the bounded source audit. The
    # supplied supplement includes another article version; do not blend it.
    studies=[]
    for index,(first,last) in enumerate(((53,66),(67,80)),1):
        old=next(s for s in packet['studies'] if s['study_id']==f'study:S{index}')
        selected=[u for u in old['units'] if first<=int(u['unit_id'][1:])<=last]
        assert len(selected)==last-first+1
        assert all(u['source_locator'].startswith('main_') for u in selected)
        studies.append({**old,'units':selected})
    scoped={**packet,'studies':studies,'unassigned_source_units':[]}
    contexts,_,_=prepare_contexts(scoped,{'hypotheses':[],'definitions':[],'study_links':[],'boundary_concerns':[]})
    return source,binding,contexts


def results_context(context,index):
    result=deepcopy(context)
    # The parser labelled these printed subheadings unclassified. This scope
    # includes Results/discussion plus the second study's empirical reweighting.
    first=61 if index==1 else 74
    result['study']['units']=[u for u in context['study']['units'] if int(u['unit_id'][1:])>=first]
    return result


def preparation():
    source,binding,contexts=inputs();rid,recipe=resolve_recipe(CANDIDATE)
    ledger,budget=VLLMLedger(CANDIDATE),VLLMBudget.load()
    assert not ledger.totals().unresolved
    assert ledger.totals().requests+16<=budget.request_cap
    assert ledger.totals_for_article(WORK).requests+16<=budget.article_limit(CANDIDATE,WORK)
    count=local_tokenizer(recipe['vllm']);stages=[]
    for i,context in enumerate(contexts,1):
        for kind in ('measures','design'):
            body,_,_=body_for(kind,context,[],recipe,[])
            estimate=count(json.dumps(body['messages'],ensure_ascii=False))+256
            assert estimate+body['max_tokens']<98304
            stages.append({'study':i,'stage':kind,'prompt_estimate':estimate})
    return contexts,rid,recipe,{'planned_calls':8,'maximum_attempts':16,'source_packet_sha256':binding['source_packet_sha256'],
        'scope':'two main-article studies only; no supplement version blending, front-end or full-article coverage claim',
        'production_accepted':False,'stages':stages}


def main():
    p=argparse.ArgumentParser();p.add_argument('--live',action='store_true');p.add_argument('--base-url');args=p.parse_args()
    contexts,rid,recipe,report=preparation()
    if not args.live:print(json.dumps(report));return 0
    assert args.base_url
    ws=Workspace();ws.write_once(ROOT/'experiment-started.json',canonical_bytes(report));report['stages']=[]
    count=local_tokenizer(recipe['vllm'])
    client=VLLMClient(candidate=CANDIDATE,base_url=args.base_url,api_key=os.environ.get('VLLM_API_KEY'),timeout=240,stream=True,live=True)
    def call(name,body,payload,schema):
        assert count(json.dumps(body['messages'],ensure_ascii=False))+256+body['max_tokens']<98304
        ws.write_once(ROOT/(name+'-request.json'),canonical_bytes(body));start=time.monotonic()
        response=client.chat_completion(work_id=WORK,batch_id=ROOT.name+'-'+name,packet=payload,request_body=body,
            source_sha256s=_sources_for(WORK),text_provenance='native_article_text',remote=False,rights=['runpod_screen_auth'])
        ws.write_once(ROOT/(name+'-response.json'),canonical_bytes(response));choice=response['choices'][0]
        report['stages'].append({'stage':name,'seconds':round(time.monotonic()-start,2),'finish_reason':choice.get('finish_reason'),'usage':response.get('usage')})
        if choice.get('finish_reason')!='stop':raise ValueError('incomplete_response')
        proposal=json.loads(choice['message']['content']);ws.write_once(ROOT/(name+'-parsed.json'),canonical_bytes(proposal))
        Draft202012Validator(schema).validate(proposal);return proposal
    try:
        for i,context in enumerate(contexts,1):
            stem=f'study-{i}'
            body,payload,schema=body_for('measures',context,[],recipe,[])
            proposal=call(stem+'-measures',body,payload,schema)
            base=_initial_record(context,proposal)
            ws.write_once(ROOT/(stem+'-measures-materialized.json'),canonical_bytes(base))
            if not base['mechanical_valid']:
                report['stages'][-1]['errors']=base['errors'];continue
            packet=measure_binding_packet(base['record'],context);schema=measure_binding_schema(packet)
            proposal=call(stem+'-fields',make_body(recipe,packet,schema,MEASURE_INSTRUCTION),packet,schema)
            bound=apply_measure_binding(base['record'],proposal,packet,context)
            ws.write_once(ROOT/(stem+'-measure-bound.json'),canonical_bytes(bound))
            body,payload,schema=body_for('design',context,[],recipe,[])
            proposal=call(stem+'-design',body,payload,schema)
            designed=_add_design(bound['record'],context,proposal)
            ws.write_once(ROOT/(stem+'-design-materialized.json'),canonical_bytes(designed))
            body,payload,schema,units,choices=grouped_body(recipe,designed['record'],results_context(context,i),[])
            proposal=call(stem+'-results',body,payload,schema)
            result=materialize_paragraph_results(designed['record'],proposal,units,choices,[],context['units'],grouped=True,resolve_choice=resolve_statistic_choice)
            result['errors']+=designed['errors'];result['mechanical_valid']=not result['errors']
            result['manipulations']=designed['manipulations'];result['manipulation_copies']=designed['manipulation_copies']
            ws.write_once(ROOT/(stem+'-final.json'),canonical_bytes(result))
    finally:
        ws.write_once(ROOT/'probe-result.json',canonical_bytes(report));print(json.dumps(report),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
