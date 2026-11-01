"""Compare changed Results scaffolding while reusing immutable section outputs.

This is a development experiment, never a scientific model-lock procedure.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

from mosaic.canonical import canonical_bytes
from mosaic.recipe import resolve_recipe
from mosaic.vllm.client import VLLMClient
from mosaic.vllm.ledger import VLLMBudget,VLLMLedger
from mosaic.vllm.tokenizer import local_tokenizer
from mosaic.workspace import Workspace
from scripts.integrated_extraction_trial import (
    CANDIDATE,ALLOWED_WORKS,load_inputs,fit_task,request_allowance,IncompleteExtraction,
    compile_section_task)
from scripts.section_comparison_probe import planning_overview
from scripts.section_source_context import prepare_contexts
from scripts.numbered_results_routing import route_numbered_results
from scripts.scoped_results_trial import prepare_task,materialize,bounded_study_payload,binding_eligible_tests
from scripts.statistic_binding_trial import prepare_task as prepare_bindings,materialize as materialize_bindings
from scripts.stress_screen_authorization import authorization
from scripts.measure_wording_trial import prepare_task as prepare_wording, materialize as materialize_wording


def load_scope(scope):
    ws=Workspace();manifest=json.loads(ws.check(scope['source_manifest']).read_bytes())
    source,binding,ordered=load_inputs(manifest)
    original={u['unit_id']:u for u in source['units']}
    routed=route_numbered_results(ordered)
    matches=[s for s in routed['studies'] if s['study_id']==scope['study_id']]
    if len(matches)!=1:raise ValueError('study_not_uniquely_located')
    primary_ids={u['unit_id'] for u in matches[0]['units']}
    if scope.get('bounded_study'):
        prepared,_,_=prepare_contexts(routed,planning_overview(routed))
        matches=[s for s in prepared if s['study']['study_id']==scope['study_id']]
        if len(matches)!=1:raise ValueError('study_not_uniquely_located')
        payload=bounded_study_payload(matches[0]);previous=None
    else:
        previous=json.loads(ws.check(scope['upstream']).read_bytes())
        payload={'physical_study_id':scope['study_id'],
            'source':deepcopy(previous['original_source_units']),
            'study_method_context':{}}
        if previous.get('original_context_units'):
            payload['source_context']=deepcopy(previous['original_context_units'])
    if any(u['unit_id'] not in primary_ids for u in payload['source']):
        raise ValueError('result_source_outside_named_study')
    for unit in payload['source']+payload.get('source_context',[]):
        if unit['unit_id'] not in original or any(unit.get(k)!=original[unit['unit_id']].get(k)
                for k in ('text','source_locator')):
            raise ValueError('replayed_source_identity_changed')
    return manifest,binding,payload,previous


def scoped_binding_task(recipe,upstream,*,operator_candidates=None,spelled_coefficients=False):
    # Remove task-local scope pointers before the attachment menu is constructed.
    # Scientific identities retain the source-checked phrase and proposed kind.
    prepared=deepcopy(upstream)
    for record in prepared['records']:
        for test in record['tests']:
            test['proposal']['analysis_scope']=[{'kind':s['proposed_kind'],'quote':s['text']}
                for s in test.get('scope_copies',[])]
    task=prepare_bindings(recipe,prepared,slim=True,quantities=True,located=True,named=True,operator_candidates=operator_candidates,spelled_coefficients=spelled_coefficients)
    task['upstream']=deepcopy(upstream)
    for test in task['payload']['tests']:
        if test['prior_proposal']['kind']=='main_effect':
            props=task['schema']['properties'][test['test_id']]['properties']
            props['relationship']={'enum':['main_effect','unresolved']}
            props['author_direction']={'enum':['no_difference','mixed','not_stated','source_conflict','unresolved']}
    request=deepcopy(task['body'])
    request['messages'][0]['content']=request['messages'][0]['content'].split('\nRESPONSE_SCHEMA:\n',1)[0]
    request['messages'][0]['content']+='\nSCOPED ANALYSES: analysis_scope contains source-copied identifying phrases for the same test. Distinguish every sample, division, time, outcome operationalization and alternative specification. A similar effect or same qualitative result under another specification never supplies its numerical estimate. Select no numeric binding unless the number belongs to that exact analysis. Shared p thresholds can apply to several tests only when the source explicitly says so. Do not treat R-squared or a model-level p-value as the coefficient-level test. An omnibus main_effect has no continuous positive or negative association direction: retain main_effect only when supported, otherwise unresolved with upstream_kind_conflict. Confidence intervals without a printed level retain unknown level.'
    task.update(compile_section_task(request,task['payload'],task['schema']))
    return task


def wording_task(recipe,scope):
    ws=Workspace();record=json.loads(ws.check(scope['wording_record']).read_bytes())['record']
    manifest=json.loads(ws.check(scope['source_manifest']).read_bytes())
    _,_,ordered=load_inputs(manifest)
    routed=route_numbered_results(ordered)
    prepared,_,_=prepare_contexts(routed,planning_overview(routed))
    matches=[p for p in prepared if p['study']['study_id']==scope['study_id']]
    if len(matches)!=1:raise ValueError('wording_study_not_unique')
    return prepare_wording(recipe,record,matches[0])


def run(manifest_path,out,*,live=False,base_url=None):
    ws=Workspace();scopes=json.loads(ws.check(manifest_path).read_bytes());root=Path(out)
    rid,recipe=resolve_recipe(CANDIDATE);count=local_tokenizer(recipe['vllm'])
    ledger=VLLMLedger(CANDIDATE)
    report={'recipe_id':rid,'stages':[],'scopes':[],'production_accepted':False,'scientifically_accepted':False}
    if live:
        if not base_url:raise ValueError('base_url_required')
        ws.write_once(root/'started.json',canonical_bytes(report))
    try:
        loaded=[(scope,*load_scope(scope)) for scope in scopes]
        for work in {row[1]['work'] for row in loaded}:
            budget=VLLMBudget.load() if work in ALLOWED_WORKS else authorization(work,CANDIDATE)[2]
            request_allowance(sum(2+bool(row[0].get('wording_record')) for row in loaded if row[1]['work']==work),ledger.totals().requests,
                ledger.totals_for_article(work).requests,budget.request_cap,budget.article_limit(CANDIDATE,work))
        if ledger.totals().unresolved:raise ValueError('unresolved_prior_call')
        for scope,manifest,binding,payload,previous in loaded:
            name=scope['name'];work=manifest['work'];budget=VLLMBudget.load();fence=None
            registry='config/screen_sources_v1.json'
            if work not in ALLOWED_WORKS:registry,_,budget,fence=authorization(work,CANDIDATE)
            sources=json.loads(ws.check(registry).read_bytes())['works'][work]['sources']
            client=VLLMClient(candidate=CANDIDATE,base_url=base_url,api_key=os.environ.get('VLLM_API_KEY'),
                timeout=240,stream=True,live=True,budget=budget,scope=fence) if live else None
            row={'name':name,'work':work,'study_id':scope['study_id'],
                'source_packet_sha256':binding['source_packet_sha256'],
                'previous_results_sha256':hashlib.sha256(canonical_bytes(previous)).hexdigest() if previous else None,
                'reused_section_artifacts':{p:hashlib.sha256(ws.check(p).read_bytes()).hexdigest()
                    for p in scope.get('reused_sections',[])}}
            report['scopes'].append(row)
            def call(stage,task):
                task,estimate,factored=fit_task(task,count)
                entry={'name':stage,'prompt_estimate':estimate,'lossless_factoring':factored}
                report['stages'].append(entry)
                if not live:return None
                ws.write_once(root/(stage+'-request.json'),canonical_bytes(task['body']))
                start=time.monotonic()
                response=client.chat_completion(work_id=work,batch_id=root.parent.name+'-'+stage,
                    packet=task['payload'],request_body=task['body'],source_sha256s=[s['sha256'] for s in sources],
                    text_provenance='native_article_text',remote=False,rights=['runpod_screen_auth'])
                ws.write_once(root/(stage+'-response.json'),canonical_bytes(response))
                choice=response['choices'][0];entry.update(seconds=round(time.monotonic()-start,2),
                    finish_reason=choice.get('finish_reason'),usage=response.get('usage'))
                if choice.get('finish_reason')!='stop':raise IncompleteExtraction(choice.get('finish_reason'))
                proposal=json.loads(choice['message']['content'])
                ws.write_once(root/(stage+'-parsed.json'),canonical_bytes(proposal))
                return proposal
            task=prepare_task(recipe,payload,bounded_study=scope.get('bounded_study',False))
            proposal=call(name+'-results',task)
            if not live:
                if scope.get('wording_record'):call(name+'-wording',wording_task(recipe,scope))
                row['planned_calls_at_most']=2+bool(scope.get('wording_record'))
                continue
            upstream=materialize(proposal,task)
            ws.write_once(root/(name+'-results-materialized.json'),canonical_bytes(upstream))
            eligible,excluded=binding_eligible_tests(upstream);row['excluded_tests']=excluded
            if scope.get('wording_record'):
                item_task=wording_task(recipe,scope)
                wording=materialize_wording(call(name+'-wording',item_task),item_task)
                ws.write_once(root/(name+'-wording-materialized.json'),canonical_bytes(wording))
            if not any(r['tests'] for r in eligible['records']):
                row['outcome']='no_valid_test_proposals';continue
            task=scoped_binding_task(recipe,eligible)
            bound=materialize_bindings(call(name+'-attachments',task),task)
            kinds={t['test_id']:t['proposal']['kind'] for r in eligible['records'] for t in r['tests']}
            for attachment in bound['attachments']:
                if kinds[attachment['test_id']]=='main_effect' and attachment['proposal']['relationship']!='main_effect':
                    attachment['role_resolution'].update(status='conflicting_proposals_abstain',kind_specific_roles_usable=False)
            bound['excluded_tests']=excluded
            ws.write_once(root/(name+'-attachments-materialized.json'),canonical_bytes(bound))
            row['outcome']='completed_with_exclusions' if excluded else 'completed_not_accepted'
        report['outcome']='completed_not_accepted' if live else 'planning_only'
    finally:
        if live:ws.write_once(root/'report.json',canonical_bytes(report))
    return report
