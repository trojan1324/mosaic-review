"""One uncurated development sequence from source packets to study proposals.

Results depend on original Methods, never on a successful measure inventory.
This is an engineering experiment, not scientific acceptance or production.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

from jsonschema import Draft202012Validator
from mosaic import REPO_ROOT
from mosaic.canonical import canonical_bytes
from mosaic.recipe import resolve_recipe
from mosaic.vllm.client import VLLMClient
from mosaic.vllm.ledger import VLLMBudget, VLLMLedger
from mosaic.vllm.tokenizer import local_tokenizer
from mosaic.workspace import Workspace
from scripts.section_comparison_probe import planning_overview, _initial_record
from scripts.structured_extraction_probe import overview_schema, present
from scripts.section_extraction_continuation import body_for
from scripts.section_extraction_contract import whole_proposal
from scripts.section_field_contract import measure_binding_packet, measure_binding_schema, apply_measure_binding
from scripts.section_field_probe import MEASURE_INSTRUCTION, make_body, xgr
from scripts.section_source_context import prepare_contexts
from scripts.selected_study_contract import materialize_selected
from scripts.results_heading_context import main_results_payload
from scripts.numbered_results_routing import route_numbered_results
from scripts.typed_test_decomposition import prepare_task as prepare_tests, materialize as materialize_tests
from scripts.statistic_binding_trial import prepare_task as prepare_bindings, materialize as materialize_bindings
from scripts.compact_section_payload import compact, expand
from scripts.source_sentence_choices import coherent_units

CANDIDATE = 'dev_vllm_screen_qwen36_27b_v1'


def _legacy_authorization():
    """Load private-screen authorization only for the legacy CLI path."""
    from scripts.audit_model_inputs import TUNING_WORKS
    from scripts.stress_screen_authorization import authorization as stress_authorization
    return TUNING_WORKS, stress_authorization


def __getattr__(name: str):
    """Retain the legacy roster export without loading it for public imports."""
    if name == "ALLOWED_WORKS":
        from scripts.audit_model_inputs import TUNING_WORKS
        return TUNING_WORKS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class IncompleteExtraction(ValueError):
    """A preserved model response ended before producing a complete proposal."""

    def __init__(self, finish_reason):
        super().__init__('incomplete_response:' + str(finish_reason))
        self.finish_reason = finish_reason


def overview_body(recipe, source, ordered):
    """Ask only for context the structural router can actually accept.

    Study membership and supplemental routing are already computed from source
    structure. Generating them again wastes output and never changes membership.
    The complete source remains in its immutable packet and study branches.
    """
    units = ordered['unassigned_source_units']
    if not units:
        raise ValueError('no_eligible_overview_source')
    schema = overview_schema(units, ordered['studies'])
    schema['properties'] = {k: schema['properties'][k] for k in ('hypotheses', 'definitions')}
    schema['required'] = list(schema['properties'])
    payload = {'task': 'FRONT_END_CONTEXT', 'ordered_source': [present(u) for u in units]}
    instruction = (
        'Identify explicitly stated hypotheses and construct definitions in the supplied unassigned article context. '
        'Cite only the evidence needed for each statement, not every related paragraph. '
        'Do not infer a hypothesis from a reported result or summarize the article. '
        'Study boundaries and supplemental routing are handled separately; do not generate routing. '
        'Return concise JSON matching RESPONSE_SCHEMA.')
    request = {'model': recipe['vllm']['served_model_name'], 'temperature': 0, 'seed': 0,
               'max_tokens': 4096,
               'messages': [{'role': 'system', 'content': instruction},
                            {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]}
    return request, payload, schema


def execute_overview(source, ordered, recipe, call):
    status = {'eligible_units': len(ordered['unassigned_source_units']),
              'semantic_accuracy_unassessed': True}
    empty = planning_overview(ordered)
    if not status['eligible_units']:
        return empty, {**status, 'status': 'not_applicable'}
    task = compile_section_task(*overview_body(recipe, source, ordered))
    try:
        proposal = call('overview', task)
    except IncompleteExtraction as exc:
        # Never parse partial JSON or report empty hypotheses as an absence claim.
        # Transport, authentication, budget and unexpected validation errors still
        # propagate; this is not a new retry permission.
        return empty, {**status, 'status': 'incomplete', 'finish_reason': exc.finish_reason,
                       'hypotheses_and_definitions': 'unresolved'}
    if proposal is None:
        return empty, {**status, 'status': 'planning_only'}
    return {**proposal, 'study_links': [], 'boundary_concerns': []}, {**status, 'status': 'complete'}


def load_inputs(manifest):
    allowed_works, stress_authorization = _legacy_authorization()
    if manifest['candidate'] != CANDIDATE:
        raise ValueError('unapproved_candidate_or_work')
    stress = None
    if manifest['work'] not in allowed_works:
        try:
            stress = stress_authorization(manifest['work'], CANDIDATE)
        except ValueError:
            raise ValueError('unapproved_candidate_or_work') from None
    ws = Workspace()
    source, binding, ordered = [json.loads(ws.check(manifest[key]).read_bytes())
                                for key in ('source', 'source_binding', 'ordered_context')]
    if stress and binding.get('registry_sha256') != hashlib.sha256(stress[0].read_bytes()).hexdigest():
        raise ValueError('stress_input_registry_changed')
    if source.get('work_id') != manifest['work'] or binding.get('source_packet_sha256') != hashlib.sha256(canonical_bytes(source)).hexdigest():
        raise ValueError('source_binding_identity_changed')
    rows = [u for s in ordered['studies'] for u in s['units']] + ordered['unassigned_source_units']
    lookup = {u['unit_id']: u for u in rows}
    original = {u['unit_id']: u for u in source['units']}
    if len(rows) != len(lookup) or len(original) != len(source['units']) or set(lookup) != set(original):
        raise ValueError('source_roster_changed')
    if any(any(lookup[uid].get(k) != v for k, v in unit.items()) for uid, unit in original.items()):
        raise ValueError('source_content_or_metadata_changed')
    if ordered.get('coverage', {}).get('omitted_units') != 0 or ordered.get('coverage', {}).get('duplicated_units') != 0:
        raise ValueError('ordered_source_loss')
    return source, binding, ordered


def request_allowance(calls, global_used, article_used, global_cap, article_cap):
    """Require room for planned calls; retries share the existing hard caps."""
    if global_used + calls > global_cap:
        raise ValueError('global_budget_refused')
    if article_used + calls > article_cap:
        raise ValueError('article_budget_refused')
    return min(2 * calls, global_cap - global_used, article_cap - article_used)


def result_chunks(study, *, maximum_groups=40):
    if maximum_groups < 1: raise ValueError('invalid_group_limit')
    payload=main_results_payload(study)
    groups=coherent_units(payload['source'])
    if len(groups)<=maximum_groups:return [payload]
    chunks=[]
    for start in range(0,len(groups),maximum_groups):
        units=[u for g in groups[start:start+maximum_groups] for u in g['original_units']]
        chunks.append({**deepcopy(payload),'source':units})
    if [u for c in chunks for u in c['source']] != payload['source']:
        raise ValueError('result_partition_changed_source')
    return chunks


def prompt_estimate(request, count):
    messages=request['messages']
    if len(messages)!=2 or [m['role'] for m in messages]!=['system','user'] or any(not isinstance(m['content'],str) for m in messages):
        raise ValueError('unsupported_chat_sizing')
    # Count what the model sees, not a JSON-escaped representation of the HTTP
    # messages array. Reserve 256 tokens for the pinned two-message chat wrapper.
    return sum(count(m['content']) for m in messages)+256


def fit_task(task, count, *, compact_prompts=False, context_limit=98304):
    """Factor repetition only when needed; never shorten source text or enums."""
    if compact_prompts:
        from scripts.compact_extraction_prompt import compact_task
        task = compact_task(task, count)
    request=task['body']
    original=prompt_estimate(request,count)
    if type(context_limit) is not int or context_limit <= request['max_tokens']:
        raise ValueError('context_limit_refused')
    if original+request['max_tokens']<context_limit and task['payload'].get('task')!='FRONT_END_CONTEXT':
        return task,original,False
    if 'RESPONSE_SCHEMA:' not in request['messages'][0]['content']:
        raise ValueError('context_limit_refused')
    payload=compact(task['payload'])
    if expand(payload)!=task['payload']:raise ValueError('payload_roundtrip_failed')
    result=deepcopy(task);result['payload']=payload;request=result['body']
    instruction=request['messages'][0]['content'].split('\nRESPONSE_SCHEMA:\n',1)[0]
    instruction+='\nSection paths are factored without changes: each source section_path_ids list refers to the complete ordered heading path in section_path_catalog. All original lines remain supplied.'
    request['messages'][0]['content']=instruction+'\nRESPONSE_SCHEMA:\n'+json.dumps(task['schema'],ensure_ascii=False,separators=(',',':'))
    request['messages'][1]['content']=json.dumps(payload,ensure_ascii=False,separators=(',',':'))
    if compact_prompts:
        result=compact_task(result,count)
        request=result['body']
    estimate=prompt_estimate(request,count)
    if estimate+request['max_tokens']>=context_limit:raise ValueError('context_limit_refused_after_lossless_factoring')
    return result,estimate,True


def compile_section_task(request, payload, schema):
    """Apply the previously verified decoder fixes consistently to old builders."""
    request = deepcopy(request)
    request.pop('reasoning_effort', None)
    request['chat_template_kwargs'] = {'enable_thinking': False}
    grammar = str(xgr.Grammar.from_json_schema(json.dumps(schema, ensure_ascii=False), max_whitespace_cnt=16))
    xgr.Grammar.from_ebnf(grammar)
    request['structured_outputs'] = {'grammar': grammar}
    # The schema shown to the model and native decoder have identical member order.
    marker = '\nRESPONSE_SCHEMA:\n'
    instruction = request['messages'][0]['content'].split(marker, 1)[0]
    # Retain the continuation's direction guidance, which follows its schema.
    if '\nDIRECTION CHECK:' in request['messages'][0]['content']:
        instruction += '\nDIRECTION CHECK:' + request['messages'][0]['content'].split('\nDIRECTION CHECK:', 1)[1]
    request['messages'][0]['content'] = instruction + marker + json.dumps(schema, ensure_ascii=False)
    return {'body': request, 'payload': payload, 'schema': schema}


def design_record(study, proposal):
    primary = study['study']
    combined = whole_proposal({'design': proposal}, primary['study_id'], primary['label'])
    return materialize_selected(combined, study['units'], study['choices'], [],
                                [u['unit_id'] for u in primary['units']], study['reuse_links'])


def execute_study(study, hypotheses, recipe, call, save, *, scoped_results=False, complete_inventory=False):
    """Independent section branches; failed measure copying cannot erase Results."""
    result = {'production_accepted': False, 'scientifically_accepted': False, 'problems': []}
    for kind in ('measures', 'design'):
        task = compile_section_task(*body_for(kind, study, hypotheses, recipe, []))
        proposal = call(kind, task)
        try:
            materialized = _initial_record(study, proposal) if kind == 'measures' else design_record(study, proposal)
            result[kind] = materialized
            save(kind + '-materialized', materialized)
            if not materialized['mechanical_valid']:
                result['problems'].append(kind + '_materialization_incomplete')
                continue
            if kind == 'measures':
                packet = measure_binding_packet(materialized['record'], study)
                if packet['slots']:
                    schema = measure_binding_schema(packet)
                    task = compile_section_task(make_body(recipe, packet, schema, MEASURE_INSTRUCTION), packet, schema)
                    bound_proposal = call('measure-fields', task)
                    bound = apply_measure_binding(materialized['record'], bound_proposal, packet, study)
                    result['measure_fields'] = bound
                    save('measure-fields-materialized', bound)
                else:
                    result['problems'].append('no_measure_slots')
        except (ValueError, KeyError) as exc:
            result['problems'].append(kind + '_materialization_refused:' + type(exc).__name__)
    if scoped_results:
        from scripts.scoped_study_execution import execute_results
        findings=execute_results(study,recipe,call,save,complete_inventory=complete_inventory)
        result['problems'].extend(findings.pop('problems'))
        result.update(findings)
        return result
    # Do not consume the model's Methods inventory, design, or interpretation here.
    chunks=result_chunks(study)
    result['result_chunk_count']=len(chunks)
    if len(chunks)>1:
        result['result_chunks']=[];result['attachment_chunks']=[]
        result['cross_chunk_relations_unassessed']=True
    for i,payload in enumerate(chunks,1):
        suffix='' if len(chunks)==1 else '-part-'+str(i)
        task = prepare_tests(recipe, payload, evidence_mode='coherent')
        proposal = call('results'+suffix, task)
        upstream = materialize_tests(proposal, task)
        if len(chunks)==1:result['results']=upstream
        else:result['result_chunks'].append(upstream)
        save('results'+suffix+'-materialized', upstream)
        if any(t.get('copy_errors') for r in upstream['records'] for t in r['tests']):
            result['problems'].append('results_source_copy_errors'+suffix)
            continue
        if not any(r['tests'] for r in upstream['records']):
            result['problems'].append('no_result_tests'+suffix)
            continue
        task = prepare_bindings(recipe, upstream, slim=True, quantities=True, located=True)
        proposal = call('attachments'+suffix, task)
        bound=materialize_bindings(proposal,task)
        if len(chunks)==1:result['attachments']=bound
        else:result['attachment_chunks'].append(bound)
        save('attachments'+suffix+'-materialized', bound)
    return result


def run(manifest_path, out, *, live=False, base_url=None, allow_partial_studies=False, scoped_results=False, complete_inventory=False, compact_prompts=False):
    if complete_inventory and not scoped_results:raise ValueError('inventory_completion_requires_scoped_results')
    allowed_works, stress_authorization = _legacy_authorization()
    ws = Workspace(); manifest = json.loads(ws.check(manifest_path).read_bytes())
    source, binding, ordered = load_inputs(manifest)
    ordered = route_numbered_results(ordered)
    rid, recipe = resolve_recipe(CANDIDATE)
    ledger, budget = VLLMLedger(CANDIDATE), VLLMBudget.load()
    source_registry = REPO_ROOT/'config/screen_sources_v1.json'
    scope = None
    if manifest['work'] not in allowed_works:
        source_registry, _, budget, scope = stress_authorization(manifest['work'], CANDIDATE)
    planned_studies,_,_=prepare_contexts(ordered,planning_overview(ordered))
    all_study_count=len(planned_studies);skipped=[]
    if allow_partial_studies:
        from scripts.partial_study_plan import independently_runnable_studies
        planned_studies,skipped=independently_runnable_studies(planned_studies)
    calls = (1 + 3 * len(planned_studies) + ((5 if complete_inventory else 4) if scoped_results else 2) * sum(len(result_chunks(s)) for s in planned_studies)) if planned_studies else 0
    if ledger.totals().unresolved:
        raise ValueError('global_budget_refused')
    attempts = request_allowance(calls, ledger.totals().requests,
        ledger.totals_for_article(manifest['work']).requests, budget.request_cap,
        budget.article_limit(CANDIDATE, manifest['work']))
    root = Path(out); count = local_tokenizer(recipe['vllm'])
    report = {'candidate': CANDIDATE, 'recipe_id': rid, 'planned_calls_at_most': calls,
              'maximum_attempts': attempts, 'source_packet_sha256': binding['source_packet_sha256'],
              'production_accepted': False, 'scientifically_accepted': False, 'stages': [], 'studies': []}
    report.update(all_study_count=all_study_count,skipped_studies=skipped,
                  allow_partial_studies=allow_partial_studies,scoped_results=scoped_results,complete_inventory=complete_inventory)
    if compact_prompts: report['prompt_mode'] = 'lossless-prompt-v1'
    if not planned_studies:
        report['outcome']='no_study_with_admitted_results'
        if live:ws.write_once(root/'report.json',canonical_bytes(report))
        return report
    if live:
        if not base_url: raise ValueError('base_url_required')
        ws.write_once(root/'started.json', canonical_bytes(report))
        client = VLLMClient(candidate=CANDIDATE, base_url=base_url, api_key=os.environ.get('VLLM_API_KEY'), timeout=240, stream=True, live=True, budget=budget, scope=scope)
    sources = json.loads(source_registry.read_bytes())['works'][manifest['work']]['sources']

    def call(name, task):
        task,estimate,compacted=fit_task(task,count,compact_prompts=compact_prompts)
        entry = {'stage': name, 'prompt_estimate': estimate,'lossless_factoring':compacted}; report['stages'].append(entry)
        if compact_prompts: entry['prompt_compaction'] = task['prompt_compaction']
        if not live: return None
        ws.write_once(root/(name+'-request.json'), canonical_bytes(task['body']))
        start = time.monotonic()
        response = client.chat_completion(work_id=manifest['work'], batch_id=root.parent.name+'-'+name,
            packet=task['payload'], request_body=task['body'], source_sha256s=[s['sha256'] for s in sources],
            text_provenance='native_article_text', remote=False, rights=['runpod_screen_auth'])
        ws.write_once(root/(name+'-response.json'), canonical_bytes(response))
        choice = response['choices'][0]
        entry.update(seconds=round(time.monotonic()-start,2), finish_reason=choice.get('finish_reason'), usage=response.get('usage'))
        if choice.get('finish_reason') != 'stop': raise IncompleteExtraction(choice.get('finish_reason'))
        proposal = json.loads(choice['message']['content'])
        ws.write_once(root/(name+'-parsed.json'), canonical_bytes(proposal))
        Draft202012Validator(task['schema']).validate(proposal)
        return proposal

    try:
        overview, report['overview'] = execute_overview(source, ordered, recipe, call)
        studies, accepted, routing = prepare_contexts(ordered, overview)
        if allow_partial_studies:
            studies,actual_skipped=independently_runnable_studies(studies)
            if actual_skipped!=skipped or [s['study']['study_id'] for s in studies]!=[s['study']['study_id'] for s in planned_studies]:
                raise ValueError('runnable_study_roster_changed')
        if live: ws.write_once(root/'routing.json', canonical_bytes(routing))
        for index, study in enumerate(studies, 1):
            stem = 'study-' + str(index)
            if not live:
                for kind in ('measures','design'):
                    call(stem+'-'+kind, compile_section_task(*body_for(kind,study,accepted['hypotheses'],recipe,[])))
                chunks=result_chunks(study)
                for i,payload in enumerate(chunks,1):
                    suffix='' if len(chunks)==1 else '-part-'+str(i)
                    if scoped_results:
                        from scripts.scoped_results_trial import prepare_task as prepare_scoped
                        task=prepare_scoped(recipe,payload)
                        if complete_inventory:
                            from scripts.comparison_member_evidence import finalize_task
                            task=finalize_task(task)
                        call(stem+'-results'+suffix,task)
                    else:call(stem+'-results'+suffix, prepare_tests(recipe,payload,evidence_mode='coherent'))
                continue
            result = execute_study(study, accepted['hypotheses'], recipe,
                lambda name, task: call(stem+'-'+name, task),
                lambda name, value: ws.write_once(root/(stem+'-'+name+'.json'), canonical_bytes(value)),
                scoped_results=scoped_results,complete_inventory=complete_inventory)
            ws.write_once(root/(stem+'-bundle.json'), canonical_bytes(result))
            complete=('attachments' in result if result['result_chunk_count']==1 else len(result['attachment_chunks'])==result['result_chunk_count'])
            report['studies'].append({'study_id': study['study']['study_id'], 'stem': stem,
                                     'problems': result['problems'], 'attachments_complete': complete,
                                     'result_chunk_count':result['result_chunk_count']})
        report['outcome'] = 'sequence_completed_not_accepted' if live else 'static_planning_only'
        if live and (report['overview']['status'] == 'incomplete' or skipped):
            report['outcome'] = 'sequence_partial_not_accepted'
    finally:
        if live: ws.write_once(root/'report.json', canonical_bytes(report))
    return report
