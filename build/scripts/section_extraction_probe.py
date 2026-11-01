"""Bounded source-first tests of the preceding iteration's concrete failures."""
import argparse
import hashlib
import json
import os
import sys
import time
from mosaic import REPO_ROOT
from mosaic.canonical import canonical_bytes
from mosaic.vllm.client import VLLMClient
from mosaic.vllm.ledger import VLLMLedger, VLLMBudget
from mosaic.vllm.tokenizer import local_tokenizer
from mosaic.workspace import Workspace

sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / 'scratch/decoder-earlier'), str(REPO_ROOT / 'scratch/decoder-runtime')]
import xgrammar as xgr
from scripts.structured_extraction_probe import prepare, present, CANDIDATE, WORK
from scripts.structured_context_routing import route_context
from scripts.study_scope_guards import build_measure_reuse_links
from scripts.study_value_selections import build_value_choices
from scripts.selected_study_contract import materialize_selected
from scripts.section_extraction_contract import section_schema, whole_proposal

ROOT = REPO_ROOT / 'scratch/section-focused-v1'
TASKS = [(1, 'measures'), (2, 'measures'), (2, 'design'), (2, 'results')]


def preparation():
    packet, binding, _, rid, recipe = prepare()
    overview = json.loads((REPO_ROOT / 'scratch/structured-extraction-v1/overview-parsed.json').read_bytes())
    studies, accepted, routing = route_context(packet, overview)
    budget, ledger = VLLMBudget.load(), VLLMLedger(CANDIDATE)
    assert not ledger.totals().unresolved
    assert ledger.totals().requests + len(TASKS)*2 <= budget.request_cap
    assert ledger.totals_for_article(WORK).requests + len(TASKS)*2 <= budget.max_requests_per_article
    prepared = []
    for study in studies:
        others = [{**other, 'units': [u for u in other['units'] if any('measure' in (n['role']+' '+n['title']).lower() for p in u.get('section_paths', []) for n in p)]}
                  for other in studies if other['study_id'] != study['study_id']]
        links = build_measure_reuse_links(study['units'], others)
        needed = {link['reference_unit_id'] for link in links}
        reference = [u for other in others for u in other['units'] if u['unit_id'] in needed]
        units = study['units'] + reference
        prepared.append({'study': study, 'units': units, 'reference': reference, 'reuse_links': links, 'choices': build_value_choices(units)})
    return prepared, accepted['hypotheses'], binding, rid, recipe


def request_body(kind, data, hypotheses, recipe, variable_roster):
    ids = [u['unit_id'] for u in data['study']['units']]
    schema = section_schema(kind, data['units'], data['choices'], [h['id'] for h in hypotheses], ids,
                            [v['id'] for v in variable_roster])
    instruction = (REPO_ROOT / 'config/protocols/section_extraction_v1.txt').read_text(encoding='utf-8')
    instruction += '\nRESPONSE_SCHEMA:\n' + json.dumps(schema, ensure_ascii=False)
    payload = {'task': kind.upper(), 'study_id': data['study']['study_id'], 'label': data['study']['label'],
               'primary_source': [present(u) for u in data['study']['units']],
               'reference_context_only': [present(u) for u in data['reference']],
               'reuse_links': data['reuse_links'], 'numeric_choices': data['choices'],
               'hypotheses': hypotheses, 'variable_roster': variable_roster}
    grammar = str(xgr.Grammar.from_json_schema(canonical_bytes(schema).decode(), max_whitespace_cnt=8))
    xgr.Grammar.from_ebnf(grammar)
    body = {'model': recipe['vllm']['served_model_name'], 'temperature': 0, 'seed': 0,
            'reasoning_effort': 'high', 'max_tokens': 16384,
            'messages': [{'role': 'system', 'content': instruction}, {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}],
            'structured_outputs': {'grammar': grammar}}
    return body, payload, schema


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--live', action='store_true'); parser.add_argument('--base-url')
    args = parser.parse_args()
    studies, hypotheses, binding, rid, recipe = preparation()
    report = {'candidate': CANDIDATE, 'recipe_id': rid, 'planned_calls': len(TASKS), 'reasoning_effort': 'high',
              'source_packet_sha256': binding['source_packet_sha256'], 'semantic_accuracy_unassessed': True,
              'production_accepted': False, 'stages': [], 'not_a_complete_article_run': True}
    count = local_tokenizer(recipe['vllm'])
    if not args.live:
        for i, kind in TASKS:
            body, payload, schema = request_body(kind, studies[i-1], hypotheses, recipe, [])
            estimate = count(json.dumps(body['messages'], ensure_ascii=False)) + 256
            assert estimate + body['max_tokens'] < 98304
            report['stages'].append({'study': i, 'task': kind, 'prompt_estimate': estimate, 'schema_sha256': hashlib.sha256(canonical_bytes(schema)).hexdigest()})
        print(json.dumps(report)); return 0
    assert args.base_url and not (ROOT/'experiment-started.json').exists()
    ws = Workspace(); ws.write_once(ROOT/'experiment-started.json', canonical_bytes(report))
    client = VLLMClient(candidate=CANDIDATE, base_url=args.base_url, api_key=os.environ.get('VLLM_API_KEY'), timeout=240, stream=True, live=True)
    ledger = VLLMLedger(CANDIDATE); before = ledger.totals().requests
    sources = json.loads((REPO_ROOT/'config/screen_sources_v1.json').read_bytes())['works'][WORK]['sources']
    parts = {i: {} for i in (1,2)}
    try:
        for i, kind in TASKS:
            data = studies[i-1]
            roster = parts[i].get('measures', {}).get('variables', []) if kind == 'results' else []
            body, payload, schema = request_body(kind, data, hypotheses, recipe, roster)
            assert count(json.dumps(body['messages'], ensure_ascii=False)) + 256 + body['max_tokens'] < 98304
            name = f'study-{i}-{kind}'
            ws.write_once(ROOT/(name+'-request.json'), canonical_bytes(body))
            start = time.monotonic()
            response = client.chat_completion(work_id=WORK, batch_id='section-v1-'+name, packet=payload, request_body=body,
                source_sha256s=[s['sha256'] for s in sources], text_provenance='native_article_text', remote=False, rights=['runpod_screen_auth'])
            ws.write_once(ROOT/(name+'-response.json'), canonical_bytes(response))
            choice = response['choices'][0]
            stats = {'stage': name, 'seconds': round(time.monotonic()-start,2), 'usage': response.get('usage'), 'finish_reason': choice.get('finish_reason')}
            report['stages'].append(stats)
            if choice.get('finish_reason') != 'stop':
                stats['outcome'] = 'incomplete_response'; continue
            proposal = json.loads(choice['message']['content'])
            ws.write_once(ROOT/(name+'-parsed.json'), canonical_bytes(proposal))
            parts[i][kind] = proposal
            merged = whole_proposal(parts[i], data['study']['study_id'], data['study']['label'])
            try:
                result = materialize_selected(merged, data['units'], data['choices'], [h['id'] for h in hypotheses],
                                               [u['unit_id'] for u in data['study']['units']], data['reuse_links'])
                ws.write_once(ROOT/(name+'-materialized.json'), canonical_bytes(result))
                stats.update(mechanical_valid=result['mechanical_valid'], errors=result['errors'], numeric_selections=len(result['numeric_selections']),
                             manipulation_copies=len(result['manipulation_copies']))
            except (ValueError, KeyError) as exc:
                stats['materialization_refused'] = type(exc).__name__+':'+str(exc)[:300]
        report['outcome'] = 'bounded_section_sequence_finished'
    finally:
        report['inference_attempts'] = ledger.totals().requests-before
        ws.write_once(ROOT/'probe-result.json', canonical_bytes(report)); print(json.dumps(report),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
