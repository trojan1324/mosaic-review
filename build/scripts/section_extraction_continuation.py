"""Versioned failure-directed continuation with complete task dependencies."""
import argparse
import hashlib
import json
import os
import time
from mosaic import REPO_ROOT
from mosaic.canonical import canonical_bytes
from mosaic.vllm.client import VLLMClient
from mosaic.vllm.ledger import VLLMBudget, VLLMLedger
from mosaic.vllm.tokenizer import local_tokenizer
from mosaic.workspace import Workspace
from scripts.structured_extraction_probe import prepare, CANDIDATE, WORK
from scripts.section_extraction_probe import request_body
from scripts.section_source_context import prepare_contexts
from scripts.section_extraction_contract import whole_proposal
from scripts.selected_study_contract import materialize_selected

ROOT = REPO_ROOT / 'scratch/section-focused-v2'
TASKS = [(1, 'measures'), (1, 'design'), (1, 'results'), (2, 'measures'), (2, 'results')]


def preparation():
    packet, binding, _, rid, recipe = prepare()
    overview = json.loads((REPO_ROOT / 'scratch/structured-extraction-v1/overview-parsed.json').read_bytes())
    studies, accepted, routing = prepare_contexts(packet, overview)
    budget, ledger = VLLMBudget.load(), VLLMLedger(CANDIDATE)
    assert not ledger.totals().unresolved
    assert ledger.totals().requests + len(TASKS)*2 <= budget.request_cap
    assert ledger.totals_for_article(WORK).requests + len(TASKS)*2 <= budget.article_limit(CANDIDATE, WORK)
    return studies, accepted['hypotheses'], binding, rid, recipe


def body_for(kind, data, hypotheses, recipe, roster):
    body, payload, schema = request_body(kind, data, hypotheses, recipe, roster)
    body['reasoning_effort'] = 'medium'
    body['messages'][0]['content'] += (
        '\nDIRECTION CHECK: A statement that X predicts Y, that an indirect effect exists, or that an effect remains significant '
        'does not establish its sign. Use direction=unclear when the assigned finding has no signed coefficient or explicit '
        'increase/decrease/positive/negative wording. Never borrow a direction from another model or theoretical expectation. '
        'For RESULTS populate analysis-specific roles using the supplied variable IDs; when models have different roles, '
        'represent them separately. Statistical significance and direction are separate fields.'
    )
    return body, payload, schema


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--live', action='store_true'); parser.add_argument('--base-url')
    args = parser.parse_args()
    studies, hypotheses, binding, rid, recipe = preparation()
    report = {'candidate': CANDIDATE, 'recipe_id': rid, 'planned_calls': len(TASKS), 'reasoning_effort': 'medium',
              'source_packet_sha256': binding['source_packet_sha256'], 'semantic_accuracy_unassessed': True,
              'production_accepted': False, 'stages': [], 'reused_tasks': [{'study': 2, 'task': 'design',
                'path': 'scratch/section-focused-v1/study-2-design-parsed.json', 'reasoning_effort': 'high'}]}
    count = local_tokenizer(recipe['vllm'])
    if not args.live:
        for i, kind in TASKS:
            body, payload, schema = body_for(kind, studies[i-1], hypotheses, recipe, [])
            estimate = count(json.dumps(body['messages'], ensure_ascii=False)) + 256
            assert estimate + body['max_tokens'] < 98304
            report['stages'].append({'study': i, 'task': kind, 'prompt_estimate': estimate,
                                    'schema_sha256': hashlib.sha256(canonical_bytes(schema)).hexdigest()})
        print(json.dumps(report)); return 0
    assert args.base_url and not (ROOT/'experiment-started.json').exists()
    ws = Workspace(); ws.write_once(ROOT/'experiment-started.json', canonical_bytes(report))
    client = VLLMClient(candidate=CANDIDATE, base_url=args.base_url, api_key=os.environ.get('VLLM_API_KEY'), timeout=240, stream=True, live=True)
    ledger = VLLMLedger(CANDIDATE); before = ledger.totals().requests
    sources = json.loads((REPO_ROOT/'config/screen_sources_v1.json').read_bytes())['works'][WORK]['sources']
    parts = {1: {}, 2: {'design': json.loads((REPO_ROOT/report['reused_tasks'][0]['path']).read_bytes())}}
    try:
        for i, kind in TASKS:
            data = studies[i-1]
            if kind == 'results' and 'measures' not in parts[i]:
                report['stages'].append({'stage': f'study-{i}-results', 'outcome': 'blocked_by_incomplete_variable_inventory'})
                continue
            roster = parts[i].get('measures', {}).get('variables', []) if kind == 'results' else []
            body, payload, schema = body_for(kind, data, hypotheses, recipe, roster)
            assert count(json.dumps(body['messages'], ensure_ascii=False)) + 256 + body['max_tokens'] < 98304
            name = f'study-{i}-{kind}'
            ws.write_once(ROOT/(name+'-request.json'), canonical_bytes(body))
            start = time.monotonic()
            response = client.chat_completion(work_id=WORK, batch_id='section-v2-'+name, packet=payload, request_body=body,
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
            result = materialize_selected(merged, data['units'], data['choices'], [h['id'] for h in hypotheses],
                                           [u['unit_id'] for u in data['study']['units']], data['reuse_links'])
            ws.write_once(ROOT/(name+'-materialized.json'), canonical_bytes(result))
            stats.update(mechanical_valid=result['mechanical_valid'], errors=result['errors'], numeric_selections=len(result['numeric_selections']),
                         manipulation_copies=len(result['manipulation_copies']))
        report['complete_studies'] = [i for i in parts if set(parts[i]) == {'measures','design','results'}]
        report['outcome'] = 'complete_tasks' if len(report['complete_studies']) == 2 else 'incomplete_tasks'
    finally:
        report['inference_attempts'] = ledger.totals().requests-before
        ws.write_once(ROOT/'probe-result.json', canonical_bytes(report)); print(json.dumps(report),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
