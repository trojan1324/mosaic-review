"""Bounded overview / coherent study / completeness engineering experiment.

Consumes the immutable source union of a completed development comparison.
No production promotion, answer-key access, or student queue is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

from mosaic import REPO_ROOT
from mosaic.canonical import canonical_bytes
from mosaic.recipe import resolve_recipe
from mosaic.vllm.client import VLLMClient
from mosaic.vllm.ledger import VLLMBudget, VLLMLedger
from mosaic.vllm.tokenizer import local_tokenizer
from mosaic.workspace import Workspace

sys.path.insert(0, str(REPO_ROOT))
from scripts.structured_extraction_contract import study_schema, audit_record

ROOT = REPO_ROOT / 'scratch/structured-extraction-v1'
CANDIDATE = 'dev_vllm_screen_gptoss120b_v1'
WORK = 'KS-DOC-000023'


def obj(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties),
            'additionalProperties': False}


def array(items):
    return {'type': 'array', 'items': items}


def overview_schema(units, studies):
    ids = array({'type': 'string', 'enum': [u['unit_id'] for u in units]})
    text = {'type': 'string'}
    return obj({
        'hypotheses': array(obj({'id': text, 'label': text, 'source_units': ids})),
        'definitions': array(obj({'label': text, 'source_units': ids})),
        'study_links': array(obj({
            'study_id': {'type': 'string', 'enum': [s['study_id'] for s in studies]},
            'supplement_unit_ids': ids, 'link_evidence_unit_ids': ids, 'reason': text})),
        'boundary_concerns': array(text)})


def present(unit):
    """Number original lines for selection; do not rewrite source bytes."""
    return {'unit_id': unit['unit_id'], 'locator': unit['source_locator'],
            'section_paths': unit.get('section_paths', []),
            'lines': [{'line': i, 'text': line} for i, line in enumerate(unit['text'].splitlines(keepends=True), 1)]}


def prepare():
    packet = json.loads((ROOT / 'ordered-context.json').read_bytes())
    binding = json.loads((ROOT / 'source-binding.json').read_bytes())
    source = json.loads((ROOT / 'source.json').read_bytes())
    assert hashlib.sha256(canonical_bytes(source)).hexdigest() == binding['source_packet_sha256']
    assert source['work_id'] == WORK and binding['candidate'] == CANDIDATE
    units = [u for study in packet['studies'] for u in study['units']] + packet['unassigned_source_units']
    assert len(units) == len(source['units']) == len({u['unit_id'] for u in units})
    assert {u['unit_id']: u['text'] for u in units} == {u['unit_id']: u['text'] for u in source['units']}
    rid, recipe = resolve_recipe(CANDIDATE)
    assert rid == binding['baseline']['recipe_id']
    return packet, binding, units, rid, recipe


def assemble_studies(packet, overview):
    """Keep inventory studies separate; attach only explicitly labelled links.

    This is an auditable routing constraint, not proof of semantic ownership.
    An explicit label is necessary but not sufficient for a scientific verdict.
    """
    unassigned = {u['unit_id']: u for u in packet['unassigned_source_units']}
    hypotheses = overview['hypotheses']
    if len({h['id'] for h in hypotheses}) != len(hypotheses):
        raise ValueError('duplicate_hypothesis_identity')
    context_ids = {uid for h in hypotheses + overview['definitions'] for uid in h['source_units']}
    if not context_ids <= set(unassigned):
        raise ValueError('overview_uses_empirical_study_as_front_end')
    routed, decisions = [], []
    for study in packet['studies']:
        attached = set()
        for link in overview['study_links']:
            if link['study_id'] != study['study_id']:
                continue
            evidence = link['link_evidence_unit_ids']
            chosen = link['supplement_unit_ids']
            allowed = bool(evidence and chosen) and set(evidence + chosen) <= set(unassigned)
            label = ' '.join(study['label'].lower().split())
            explicit = allowed and any(label in ' '.join(unassigned[uid]['text'].lower().split()) for uid in evidence)
            accepted = bool(allowed and explicit)
            decisions.append({**link, 'routing_constraint_pass': accepted,
                              'semantic_ownership_unassessed': True})
            if accepted:
                attached.update(chosen + evidence)
        # Supplemental units retain their own recorded reading sequence.
        units = study['units'] + [u for uid, u in unassigned.items() if uid in attached]
        context = [u for uid, u in unassigned.items() if uid in context_ids and uid not in attached]
        routed.append({**study, 'units': units, 'front_end_context': context})
    return routed, decisions


def request_body(recipe, stage, payload, schema):
    instruction = (REPO_ROOT / 'config/protocols/structured_extraction_v1.txt').read_text(encoding='utf-8')
    return {'model': recipe['vllm']['served_model_name'], 'temperature': 0, 'seed': 0,
            'reasoning_effort': 'low', 'max_tokens': 16384,
            'messages': [{'role': 'system', 'content': instruction},
                         {'role': 'user', 'content': json.dumps({'stage': stage, **payload}, ensure_ascii=False)}],
            'response_format': {'type': 'json_schema', 'json_schema': {'name': 'structured_extraction', 'schema': schema}}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--base-url')
    args = parser.parse_args()
    packet, binding, units, rid, recipe = prepare()
    planned = 1 + 2 * len(packet['studies'])
    overview_payload = {'inventory_studies': [{k: s[k] for k in ('study_id', 'label', 'inventory_status', 'ambiguity')} for s in packet['studies']],
                        'study_section_unit_ids': {s['study_id']: [u['unit_id'] for u in s['units']] for s in packet['studies']},
                        'article_source': [present(u) for u in units]}
    schema = overview_schema(units, packet['studies'])
    body = request_body(recipe, 'overview', overview_payload, schema)
    count = local_tokenizer(recipe['vllm'])
    token_estimate = count(json.dumps(body, ensure_ascii=False))
    assert token_estimate + body['max_tokens'] < 98304
    ledger, budget = VLLMLedger(CANDIDATE), VLLMBudget.load()
    assert not ledger.totals().unresolved
    assert ledger.totals().requests + planned * 2 <= budget.request_cap
    assert ledger.totals_for_article(WORK).requests + planned * 2 <= budget.max_requests_per_article
    report = {'candidate': CANDIDATE, 'recipe_id': rid, 'planned_logical_requests': planned,
              'source_packet_sha256': binding['source_packet_sha256'], 'source_units': len(units),
              'overview_local_request_json_tokens': token_estimate, 'output_allowance_per_call': 16384,
              'semantic_accuracy_unassessed': True, 'release_ready': False, 'stages': []}
    if not args.live:
        print(json.dumps(report)); return 0
    assert args.base_url and not (ROOT / 'experiment-started.json').exists()
    workspace = Workspace()
    workspace.write_once(ROOT / 'experiment-started.json', canonical_bytes(report))
    client = VLLMClient(candidate=CANDIDATE, base_url=args.base_url, api_key=os.environ.get('VLLM_API_KEY'),
                        timeout=240, stream=True, live=True)
    source_hashes = json.loads((REPO_ROOT / 'config/screen_sources_v1.json').read_bytes())['works'][WORK]['sources']

    def call(stage, payload, schema):
        body = request_body(recipe, stage, payload, schema)
        assert count(json.dumps(body, ensure_ascii=False)) + body['max_tokens'] < 98304
        workspace.write_once(ROOT / (stage + '-request.json'), canonical_bytes(body))
        started = time.monotonic()
        response = client.chat_completion(work_id=WORK, batch_id='structured-v1-' + stage,
            packet=payload, request_body=body, source_sha256s=[s['sha256'] for s in source_hashes],
            text_provenance='native_article_text', remote=False, rights=['runpod_screen_auth'])
        workspace.write_once(ROOT / (stage + '-response.json'), canonical_bytes(response))
        choice = response['choices'][0]
        stats = {'stage': stage, 'seconds': round(time.monotonic() - started, 2),
                 'usage': response.get('usage'), 'finish_reason': choice.get('finish_reason')}
        report['stages'].append(stats)
        assert choice.get('finish_reason') == 'stop', 'incomplete_response'
        parsed = json.loads(choice['message']['content'])
        import jsonschema
        jsonschema.validate(parsed, schema)
        workspace.write_once(ROOT / (stage + '-parsed.json'), canonical_bytes(parsed))
        return parsed

    before = ledger.totals().requests
    try:
        overview = call('overview', overview_payload, schema)
        studies, decisions = assemble_studies(packet, overview)
        workspace.write_once(ROOT / 'routing-decisions.json', canonical_bytes(decisions))
        hypothesis_ids = [h['id'] for h in overview['hypotheses']]
        for i, study in enumerate(studies, 1):
            offered = study['units']
            payload = {'study_id': study['study_id'], 'label': study['label'],
                       'boundary_ambiguity': study['ambiguity'],
                       'hypotheses': overview['hypotheses'],
                       'front_end_context_only': [present(u) for u in study['front_end_context']],
                       'ordered_study_source': [present(u) for u in offered]}
            schema = study_schema([u['unit_id'] for u in offered], hypothesis_ids)
            record = call(f'study-{i}-extract', payload, schema)
            for phase in ('extract', 'complete'):
                if phase == 'complete':
                    payload = {**payload, 'previous_record': record, 'mechanical_feedback': audit}
                    record = call(f'study-{i}-complete', payload, schema)
                audit = audit_record(record, offered, hypothesis_ids)
                if record['study_id'] != study['study_id'] or record['label'] != study['label']:
                    audit['errors'].append({'code': 'study_identity_changed', 'path': '/study_id'})
                    audit['mechanical_valid'] = False
                workspace.write_once(ROOT / f'study-{i}-{phase}-audit.json', canonical_bytes(audit))
                report['stages'][-1]['mechanical_valid'] = audit['mechanical_valid']
                report['stages'][-1]['mechanical_error_count'] = len(audit['errors'])
        report['outcome'] = 'bounded_sequence_completed'
    except Exception as exc:
        report['outcome'] = 'stopped'
        report['error_type'] = type(exc).__name__
        raise
    finally:
        report['inference_attempts'] = ledger.totals().requests - before
        workspace.write_once(ROOT / 'probe-result.json', canonical_bytes(report))
        print(json.dumps(report), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
