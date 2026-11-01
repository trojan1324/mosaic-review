"""Opt-in complete candidate driver: upstream, then actual-answer dependents.

No rentals or cluster operations. The caller supplies admitted sources, scope,
fixed cumulative budget, endpoint and absolute deadline. Model/recipe selection
and scientific qualification remain external. Historical experimental answers
are not combined; any reuse needs an explicitly frozen, verified origin.
"""
from copy import deepcopy
from hashlib import sha256
from importlib import metadata
import json
import math
import platform
import time

from jsonschema import ValidationError

from mosaic import REPO_ROOT
from mosaic.canonical import canonical_bytes
from mosaic.vllm.ledger import VLLMLedger, VLLMHardStop
from mosaic.vllm.cache import load_cached_response
from mosaic.vllm.phased import PhasedRun, digest
from mosaic.vllm.reservation import require
from mosaic.vllm.tokenizer import local_tokenizer
from mosaic.vllm.unavailable import unavailable_outcome
from scripts.complete_extraction_sequence import finalize_paper, prepare_measure_tasks
from scripts.integrated_extraction_trial import compile_section_task, fit_task
from scripts.study_fact_selections import INLINE_VERSION, prepare as prepare_facts, finalize_with_study_facts
from scripts.study_fact_selections import RESPONSE_POLICY, retain_schema_failure
from scripts.study_statistic_annotations import annotate
from scripts.extraction_cache import context as cache_context, verify_context
from scripts.core_evidence_trial import prepare as prepare_core
from scripts.principal_statement_trial import prepare_attributed


VERSION = 'complete-extraction-driver/v1'


def implementation_identity():
    """Hash local implementation/schema files, never sources or protected keys."""
    files = sorted([*REPO_ROOT.glob('mosaic/**/*.py'), *REPO_ROOT.glob('scripts/*.py'),
                    *REPO_ROOT.glob('schemas/*.json')])
    return {'files': {p.relative_to(REPO_ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in files},
        'python': platform.python_version(), 'packages': {name: metadata.version(name)
            for name in ('jsonschema', 'xgrammar', 'tokenizers')}}


def prepare_program(papers, recipe, *, candidate, run_id, max_measure_stages_per_paper,
                    max_workers=1, max_request_seconds=180, max_attempts=1,
                    fixture_mode=False, allow_new_requests=True, context_limit=98304):
    """Freeze a finite selected sequence without provider or tokenizer calls."""
    require(isinstance(papers, list) and papers and isinstance(candidate, str) and candidate
            and isinstance(run_id, str) and run_id and type(fixture_mode) is bool
            and type(allow_new_requests) is bool, 'complete_program_identity_required')
    require(type(max_measure_stages_per_paper) is int and max_measure_stages_per_paper >= 0
            and type(max_workers) is int and 1 <= max_workers <= 32
            and type(max_attempts) is int and max_attempts in (1, 2)
            and type(max_request_seconds) in (int, float) and math.isfinite(max_request_seconds)
            and max_request_seconds > 0, 'complete_program_bound_invalid')
    require(type(context_limit) is int and context_limit > 6144, 'complete_program_context_limit_invalid')
    work_ids = [p['work_id'] for p in papers]
    paper_ids = [p['plan']['paper']['paper_id'] for p in papers]
    require(len(set(work_ids)) == len(papers) and len(set(paper_ids)) == len(papers), 'complete_program_duplicate_paper')
    for paper in papers:
        plan = paper['plan']
        finalize_paper(plan, findings_proposal=None, core_proposal=None, measure_responses={})
        require(plan['core_task']['body']['model'] == recipe['vllm']['served_model_name']
                and paper['source_sha256s'] == [plan['paper']['source_pdf_sha256']], 'complete_program_source_or_model_changed')
        require(paper['text_provenance'] == ('fixture_replay' if fixture_mode else 'native_article_text'),
                'complete_program_provenance_changed')
    require(fixture_mode or recipe['vllm'].get('local_tokenizer') is not None, 'complete_program_pinned_tokenizer_required')
    return {'version': VERSION, 'papers': deepcopy(papers), 'recipe': deepcopy(recipe), 'candidate': candidate,
        'run_id': run_id, 'max_measure_stages_per_paper': max_measure_stages_per_paper,
        'max_workers': max_workers, 'max_request_seconds': max_request_seconds, 'max_attempts': max_attempts,
        'maximum_output_tokens_per_request': 6144, 'context_limit': context_limit,
        'compact_upstream_prompts': True,
        'study_fact_version': INLINE_VERSION, 'study_fact_response_policy': RESPONSE_POLICY, 'fixture_mode': fixture_mode,
        'allow_new_requests': allow_new_requests,
        'implementation': implementation_identity(), 'scientifically_accepted': False, 'production_accepted': False}


def _request(program, paper, request_id, task, count, *, binding, parent_context=None):
    if 'request' in task:
        body = deepcopy(task['request'])
        body.pop('response_schema', None)
        body['model'] = program['recipe']['vllm']['served_model_name']
        compiled = compile_section_task(body, task['payload'], task['schema'])
    else:
        compiled = deepcopy(task)
    kind = parent_context['kind'] if parent_context else request_id.rsplit('-', 1)[-1]
    compiled, tokens, _ = fit_task(compiled, count,
        compact_prompts=program['compact_upstream_prompts'] and kind in ('core', 'findings'),
        context_limit=program['context_limit'])
    require(compiled['body']['max_tokens'] <= program['maximum_output_tokens_per_request'], 'complete_output_bound_exceeded')
    result = {'id': request_id, 'work_id': paper['work_id'], 'batch_id': program['run_id'] + '-' + request_id,
        'task': compiled, 'binding_sha256': digest({'recipe_sha256': digest(program['recipe']), 'parent': binding,
            'plan_sha256': digest(paper['plan']), 'task_sha256': digest(task)}),
        'prompt_tokens': tokens, 'context_limit': program['context_limit'],
        'source_sha256s': deepcopy(paper['source_sha256s']), 'text_provenance': paper['text_provenance'],
        'remote': paper['remote'], 'rights': deepcopy(paper['rights']),
        'max_request_seconds': program['max_request_seconds'], 'max_attempts': program['max_attempts'],
        'stream': not program['fixture_mode']}
    result['cache_context'] = parent_context or cache_context(paper['plan'], kind)
    if kind == 'study_facts':
        require(program.get('study_fact_response_policy') == RESPONSE_POLICY, 'complete_response_policy_changed')
        result['response_policy'] = RESPONSE_POLICY
    entry = program.get('cache_entries', {}).get(result['binding_sha256'])
    expected = [e for e in program.get('cache_entries', {}).values() if e['request_id'] == request_id]
    require(not expected or len(expected) == 1 and entry == expected[0], 'cache_expected_binding_changed')
    if entry is not None:
        result['cache_entry_sha256'] = digest(entry)
    omitted = program.get('unavailable_entries', {}).get(result['binding_sha256'])
    expected = [e for e in program.get('unavailable_entries', {}).values() if e['request_id'] == request_id]
    require(not expected or len(expected) == 1 and omitted == expected[0], 'unavailable_expected_binding_changed')
    require(entry is None or omitted is None, 'unavailable_cache_overlap')
    if omitted is not None:
        result['unavailable'] = deepcopy(omitted)
    require(program.get('allow_new_requests', True) or entry is not None or omitted is not None,
            'complete_unlisted_live_request')
    return result


def _upstream_task(program, paper, stage):
    # Canonical saved JSON sorts object keys, while the prompt and decoder
    # grammar preserve builder order. Rebuild that order without changing any
    # contract value before applying the previously tested prompt compaction.
    builder = prepare_core if stage == 'core' else prepare_attributed
    task = builder(program['recipe'], paper['plan']['source']['units'])
    require(digest(task) == digest(paper['plan'][stage + '_task']), 'complete_upstream_contract_changed')
    return task


def _cached(program, requests, workspace, scope):
    return {r['id']: load_cached_response(program['cache_entries'][r['binding_sha256']], r,
        candidate=program['candidate'], recipe_id='recipe-' + digest(program['recipe']),
        workspace=workspace, scope=scope, verify_context=verify_context,
        retain_schema_failure=retain_schema_failure if r['cache_context']['kind'] == 'study_facts' else None)
        for r in requests if 'cache_entry_sha256' in r}


def _unavailable(requests, workspace, scope):
    return {r['id']: unavailable_outcome(r, workspace=workspace, scope=scope)
            for r in requests if 'unavailable' in r}


def _reserve_whole_program(program, ledger, budget, upstream, upstream_cache, known_children, child_cache,
                           unavailable):
    """Credit verified cache and validated no-send stages, never unknown children."""
    counts = {p['work_id']: 0 for p in program['papers']}
    outputs = seconds = 0
    for request in upstream + known_children:
        if request['id'] in upstream_cache or request['id'] in child_cache or request['id'] in unavailable:
            continue
        counts[request['work_id']] += request['max_attempts']
        outputs += request['max_attempts'] * request['task']['body']['max_tokens']
        seconds += request['max_attempts'] * request['max_request_seconds']
    for request in upstream:
        if (request['id'].endswith('-core') and request['id'] not in upstream_cache
                and request['id'] not in unavailable):
            maximum = (1 + program['max_measure_stages_per_paper']) * program['max_attempts']
            counts[request['work_id']] += maximum
            outputs += maximum * program['maximum_output_tokens_per_request']
            seconds += maximum * program['max_request_seconds']
    total = sum(counts.values())
    prior = ledger.totals()
    require(not prior.unresolved and prior.requests + total <= budget.request_cap
            and prior.output_tokens + outputs <= budget.output_token_cap_total
            and prior.wall_clock_seconds + seconds <= budget.wall_clock_cap_seconds,
            'complete_program_cumulative_budget')
    for paper in program['papers']:
        require(ledger.totals_for_article(paper['work_id']).requests + counts[paper['work_id']]
                <= budget.article_limit(program['candidate'], paper['work_id']), 'complete_program_article_budget')
    return {'maximum_attempts': total, 'maximum_attempts_per_paper': counts,
        'maximum_output_tokens': outputs, 'maximum_request_seconds': seconds, 'prior_totals': vars(prior),
        'verified_cache_hits': len(upstream_cache) + len(child_cache),
        'declared_unavailable_stages': len(unavailable), 'historical_counters_unchanged': True}


def _dependents(program, upstream_outputs, count):
    dependent, preparation, plans, fact_tasks = [], [], {}, {}
    by_upstream = {r['id']: r for r in upstream_outputs}
    for slot, paper in enumerate(program['papers'], 1):
        prefix = f'paper-{slot:03d}'
        core = by_upstream.get(prefix + '-core', {}).get('answer')
        if core is None:
            preparation.append({'id': prefix + '-dependents', 'status': 'not_started', 'reason': 'core_unavailable'})
            continue
        try:
            sequence = prepare_measure_tasks(paper['plan'], core)
            plans[prefix] = sequence
            measure_stages = [s for s in sequence['measure_stages'] if s['status'] == 'awaiting_response']
            if len(measure_stages) > program['max_measure_stages_per_paper']:
                preparation.append({'id': prefix + '-measures', 'status': 'not_started', 'reason': 'measure_stage_bound_exceeded',
                                    'required': len(measure_stages), 'maximum': program['max_measure_stages_per_paper']})
                measure_stages = []
            for index, stage in enumerate(measure_stages):
                request_id = prefix + f'-measure-{index:03d}'
                try:
                    request = _request(program, paper, request_id, stage['task'], count,
                        binding={'core_proposal_sha256': digest(core), 'measure_stage_id': stage['id']},
                        parent_context=cache_context(paper['plan'], 'measure', core=core,
                            inventory=sequence['core']['inventory'], stage=stage))
                    dependent.append(request)
                    stage['runner_request_id'] = request_id
                except ValueError as exc:
                    preparation.append({'id': request_id, 'status': 'not_prepared', 'error_type': type(exc).__name__})
            fact_task = prepare_facts(paper['plan'], core, version=program['study_fact_version'])
            fact_tasks[prefix] = fact_task
            if fact_task['payload']['targets']:
                dependent.append(_request(program, paper, prefix + '-study-facts', fact_task, count,
                    binding={'core_proposal_sha256': digest(core)},
                    parent_context=cache_context(paper['plan'], 'study_facts', core=core)))
            else:
                preparation.append({'id': prefix + '-study-facts', 'status': 'not_started', 'reason': 'no_core_targets'})
        except (ValueError, KeyError, ValidationError) as exc:
            preparation.append({'id': prefix + '-dependents', 'status': 'not_prepared', 'error_type': type(exc).__name__})
    return dependent, preparation, plans, fact_tasks


def _finalize(plan, core, findings, measures, fact_task, fact_outcome):
    fact_answer = fact_outcome.get('answer')
    if (fact_outcome.get('status') == 'invalid_response'
            and fact_outcome.get('schema_validation', {}).get('valid') is False):
        # Only this additional-fact stage has per-fact quarantine. The
        # materializer rejects wrong targets/containers and capacity violations
        # before considering individual siblings; the proposal is never repaired.
        fact_answer = fact_outcome['parsed_answer']
    report = finalize_with_study_facts(plan, findings_proposal=findings, core_proposal=core,
        measure_responses=measures, study_fact_task=fact_task, study_fact_proposal=fact_answer)
    if 'schema_validation' in fact_outcome:
        if fact_answer is None and report['study_fact_stage']['status'] == 'unavailable':
            report['study_fact_stage'] = {'status': 'invalid_proposal',
                'reason': 'study_fact_response_refused', 'error_type': 'ValueError'}
        report['study_fact_stage']['response_validation'] = deepcopy(fact_outcome['schema_validation'])
    if report['study_fact_stage']['status'] == 'materialized_source_bound_proposal':
        report['statistic_annotation'] = annotate(fact_task, fact_answer, plan, core,
                                                  report['study_fact_stage']['materialized'])
    return report


def run_program(program, *, expected_program_sha256, workspace, receipt_root, ledger_root,
                expected_ledger_sha256, budget, scope, deadline_epoch, base_url=None, api_key=None,
                live=False, resume=False, fixture_transport=None, fixture_count=None):
    """Run/replay both phases and retain a report for every frozen paper.

    Test injection is allowed only for a wholly synthetic program. Native runs
    always use the recipe's verified local tokenizer and DeadlineTransport.
    Only explicitly bound historical entries can be cached. No answer search,
    decoder repair loop or target union is performed during execution.
    """
    program = deepcopy(program)
    require(program.get('version') == VERSION and digest(program) == expected_program_sha256, 'complete_program_changed')
    require(digest(program['implementation']) == digest(implementation_identity()), 'complete_implementation_changed')
    require(program.get('study_fact_response_policy') == RESPONSE_POLICY, 'complete_response_policy_changed')
    require(type(program.get('allow_new_requests', True)) is bool, 'complete_request_policy_invalid')
    entries = program.get('cache_entries', {})
    require(isinstance(entries, dict) and all(key == entry['binding_sha256'] for key, entry in entries.items())
            and len({entry['request_id'] for entry in entries.values()}) == len(entries), 'cache_roster_invalid')
    omissions = program.get('unavailable_entries', {})
    require(isinstance(omissions, dict) and all(isinstance(e, dict) and key == e.get('binding_sha256')
            and isinstance(e.get('request_id'), str) for key, e in omissions.items())
            and len({e['request_id'] for e in omissions.values()}) == len(omissions), 'unavailable_roster_invalid')
    require(not set(entries) & set(omissions)
            and not {e['request_id'] for e in entries.values()} & {e['request_id'] for e in omissions.values()},
            'unavailable_cache_overlap')
    require(program['fixture_mode'] or fixture_transport is None and fixture_count is None,
            'complete_native_test_injection_refused')
    require(not program['fixture_mode'] or fixture_transport is not None, 'complete_fixture_transport_required')
    count = fixture_count if program['fixture_mode'] else local_tokenizer(program['recipe']['vllm'])
    require(callable(count), 'complete_tokenizer_required')
    root = workspace.check(receipt_root)
    for paper in program['papers']:
        scope.check(paper['work_id'], paper['source_sha256s'], paper['text_provenance'],
                    remote=paper['remote'], rights=paper['rights'])
    started = time.monotonic()
    upstream = []
    preparation = []
    for slot, paper in enumerate(program['papers'], 1):
        for stage in ('findings', 'core'):
            request_id = f'paper-{slot:03d}-{stage}'
            try:
                upstream.append(_request(program, paper, request_id, _upstream_task(program, paper, stage), count,
                                         binding={'stage': 'upstream'}))
            except ValueError as exc:
                preparation.append({'id': request_id, 'status': 'not_prepared', 'error_type': type(exc).__name__})
    prep_seconds = time.monotonic() - started
    upstream_cache = _cached(program, upstream, workspace, scope)
    upstream_unavailable = _unavailable(upstream, workspace, scope)
    known_children, _, _, _ = _dependents(program, [v.outcome for v in upstream_cache.values()], count)
    known_child_cache = _cached(program, known_children, workspace, scope)
    known_unavailable = {**upstream_unavailable, **_unavailable(known_children, workspace, scope)}
    dependent = []
    dependent_plans = {}
    fact_tasks = {}
    upstream_outputs = []
    dependent_outputs = []
    failure = None
    prior_report = None
    run = None
    with PhasedRun(candidate=program['candidate'], ledger_root=ledger_root, receipt_root=root / 'phases',
            workspace=workspace, budget=budget, phase_ids=['upstream', 'dependent'],
            expected_ledger_sha256=expected_ledger_sha256, plan_sha256=expected_program_sha256,
            deadline_epoch=deadline_epoch, max_workers=program['max_workers'], resume=resume) as run:
        if not resume:
            ledger = VLLMLedger(program['candidate'], root=workspace.check(ledger_root), workspace=workspace)
            reservation = _reserve_whole_program(program, ledger, budget, upstream, upstream_cache,
                                                 known_children, known_child_cache, known_unavailable)
            workspace.write_once(root / 'program.json', canonical_bytes(program))
            workspace.write_once(root / 'program-reservation.json', canonical_bytes(reservation))
        else:
            require(sha256((root / 'program.json').read_bytes()).hexdigest() == expected_program_sha256,
                    'complete_saved_program_changed')
        try:
            upstream_outputs = run.run_phase('upstream', upstream, scope=scope, base_url=base_url,
                api_key=api_key, live=live, transport=fixture_transport, cached_responses=upstream_cache)
        except Exception as exc:
            failure = exc.code if isinstance(exc, VLLMHardStop) else type(exc).__name__
            path = root / 'phases/upstream/outcomes.json'
            if path.exists(): upstream_outputs = json.loads(path.read_bytes())
        by_upstream = {r['id']: r for r in upstream_outputs}
        dependent, child_preparation, dependent_plans, fact_tasks = _dependents(program, upstream_outputs, count)
        preparation.extend(child_preparation)
        # A stop forbids new sends, not replay of an already-closed dependent
        # phase. Its outcomes must survive exact restart with that same stop.
        if not failure and (not run.stop_reason or len(run.closed) == 2):
            try:
                dependent_outputs = run.run_phase('dependent', dependent, scope=scope, base_url=base_url,
                    api_key=api_key, live=live, transport=fixture_transport,
                    cached_responses=_cached(program, dependent, workspace, scope))
            except Exception as exc:
                failure = exc.code if isinstance(exc, VLLMHardStop) else type(exc).__name__
                path = root / 'phases/dependent/outcomes.json'
                if path.exists(): dependent_outputs = json.loads(path.read_bytes())
        by_dependent = {r['id']: r for r in dependent_outputs}
        papers = []
        for slot, paper in enumerate(program['papers'], 1):
            prefix = f'paper-{slot:03d}'
            core = by_upstream.get(prefix + '-core', {}).get('answer')
            finding = by_upstream.get(prefix + '-findings', {}).get('answer')
            measures = {s['id']: by_dependent[s['runner_request_id']]['answer']
                for s in dependent_plans.get(prefix, {}).get('measure_stages', [])
                if s.get('runner_request_id') in by_dependent and by_dependent[s['runner_request_id']]['answer'] is not None}
            report = _finalize(paper['plan'], core, finding, measures, fact_tasks.get(prefix),
                               by_dependent.get(prefix + '-study-facts', {}))
            paper_omissions = [o for o in upstream_outputs + dependent_outputs
                              if o['id'].startswith(prefix + '-') and o.get('origin') == 'declared_unavailable']
            if paper_omissions:
                report['declared_unavailable_stages'] = [{'request_id': o['id'], 'reason': o['reason'],
                    'declaration_sha256': o['unavailable_receipt']['declaration_sha256']} for o in paper_omissions]
            workspace.write_once(root / (prefix + '-report.json'), canonical_bytes(report))
            papers.append({'paper_id': paper['plan']['paper']['paper_id'], 'work_id': paper['work_id'],
                'report_path': prefix + '-report.json', 'report_sha256': digest(report)})
        result = {'version': VERSION, 'program_sha256': expected_program_sha256, 'papers': papers,
            'preparation_outcomes': preparation, 'upstream_outcomes': upstream_outputs,
            'dependent_outcomes': dependent_outputs, 'prepared_dependent_requests': len(dependent),
            'unsent_dependent_ids': [r['id'] for r in dependent if r['id'] not in by_dependent],
            'unused_cache_request_ids': sorted(e['request_id'] for e in entries.values()
                if e['request_id'] not in {o['id'] for o in upstream_outputs + dependent_outputs
                                         if o.get('origin') == 'historical_cache'}),
            'declared_unavailable_request_ids': sorted(o['id'] for o in upstream_outputs + dependent_outputs
                                                       if o.get('origin') == 'declared_unavailable'),
            'unused_unavailable_request_ids': sorted(e['request_id'] for e in omissions.values()
                if e['request_id'] not in {o['id'] for o in upstream_outputs + dependent_outputs
                                         if o.get('origin') == 'declared_unavailable'}),
            'closed_phases': [digest(c) for c in run.closed], 'stop_reason': failure or run.stop_reason,
            'scientifically_accepted': False, 'production_accepted': False, 'complete_extraction_claimed': False,
            'timing': {'preparation_seconds': prep_seconds, 'this_invocation_seconds': time.monotonic() - started,
                'includes_resumed_prior_provider_time': False, 'complete_paper_throughput_established': False},
            'limits': ['All source-bound records remain fallible proposals; software checks are not qualification.',
                'Closed phases may replay; interrupted sends require audit and never reset counters.',
                'Historical responses retain their declared origin, parent differences and original usage; no answer union is made.',
                'Preparation, unknown sends, unavailable stages and absent model targets remain unresolved.']}
        report_path = root / 'report.json'
        if report_path.exists():
            prior_report = json.loads(report_path.read_bytes())
            require(digest({k: v for k, v in prior_report.items() if k != 'timing'})
                    == digest({k: v for k, v in result.items() if k != 'timing'}), 'complete_report_changed')
            return prior_report
        workspace.write_once(report_path, canonical_bytes(result))
        return result
