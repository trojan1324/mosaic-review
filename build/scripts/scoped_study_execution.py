"""One original-source execution of the tested scoped Results sequence."""
from scripts.integrated_extraction_trial import result_chunks
from scripts.scoped_results_trial import prepare_task,materialize
from scripts.scoped_results_comparison import scoped_binding_task
from scripts.statistic_binding_trial import materialize as materialize_bindings
from scripts.result_field_isolation import prepare_numeric_view,prepare_scope_repair,materialize_scope_repair
from scripts.unassigned_statistic_trial import prepare_task as prepare_delta,materialize as materialize_delta,quarantine_unresolved_numbers
from scripts.numeric_context_guard import filter_attachments,filter_delta


def execute_results(study,recipe,call,save,*,complete_inventory=False):
    return execute_payloads(result_chunks(study),recipe,call,save,complete_inventory=complete_inventory)


def execute_payloads(chunks,recipe,call,save,*,complete_inventory=False,bounded_study=False,operator_candidates=None):
    """Shared sequence for independently source-verified bounded Results packets."""
    result={'result_chunk_count':len(chunks),'problems':[],'scientifically_accepted':False,'production_accepted':False}
    if len(chunks)>1:
        result.update(result_chunks=[],attachment_chunks=[],cross_chunk_relations_unassessed=True)
    for index,payload in enumerate(chunks,1):
        suffix='' if len(chunks)==1 else '-part-'+str(index)
        task=prepare_task(recipe,payload,bounded_study=bounded_study)
        if complete_inventory:
            from scripts.comparison_member_evidence import finalize_task, materialize as witnessed_materialize
            task=finalize_task(task)
            upstream=witnessed_materialize(call('results'+suffix,task),task)
        else:upstream=materialize(call('results'+suffix,task),task)
        save('results'+suffix+'-materialized',upstream)
        if complete_inventory:
            from scripts.result_inventory_completion import prepare_task as prepare_inventory, materialize as materialize_inventory
            task=prepare_inventory(recipe,payload,upstream,
                task_factory=lambda r,p:finalize_task(prepare_task(r,p,bounded_study=bounded_study)))
            completion=materialize_inventory(call('inventory'+suffix,task),task)
            save('inventory'+suffix+'-materialized',completion)
            if any(q['reason'] in ('foreign_new_test_source','foreign_unresolved_source')
                   for q in completion['quarantined_proposals']):
                result['problems'].append('inventory_source_proposals_quarantined'+suffix)
            upstream=completion['merged_upstream']
            result.setdefault('inventory_completions',[]).append(completion)
        if len(chunks)==1:result['results']=upstream
        else:result['result_chunks'].append(upstream)
        try:repair_task=prepare_scope_repair(recipe,upstream)
        except ValueError as exc:
            if str(exc)!='no_essential_scope_phrase_to_repair':raise
        else:
            repaired=materialize_scope_repair(call('scope'+suffix,repair_task),repair_task)
            save('scope'+suffix+'-materialized',repaired)
            upstream=repaired['proposed_results']
        isolation=prepare_numeric_view(upstream,isolate_scope=True)
        save('field-isolation'+suffix,isolation)
        if isolation['excluded_tests']:result['problems'].append('unresolved_result_identities'+suffix)
        if isolation['excluded_fields']:result['problems'].append('unresolved_optional_result_fields'+suffix)
        eligible=isolation['numeric_view']
        if not any(r['tests'] for r in eligible['records']):
            result['problems'].append('no_eligible_result_tests'+suffix);continue
        allowed_units={u['unit_id'] for g in eligible['source_group_provenance'] for u in g['original_units']}
        operators=[c for c in (operator_candidates or []) if c['unit_id'] in allowed_units]
        task=scoped_binding_task(recipe,eligible,operator_candidates=operators)
        raw_bound=materialize_bindings(call('attachments'+suffix,task),task)
        kinds={t['test_id']:t['proposal']['kind'] for r in eligible['records'] for t in r['tests']}
        for attachment in raw_bound['attachments']:
            if kinds[attachment['test_id']]=='main_effect' and attachment['proposal']['relationship']!='main_effect':
                attachment['role_resolution'].update(status='conflicting_proposals_abstain',kind_specific_roles_usable=False)
        save('attachments'+suffix+'-raw-materialized',raw_bound)
        bound=filter_attachments(quarantine_unresolved_numbers(raw_bound))
        save('attachments'+suffix+'-materialized',bound)
        if len(chunks)==1:result['attachments']=bound
        else:result['attachment_chunks'].append(bound)
        try:task=prepare_delta(recipe,eligible,bound,continuations=True,operator_candidates=operators)
        except ValueError as exc:
            if str(exc)!='no_unassigned_statistic_choices':raise
            continue
        raw_delta=materialize_delta(call('numeric-delta'+suffix,task),task)
        save('numeric-delta'+suffix+'-raw-materialized',raw_delta)
        delta=filter_delta(raw_delta);delta['field_isolation']=isolation
        save('numeric-delta'+suffix+'-materialized',delta)
        result.setdefault('numeric_deltas',[]).append(delta)
    return result
