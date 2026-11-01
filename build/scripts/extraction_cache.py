"""Frozen source/parent proofs for reusing one declared extraction response."""
from copy import deepcopy
import re

from mosaic.vllm.cache import digest
from mosaic.vllm.reservation import require
from scripts.measure_catalog import build_catalog


def context(plan, kind, *, core=None, inventory=None, stage=None):
    out = {'version': 'extraction-cache-context/v1', 'kind': kind, 'paper': deepcopy(plan['paper']),
        'plan_sha256': digest(plan), 'source_units_sha256': digest(plan['source']['units']),
        'source_policy_sha256': plan['source']['source_policy_sha256']}
    if kind in ('measure', 'study_facts'):
        require(core is not None, 'cache_parent_core_required')
        out['core_proposal_sha256'] = digest(core)
    if kind == 'measure':
        require(inventory is not None and stage is not None, 'cache_target_parent_required')
        out.update(inventory=deepcopy(inventory), base_target=deepcopy(stage['version']),
                   expanded_target=deepcopy(stage['expanded_version']))
        _measure_parent(out)
    return out


def _measure_parent(value):
    inventory, base = value['inventory'], value['base_target']
    require(digest(inventory['paper']) == digest(value['paper'])
            and inventory['source_identity']['eligible_source_units_sha256'] == value['source_units_sha256']
            and inventory['source_identity']['source_policy_sha256'] == value['source_policy_sha256'],
            'cache_inventory_source_changed')
    versions = build_catalog([inventory])['versions']
    require(sum(digest(v) == digest(base) for v in versions) == 1, 'cache_target_not_in_actual_parent')
    study = next(s for s in inventory['studies'] if s['id'] == base['study_id'])
    expanded = value['expanded_target']
    require(expanded['id'] == base['id'] and expanded['field_context_receipt']['base_version_sha256'] == digest(base),
            'cache_context_base_changed')
    return {k: v for k, v in study.items() if k != 'measures'}


def verify_context(old, new, old_task, new_task):
    require(old.get('version') == new.get('version') == 'extraction-cache-context/v1', 'cache_context_version_invalid')
    require(old['kind'] == new['kind'] and old['kind'] in ('core', 'findings', 'measure', 'study_facts'),
            'cache_stage_changed')
    require(all(digest(old[k]) == digest(new[k]) for k in ('paper', 'source_units_sha256', 'source_policy_sha256')),
            'cache_parent_source_changed')
    packet_same = digest(old_task['payload']) == digest(new_task['payload'])
    receipt = {'kind': old['kind'], 'old_context_sha256': digest(old), 'current_context_sha256': digest(new),
        'packet_identical': packet_same, 'id_remapped': False, 'scientifically_accepted': False}
    if old['kind'] in ('core', 'findings'):
        require(packet_same and digest(old_task) == digest(new_task), 'cache_upstream_contract_changed')
    elif old['kind'] == 'study_facts':
        require(packet_same and old['core_proposal_sha256'] == new['core_proposal_sha256'],
                'cache_additional_parent_changed')
        if digest(old_task) != digest(new_task):
            # A local plan pointer is not a model input. The full task except
            # that pointer must still match, including source values, target
            # roster, schema and complete request body. Never remap core IDs.
            for task, parent in ((old_task, old), (new_task, new)):
                require(isinstance(task.get('base_plan_sha256'), str)
                    and re.fullmatch(r'[0-9a-f]{64}', task['base_plan_sha256']) is not None
                    and task['base_plan_sha256'] == parent.get('plan_sha256')
                    and task.get('base_core_sha256') == parent.get('core_proposal_sha256'),
                    'cache_additional_plan_reference_changed')
            stripped = [{k: v for k, v in task.items() if k != 'base_plan_sha256'}
                        for task in (old_task, new_task)]
            require(digest(stripped[0]) == digest(stripped[1]), 'cache_additional_parent_changed')
            receipt.update(plan_reference_remapped=True, old_plan_sha256=old['plan_sha256'],
                current_plan_sha256=new['plan_sha256'],
                shared_task_without_plan_reference_sha256=digest(stripped[0]),
                full_parent_equivalence_claimed=False, original_answer_unchanged=True)
    else:
        old_study, new_study = _measure_parent(old), _measure_parent(new)
        for task, value in ((old_task, old), (new_task, new)):
            # Older saved tasks carry these actual parent records in addition
            # to the wire contract. They cannot be exchanged for a new proof.
            for field, proof in (('base_version', 'base_target'), ('source_version', 'expanded_target')):
                require(field not in task or digest(task[field]) == digest(value[proof]),
                        'cache_historical_target_changed')
        require(digest(old_study['identifier_evidence']) == digest(new_study['identifier_evidence'])
                and old_study['id'] == new_study['id'], 'cache_study_identity_changed')
        old_base, new_base = old['base_target'], new['base_target']
        require(digest({k: v for k, v in old_base.items() if k != 'id'})
                == digest({k: v for k, v in new_base.items() if k != 'id'}), 'cache_target_changed')
        left, right = deepcopy(old['expanded_target']), deepcopy(new['expanded_target'])
        for value in (left, right):
            value.pop('id')
            value['field_context_receipt'].pop('base_version_sha256')
        require(digest(left) == digest(right), 'cache_expanded_target_changed')
        require(old_task['payload']['version_id'] == old_base['id']
                and new_task['payload']['version_id'] == new_base['id'], 'cache_packet_target_changed')
        require(digest({k: v for k, v in old_task['payload'].items() if k != 'version_id'})
                == digest({k: v for k, v in new_task['payload'].items() if k != 'version_id'}), 'cache_packet_changed')
        # Parent design/population/sample proposals are not inputs to this
        # identical measure request. Retain their changes explicitly; the
        # reused measure answer never confirms those different assertions.
        receipt.update(id_remapped=old_base['id'] != new_base['id'], old_target_id=old_base['id'],
            current_target_id=new_base['id'], old_inventory_sha256=digest(old['inventory']),
            current_inventory_sha256=digest(new['inventory']), old_core_sha256=old['core_proposal_sha256'],
            current_core_sha256=new['core_proposal_sha256'],
            parent_fields_changed=sorted(k for k in set(old_study) | set(new_study)
                if digest(old_study.get(k)) != digest(new_study.get(k))),
            full_parent_equivalence_claimed=False, study_identity_evidence_identical=True,
            original_answer_unchanged=True)
    return receipt
