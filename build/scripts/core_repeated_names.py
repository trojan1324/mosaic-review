"""Preserve repeated identical source spellings without assigning identity.

This optional derivative repairs a name locator only. All repeated positions
remain witnesses to one model proposal, not evidence of a single instrument.
Historical name recovery and source text are unchanged.
"""
from copy import deepcopy
from hashlib import sha256
import re

from mosaic.canonical import canonical_bytes
from scripts.core_evidence_recovery import _normalised_pattern, source_name_mapping
from scripts.core_evidence_trial import validate

VERSION = 'repeated-source-name-witnesses/v1'
MATCH_TYPE = 'repeated_identical_case_or_layout_spelling'


def digest(value):
    return sha256(canonical_bytes(value)).hexdigest()


def _require(condition, code):
    if not condition:
        raise ValueError(code)


def repeated_name_mapping(name, text):
    """Recover only a refused name with several byte-identical raw spellings."""
    if not isinstance(name, str) or not name or source_name_mapping(name, text) is not None:
        return None
    pattern = _normalised_pattern(name)
    matches = list(re.finditer(pattern, text, re.IGNORECASE)) if pattern else []
    if len(matches) < 2 or len({m.group() for m in matches}) != 1:
        return None
    spans = [{'start': m.start(), 'end': m.end(), 'text': m.group()} for m in matches]
    return {'proposal_name': name, 'source_name': spans[0]['text'], 'match_type': MATCH_TYPE,
            'source_span': deepcopy(spans[0]), 'all_source_spans': spans,
            'representative_span_rule': 'first_occurrence_for_display_only',
            'source_text_sha256': sha256(text.encode()).hexdigest(),
            'source_text_unchanged': True, 'semantic_identity_verified': False}


def recover_repeated_names(proposal, task, baseline):
    """Extend an already validated v3/v4/v5 recovery, retaining all old fields.

    The inventory builder independently replays both the baseline and this
    derivative. Only one cited statement may supply the repeated spelling;
    alternative statements are not silently selected or combined.
    """
    _require(baseline.get('version') in {'core-name-witnesses-v3', 'core-name-witnesses-v4',
                                        'core-name-witnesses-v5'}, 'repeated_name_base_version_invalid')
    _require('repeated_name_policy' not in baseline, 'repeated_name_already_applied')
    _require(baseline.get('original_task_sha256') == digest(task)
             and baseline.get('original_proposal_sha256') == digest(proposal), 'repeated_name_base_input_changed')
    _require(all(baseline.get(flag) is False for flag in (
        'scientifically_accepted', 'production_accepted', 'semantic_labels_verified', 'coverage_verified')),
        'repeated_name_unverified_base_required')
    table = validate(task)
    result = deepcopy(baseline)
    repairs, fixed = [], set()
    for refusal in baseline['quarantined_fields']:
        if refusal['reason'] != 'name_not_exact_or_uniquely_normalisable_in_cited_sentence':
            continue
        keys = refusal['path'].split('/')[1:]
        kind, index, children, child_index, name_key = keys
        index, child_index = int(index), int(child_index)
        field = proposal[kind][index][children][child_index]
        evidence_key = 'description_sentence_ids' if name_key == 'name' else 'role_sentence_ids'
        alternatives = []
        for sid in dict.fromkeys([field['name_sentence_id']] + field[evidence_key]):
            mapping = repeated_name_mapping(field[name_key], table[sid]['text'])
            if mapping is not None:
                alternatives.append({'sentence_id': sid, 'mapping': mapping})
        if len(alternatives) != 1:
            continue
        selected = alternatives[0]
        sid, mapping = selected['sentence_id'], selected['mapping']
        if re.fullmatch(r'control\s+variables?', mapping['source_name'], re.I):
            continue
        identifier = (f'study-{index}-measure-{child_index}' if name_key == 'name'
                      else f'analysis-{index}-role-{child_index}')
        common = {'id': identifier, 'name_mapping': deepcopy(mapping), 'name_evidence': [deepcopy(table[sid])]}
        if name_key == 'name':
            recovered = {**common, 'name': mapping['source_name'],
                'description_evidence': [deepcopy(table[s]) for s in field['description_sentence_ids']],
                'item_evidence': [deepcopy(table[s]) for s in field['item_sentence_ids']],
                'status': 'candidate_source_backed', 'semantic_label_verified': False}
        else:
            recovered = {**common, 'variable_name': mapping['source_name'], 'role': field['role'],
                'role_evidence': [deepcopy(table[s]) for s in field['role_sentence_ids']],
                'status': 'candidate_role_unverified', 'semantic_role_verified': False,
                'name_and_role_evidence_are_separate': sid not in field['role_sentence_ids']}
        _require(not any(r['id'] == identifier for r in result[kind][index][children]),
                 'repeated_name_existing_field_changed')
        result[kind][index][children].append(recovered)
        result[kind][index][children].sort(key=lambda r: int(r['id'].rsplit('-', 1)[-1]))
        fixed.add(refusal['path'])
        repairs.append({'field_path': '/' + '/'.join(keys[:-1]), 'original_name': field[name_key],
            'original_name_sentence_id': field['name_sentence_id'], 'replacement_name_sentence_id': sid,
            'mapping': deepcopy(mapping), 'semantic_identity_verified': False})
    result['quarantined_fields'] = [r for r in baseline['quarantined_fields'] if r['path'] not in fixed]
    result.update(repeated_name_policy=VERSION, repeated_name_base_record_sha256=digest(baseline),
        repeated_name_original_quarantined_fields=deepcopy(baseline['quarantined_fields']),
        repeated_name_repairs=repairs, all_prior_fields_unchanged=True)
    return result


def validate_bound_repeated_evidence(evidence, *, name, name_evidence):
    """Reconstruct every repeated position before accepting a catalog witness."""
    _require(isinstance(evidence, dict) and evidence.get('version') == VERSION
             and evidence.get('scientifically_accepted') is False
             and evidence.get('source_text_unchanged') is True, 'repeated_name_bound_evidence_changed')
    mapping = evidence.get('mapping')
    _require(isinstance(mapping, dict) and mapping.get('source_name') == name,
             'repeated_name_bound_name_changed')
    matches = [s for s in name_evidence if repeated_name_mapping(mapping.get('proposal_name'), s['quote']) == mapping]
    _require(len(matches) == 1, 'repeated_name_bound_mapping_changed')
