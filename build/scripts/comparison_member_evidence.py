"""Source witnesses for the two members of a reported simple contrast.

This is deliberately an opt-in wrapper around the existing coherent Results
task.  It does not decide whether a proposed label is scientifically correct:
it requires a separate, copied source phrase for each member and records the
two things separately.
"""
from copy import deepcopy
from jsonschema import Draft202012Validator

from scripts.integrated_extraction_trial import compile_section_task
from scripts.source_quote_locator import locate_quote
from scripts.source_sentence_choices import original_segments, sentence_choices


MEMBER_GUIDE = """\
CONTRAST MEMBERS: For every simple_contrast, level_a and level_b are objects.
label is your proposed readable label. quote is a short contiguous phrase
copied from the selected original sentence that identifies that member.
evidence_selection identifies the supplied sentence. The copied quote, not a
generated label such as 'reference group', is the evidence for the member.
Use a source_context sentence only when that sentence is explicitly included
in the test's support. If either member cannot be tied to a supplied sentence,
set that member to null. Null explicitly means members_not_stated; never invent
a member to fill the schema. The result sentence is retained for inspection.
"""


def _variants(schema):
    return schema['$defs']['paragraph']['anyOf'][0]['properties']['tests']['items']['anyOf']


def _member_schema():
    return {'type': 'object', 'properties': {
        'label': {'type': 'string', 'minLength': 1, 'maxLength': 300},
        'quote': {'type': 'string', 'minLength': 1, 'maxLength': 300},
        'evidence_selection': {'$ref': '#/$defs/selection'},
    }, 'required': ['label', 'quote', 'evidence_selection'], 'additionalProperties': False}


def finalize_task(task):
    """Return an opt-in task whose contrast levels carry source witnesses.

    The caller supplies a coherent task already built by scoped_results_trial.
    Recompiling makes the prompt's schema and constrained decoder identical.
    """
    output = deepcopy(task)
    schema = output['schema']
    output['pre_member_schema'] = deepcopy(schema)
    schema['$defs']['contrast_member_evidence'] = _member_schema()
    matched = False
    for variant in _variants(schema):
        if variant['properties'].get('kind') != {'const': 'simple_contrast'}:
            continue
        matched = True
        for key in ('level_a', 'level_b'):
            variant['properties'][key] = {'anyOf': [{'$ref': '#/$defs/contrast_member_evidence'}, {'type': 'null'}]}
    if not matched:
        raise ValueError('simple_contrast_variant_missing')
    request = deepcopy(output['body'])
    instruction = request['messages'][0]['content'].split('\nRESPONSE_SCHEMA:\n', 1)[0]
    request['messages'][0]['content'] = instruction + '\n' + MEMBER_GUIDE
    output.update(compile_section_task(request, output['payload'], schema))
    output['comparison_member_evidence_version'] = 'source-witness-v1'
    return output


def proposal_for_baseline(proposal):
    """Project witnessed member objects to legacy label strings for copying.

    Validate ``proposal`` against the finalized schema first.  The existing
    baseline materializer has a legacy string-only contrast schema and only
    needs the proposed labels while it copies result evidence.
    """
    output = deepcopy(proposal)
    for paragraph in output.values():
        for test in paragraph.get('tests', []):
            if test.get('kind') == 'simple_contrast':
                for level in ('level_a', 'level_b'):
                    value = test.get(level)
                    if isinstance(value, dict):
                        test[level] = value.get('label')
                    elif value is None:
                        test[level] = 'members_not_stated'
    return output


def materialize(proposal, task):
    """Validate the witnessed schema, then use the unchanged source copier."""
    from scripts.scoped_results_trial import materialize as scoped_materialize
    Draft202012Validator(task['schema']).validate(proposal)
    legacy_task = {**task, 'schema': task['pre_member_schema']}
    copied = scoped_materialize(proposal_for_baseline(proposal), legacy_task)
    return attach_member_evidence(proposal, task, copied)


def _append_error(test, error):
    if error not in test['copy_errors']:
        test['copy_errors'].append(error)


def _copy_member(member, *, own_unit_id, supported_choices, choices, groups):
    """Copy one member witness or return its conservative unresolved reason."""
    if not isinstance(member, dict):
        return None, 'simple_contrast_member_evidence_missing'
    label, quote, selection = member.get('label'), member.get('quote'), member.get('evidence_selection')
    if not isinstance(label, str) or not isinstance(quote, str) or not isinstance(selection, dict):
        return None, 'simple_contrast_member_evidence_missing'
    choice = choices.get(selection.get('choice_id'))
    if choice is None or choice['unit_id'] not in groups:
        return None, 'simple_contrast_member_selection_invalid'
    if choice['unit_id'] != own_unit_id and choice['choice_id'] not in supported_choices:
        return None, 'simple_contrast_member_context_not_explicitly_supported'
    span = locate_quote(quote, choice['text'], allow_alphabetic_ligatures=True)
    if span is None:
        return None, 'simple_contrast_member_quote_not_in_selected_evidence'
    start, end = choice['start'] + span['start'], choice['start'] + span['end']
    group = groups[choice['unit_id']]
    return {'label_proposal': label, 'unit_id': choice['unit_id'], 'choice_id': choice['choice_id'],
            'start': start, 'end': end, 'text': group['text'][start:end],
            'source_segments': original_segments(group, start, end),
            'contextual_link_proposed': choice['unit_id'] != own_unit_id}, None


def attach_member_evidence(proposal, task, materialized):
    """Attach copied witnesses and an eligibility flag without dropping audit rows.

    This accepts legacy/raw proposals too, but treats them as unresolved.  The
    finalized schema returned by :func:`finalize_task` makes witnesses mandatory
    for new model output; this fallback keeps historical output inspectable.
    """
    output = deepcopy(materialized)
    choices = {c['choice_id']: c for c in sentence_choices(
        task['payload']['source'] + task['payload'].get('source_context', []))}
    groups = {g['unit_id']: g for g in task['source_groups']}
    for record in output['records']:
        uid = record['source']['unit_id']
        raw_tests = proposal.get(uid, {}).get('tests', [])
        for rendered, raw in zip(record['tests'], raw_tests):
            if raw.get('kind') != 'simple_contrast':
                continue
            supported = {item.get('choice_id') for item in raw.get('support', []) if isinstance(item, dict)}
            copied = {}
            for level in ('level_a', 'level_b'):
                witness, error = _copy_member(raw.get(level), own_unit_id=uid,
                    supported_choices=supported, choices=choices, groups=groups)
                if error:
                    _append_error(rendered, error)
                else:
                    copied[level] = witness
            bound = len(copied) == 2
            if bound and all(copied['level_a'][key] == copied['level_b'][key]
                             for key in ('unit_id', 'start', 'end')):
                _append_error(rendered, 'simple_contrast_member_witnesses_not_distinct')
                bound = False
            rendered['contrast_member_evidence'] = {
                'status': 'source_bound' if bound else 'unresolved',
                'level_a': copied.get('level_a'), 'level_b': copied.get('level_b'),
            }
            rendered['numeric_comparison_claim_eligible'] = bound
    output['comparison_member_evidence_version'] = task.get(
        'comparison_member_evidence_version', 'source-witness-v1')
    return output
