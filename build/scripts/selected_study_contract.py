"""Study extraction schema whose numeric values and wording are source choices."""
from copy import deepcopy
from jsonschema import Draft202012Validator
from scripts.structured_extraction_contract import study_schema, audit_record, _obj, _array, _string
from scripts.study_value_selections import resolve_value_choice
from scripts.study_scope_guards import reuse_allowed, count_choices
from scripts.source_line_schema import bounded_selection_schema


def selected_study_schema(units, choices, hypotheses, primary_unit_ids=None):
    primary = list(primary_unit_ids or [u['unit_id'] for u in units])
    schema = deepcopy(study_schema(primary, hypotheses))
    unit = {'type': 'string', 'enum': primary}
    fact = _obj({'label': _string(256), 'value_choice_id': {'type': 'string', **({'enum': [c['choice_id'] for c in choices]} if choices else {})},
                 'link_source_units': _array(unit, 16)})
    schema['$defs'] = {'selected_fact': fact}
    # Item wording is always constrained to its primary study source.  Value
    # selections retain their separate, explicitly audited reuse mechanism.
    schema['$defs']['bounded_selection'] = bounded_selection_schema(units, primary)
    count_fact = deepcopy(fact)
    counts = count_choices(units)
    count_fact['properties']['value_choice_id'] = {'type': 'string', **({'enum': [c['choice_id'] for c in counts]} if counts else {})}
    schema['$defs']['count_fact'] = count_fact
    design = schema['properties']['design']['items']['properties']
    design['label'] = {'type': 'string', 'enum': ['design', 'conditions', 'population']}
    sample_fact = deepcopy(fact)
    sample_fact['properties']['label'] = {'type': 'string', 'enum': ['analysed_n', 'excluded_n']}
    schema['properties']['sample_sizes'] = _array(sample_fact, 16)
    measure = schema['properties']['variables']['items']['properties']['measures']['items']['properties']
    measure['n_items'] = {'anyOf': [{'$ref': '#/$defs/count_fact'}, {'type': 'null'}]} if counts else {'type': 'null'}
    measure['reliability']['items'] = {'$ref': '#/$defs/selected_fact'}
    schema['properties']['analyses']['items']['properties']['findings']['items']['properties']['statistics']['items'] = {'$ref': '#/$defs/selected_fact'}
    selection = {'$ref': '#/$defs/bounded_selection'}
    measure['items']['items'] = selection
    measure['anchors']['items'] = selection
    schema['properties']['manipulations'] = _array(_obj({'label': _string(1000), 'source_units': _array(unit, 32, 1),
                                                       'wording': _array(selection, 64, 1)}), 32)
    schema['required'] = list(schema['properties'])
    return schema


def materialize_selected(proposal, units, choices, hypotheses, primary_unit_ids, reuse_links=()):
    schema = selected_study_schema(units, choices, hypotheses, primary_unit_ids)
    # Decoders receive the exact bounded schema above.  Materialization keeps
    # the historical error-reporting path for a stale/out-of-range selection,
    # which is marked mechanically invalid below instead of raising before the
    # copied-range audit can record it.  Unit identity remains constrained.
    materializer_schema = deepcopy(schema)
    for branch in materializer_schema['$defs']['bounded_selection']['oneOf']:
        branch['properties']['first_line'].pop('maximum')
        branch['properties']['last_line'].pop('maximum')
    Draft202012Validator(materializer_schema).validate(proposal)
    primary = set(primary_unit_ids)
    copies = []

    def resolve(value, path):
        if isinstance(value, dict):
            if 'value_choice_id' in value:
                choice = resolve_value_choice(value['value_choice_id'], choices, units)
                links = value['link_source_units']
                if choice['unit_id'] not in primary and not reuse_allowed(path, value['value_choice_id'], links, primary, reuse_links):
                    raise ValueError('cross_study_selection_without_primary_link')
                copies.append({'path': path, **choice, 'link_source_units': links})
                return {'label': value['label'], 'value': choice['text'], 'source_units': list(dict.fromkeys([choice['unit_id'], *links]))}
            return {key: resolve(child, path + '/' + key) for key, child in value.items()}
        if isinstance(value, list):
            return [resolve(child, path + '/' + str(i)) for i, child in enumerate(value)]
        return value
    resolved = resolve(proposal, '')
    samples = resolved.pop('sample_sizes')
    if any(s['label'] not in ('analysed_n', 'excluded_n') for s in samples):
        raise ValueError('unsupported_sample_size_kind')
    design_length = len(resolved['design'])
    for copied in copies:
        if copied['path'].startswith('/sample_sizes/'):
            index = int(copied['path'].rsplit('/', 1)[1])
            copied['path'] = '/design/' + str(design_length + index)
    resolved['design'].extend(samples)
    manipulations = resolved.pop('manipulations')
    audit = audit_record(resolved, units, hypotheses)
    errors = []
    for error in audit['errors']:
        if error['code'] != 'numeric_token_not_in_source' or not any(c['path'] + '/value' == error['path'] for c in copies):
            errors.append(error)
        # Every numeric field in this schema was already resolved from an exact
        # verified catalog selection; preserve any legacy typography flag.
    wording = []
    text_by_id = {u['unit_id']: u['text'] for u in units}
    for i, manipulation in enumerate(manipulations):
        for j, selection in enumerate(manipulation['wording']):
            if selection['unit_id'] not in manipulation['source_units']:
                errors.append({'code': 'manipulation_source_mismatch', 'path': f'/manipulations/{i}/wording/{j}'})
            text = text_by_id[selection['unit_id']]
            lines = text.splitlines(keepends=True)
            start, end = selection['first_line'], selection['last_line']
            if not 1 <= start <= end <= len(lines):
                errors.append({'code': 'invalid_line_range', 'path': f'/manipulations/{i}/wording/{j}'})
                continue
            offset = sum(map(len, lines[:start - 1]))
            copied = ''.join(lines[start - 1:end])
            wording.append({'path': f'/manipulations/{i}/wording/{j}', **selection, 'unit_start': offset,
                            'unit_end': offset + len(copied), 'copied_text': copied})
    return {'record': resolved, 'manipulations': manipulations, 'numeric_selections': copies, 'manipulation_copies': wording,
            'legacy_audit': audit, 'errors': errors, 'mechanical_valid': not errors,
            'semantic_accuracy_unassessed': True, 'production_accepted': False}
