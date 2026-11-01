"""Narrow measure-field binding and typed whole-expression results contracts."""
from copy import deepcopy
from jsonschema import Draft202012Validator
from scripts.structured_extraction_contract import study_schema, _obj, _array, _string, audit_record
from scripts.study_scope_guards import count_choices, reuse_allowed
from scripts.typed_statistic_choices import build_statistic_choices, resolve_statistic_choice
from scripts.direction_support import direction_evidence_schema, validate_direction_evidence


def statistic_facts(choice):
    components = {c['name']: c for c in choice['components']}
    if choice['statistic_type'] == 'confidence_interval':
        fields = [('CI lower', components['lower']['text']), ('CI upper', components['upper']['text'])]
    elif 'value' not in components:
        values = [c for c in choice['components'] if c['name'].startswith('value_')]
        if len(values) == 1:
            fields = [(choice['printed_label'], values[0]['text'])]
            if choice['statistic_type'] == 'p_value':
                fields = [(choice['printed_label'], choice['text'][len(choice['printed_label']):].strip())]
            fields += [(name,c['text']) for name,c in components.items() if name.startswith('df_') or name == 'sample_n']
        else:
            # A list or unclassified expression stays grouped. Its order is
            # printed evidence, not an inferred mapping to named conditions.
            fields = [(choice['printed_label'], choice['text'])]
    else:
        value = components['value']['text']
        if choice['statistic_type'] == 'p_value':
            value = choice['text'][len(choice['printed_label']):].strip()
        fields = [(choice['printed_label'], value)]
        fields += [(name, c['text']) for name, c in components.items() if name.startswith('df_')]
    return [{'label': label, 'value': value, 'source_units': [choice['unit_id']]} for label, value in fields]


def measure_binding_packet(record, context):
    all_counts = count_choices(context['units'])
    counts = [{**row, 'original_choice_id': row['choice_id'], 'choice_id': f'count{i:03d}'} for i, row in enumerate(all_counts, 1)]
    statistics = build_statistic_choices(context['units'])
    slots = []
    for i, variable in enumerate(record['variables']):
        for j, measure in enumerate(variable['measures']):
            ids = set(measure['source_units'])
            links = [r for r in context['reuse_links'] if r['primary_unit_id'] in ids]
            allowed_external = {cid for link in links for cid in link['allowed_choice_ids']}
            offered_counts = [c for c in counts if c['unit_id'] in ids or c['original_choice_id'] in allowed_external]
            offered_reliability = [c for c in statistics if c['unit_id'] in ids and c['statistic_type'] == 'reliability']
            offered_ids = ids | {c['unit_id'] for c in offered_counts}
            slots.append({'slot_id': f'm{len(slots)+1}', 'variable_name': variable['name'], 'measure_name': measure['name'],
                'path': [i,j], 'count_choices': offered_counts, 'reliability_choices': offered_reliability,
                'source': [u for u in context['units'] if u['unit_id'] in offered_ids], 'reuse_links': links})
    return {'slots': slots}


def measure_binding_schema(packet):
    return _obj({s['slot_id']: _obj({
        'count_choice': {'type': 'string', 'enum': ['not_reported','ambiguous'] + [c['choice_id'] for c in s['count_choices']]},
        'reliability_choices': _array({'type': 'string', **({'enum':[c['choice_id'] for c in s['reliability_choices']]} if s['reliability_choices'] else {})}, len(s['reliability_choices'])),
        'note': {'type':'string','maxLength':1000}}) for s in packet['slots']})


def apply_measure_binding(record, proposal, packet, context):
    Draft202012Validator(measure_binding_schema(packet)).validate(proposal)
    # Rebuild the entire offered slot binding before copying any source value.
    if measure_binding_packet(record, context) != packet:
        raise ValueError('measure_binding_packet_changed')
    result, changes = deepcopy(record), []
    statistics = build_statistic_choices(context['units'])
    for slot in packet['slots']:
        choice = proposal[slot['slot_id']]
        i,j = slot['path']; measure = result['variables'][i]['measures'][j]
        count = next((c for c in slot['count_choices'] if c['choice_id'] == choice['count_choice']), None)
        if count is not None:
            primary_ids = [u['unit_id'] for u in context['study']['units']]
            links = [r['primary_unit_id'] for r in slot['reuse_links'] if count['original_choice_id'] in r['allowed_choice_ids']]
            if count['unit_id'] not in primary_ids and not reuse_allowed(f'/variables/{i}/measures/{j}/n_items', count['original_choice_id'], links, primary_ids, slot['reuse_links']):
                raise ValueError('invalid_external_count_binding')
            new = {'label':'Number of items','value':count['text'],'source_units':list(dict.fromkeys([count['unit_id'],*links]))}
            changes.append({'slot':slot['slot_id'],'field':'n_items','before':measure['n_items'],'after':new})
            measure['n_items'] = new
        # A missing proposal never erases a previously populated field.
        for cid in choice['reliability_choices']:
            selected = resolve_statistic_choice(cid, statistics, context['units'])
            if selected not in slot['reliability_choices']:
                raise ValueError('reliability_outside_measure_source')
            for fact in statistic_facts(selected):
                if not any(old['value'] == fact['value'] and old['source_units'] == fact['source_units'] for old in measure['reliability']):
                    measure['reliability'].append(fact)
                    changes.append({'slot':slot['slot_id'],'field':'reliability','after':fact})
    return {'record':result,'changes':changes,'semantic_accuracy_unassessed':True,'production_accepted':False}


def result_units(context):
    primary = context['study']['units']
    selected = [u for u in primary if any(n['role'] == 'results' for p in u.get('section_paths',[]) for n in p)]
    return selected or primary


def typed_results_schema(units, choices, hypotheses, variables):
    schema = study_schema([u['unit_id'] for u in units], hypotheses)
    schema['properties'] = {k:schema['properties'][k] for k in ('study_id','label','analyses','hypothesis_support','gaps')}
    schema['required'] = list(schema['properties'])
    analysis = schema['properties']['analyses']['items']['properties']
    analysis['roles']['items']['properties']['variable_id'] = {'type':'string','enum':[v['id'] for v in variables]}
    finding = analysis['findings']['items']
    del finding['properties']['statistics']
    finding['properties']['statistic_choices'] = _array({'type':'string', **({'enum':[c['choice_id'] for c in choices]} if choices else {})},len(choices))
    finding['properties']['statistic_choices']['uniqueItems'] = True
    lexical = direction_evidence_schema([u['unit_id'] for u in units], 0)['oneOf'][0]
    estimate = _obj({'kind': {'const': 'selected_estimate'}, 'choice_id': {'type': 'string',
        'enum': [c['choice_id'] for c in choices if c['statistic_type'] in ('coefficient', 'correlation')]}})
    evidence = [lexical, {'type': 'null'}]
    if estimate['properties']['choice_id']['enum']:
        evidence.insert(1, estimate)
    finding['properties']['direction_evidence'] = {'anyOf': evidence}
    finding['required'] = list(finding['properties'])
    return schema


def materialize_results(base, proposal, units, choices, hypotheses, all_units, *, resolve_choice=resolve_statistic_choice):
    Draft202012Validator(typed_results_schema(units, choices, hypotheses, base['variables'])).validate(proposal)
    if proposal['study_id'] != base['study_id'] or proposal['label'] != base['label']:
        raise ValueError('changed_study_identity')
    result = deepcopy(base); result['analyses'] = deepcopy(proposal['analyses'])
    result['hypothesis_support'] = deepcopy(proposal['hypothesis_support'])
    result['gaps'] += deepcopy(proposal['gaps'])
    copies, direction_checks, errors = [], [], []
    for i,a in enumerate(result['analyses']):
        for j,f in enumerate(a['findings']):
            f['statistics'] = []
            selected_indexes = {}
            for cid in f.pop('statistic_choices'):
                selected = resolve_choice(cid, choices, units)
                if selected['unit_id'] not in f['source_units']:
                    raise ValueError('statistic_source_not_cited_by_finding')
                facts = statistic_facts(selected)
                selected_indexes[cid] = (len(f['statistics']), selected['unit_id'])
                for fact in facts:
                    copies.append({'path':f'/analyses/{i}/findings/{j}/statistics/{len(f["statistics"])}', 'choice':selected})
                    f['statistics'].append(fact)
            evidence = f.pop('direction_evidence')
            witness = evidence
            if evidence and evidence.get('kind') == 'selected_estimate':
                selected = selected_indexes.get(evidence['choice_id'])
                if selected is None:
                    errors.append({'code': 'direction_choice_not_among_finding_statistics',
                                   'path': f'/analyses/{i}/findings/{j}/direction'})
                    witness = None
                else:
                    witness = {'kind': 'signed_estimate', 'statistic_index': selected[0], 'unit_id': selected[1]}
            check = validate_direction_evidence(f, witness, units)
            direction_checks.append({'analysis':i,'finding':j,'evidence':evidence,**check})
            errors += [{'code':e,'path':f'/analyses/{i}/findings/{j}/direction'} for e in check['errors']]
    audit = audit_record(result, all_units, hypotheses)
    copied_paths = {c['path']+'/value' for c in copies} | verified_count_paths(result, all_units)
    errors += [e for e in audit['errors'] if not (e['code']=='numeric_token_not_in_source' and e['path'] in copied_paths)]
    return {'record':result,'typed_copies':copies,'direction_checks':direction_checks,'errors':errors,
            'mechanical_valid':not errors,'semantic_accuracy_unassessed':True,'production_accepted':False,'legacy_audit':audit}


def verified_count_paths(record, units):
    """Exempt only exact printed count expressions from legacy typography flags."""
    counts = count_choices(units)
    paths = set()
    for i, variable in enumerate(record['variables']):
        for j, measure in enumerate(variable['measures']):
            fact = measure.get('n_items')
            if fact and any(c['text'] == fact['value'] and c['unit_id'] in fact['source_units'] for c in counts):
                paths.add(f'/variables/{i}/measures/{j}/n_items/value')
    return paths
