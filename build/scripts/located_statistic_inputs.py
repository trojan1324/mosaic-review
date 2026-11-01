"""Locate test and numeric evidence explicitly in an attachment request.

Pure request scaffolding. Identical semantic descriptions on the same source
sentence cannot be distinguished by their array position or arbitrary test ID.
"""
from collections import defaultdict
from copy import deepcopy
import json


def located_inputs(packet, schema, upstream, atoms, witnesses):
    packet, schema = deepcopy(packet), deepcopy(schema)
    source = {r['source']['unit_id']: r['source']['text'] for r in upstream['records']}
    original = {t['test_id']: t for r in upstream['records'] for t in r['tests']}
    lookup = {a['id']: a for a in atoms}
    identities = defaultdict(list)

    def evidence(uid, span):
        if source[uid][span['start']:span['end']] != span['text']:
            raise ValueError('test_citation_changed')
        return [c['choice_id'] for c in witnesses if c['unit_id'] == uid
                and c['start'] < span['end'] and span['start'] < c['end']]

    for row in packet['tests']:
        test = original[row['test_id']]; uid = row['source_unit_id']
        selected = test['result_copy']
        row['result_text'] = selected['text']
        row['result_evidence_ids'] = evidence(uid, selected)
        # Earlier task-local sentence IDs must not masquerade as identifiers in
        # this new sentence menu. Remap original spans, never the numeric suffix.
        row['support_evidence_ids'] = []
        for support in test.get('support_copies', []):
            span = support['original_span']; sid = support['unit_id']
            row['support_evidence_ids'].extend(c['choice_id'] for c in witnesses
                if c['unit_id'] == sid and c['start'] < span['end'] and span['start'] < c['end'])
        prior = row['prior_proposal']
        for name in ('result_selection', 'support', 'unspecified_controls', 'population'):
            prior.pop(name, None)
        for control in prior.get('controls', []):
            control.pop('evidence_selection', None)
        identity = {k: v for k, v in prior.items() if k != 'uncertainty'}
        identities[(uid, tuple(row['result_evidence_ids']),
                    json.dumps(identity, sort_keys=True, ensure_ascii=False).casefold())].append(row)

    for choice in packet['statistic_choices']:
        atom = lookup[choice['id']]; exp = atom['expression']
        span = atom['selected_component'] or exp
        choice['evidence_ids'] = [c['choice_id'] for c in witnesses
            if c['unit_id'] == exp['unit_id'] and c['start'] <= span['start'] and span['end'] <= c['end']]

    for group in identities.values():
        if len(group) < 2:
            continue
        ids = [r['test_id'] for r in group]
        for row in group:
            row['indistinguishable_test_ids'] = ids
            row['allowed_statistic_ids'] = []
            props = schema['properties'][row['test_id']]['properties']
            props['statistic_bindings'] = {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 0}
            props['issues'] = {'type': 'array', 'items': {'const': 'group_attribution_unresolved'},
                               'minItems': 1, 'maxItems': 1}
    return packet, schema
