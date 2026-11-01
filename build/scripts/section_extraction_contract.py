"""Narrow source-first section tasks; no critique or rewrite of previous output."""
from copy import deepcopy
from scripts.selected_study_contract import selected_study_schema

FIELDS = {'measures': ['study_id', 'label', 'variables', 'gaps'],
          'design': ['study_id', 'label', 'design', 'sample_sizes', 'manipulations', 'gaps'],
          'results': ['study_id', 'label', 'analyses', 'hypothesis_support', 'gaps']}


def section_schema(kind, units, choices, hypotheses, primary_unit_ids, variable_ids=()):
    full = selected_study_schema(units, choices, hypotheses, primary_unit_ids)
    schema = deepcopy(full)
    schema['properties'] = {key: full['properties'][key] for key in FIELDS[kind]}
    schema['required'] = list(schema['properties'])
    if kind == 'results' and variable_ids:
        schema['properties']['analyses']['items']['properties']['roles']['items']['properties']['variable_id'] = {'type': 'string', 'enum': list(variable_ids)}
    return schema


def whole_proposal(parts, study_id, label):
    value = {'study_id': study_id, 'label': label, 'design': [], 'sample_sizes': [], 'variables': [],
             'analyses': [], 'hypothesis_support': [], 'gaps': [], 'manipulations': []}
    for kind, part in parts.items():
        if part['study_id'] != study_id or part['label'] != label:
            raise ValueError('study_identity_changed')
        for field in FIELDS[kind]:
            if field in ('study_id', 'label'):
                continue
            value[field].extend(part[field])
    return value
