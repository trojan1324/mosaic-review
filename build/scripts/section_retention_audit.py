"""Read-only iteration differences; unchanged counts never prove accuracy."""
from collections import Counter
import re
import unicodedata


def normalized_literal(value):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', value).replace('−', '-'))


def statistics(record):
    return Counter((tuple(sorted(stat['source_units'])), normalized_literal(stat['value']))
                   for analysis in record['analyses'] for finding in analysis['findings']
                   for stat in finding['statistics'])


def item_ranges(record):
    return Counter((item['unit_id'], item['first_line'], item['last_line'])
                   for variable in record['variables'] for measure in variable['measures']
                   for item in measure['items'])


def compare(previous, current):
    old_stats, new_stats = statistics(previous), statistics(current)
    old_items, new_items = item_ranges(previous), item_ranges(current)
    missing = lambda old, new: [{'source_key': list(key), 'count': n} for key, n in (old-new).items()]
    by_id = {v['id']: v['name'] for v in current['variables']}
    roles = [{'analysis_id': a['id'], 'variable': by_id.get(r['variable_id']), 'role': r['role'],
              'source_units': r['source_units']} for a in current['analyses'] for r in a['roles']]
    return {'previous_statistics': sum(old_stats.values()), 'current_statistics': sum(new_stats.values()),
            'missing_or_changed_statistic_occurrences': missing(old_stats, new_stats),
            'previous_item_selections': sum(old_items.values()), 'current_item_selections': sum(new_items.values()),
            'missing_or_changed_item_ranges': missing(old_items, new_items),
            'variables': [{'name': v['name'], 'measure_counts': [m['n_items'] for m in v['measures']]} for v in current['variables']],
            'analysis_roles': roles,
            'analyses_without_roles': [a['id'] for a in current['analyses'] if a['findings'] and not a['roles']],
            'semantic_accuracy_unassessed': True, 'production_accepted': False,
            'limitations': ['Matching literals do not prove correct scientific association.',
                           'Changed line boundaries or notation may preserve the same information.',
                           'Never restore a removed claim without checking its source.']}


def pool_manipulations(materialized):
    """Losslessly share exact repeated source selections; keep every condition."""
    fragments, keys, conditions = [], {}, []
    for index, manipulation in enumerate(materialized['manipulations']):
        selected = []
        expected = len(manipulation['wording'])
        for copy in materialized['manipulation_copies']:
            if not copy['path'].startswith(f'/manipulations/{index}/wording/'):
                continue
            key = (copy['unit_id'], copy['unit_start'], copy['unit_end'], copy['copied_text'])
            if key not in keys:
                keys[key] = f'fragment:{len(keys)+1}'
                fragments.append({'id': keys[key], **{k: copy[k] for k in ('unit_id','unit_start','unit_end','copied_text')}})
            selected.append(keys[key])
        if len(selected) != expected:
            raise ValueError('incomplete_manipulation_copies')
        conditions.append({'label': manipulation['label'], 'source_units': manipulation['source_units'], 'fragments': selected})
    return {'fragments': fragments, 'conditions': conditions, 'semantic_accuracy_unassessed': True}
