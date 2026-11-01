"""Select Results from an admitted ordered study, keeping article versions apart.

This is an extraction view only. It never changes the captured section tree or
source bytes. Later subsections remain available for empirical follow-ups;
their physical location is not proof that every finding belongs to this study.
"""
import re
from mosaic.adapted.e.headings import split_section_number

_RESULT_TITLE = re.compile(r'^(?:results?|findings?)(?:\s*(?:and|&|/)\s*discussion)?$', re.I)


def _study_roots(unit):
    return {node.get('title') for path in unit.get('section_paths', [])
            for node in path if node.get('role') == 'study' and node.get('title')}


def select_result_scope(study, *, source_label='main'):
    """Return an explicit source-version scope, or a missing-heading status.

    Source labels come from the already admitted packet, not a new filesystem
    lookup. An unresolved supplement is never pooled just because its study
    label matches. Callers decide how to handle a genuinely absent heading.
    """
    if not isinstance(source_label, str) or not source_label:
        raise ValueError('source_label_required')
    units = [u for u in study['units'] if u['source_locator'].startswith(source_label + '_')]
    start = None
    for index, unit in enumerate(units):
        nodes = [node for path in unit.get('section_paths', []) for node in path]
        if any(node.get('role') == 'results' or
               _RESULT_TITLE.fullmatch(split_section_number(node.get('title', ''))[1].strip())
               for node in nodes):
            start = index
            break
    selected = [] if start is None else units[start:]
    # An admitted study bucket can also contain a later appendix headed
    # "Study 2: Scenario" or "Study 2: Developed Measures" in the same PDF.
    # Its shared study number is not a continuation of the physical Results
    # section. Keep later empirical subsections under the actual starting root.
    roots = set() if not selected else _study_roots(selected[0])
    if roots:
        selected = [u for u in selected if not _study_roots(u) or roots & _study_roots(u)]
    chosen = {u['unit_id'] for u in selected}
    return {'study_id': study['study_id'], 'source_label': source_label,
        'status': 'missing_results_heading' if start is None else 'selected',
        'units': selected, 'start_unit_id': None if start is None else units[start]['unit_id'],
        'excluded_unit_ids': [u['unit_id'] for u in study['units'] if u['unit_id'] not in chosen],
        'physical_study_heading_roots': sorted(roots),
        'heading_scope_unassessed_unit_ids': [u['unit_id'] for u in selected if not _study_roots(u)],
        'study_attribution_unassessed': True}
