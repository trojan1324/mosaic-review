"""Attach only explicitly named earlier-study measure-count dependencies.

Heading inventories may put a measure paragraph beside, rather than beneath,
the Measures heading. Eligibility therefore comes from the actual count-bearing
source paragraph and explicit named reuse, not a heading-title substring.
"""
import re
from scripts.structured_context_routing import route_context
from scripts.study_scope_guards import build_measure_reuse_links, count_choices
from scripts.study_value_selections import build_value_choices


def prepare_contexts(packet, overview):
    studies, accepted, routing = route_context(packet, overview)
    prepared = []
    for study in studies:
        others = []
        for other in studies:
            if other['study_id'] == study['study_id']:
                continue
            # Inspect only the already-admitted union. Explicit item-count text
            # can establish an eligible dependency despite imperfect headings.
            units = [u for u in other['units'] if count_choices([u])]
            others.append({**other, 'units': units})
        links = build_measure_reuse_links(study['units'], others)
        needed = {link['reference_unit_id'] for link in links}
        reference = [u for other in others for u in other['units'] if u['unit_id'] in needed]
        explicit_mentions = {u['unit_id'] for u in study['units'] if re.search(r'\bsame\s+(?:\w+\s+){0,4}(?:measure|scale)\b[\s\S]{0,200}\bstudy\s+\d+\b', u['text'], re.I)}
        resolved_mentions = {link['primary_unit_id'] for link in links}
        units = study['units'] + reference
        prepared.append({'study': study, 'units': units, 'reference': reference, 'reuse_links': links,
                         'unresolved_reuse_mentions': sorted(explicit_mentions-resolved_mentions),
                         'choices': build_value_choices(units)})
    return prepared, accepted, routing
