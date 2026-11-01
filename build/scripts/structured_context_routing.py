"""Bounded structural routing for the staged development experiment.

Unaccepted overview context stays in the audit; it cannot abort unrelated
study extraction. Supplemental headings, not a broad model-selected bundle,
bound candidate attachments. Membership remains a proposal to inspect.
"""
import re
from collections import Counter


def route_context(packet, overview):
    unassigned = {u['unit_id']: u for u in packet['unassigned_source_units']}
    accepted = {'hypotheses': [], 'definitions': []}
    rejected = []
    duplicate_hypotheses = {key for key, n in Counter(h['id'] for h in overview['hypotheses']).items() if n > 1}
    for kind in accepted:
        for row in overview[kind]:
            ids = row['source_units']
            if (not ids or not set(ids) <= set(unassigned)
                    or (kind == 'hypotheses' and row['id'] in duplicate_hypotheses)):
                rejected.append({'kind': kind, 'proposal': row, 'reason': 'ineligible_front_end_context'})
            else:
                accepted[kind].append(row)
    supplemental = [u for u in unassigned.values()
                    if any(h['kind'] == 'supplemental_study' for h in u.get('study_hints', []))]
    sections = []
    for unit in supplemental:
        paths = unit.get('section_paths', [])
        if len(paths) != 1 or not paths[0]:
            sections.append({'key': None, 'role': None, 'units': [unit]})
            continue
        leaf = paths[0][-1]
        key = leaf['node_id']
        if sections and sections[-1]['key'] == key:
            sections[-1]['units'].append(unit)
        else:
            sections.append({'key': key, 'role': leaf['role'], 'units': [unit]})
    links = []
    previous = None
    previous_source = None
    for section in sections:
        units = section['units']
        sources = {h['id'] for u in units for h in u.get('study_hints', []) if h['kind'] == 'supplemental_study'}
        source = tuple(sorted(sources))
        if source != previous_source:
            previous = None
        mentions = []
        for study in packet['studies']:
            pattern = r'(?<!\w)' + r'\s+'.join(re.escape(p) for p in study['label'].split()) + r'(?!\w)'
            evidence = [u['unit_id'] for u in units if re.search(pattern, u['text'], re.I)]
            if evidence:
                mentions.append((study['study_id'], evidence))
        if section['key'] and len(mentions) == 1:
            target, evidence = mentions[0]
            previous = (target, evidence)
        elif section['key'] and not mentions and section['role'] == 'results' and previous:
            target, evidence = previous
        else:
            previous = None
            previous_source = source
            continue
        links.append({'study_id': target, 'section_id': section['key'],
                      'unit_ids': [u['unit_id'] for u in units],
                      'link_evidence_unit_ids': evidence, 'semantic_ownership_unassessed': True})
        previous_source = source
    context_ids = {uid for row in accepted['hypotheses'] + accepted['definitions'] for uid in row['source_units']}
    routed = []
    for study in packet['studies']:
        ids = {uid for link in links if link['study_id'] == study['study_id'] for uid in link['unit_ids']}
        routed.append({**study, 'units': study['units'] + [u for uid, u in unassigned.items() if uid in ids],
                       'front_end_context': [u for uid, u in unassigned.items() if uid in context_ids and uid not in ids]})
    return routed, accepted, {'rejected_front_end': rejected, 'structural_links': links,
                             'superseded_model_links': overview['study_links'],
                             'boundary_concerns': overview['boundary_concerns'],
                             'semantic_accuracy_unassessed': True}
