"""Pure source-packet projection shared by private and public capture paths.

No registry, sealed-set, recipe, Run, provider, or filesystem write dependency
is imported here.  Inputs are already captured/inventoried package objects.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib

from mosaic.canonical import canonical_bytes
from mosaic.inventory.section_tree import build_source_tree
from mosaic.vllm.context import build_unit_batches, context_packet
from scripts.study_context_packets import build_packets
from scripts.source_heading_projection import wrapped_study_parents, project_wrapped_units, inline_heading_witness


def internal_table_headings(tree, objects, source_regions):
    """Exclude only fully enclosed, unclassified table labels from routing.

    Section trees, source text, source policy and admitted unit spans remain
    untouched. A missing/partial box or a genuine section role proves nothing.
    """
    excluded = {}
    for node in tree.nodes:
        h = node.heading
        if node.role != 'unclassified' or h is None:
            continue
        region = objects[h.text_region_id]
        lines = [line for block in region.get('blocks', []) for line in block.get('lines', [])
                 if line['start'] < h.end and h.start < line['end']]
        if not lines or any(not line.get('bbox') for line in lines):
            continue
        covered = {i for line in lines for i in range(max(h.start, line['start']), min(h.end, line['end']))}
        if any(i not in covered and not region['text'][i].isspace() for i in range(h.start, h.end)):
            continue
        for table in source_regions:
            box = table.get('bbox')
            if table['text_region_id'] != h.text_region_id or not box:
                continue
            if all(box[0] - 2 <= ln['bbox'][0] < ln['bbox'][2] <= box[2] + 2
                   and box[1] + 2 <= ln['bbox'][1] < ln['bbox'][3] <= box[3] - 2 for ln in lines):
                excluded[node.node_id] = {'table_id': table['table_id'], 'text_region_id': h.text_region_id,
                    'start': h.start, 'end': h.end, 'reason': 'unclassified_heading_fully_inside_table_region'}
                break
    return excluded


def snapshot(builder, policy, work_id, max_input_tokens, *, source_regions=(), repair_headings=False,
             source_bytes=None):
    units, hints, policies = {}, defaultdict(set), set()
    omitted_lanes = defaultdict(int)
    for _, study in sorted(builder.of_type('study')):
        if study['kind'] == 'paper_level':
            continue
        for batch in build_unit_batches(builder, study, max_input_tokens=max_input_tokens, source_policy=policy):
            packet = context_packet(builder, batch, study, already_proposed=set())
            if packet['source_lane'] != 'body_candidate':
                omitted_lanes[packet['source_lane']] += len(packet['unit_catalog'])
                continue
            policies.add(packet['source_policy_sha256'])
            for unit in packet['unit_catalog']:
                key = (packet['text_region_id'], unit['unit_id'])
                if key in units and units[key] != unit:
                    raise ValueError('source_unit_changed')
                units[key] = unit
                hints[key].add((packet['study_id'], packet['study_kind']))
    if not units or len(policies) != 1:
        raise ValueError('missing_or_inconsistent_admitted_source')
    rows, bindings = [], []
    for index, (key, unit) in enumerate(sorted(units.items(), key=lambda item: item[1]['unit_id']), 1):
        uid = f'U{index:03d}'
        rows.append({'unit_id': uid, 'source_locator': unit['unit_id'], 'page': unit['page'], 'text': unit['text'],
                     'study_hints': [{'id': sid, 'kind': kind} for sid, kind in sorted(hints[key])]})
        bindings.append({'unit_id': uid, 'text_region_id': key[0], 'original_unit_id': key[1],
                         'start': unit['start'], 'end': unit['end'],
                         'text_sha256': hashlib.sha256(unit['text'].encode()).hexdigest()})
    source = {'work_id': work_id, 'source_policy_sha256': next(iter(policies)), 'units': rows,
              'scope': 'Union of admitted body-candidate units from the production batch builder; no withheld source added.'}
    objects = builder.objects; trees = {}; paths = {}; excluded = {}; wrapped = {}
    for binding in bindings:
        sid = objects[binding['text_region_id']]['source_id']
        if sid not in trees:
            trees[sid] = build_source_tree(objects, sid)
            excluded.update(internal_table_headings(trees[sid], objects, source_regions))
            if repair_headings:
                wrapped.update(wrapped_study_parents(trees[sid], objects))
        tree = trees[sid]; nodes = {n.node_id: n for n in tree.nodes}; linked = []
        for node in tree.nodes:
            if any(seg.text_region_id == binding['text_region_id'] and seg.start < binding['end'] and binding['start'] < seg.end for seg in node.segments):
                chain = []; current = node
                while current:
                    if current.node_id not in excluded:
                        chain.append({'title': current.title, 'role': current.role, 'node_id': current.node_id})
                    current = nodes.get(current.parent_id)
                linked.append(list(reversed(chain)))
        paths[binding['unit_id']] = linked
    ordered = build_packets(rows, bindings, objects, paths)
    if repair_headings:
        # Retain the private optional heading projection unchanged in behavior.
        unit_order = {}
        for b in bindings:
            region = objects[b['text_region_id']]; page = objects.get(region.get('page_id'), {})
            membership = objects[region['source_id']]['membership'][0]
            block = next(block for block in region['blocks'] if block['start'] <= b['start'] < b['end'] <= block['end'])
            unit_order[b['unit_id']] = (membership['role'] != 'main_article', membership['source_label'],
                page.get('physical_index', -1), region['reading_order'].index(block['block_index']), b['start'])
        ordered = project_wrapped_units(ordered, wrapped, unit_order)
        import pymupdf
        docs = {}
        try:
            for study in ordered['studies']:
                for unit in study['units']:
                    if not unit['text'].startswith(('Results.', 'Discussion.', 'Method.', 'Methods.')):
                        continue
                    b = unit['source_binding']; region = objects[b['text_region_id']]; page = objects.get(region.get('page_id')); sid = region['source_id']
                    if not page or not unit['source_locator'].startswith('main_'):
                        continue
                    if sid not in docs:
                        raw = None if source_bytes is None else source_bytes.get(sid)
                        if source_bytes is not None:
                            if not isinstance(raw, bytes) or hashlib.sha256(raw).hexdigest() != objects[sid]['artifact']['sha256']:
                                raise ValueError('source_packet_source_bytes_hash_mismatch')
                            docs[sid] = pymupdf.open(stream=raw, filetype='pdf')
                        else:
                            docs[sid] = pymupdf.open(objects[sid]['artifact']['acquisition_path'])
                    witness = inline_heading_witness(region, b['start'], b['end'], docs[sid][page['physical_index']])
                    if witness:
                        unit['inline_heading'] = {**witness, 'text_region_id': b['text_region_id']}
        finally:
            for doc in docs.values():
                doc.close()
    digest = hashlib.sha256(canonical_bytes(source)).hexdigest()
    ordered['source_packet_sha256'] = digest
    binding = {'work_id': work_id, 'source_packet_sha256': digest, 'unit_bindings': bindings,
               'omitted_nonbody_lanes': dict(omitted_lanes)}
    if excluded:
        ordered['routing_excluded_table_headings'] = excluded
        binding['routing_excluded_table_headings'] = excluded
    return source, binding, ordered
