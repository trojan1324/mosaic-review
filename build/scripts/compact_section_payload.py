"""Lossless factoring of repeated section paths in large source menus."""
from copy import deepcopy
from mosaic.canonical import canonical_bytes

MENU_KEYS = ('ordered_source','primary_source','reference_context_only')


def compact(payload):
    result=deepcopy(payload);catalog=[];seen={}
    if 'section_path_catalog' in result:raise ValueError('already_compacted')
    for key in MENU_KEYS:
        for unit in result.get(key,[]):
            if 'section_paths' not in unit:continue
            if 'section_path_ids' in unit:raise ValueError('ambiguous_section_encoding')
            ids=[]
            for path in unit.pop('section_paths'):
                fingerprint=canonical_bytes(path)
                if fingerprint not in seen:
                    pid='path'+str(len(catalog)+1)
                    seen[fingerprint]=pid;catalog.append({'id':pid,'path':path})
                ids.append(seen[fingerprint])
            unit['section_path_ids']=ids
    result['section_path_catalog']=catalog
    return result


def expand(payload):
    result=deepcopy(payload);catalog=result.pop('section_path_catalog')
    paths={p['id']:p['path'] for p in catalog}
    if len(paths)!=len(catalog):raise ValueError('duplicate_section_path')
    for key in MENU_KEYS:
        for unit in result.get(key,[]):
            if 'section_path_ids' in unit:
                unit['section_paths']=[deepcopy(paths[pid]) for pid in unit.pop('section_path_ids')]
    return result
