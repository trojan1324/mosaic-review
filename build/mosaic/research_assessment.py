"""Complete draft-surface inventory for advisory checks, never an accuracy score.

Exact spans are explicitly supplied annotations, not inferred atomic facts. The
inventory includes uncited titles, headings, gaps and suggestions. A checked
paragraph cannot confer a status on another surface or an uncovered clause.
Source capture, claim interpretation and provider-origin checks are external.
"""
from hashlib import sha256

from mosaic.canonical import canonical_bytes


VERSION = 'research-assessment-manifest/v1'


def digest(value):
    return sha256(canonical_bytes(value)).hexdigest()


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _surfaces(draft, sources):
    _require(isinstance(draft, dict) and set(draft) == {'title', 'sections', 'gaps'},
             'unsupported_draft_shape')
    out = []
    def add(pointer, kind, text, citations=()):
        _require(isinstance(text, str), 'surface_text_required')
        _require(isinstance(citations, (list, tuple)) and all(isinstance(c, str) for c in citations)
                 and len(citations) == len(set(citations)) and set(citations) <= set(sources),
                 'unknown_or_duplicate_citation')
        out.append({'id': pointer, 'kind': kind, 'text': text,
                    'text_sha256': sha256(text.encode('utf-8')).hexdigest(),
                    'citations': list(citations)})
    add('/draft/title', 'title', draft['title'])
    _require(isinstance(draft['sections'], list) and isinstance(draft['gaps'], list),
             'unsupported_draft_shape')
    for i, section in enumerate(draft['sections']):
        _require(isinstance(section, dict) and set(section) == {'heading', 'paragraphs'}
                 and isinstance(section['paragraphs'], list), 'unsupported_section_shape')
        add(f'/draft/sections/{i}/heading', 'section_heading', section['heading'])
        for j, paragraph in enumerate(section['paragraphs']):
            _require(isinstance(paragraph, dict) and set(paragraph) == {'text', 'kind', 'citations'}
                     and paragraph['kind'] in ('source_summary', 'research_suggestion'),
                     'unsupported_paragraph_shape')
            add(f'/draft/sections/{i}/paragraphs/{j}/text', paragraph['kind'],
                paragraph['text'], paragraph['citations'])
    for i, text in enumerate(draft['gaps']):
        add(f'/draft/gaps/{i}', 'gap_statement', text)
    return out


def content_identity(result):
    """Bind the entire draft and supplied source contexts, including order."""
    return digest({'draft': result['draft'], 'sources': result['sources']})


def _uncovered(text, ranges):
    # Character coverage is bookkeeping only. Even complete span coverage does
    # not establish complete atomization or assessment of every factual clause.
    end = 0; missing = []
    for start, stop in sorted(ranges):
        if start > end and text[end:start].strip():
            missing.append([end, start])
        end = max(end, stop)
    if text[end:].strip():
        missing.append([end, len(text)])
    return missing


def build_manifest(result, annotations=None):
    """Freeze exhaustive surfaces and optional exact, explicitly annotated spans.

    annotations = {content_sha256, claims: [{id, surface_id, start, end,
      text_sha256, citations, kind}]}. No live call or semantic label is made.
    Annotation kinds distinguish source claims, synthesis, suggestions and
    retrieval gaps; they are author-supplied classifications, not verified facts.
    """
    rows = result['sources']
    _require(isinstance(rows, list) and all(isinstance(s, dict)
             and isinstance(s.get('citation_id'), str) for s in rows), 'source_identity_required')
    sources = {s['citation_id']: s for s in rows}
    _require(len(sources) == len(rows), 'duplicate_source_identity')
    surfaces = _surfaces(result['draft'], sources)
    by_id = {s['id']: s for s in surfaces}; identity = content_identity(result)
    annotations = annotations if annotations is not None else {'content_sha256': identity, 'claims': []}
    _require(isinstance(annotations, dict) and set(annotations) == {'content_sha256', 'claims'}
             and annotations['content_sha256'] == identity, 'stale_or_invalid_annotations')
    _require(isinstance(annotations['claims'], list), 'invalid_claim_manifest')
    claims = []; seen = set(); ranges = {s['id']: [] for s in surfaces}
    for claim in annotations['claims']:
        _require(isinstance(claim, dict) and set(claim) ==
                 {'id', 'surface_id', 'start', 'end', 'text_sha256', 'citations', 'kind'},
                 'invalid_claim_shape')
        _require(isinstance(claim['id'], str) and claim['id'] and claim['id'] not in seen,
                 'duplicate_or_missing_claim_id')
        seen.add(claim['id'])
        _require(isinstance(claim['surface_id'], str) and claim['surface_id'] in by_id,
                 'unknown_claim_surface')
        surface = by_id[claim['surface_id']]; start, end = claim['start'], claim['end']
        _require(type(start) is int and type(end) is int and 0 <= start < end <= len(surface['text']),
                 'invalid_claim_offsets')
        text = surface['text'][start:end]
        _require(text.strip() and sha256(text.encode('utf-8')).hexdigest() == claim['text_sha256'],
                 'claim_text_changed')
        citations = claim['citations']
        _require(isinstance(citations, list) and all(isinstance(c, str) for c in citations)
                 and len(citations) == len(set(citations)) and set(citations) <= set(sources),
                 'unknown_or_duplicate_claim_citation')
        _require(claim['kind'] in ('source_claim', 'synthesis', 'suggestion', 'retrieval_gap'),
                 'unsupported_claim_kind')
        _require(claim['kind'] != 'source_claim' or citations, 'source_claim_needs_citation')
        ranges[surface['id']].append((start, end))
        # Sources are frozen individually. A single supporting citation cannot
        # make a multi-source synthesis or the complete draft checked.
        checks = [{'citation_id': c, 'source_sha256': digest(sources[c]),
                   'check_input_sha256': digest({'content_sha256': identity, 'claim': claim,
                                                'source': sources[c]})}
                  for c in citations]
        claims.append({**claim, 'citations': list(citations), 'text': text,
                       'source_checks': checks, 'assessment_status': 'unchecked',
                       'atomicity_verified': False, 'scientifically_accepted': False})
    for surface in surfaces:
        surface['unannotated_ranges'] = _uncovered(surface['text'], ranges[surface['id']])
        surface['assessment_status'] = 'unchecked'
        surface['claim_ids'] = [c['id'] for c in claims if c['surface_id'] == surface['id']]
    return {'version': VERSION, 'content_sha256': identity, 'surfaces': surfaces, 'claims': claims,
            'surface_count': len(surfaces), 'annotated_claim_count': len(claims),
            'surfaces_with_unannotated_text': sum(bool(s['unannotated_ranges']) for s in surfaces),
            'assessed_claim_count': 0, 'atomic_claim_coverage': None,
            'source_support_verified': False, 'scientifically_accepted': False,
            'accuracy_percentage': None,
            'limits': ['Character-span coverage is not factual-clause or atomic-claim coverage.',
                       'All source fidelity, identity/scope and semantic support remain separate checks.',
                       'A source hash binds supplied bytes; it does not independently validate capture.',
                       'Titles, headings, gaps and suggestions never inherit paragraph assessments.']}
