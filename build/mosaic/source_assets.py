"""Exact-identity access to explicitly rooted public bundle PDFs.

This export adapter retains MOSAIC's manifest, path and byte-identity checks.
It has no private source roster or implicit home/Dropbox/legacy search roots.
"""
from hashlib import sha256
import json
from pathlib import Path


def checked_path(locator, allowed_roots=None):
    if not allowed_roots:
        raise ValueError('public_source_roots_required')
    roots = [Path(p).resolve() for p in allowed_roots]
    if str(locator).startswith(('\\\\', '//')):
        raise ValueError('source_pdf_path_refused')
    path = Path(locator).resolve()
    if (path.suffix.lower() != '.pdf' or not any(path.is_relative_to(root) for root in roots)
            or {'sealed', 'references', '_holdback'}.intersection(p.casefold() for p in path.parts)):
        raise ValueError('source_pdf_path_refused')
    return path


def load_assets(path):
    path = Path(path).resolve()
    data = json.loads(path.read_bytes())
    if data.get('version') != 'mosaic-source-assets/v1' or data.get('scientifically_accepted') is not False:
        raise ValueError('source_asset_manifest_invalid')
    rows = data['assets']
    if len({r['paper_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate_source_asset')
    return {r['paper_id']: {**r, 'path': str((path.parent / r['path']).resolve())}
            if r.get('path') and not Path(r['path']).is_absolute() else r for r in rows}


def pdf_bytes(assets, paper_id, *, allowed_roots=None):
    row = assets.get(paper_id)
    if row is None or row.get('availability') != 'local_file_present':
        raise FileNotFoundError('source_pdf_unavailable')
    path = checked_path(row['path'], allowed_roots)
    raw = path.read_bytes()
    if not raw.startswith(b'%PDF-') or sha256(raw).hexdigest() != row['source_pdf_sha256']:
        raise ValueError('source_pdf_identity_mismatch')
    return raw
