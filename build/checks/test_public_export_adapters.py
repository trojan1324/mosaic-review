"""Public-only adapters retain safeguards without private recipe/root defaults."""
from hashlib import sha256
import importlib.util
from pathlib import Path

import pytest


def _adapter(name):
    root = Path(__file__).resolve().parents[1]
    template = root / 'public_build_templates' / (name + '_public.py')
    # In a public export these templates occupy their actual module paths.
    path = template if template.exists() else root / 'mosaic' / (name + '.py')
    spec = importlib.util.spec_from_file_location('tested_public_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_pdf_adapter_requires_explicit_root_and_exact_identity(tmp_path):
    adapter = _adapter('source_assets')
    root = tmp_path / 'bundle'; root.mkdir()
    pdf = root / 'fictional.pdf'; pdf.write_bytes(b'%PDF-fictional-test')
    assets = {'p': {'availability': 'local_file_present', 'path': str(pdf),
                    'source_pdf_sha256': sha256(pdf.read_bytes()).hexdigest()}}
    with pytest.raises(ValueError, match='public_source_roots_required'):
        adapter.pdf_bytes(assets, 'p')
    assert adapter.pdf_bytes(assets, 'p', allowed_roots=[root]) == pdf.read_bytes()
    with pytest.raises(ValueError, match='source_pdf_path_refused'):
        adapter.pdf_bytes(assets, 'p', allowed_roots=[tmp_path / 'different'])
    pdf.write_bytes(b'%PDF-changed')
    with pytest.raises(ValueError, match='source_pdf_identity_mismatch'):
        adapter.pdf_bytes(assets, 'p', allowed_roots=[root])


def test_public_named_recipe_refuses_instead_of_reading_private_configuration():
    adapter = _adapter('recipe')
    with pytest.raises(ValueError, match='public_private_recipe_unavailable'):
        adapter.resolve_recipe('a-private-recipe')
    assert set(adapter.rapidocr_info()) == {'available', 'version'}


def test_exported_structure_relationships_have_their_vocabulary_resource():
    from mosaic.eval.structure_validator import study_edges
    result = study_edges({'study_id': 'study-new', 'study_links': [
        {'target_study_id': 'study-prior', 'relation': 'reuses_measure', 'evidence_unit_ids': ['unit-fictional']}
    ]})
    assert result == [{'key': 'study_links', 'from_study_id': 'study-new', 'to_study_id': 'study-prior',
                       'relation': 'reuses_measure', 'mapped_relation': 'reuses_measure_from',
                       'evidence_unit_ids': ['unit-fictional']}]
