"""The public plan names actual functions without inferring observed runtime support."""
import importlib
import pytest
from mosaic.public_build import STAGES, STAGE_HANDLERS

pytestmark = pytest.mark.evidence_mock

def test_public_stage_map_names_callable_implementations():
    assert set(STAGE_HANDLERS) == set(STAGES)
    for path in STAGE_HANDLERS.values():
        module, name = path.rsplit('.', 1)
        assert callable(getattr(importlib.import_module(module), name))
