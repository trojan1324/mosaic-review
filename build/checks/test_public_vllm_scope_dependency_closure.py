"""An explicit public vLLM scope must not import the private roster/scope."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_explicit_public_vllm_scope_imports_without_private_scope_or_sealed_module(tmp_path: Path):
    code = """
import sys, tempfile
from pathlib import Path
root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
class BlockPrivate:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'mosaic.vllm.scope', 'mosaic.sealed', 'mosaic.candidates.fixture'}:
            raise ImportError('private module intentionally absent')
        return None
sys.meta_path.insert(0, BlockPrivate())
from mosaic.vllm.client import VLLMClient
import mosaic.vllm.client as client_module
assert Path(client_module.__file__).resolve().is_relative_to(root)
from mosaic.vllm.ledger import VLLMBudget, VLLMLedger
class PublicScope:
    def check(self, work_id, source_sha256s, text_provenance, *, remote, rights=None):
        assert remote is False and rights == ['public_local_processing_user_declared']
with tempfile.TemporaryDirectory() as raw:
    client = VLLMClient(candidate='public-fixture', ledger=VLLMLedger('public-fixture', root=Path(raw)),
                        scope=PublicScope(), budget=VLLMBudget(1, 1, 1, 1), live=False)
    assert client.scope.__class__.__name__ == 'PublicScope'
print('public-vllm-private-scope-free')
"""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-I", "-c", code, str(root)], cwd=root,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "public-vllm-private-scope-free"
