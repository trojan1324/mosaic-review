"""Public Jev transport must not load private source-scope modules when scoped explicitly."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_explicit_public_scope_client_imports_without_private_scope_or_sealed_module():
    code = """
import sys
import tempfile
from pathlib import Path
root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
class BlockPrivate:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'mosaic.jev.scope', 'mosaic.sealed'}:
            raise ImportError('private module intentionally absent')
        return None
sys.meta_path.insert(0, BlockPrivate())
from mosaic.jev.client import JevClient
import mosaic.jev.client as client_module
assert Path(client_module.__file__).resolve().is_relative_to(root)
from mosaic.jev.ledger import Budget, JevLedger
from decimal import Decimal
class PublicScope:
    def check(self, work_id, source_sha256s, text_provenance):
        assert work_id == 'public-fixture'
with tempfile.TemporaryDirectory() as raw:
    client = JevClient(ledger=JevLedger(Path(raw) / 'ledger.jsonl'), scope=PublicScope(),
                       budget=Budget(Decimal('1'), 1), api_key='fixture')
    assert client.scope.__class__.__name__ == 'PublicScope'
print('public-client-private-scope-free')
"""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-I", "-c", code, str(root)], cwd=root,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "public-client-private-scope-free"
