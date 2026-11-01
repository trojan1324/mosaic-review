from hashlib import sha256
from pathlib import Path
import json

import pytest

from mosaic.public_model_inventory import inventory_local_model
from mosaic.public_runtime_manifest import PublicRuntimeManifestRefusal
from test_public_execution import _prepared

pytestmark = pytest.mark.evidence_mock


def test_inventory_hashes_actual_selected_tree_and_refuses_silent_replacement(tmp_path):
    workspace, _, prepared = _prepared(tmp_path)
    old = json.loads((prepared / "local-model-manifest.json").read_bytes())
    model = {k: v for k, v in old["model"].items() if k not in {"files", "tokenizer"}}
    model.update(revision="a" * 40, tokenizer_path="tokenizer.json")
    declaration = {"version": "mosaic-public-model-declaration/v1", "model": model,
                   "runtime": old["runtime"], "hardware": old["hardware"]}
    path = tmp_path / "inventoried.json"
    result = inventory_local_model(declaration, tmp_path / "weights", path, workspace=workspace)
    assert result["file_count"] == 2 and result["network_calls"] == 0
    assert not result["upstream_provenance_verified"] and not result["runtime_and_hardware_verified"]
    manifest = json.loads(path.read_bytes())
    for row in manifest["model"]["files"]:
        assert sha256((tmp_path / "weights" / row["path"]).read_bytes()).hexdigest() == row["sha256"]
    assert inventory_local_model(declaration, tmp_path / "weights", path, workspace=workspace) == result
    with pytest.raises(PublicRuntimeManifestRefusal, match="inside_weights"):
        inventory_local_model(declaration, tmp_path / "weights", tmp_path / "weights/manifest.json", workspace=workspace)
    (tmp_path / "weights/extra.bin").write_bytes(b"extra fictional bytes")
    with pytest.raises(Exception, match="immutable artifact differs"):
        inventory_local_model(declaration, tmp_path / "weights", path, workspace=workspace)
    assert json.loads(path.read_bytes()) == manifest
