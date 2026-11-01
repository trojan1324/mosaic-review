"""Inventory an explicitly selected local model tree without downloading it."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import re
from typing import Mapping, Any

from mosaic.public_runtime_manifest import VERSION, _root, _relative, _need, validate_manifest
from mosaic.workspace import Workspace


def inventory_local_model(declaration: Mapping[str, Any], weights_root: Path, output: Path,
                          *, workspace: Workspace) -> dict:
    """Compute actual local file hashes; runtime/hardware fields remain declarations.

    The revision identifies the caller's intended acquisition. Local hashing
    alone does not establish that those bytes came from that upstream revision.
    """
    _need(isinstance(declaration, Mapping) and set(declaration) == {"version", "model", "runtime", "hardware"}
          and declaration["version"] == "mosaic-public-model-declaration/v1", "public_model_declaration_invalid")
    model = declaration["model"]
    _need(isinstance(model, Mapping) and set(model) == {"id", "revision", "tokenizer_path",
          "tokenizer_runtime_version", "add_special_tokens"}, "public_model_declaration_fields_invalid")
    _need(isinstance(model["revision"], str) and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", model["revision"]) is not None,
          "public_model_revision_not_immutable")
    root = _root(weights_root)
    _need(not output.absolute().is_relative_to(root), "public_model_manifest_inside_weights_refused")
    paths = sorted(root.rglob("*"))
    _need(not any(p.is_symlink() or getattr(p, "is_junction", lambda: False)() for p in paths),
          "public_model_link_refused")
    rows = []
    for path in paths:
        if not path.is_file():
            continue
        name = path.relative_to(root).as_posix()
        _relative(name)
        _need(not any(part.casefold().startswith(".env") or part.casefold() == ".git" for part in Path(name).parts),
              "public_model_private_file_refused")
        hasher = sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        rows.append({"path": name, "size": path.stat().st_size, "sha256": hasher.hexdigest()})
    tokenizer = next((r for r in rows if r["path"] == model["tokenizer_path"]), None)
    _need(tokenizer is not None, "public_model_tokenizer_missing")
    manifest = {"version": VERSION, "model": {k: deepcopy(v) for k, v in model.items() if k != "tokenizer_path"},
                "runtime": deepcopy(declaration["runtime"]), "hardware": deepcopy(declaration["hardware"])}
    manifest["model"].update(files=rows, tokenizer={"path": tokenizer["path"], "sha256": tokenizer["sha256"]})
    verified = validate_manifest(manifest, weights_root=root)
    result_path = workspace.write_canonical_json(output, manifest)
    return {"manifest_path": str(result_path), "manifest_sha256": sha256(result_path.read_bytes()).hexdigest(),
            "model_files_sha256": verified["model_files_sha256"], "file_count": len(rows),
            "upstream_provenance_verified": False, "runtime_and_hardware_verified": False,
            "network_calls": 0, "model_calls": 0}
