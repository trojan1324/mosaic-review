"""Validation-only manifest for a future local public inference runtime.

No function here starts a server, downloads a model, opens a socket, or treats
a declaration as hardware evidence.  It validates an explicit local tree so a
separate reviewed Linux smoke can later bind to stable bytes.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping


VERSION = "mosaic-public-local-model-manifest/v1"
_HEX = set("0123456789abcdef")
_PRIVATE_PARTS = {"sealed", "references", "private", "holdback", "jev_eval", "evoke"}


class PublicRuntimeManifestRefusal(ValueError):
    """A local runtime manifest cannot establish the requested boundary."""


def _need(condition: bool, code: str) -> None:
    if not condition:
        raise PublicRuntimeManifestRefusal(code)


def _sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _relative(value: Any) -> Path:
    _need(isinstance(value, str) and value and "\\" not in value and ":" not in value and "\x00" not in value,
          "public_runtime_manifest_relative_path_required")
    posix, windows = PurePosixPath(value), PureWindowsPath(value)
    _need(not posix.is_absolute() and not windows.is_absolute() and not windows.drive
          and ".." not in posix.parts and all(part not in ("", ".") for part in value.split("/")),
          "public_runtime_manifest_relative_path_required")
    _need(not any(part.casefold() in _PRIVATE_PARTS for part in posix.parts), "public_runtime_manifest_private_path_refused")
    return Path(*posix.parts)


def _under(child: Path, parent: Path) -> bool:
    return child == parent or str(child).startswith(str(parent) + os.sep)


def _root(value: Path | str) -> Path:
    path = Path(value)
    _need(path.is_absolute() and path.is_dir(), "public_runtime_manifest_weights_root_invalid")
    current = path
    while True:
        _need(not current.is_symlink() and not getattr(current, "is_junction", lambda: False)(),
              "public_runtime_manifest_weights_root_invalid")
        if current == current.parent:
            break
        current = current.parent
    real = Path(os.path.realpath(path))
    _need(not any(part.casefold() in _PRIVATE_PARTS for part in real.parts), "public_runtime_manifest_private_path_refused")
    return real


def _file(root: Path, relative: str, size: Any, digest: Any) -> Path:
    _need(type(size) is int and size >= 0 and _sha(digest), "public_runtime_manifest_file_metadata_invalid")
    raw = root / _relative(relative)
    _need(not raw.is_symlink(), "public_runtime_manifest_file_invalid")
    target = Path(os.path.realpath(raw))
    _need(_under(target, root) and target.is_file() and not target.is_symlink(), "public_runtime_manifest_file_invalid")
    _need(target.stat().st_size == size, "public_runtime_manifest_file_size_mismatch")
    hasher = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    actual = hasher.hexdigest()
    _need(actual == digest, "public_runtime_manifest_file_hash_mismatch")
    return target


def validate_manifest(manifest: Mapping[str, Any], *, weights_root: Path | str) -> dict[str, Any]:
    """Validate all and only an explicitly selected local model tree.

    An observed-smoke declaration remains untrusted here: this pure preflight
    cannot establish that a Linux GPU actually ran it.  Therefore every result
    deliberately reports ``supported_runtime=False``.
    """
    _need(isinstance(manifest, Mapping) and manifest.get("version") == VERSION,
          "public_runtime_manifest_version_invalid")
    model, runtime, hardware = manifest.get("model"), manifest.get("runtime"), manifest.get("hardware")
    _need(isinstance(model, Mapping) and isinstance(runtime, Mapping) and isinstance(hardware, Mapping),
          "public_runtime_manifest_sections_required")
    _need(isinstance(model.get("id"), str) and model["id"] and isinstance(model.get("revision"), str) and model["revision"],
          "public_runtime_manifest_model_identity_invalid")
    _need(type(model.get("add_special_tokens")) is bool and isinstance(model.get("tokenizer_runtime_version"), str)
          and model["tokenizer_runtime_version"], "public_runtime_manifest_tokenizer_contract_invalid")
    files = model.get("files")
    _need(isinstance(files, list) and files, "public_runtime_manifest_file_roster_required")
    paths = [row.get("path") if isinstance(row, Mapping) else None for row in files]
    _need(all(isinstance(path, str) for path in paths) and len(set(paths)) == len(paths),
          "public_runtime_manifest_file_roster_invalid")
    for path in paths:
        _relative(path)
    root = _root(weights_root)
    discovered = list(root.rglob("*"))
    _need(not any(path.is_symlink() or getattr(path, "is_junction", lambda: False)() for path in discovered),
          "public_runtime_manifest_file_invalid")
    actual_paths = {path.relative_to(root).as_posix() for path in discovered if path.is_file()}
    _need(actual_paths == set(paths), "public_runtime_manifest_extra_or_missing_weight_file")
    checked = [_file(root, row["path"], row.get("size"), row.get("sha256")) for row in files]
    tokenizer = model.get("tokenizer")
    _need(isinstance(tokenizer, Mapping) and tokenizer.get("path") in set(paths) and _sha(tokenizer.get("sha256")),
          "public_runtime_manifest_tokenizer_invalid")
    tokenizer_row = next(row for row in files if row["path"] == tokenizer["path"])
    _need(tokenizer_row["sha256"] == tokenizer["sha256"], "public_runtime_manifest_tokenizer_hash_mismatch")
    _need(runtime.get("platform") == "linux" and _sha(runtime.get("container_digest")),
          "public_runtime_manifest_runtime_identity_invalid")
    wheels = runtime.get("wheels")
    _need(isinstance(wheels, list) and wheels and all(isinstance(row, Mapping) and isinstance(row.get("name"), str)
          and row["name"] and _sha(row.get("sha256")) for row in wheels), "public_runtime_manifest_wheel_lock_invalid")
    _need(isinstance(hardware.get("declared"), Mapping) and hardware["declared"],
          "public_runtime_manifest_hardware_declaration_required")
    smoke = hardware.get("observed_linux_gpu_smoke_receipt")
    smoke_declared = isinstance(smoke, Mapping) and _sha(smoke.get("sha256"))
    return {"version": VERSION, "weights_root": str(root), "model_id": model["id"], "revision": model["revision"],
            "model_file_count": len(checked), "model_files_sha256": hashlib.sha256(
                "".join(f"{row['path']}:{row['sha256']}:{row['size']}\n" for row in sorted(files, key=lambda x: x["path"])).encode()).hexdigest(),
            "tokenizer": {"path": tokenizer["path"], "sha256": tokenizer["sha256"],
                          "runtime_version": model["tokenizer_runtime_version"],
                          "add_special_tokens": model["add_special_tokens"]},
            "grammar_contract": "xgrammar Grammar.from_json_schema then Grammar.from_ebnf; enable_thinking=false",
            "smoke_receipt_declared_untrusted": smoke_declared, "supported_runtime": False,
            "reason": "A manifest preflight cannot establish an observed Linux/GPU smoke run."}


def render_launch_plan(preflight: Mapping[str, Any]) -> dict[str, Any]:
    """Render, but never execute, the exact local-only reference launch plan."""
    _need(isinstance(preflight, Mapping) and preflight.get("version") == VERSION
          and preflight.get("supported_runtime") is False, "public_runtime_manifest_preflight_required")
    return {"status": "UNVALIDATED_DO_NOT_EXECUTE", "network": {"host": "127.0.0.1", "remote": False,
            "transport": "local_linux_loopback_only"}, "command": ["python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", preflight["weights_root"], "--served-model-name", preflight["model_id"], "--host", "127.0.0.1"],
            "driver_contract": preflight["grammar_contract"], "blocked_by": preflight["reason"]}
