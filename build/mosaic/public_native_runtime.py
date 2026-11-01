"""Public native-runtime preparation boundary; no execution API."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from importlib import metadata
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import urlsplit

from mosaic.canonical import canonical_bytes
from mosaic.public_runtime_manifest import validate_manifest
from mosaic.workspace import Workspace


VERSION = "mosaic-public-native-preparation/v1"
_HEX = set("0123456789abcdef")


class PublicNativePreparationRefusal(ValueError):
    pass


def _need(value: bool, code: str) -> None:
    if not value:
        raise PublicNativePreparationRefusal(code)


def _sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _digest(value: Any) -> str:
    return sha256(canonical_bytes(value)).hexdigest()


def _no_links(path: Path) -> None:
    current = path
    while True:
        _need(not current.is_symlink() and not getattr(current, "is_junction", lambda: False)(),
              "public_native_capture_link_refused")
        if current == current.parent:
            return
        current = current.parent


def _read(root: Path, name: str) -> dict[str, Any]:
    path = root / name
    _need(path.is_file(), "public_native_capture_artifact_missing")
    _no_links(path)
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise PublicNativePreparationRefusal("public_native_capture_artifact_invalid") from exc
    _need(isinstance(value, dict), "public_native_capture_artifact_invalid")
    return value


def load_capture_envelope(capture_root: Path | str, *, workspace: Workspace) -> dict[str, Any]:
    """Load exactly four public-capture JSON artifacts plus the frozen PDF hash."""
    root = Path(capture_root)
    _need(root.is_absolute() and root.is_dir(), "public_native_capture_root_invalid")
    _no_links(root)
    receipt = _read(root, "receipt.json")
    source = _read(root, "source.json")
    binding = _read(root, "source-binding.json")
    upstream = _read(root, "prepare-upstream-input.json")
    pdf = root / "source.pdf"
    _need(pdf.is_file(), "public_native_capture_source_pdf_missing")
    _no_links(pdf)
    _need(receipt.get("version") == "public-capture-source-packet/v1"
          and receipt.get("status") == "source_packet_ready", "public_native_capture_not_ready")
    work, pdf_sha, source_sha = receipt.get("work_id"), receipt.get("source_pdf_sha256"), receipt.get("source_packet_sha256")
    _need(isinstance(work, str) and _sha(pdf_sha) and _sha(source_sha) and work == "public-" + pdf_sha,
          "public_native_capture_identity_invalid")
    _need(sha256(pdf.read_bytes()).hexdigest() == pdf_sha, "public_native_capture_pdf_hash_mismatch")
    _need(_digest(source) == source_sha and binding.get("source_packet_sha256") == source_sha,
          "public_native_capture_source_hash_mismatch")
    _need(source.get("work_id") == work and binding.get("work_id") == work
          and upstream.get("paper_id") == work and upstream.get("source_pdf_sha256") == pdf_sha
          and upstream.get("source") == source, "public_native_capture_binding_mismatch")
    return {"capture_receipt": receipt, "source": source, "source_binding": binding,
            "prepare_upstream_input": upstream}


def _loopback(value: Any) -> str:
    _need(isinstance(value, str) and value, "public_native_loopback_url_required")
    _need(re.fullmatch(r"http://127\.0\.0\.1(?::[1-9][0-9]{0,4})?(?:/v1)?/?", value) is not None,
          "public_native_loopback_url_refused")
    parsed = urlsplit(value)
    _need(parsed.scheme == "http" and parsed.hostname == "127.0.0.1" and parsed.username is None
          and parsed.password is None and not parsed.query and not parsed.fragment
          and parsed.path in ("", "/", "/v1", "/v1/"), "public_native_loopback_url_refused")
    try:
        port = parsed.port
    except ValueError as exc:
        raise PublicNativePreparationRefusal("public_native_loopback_url_refused") from exc
    _need(port is None or 1 <= port <= 65535, "public_native_loopback_url_refused")
    return "http://127.0.0.1" + (":" + str(port) if port is not None else "") + "/v1"


def _verify_tokenizer(manifest: Mapping[str, Any], preflight: Mapping[str, Any]) -> dict[str, Any]:
    """Prove the installed tokenizer package can parse the pinned local bytes."""
    model = manifest["model"]
    tokenizer = model["tokenizer"]
    try:
        installed = metadata.version("tokenizers")
        from tokenizers import Tokenizer
    except (ImportError, metadata.PackageNotFoundError) as exc:
        raise PublicNativePreparationRefusal("public_native_tokenizer_runtime_missing") from exc
    _need(installed == model["tokenizer_runtime_version"], "public_native_tokenizer_runtime_mismatch")
    path = Path(preflight["weights_root"]) / tokenizer["path"]
    data = path.read_bytes()
    _need(sha256(data).hexdigest() == tokenizer["sha256"], "public_native_tokenizer_hash_mismatch")
    try:
        parsed = Tokenizer.from_str(data.decode("utf-8"))
    except Exception as exc:
        raise PublicNativePreparationRefusal("public_native_tokenizer_parse_refused") from exc
    _need(parsed.truncation is None and parsed.padding is None, "public_native_tokenizer_parse_refused")
    return {"path": tokenizer["path"], "sha256": tokenizer["sha256"], "runtime_version": installed,
            "parsed": True, "add_special_tokens": model["add_special_tokens"]}


def _profile(value: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    _need(isinstance(value, Mapping) and value.get("version") == "mosaic-public-native-driver-profile/v1",
          "public_native_driver_profile_invalid")
    value = dict(value)
    model, driver, source = value.get("model"), value.get("driver"), value.get("source")
    _need(isinstance(model, Mapping) and isinstance(driver, Mapping) and isinstance(source, Mapping),
          "public_native_driver_profile_invalid")
    manifest_model = manifest.get("model")
    _need(isinstance(manifest_model, Mapping) and model.get("id") == manifest_model.get("id")
          and model.get("revision") == manifest_model.get("revision")
          and isinstance(model.get("served_model_name"), str) and model["served_model_name"],
          "public_native_model_manifest_mismatch")
    _need(source.get("permission_declaration") == "user_declared_permitted", "public_native_permission_required")
    for key in ("candidate", "run_id"):
        _need(isinstance(driver.get(key), str) and driver[key], "public_native_driver_profile_invalid")
    _need(type(driver.get("max_measure_stages_per_paper")) is int and driver["max_measure_stages_per_paper"] >= 0,
          "public_native_driver_limit_invalid")
    for key in ("request_cap", "output_token_cap", "article_request_cap"):
        number = driver.get(key)
        _need(type(number) is int and number > 0,
              "public_native_driver_limit_invalid")
    for key in ("max_request_seconds", "wall_clock_cap_seconds"):
        number = driver.get(key)
        _need(type(number) in (int, float) and math.isfinite(number) and number > 0,
              "public_native_driver_limit_invalid")
    context_limit = driver.get("context_limit", 98304)
    _need(type(context_limit) is int and context_limit > 6144,
          "public_native_driver_limit_invalid")
    driver = {**driver, "context_limit": context_limit}
    value["driver"] = driver
    return value


class PublicLocalScope:
    """Native local preparation scope; it carries no authority to execute."""
    def __init__(self, work_id: str, source_pdf_sha256: str) -> None:
        self.work_id, self.source_pdf_sha256 = work_id, source_pdf_sha256

    def check(self, work_id, hashes, provenance, *, remote, rights) -> None:
        _need(work_id == self.work_id and hashes == [self.source_pdf_sha256], "public_native_scope_identity_refused")
        _need(provenance == "native_article_text" and remote is False,
              "public_native_scope_transport_refused")
        _need(rights == ["public_local_processing_user_declared"], "public_native_scope_rights_refused")


def prepare_local_program(capture_envelope: Mapping[str, Any], local_model_manifest: Mapping[str, Any], weights_root: Path | str,
                          driver_profile: Mapping[str, Any], base_url: str, output: Path | str, *, workspace: Workspace) -> dict[str, Any]:
    """Freeze a native program and receipt; never call the model driver or a socket."""
    _need(isinstance(capture_envelope, Mapping), "public_native_capture_envelope_required")
    receipt, source, binding, upstream = (capture_envelope.get(name) for name in
        ("capture_receipt", "source", "source_binding", "prepare_upstream_input"))
    _need(all(isinstance(value, Mapping) for value in (receipt, source, binding, upstream)),
          "public_native_capture_envelope_required")
    # Reuse the loader's exact checks without trusting a path supplied in the envelope.
    work, pdf_sha, source_sha = receipt.get("work_id"), receipt.get("source_pdf_sha256"), receipt.get("source_packet_sha256")
    _need(receipt.get("version") == "public-capture-source-packet/v1" and receipt.get("status") == "source_packet_ready" and isinstance(work, str) and _sha(pdf_sha)
          and _sha(source_sha) and work == "public-" + pdf_sha and receipt.get("scientifically_accepted") is False,
          "public_native_capture_identity_invalid")
    _need(_digest(source) == source_sha and binding.get("source_packet_sha256") == source_sha
          and source.get("work_id") == work and binding.get("work_id") == work
          and upstream.get("paper_id") == work and upstream.get("source_pdf_sha256") == pdf_sha
          and upstream.get("source") == source, "public_native_capture_binding_mismatch")
    preflight = validate_manifest(local_model_manifest, weights_root=weights_root)
    profile = _profile(driver_profile, local_model_manifest)
    endpoint = _loopback(base_url)
    tokenizer_check = _verify_tokenizer(local_model_manifest, preflight)
    manifest_model = local_model_manifest["model"]
    tokenizer = manifest_model["tokenizer"]
    tokenizer_path = Path(preflight["weights_root"]) / tokenizer["path"]
    recipe = {"vllm": {"model": manifest_model["id"], "revision": manifest_model["revision"],
              "served_model_name": profile["model"]["served_model_name"],
              "local_tokenizer": {"model": manifest_model["id"], "revision": manifest_model["revision"],
                  "runtime_version": manifest_model["tokenizer_runtime_version"], "path": str(tokenizer_path),
                  "sha256": tokenizer["sha256"], "add_special_tokens": manifest_model["add_special_tokens"]}}}
    # The real runner will reconstruct this counter through its own default
    # Workspace.  Prove that location/bytes/parser contract now, without a
    # socket or inference call, instead of freezing a program it will reject.
    from mosaic.vllm.tokenizer import TokenizerRefused, local_tokenizer
    try:
        counter = local_tokenizer(recipe["vllm"])
        _need(callable(counter) and type(counter("native preparation check")) is int,
              "public_native_tokenizer_counter_refused")
    except (PermissionError, TokenizerRefused) as exc:
        raise PublicNativePreparationRefusal("public_native_tokenizer_workspace_or_runtime_refused") from exc
    from scripts.complete_extraction_runner import prepare_program
    from scripts.complete_extraction_sequence import prepare_upstream
    plan = prepare_upstream(recipe, dict(upstream))
    limits = profile["driver"]
    paper = {"work_id": work, "plan": plan, "source_sha256s": [pdf_sha],
             "text_provenance": "native_article_text", "rights": ["public_local_processing_user_declared"], "remote": False}
    program = prepare_program([paper], recipe, candidate=limits["candidate"], run_id=limits["run_id"],
                              max_measure_stages_per_paper=limits["max_measure_stages_per_paper"], max_workers=1,
                              max_request_seconds=limits["max_request_seconds"], max_attempts=1, fixture_mode=False,
                              context_limit=limits["context_limit"])
    scope = PublicLocalScope(work, pdf_sha)
    scope.check(work, paper["source_sha256s"], paper["text_provenance"], remote=False, rights=paper["rights"])
    root = workspace.mkdir(output)
    manifest_path = workspace.write_canonical_json(root / "local-model-manifest.json", local_model_manifest)
    profile_path = workspace.write_canonical_json(root / "driver-profile.json", profile)
    preflight_path = workspace.write_canonical_json(root / "runtime-preflight.json", {**preflight, "tokenizer_checked": tokenizer_check})
    program_path = workspace.write_canonical_json(root / "prepared-program.json", program)
    prepared = {"version": VERSION, "status": "prepared_only_execution_unvalidated", "execution_api": "intentionally_absent",
                "capture": {"work_id": work, "source_pdf_sha256": pdf_sha, "source_packet_sha256": source_sha},
                "endpoint": endpoint, "manifest_preflight_sha256": sha256(preflight_path.read_bytes()).hexdigest(),
                "local_model_manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
                "driver_profile_sha256": sha256(profile_path.read_bytes()).hexdigest(),
                "program_sha256": sha256(program_path.read_bytes()).hexdigest(), "scope": "public_local_exact_source_only",
                "tokenizer_counter_checked": True,
                "scientifically_accepted": False, "qualified": False, "model_calls": 0, "network_calls": 0,
                "limitations": ["No socket or model execution occurs during preparation.",
                                "Manifest declarations are not installed-runtime or hardware verification.",
                                "Linux/GPU smoke execution remains unobserved."]}
    receipt_path = workspace.write_canonical_json(root / "prepared-receipt.json", prepared)
    return {"preflight_path": str(preflight_path), "program_path": str(program_path),
            "receipt_path": str(receipt_path), "receipt": deepcopy(prepared)}
