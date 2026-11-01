"""Linux-only, CPU-only preflight for the public open-weights runtime.

This is deliberately before GPU allocation, model download, PDF capture, or
MOSAIC extraction.  It exercises the narrow orphan-package repair in the exact
declared base environment and leaves immutable receipts for a later model smoke.
An image string is only a declaration until an external container/runtime
receipt binds it to an observed image ID or digest.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Callable

from mosaic.public_linux_orphan_dependencies import (
    FILE_MANIFEST_CODE,
    ORPHAN_IMPORT_CODE,
    PDF_NAME,
    PDF_WHEEL,
    PDF_WHEEL_SHA,
    PDF_WHEEL_URL,
    TARGET,
    TARGET_ABSENCE_CODE,
    TARGET_RECORD_CODE,
    OrphanRepairRefusal,
    _only_nccl,
    _raw,
    _validate_pdf_metadata,
    load_plan,
    parse_freeze,
    replay_graph,
    run_orphan_dependency_gate,
    validate_baseline,
    validate_file_manifests,
    validate_orphan_pdf_stage,
    validate_record_ownership,
)
from mosaic.workspace import Workspace


VERSION = "mosaic-public-linux-environment/v1"
MAX_PREFLIGHT_SECONDS = 900
PDF_IMPORT_CODE = "from importlib.metadata import version; import fitz; assert version('pymupdf') == '1.28.2'; print('pymupdf-import-ok')"
PDF_IMPORT_STDOUT = (
    b"pymupdf-import-ok",
    b"warning: The `fitz` API is deprecated and will be removed in future. Use `import pymupdf` instead.\npymupdf-import-ok",
)
CORE_RUNTIME_IMPORT_CODE = (
    "import importlib.metadata as m, json, torch, tokenizers, xgrammar; "
    "print(json.dumps({'metadata_versions': {n:m.version(n) for n in ('torch','tokenizers','xgrammar')}, "
    "'module_versions': {'torch':torch.__version__, 'tokenizers':tokenizers.__version__}, "
    "'torch_cuda':torch.version.cuda}, sort_keys=True))"
)


class PublicLinuxEnvironmentRefusal(ValueError):
    """The CPU-only environment preflight cannot safely continue."""


def _need(value: bool, code: str) -> None:
    if not value:
        raise PublicLinuxEnvironmentRefusal(code)


def _sha(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _receipt(output: Path, label: str) -> dict[str, Any]:
    value = json.loads((output / f"{label}.json").read_bytes())
    _need(isinstance(value, dict), "linux_environment_receipt_invalid")
    return value


def _validate_command_semantics(label: str, command: list[str], *, ownership: dict[str, Any] | None = None) -> None:
    """Keep retained receipts meaningful even if their hashes are recomputed."""
    _need(len(command) >= 2, "linux_environment_retained_command_semantics_invalid")
    tail = command[1:]
    if label.endswith("-freeze") or label == "installed":
        expected = ["-m", "pip", "freeze", "--all"]
    elif label.endswith("-metadata"):
        expected = ["-m", "pip", "inspect", "--local"]
    elif label.endswith("-check"):
        expected = ["-m", "pip", "check"]
    elif label == "orphan-target-record":
        expected = ["-c", TARGET_RECORD_CODE]
    elif label in {"orphan-before-files", "orphan-after-files"}:
        expected = ["-c", FILE_MANIFEST_CODE]
    elif label == "orphan-uninstall":
        expected = ["-m", "pip", "uninstall", "--yes", TARGET]
    elif label == "orphan-post-imports":
        expected = ["-c", ORPHAN_IMPORT_CODE]
    elif label == "orphan-post-remove-absence":
        _need(isinstance(ownership, dict), "linux_environment_retained_command_semantics_invalid")
        record = validate_record_ownership(str(ownership.get("record", "")).encode())
        expected = ["-c", TARGET_ABSENCE_CODE, ownership.get("root"), json.dumps(sorted(record))]
    elif label == "install":
        _need(tail[:6] == ["-m", "pip", "install", "--no-deps", "--require-hashes", "-r"]
              and len(tail) == 7 and tail[-1].replace("\\", "/").rsplit("/", 1)[-1]
              == "requirements-linux-preflight.lock", "linux_environment_retained_command_semantics_invalid")
        return
    elif label == "post-pdf-imports":
        expected = ["-c", PDF_IMPORT_CODE]
    elif label == "core-runtime-imports":
        expected = ["-c", CORE_RUNTIME_IMPORT_CODE]
    else:
        raise PublicLinuxEnvironmentRefusal("linux_environment_retained_command_semantics_invalid")
    _need(tail == expected, "linux_environment_retained_command_semantics_invalid")


def _run_factory(*, workspace: Workspace, output: Path, deadline_epoch: float,
                 run_process: Callable[..., subprocess.CompletedProcess[bytes]]) -> tuple[Callable[..., dict[str, Any]], list[dict[str, Any]]]:
    commands: list[dict[str, Any]] = []

    def run(command: list[str], label: str, timeout: int, *, require_success: bool = True) -> dict[str, Any]:
        _need(not (output / f"{label}.json").exists(), "linux_environment_duplicate_command")
        remaining = deadline_epoch - time.time()
        _need(remaining > 1, "linux_environment_deadline_expired")
        effective_timeout = max(1, min(timeout, int(remaining)))
        timed_out = False
        try:
            completed = run_process(command, capture_output=True, timeout=effective_timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            completed = subprocess.CompletedProcess(command, -1, exc.stdout or b"", exc.stderr or b"")
        stdout = completed.stdout if isinstance(completed.stdout, bytes) else str(completed.stdout or "").encode()
        stderr = completed.stderr if isinstance(completed.stderr, bytes) else str(completed.stderr or "").encode()
        workspace.write_once(output / f"{label}.stdout.log", stdout)
        workspace.write_once(output / f"{label}.stderr.log", stderr)
        receipt = {"command": list(command), "returncode": completed.returncode, "timed_out": timed_out,
                   "stdout_sha256": _sha(stdout), "stderr_sha256": _sha(stderr)}
        workspace.write_canonical_json(output / f"{label}.json", receipt)
        commands.append(receipt)
        _need(not timed_out, "linux_environment_command_timeout")
        _need(not require_success or completed.returncode == 0, "linux_environment_command_failed")
        return receipt

    return run, commands


def _validate_core_runtime_import(raw: bytes) -> dict[str, Any]:
    """Accept import metadata without initializing CUDA or a vLLM engine."""
    try:
        observed = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise PublicLinuxEnvironmentRefusal("linux_environment_core_runtime_import_invalid") from exc
    _need(isinstance(observed, dict)
          and set(observed) == {"metadata_versions", "module_versions", "torch_cuda"},
          "linux_environment_core_runtime_import_invalid")
    metadata_versions, module_versions = observed["metadata_versions"], observed["module_versions"]
    _need(isinstance(metadata_versions, dict) and set(metadata_versions) == {"torch", "tokenizers", "xgrammar"}
          and all(isinstance(value, str) and value for value in metadata_versions.values())
          and isinstance(module_versions, dict) and set(module_versions) == {"torch", "tokenizers"}
          and all(isinstance(value, str) and value for value in module_versions.values())
          and all(module_versions[name] == metadata_versions[name] for name in module_versions)
          and (observed["torch_cuda"] is None or isinstance(observed["torch_cuda"], str)),
          "linux_environment_core_runtime_import_invalid")
    return observed


def run_linux_environment_preflight(output: Path, *, deadline_epoch: float,
                                    declared_image: str | None = None,
                                    workspace: Workspace | None = None,
                                    run_process: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run) -> dict[str, Any]:
    """Exercise the dependency repair in a Linux process with no GPU/model work.

    ``declared_image`` must match the repair plan.  It is not evidence that this
    process actually came from that image; container ownership is intentionally
    recorded as unverified until an external Docker receipt binds it.
    """
    _need(platform.system() == "Linux", "linux_environment_linux_required")
    _need(0 < deadline_epoch - time.time() <= MAX_PREFLIGHT_SECONDS,
          "linux_environment_deadline_invalid")
    plan = load_plan()
    image = declared_image or plan.image
    _need(image == plan.image, "linux_environment_declared_image_mismatch")
    if workspace is None:
        workspace = Workspace()
    destination = workspace.check(output)
    _need(not destination.exists(), "linux_environment_output_must_be_new")
    workspace.mkdir(destination)
    run, commands = _run_factory(workspace=workspace, output=destination, deadline_epoch=deadline_epoch,
                                 run_process=run_process)
    try:
        gate = run_orphan_dependency_gate(run=run, python=sys.executable, output=destination,
                                          workspace=workspace, image=image, plan=plan)
        before_pdf = parse_freeze((destination / "post-repair-freeze.stdout.log").read_bytes())
        lock = workspace.write_text(destination / "requirements-linux-preflight.lock",
                                    f"pymupdf @ {PDF_WHEEL_URL} --hash=sha256:{PDF_WHEEL_SHA}\n")
        run([sys.executable, "-m", "pip", "install", "--no-deps", "--require-hashes", "-r", str(lock)],
            "install", 120)
        run([sys.executable, "-m", "pip", "freeze", "--all"], "installed", 30)
        pdf_stage = validate_orphan_pdf_stage(run=run, python=sys.executable, output=destination,
                                              workspace=workspace, before=before_pdf, plan=plan)
        run([sys.executable, "-c", PDF_IMPORT_CODE], "post-pdf-imports", 30)
        # Importing the large Torch/CUDA libraries can exceed 30 seconds on a
        # cold, CPU-limited local VM. The overall 900-second deadline still
        # bounds this child; no model loading or GPU initialization is added.
        run([sys.executable, "-c", CORE_RUNTIME_IMPORT_CODE], "core-runtime-imports", 120)
        runtime_observation = _validate_core_runtime_import((destination / "core-runtime-imports.stdout.log").read_bytes())
        result = {
            "version": VERSION,
            "status": "environment_preflight_completed",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "declared_image": image,
            "image_provenance": "declared_not_observed; bind an external container receipt before treating this as exact-image evidence",
            "platform": {"system": platform.system(), "python": sys.version},
            "plan_sha256": plan.config_sha256,
            "dependency_gate_sha256": _sha((destination / "dependency-gate.json").read_bytes()),
            "dependency_pdf_stage_sha256": _sha((destination / "dependency-pdf-stage.json").read_bytes()),
            "core_runtime_import_sha256": _sha((destination / "core-runtime-imports.json").read_bytes()),
            "core_runtime": runtime_observation,
            "commands": commands,
            "gpu_observed": False,
            "model_downloaded": False,
            "model_inference_run": False,
            "scientifically_accepted": False,
            "limitations": ["No GPU, model weights, PDF capture, extraction, or serving occurred.",
                            "The declared image is not proven by this process without an external runtime receipt."],
        }
        workspace.write_canonical_json(destination / "preflight-outcome.json", result)
        return result
    except Exception as exc:
        failure = {"version": VERSION, "status": "environment_preflight_refused",
                   "declared_image": image, "plan_sha256": plan.config_sha256,
                   "reason": str(exc), "commands": commands, "gpu_observed": False,
                   "model_downloaded": False, "model_inference_run": False,
                   "scientifically_accepted": False}
        workspace.write_canonical_json(destination / "preflight-outcome.json", failure)
        raise


def verify_retained_linux_environment_preflight(output: Path, *, expected_config_sha256: str) -> dict[str, Any]:
    """Replay retained CPU preflight evidence without running pip or importing vLLM."""
    root = Path(output)
    outcome = json.loads((root / "preflight-outcome.json").read_bytes())
    _need(isinstance(outcome, dict) and outcome.get("version") == VERSION
          and outcome.get("status") == "environment_preflight_completed", "linux_environment_retained_outcome_invalid")
    plan = load_plan()
    _need(plan.config_sha256 == expected_config_sha256 and outcome.get("plan_sha256") == expected_config_sha256
          and outcome.get("declared_image") == plan.image, "linux_environment_retained_plan_invalid")
    _need(outcome.get("image_provenance") == "declared_not_observed; bind an external container receipt before treating this as exact-image evidence"
          and outcome.get("gpu_observed") is False and outcome.get("model_downloaded") is False
          and outcome.get("model_inference_run") is False and outcome.get("scientifically_accepted") is False,
          "linux_environment_retained_scope_invalid")
    labels = ["baseline-freeze", "baseline-metadata", "baseline-check", "orphan-target-record", "orphan-before-files",
              "orphan-uninstall", "orphan-post-remove-absence", "post-repair-freeze", "post-repair-metadata",
              "post-repair-check", "orphan-after-files", "orphan-post-imports", "install", "installed",
              "post-pdf-freeze", "post-pdf-metadata", "post-pdf-check", "post-pdf-imports", "core-runtime-imports"]
    commands = outcome.get("commands")
    _need(isinstance(commands, list) and len(commands) == len(labels), "linux_environment_retained_commands_invalid")
    ownership_for_semantics = None
    receipts: dict[str, dict[str, Any]] = {}
    for index, label in enumerate(labels):
        receipt = _receipt(root, label)
        _need(commands[index] == receipt and isinstance(receipt.get("command"), list)
              and all(isinstance(value, str) for value in receipt["command"])
              and receipt.get("timed_out") is False
              and receipt.get("returncode") == (1 if label.endswith("-check") else 0),
              "linux_environment_retained_command_invalid")
        for kind in ("stdout", "stderr"):
            raw = _raw(root, label, kind)
            _need(receipt.get(f"{kind}_sha256") == _sha(raw), "linux_environment_retained_command_hash_invalid")
        if label == "orphan-target-record":
            ownership_for_semantics = json.loads(_raw(root, label, "stdout"))
        _validate_command_semantics(label, receipt["command"], ownership=ownership_for_semantics)
        receipts[label] = receipt
    before = parse_freeze(_raw(root, "baseline-freeze", "stdout"))
    metadata = _raw(root, "baseline-metadata", "stdout")
    validate_baseline(freeze=before, metadata=metadata,
                      check={**receipts["baseline-check"], "stdout": _raw(root, "baseline-check", "stdout")}, plan=plan)
    after = parse_freeze(_raw(root, "post-repair-freeze", "stdout"))
    _need({name: version for name, version in before.items() if name != TARGET} == after,
          "linux_environment_retained_distribution_delta_invalid")
    _only_nccl({**receipts["post-repair-check"], "stdout": _raw(root, "post-repair-check", "stdout")},
               replay_graph(_raw(root, "post-repair-metadata", "stdout"), after))
    ownership = json.loads(_raw(root, "orphan-target-record", "stdout"))
    record = validate_record_ownership(str(ownership.get("record", "")).encode())
    files = ownership.get("files")
    before_manifest = json.loads(_raw(root, "orphan-before-files", "stdout"))
    after_manifest = json.loads(_raw(root, "orphan-after-files", "stdout"))
    _need(isinstance(files, dict) and isinstance(before_manifest, dict) and isinstance(after_manifest, dict),
          "linux_environment_retained_manifest_invalid")
    validate_file_manifests(before_manifest["distributions"], after_manifest["distributions"],
                            target_record=record, target_file_hashes=files)
    _need(after_manifest.get("shared_opentelemetry_files") == {
        name: values for name, values in before_manifest.get("shared_opentelemetry_files", {}).items()
        if name != TARGET}, "linux_environment_retained_shared_namespace_invalid")
    absence = json.loads(_raw(root, "orphan-post-remove-absence", "stdout"))
    _need(absence.get("absent_paths") == sorted(record), "linux_environment_retained_absence_invalid")
    post_pdf = parse_freeze(_raw(root, "post-pdf-freeze", "stdout"))
    _need(parse_freeze(_raw(root, "installed", "stdout")) == post_pdf
          and {name for name in set(after) | set(post_pdf) if after.get(name) != post_pdf.get(name)} <= {PDF_NAME},
          "linux_environment_retained_pdf_delta_invalid")
    pdf_metadata = _raw(root, "post-pdf-metadata", "stdout")
    _validate_pdf_metadata(pdf_metadata, post_pdf)
    _only_nccl({**receipts["post-pdf-check"], "stdout": _raw(root, "post-pdf-check", "stdout")},
               replay_graph(pdf_metadata, post_pdf))
    # The pinned 1.28.2 wheel emits this exact informational alias warning on
    # stdout. Keep it in the raw receipt; do not accept arbitrary extra output.
    _need(_raw(root, "post-pdf-imports", "stdout").strip() in PDF_IMPORT_STDOUT,
          "linux_environment_retained_pdf_import_invalid")
    runtime_observation = _validate_core_runtime_import(_raw(root, "core-runtime-imports", "stdout"))
    _need(outcome.get("dependency_gate_sha256") == _sha((root / "dependency-gate.json").read_bytes())
          and outcome.get("dependency_pdf_stage_sha256") == _sha((root / "dependency-pdf-stage.json").read_bytes())
          and outcome.get("core_runtime_import_sha256") == _sha((root / "core-runtime-imports.json").read_bytes())
          and outcome.get("core_runtime") == runtime_observation,
          "linux_environment_retained_binding_invalid")
    return {"version": VERSION, "environment_preflight_observed": True,
            "exact_image_observed": False, "scientifically_accepted": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser("preflight")
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--deadline-epoch", required=True, type=float)
    run.add_argument("--declared-image")
    verify = subcommands.add_parser("verify")
    verify.add_argument("--output", required=True, type=Path)
    verify.add_argument("--expected-config-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = run_linux_environment_preflight(args.output, deadline_epoch=args.deadline_epoch,
                                                 declared_image=args.declared_image)
    else:
        result = verify_retained_linux_environment_preflight(args.output,
                                                              expected_config_sha256=args.expected_config_sha256)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
