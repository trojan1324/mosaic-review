"""Executable public native-capture prefix, independent of scientific inference.

This is deliberately narrower than the proposed complete installer. It creates
all capture artifacts from explicitly registered local PDFs, with immutable
stage receipts and per-source refusals. It does not extract model responses,
build a serving database, or start the interface.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from mosaic.canonical import canonical_bytes
from mosaic.public_build import (StageExecutor, digest, register_sources,
                                 relative_path, validate_config)
from mosaic.public_capture import PublicCaptureRefused, capture_public
from mosaic.workspace import Workspace


def implementation_fingerprint(code_root: Path) -> str:
    """Bind code, schemas, runtime profile, locked dependencies and display.

    This conservative fingerprint deliberately invalidates the whole public
    build when a method or view asset changes. Private registries and run data
    are not dependency discovery inputs.
    """
    files = set([*(code_root / "mosaic").rglob("*.py"),
                 *(code_root / "scripts").glob("*.py"),
                 *(code_root / "schemas").glob("*.json")])
    files.update(p for p in (code_root / "public_frontend").rglob("*") if p.is_file())
    files.update(code_root / name for name in (
        "bootstrap.py", "start.py", "requirements-public-windows.lock",
        "config/public_runtime_reference_v1.json", "config/public_wheel_inventory_v1.json")
        if (code_root / name).is_file())
    return digest({p.relative_to(code_root).as_posix(): sha256(p.read_bytes()).hexdigest()
                   for p in sorted(files)})


def capture_build(config: Mapping[str, Any], *, project_root: Path,
                  workspace: Workspace) -> dict[str, Any]:
    """Run the offline prefix, retaining every selected source's disposition."""
    validate_config(dict(config), project_root=project_root)
    run_root = project_root / relative_path(config["run"]["root"])
    manifest = register_sources(config, project_root=project_root, workspace=workspace)
    # Pin the checkout's implementation/schema bytes without putting its paths
    # or private input registries in the report. This is a conservative local
    # fingerprint, not a substitute for the future public dependency lock.
    code_root = Path(__file__).resolve().parent.parent
    implementation = implementation_fingerprint(code_root)
    executor = StageExecutor(run_root, workspace=workspace, implementation_sha256=implementation)

    def intake_action(scoped: Workspace, attempt: Path) -> Mapping[str, Path]:
        return {"input_manifest": scoped.write_canonical_json(attempt / "input-manifest.json", manifest)}

    intake = executor.execute("intake", {"manifest_sha256": digest(manifest)},
                              dependencies={}, action=intake_action)
    source_root = Path(config["sources"]["root"])
    if not source_root.is_absolute():
        source_root = project_root / relative_path(config["sources"]["root"])

    def capture_action(scoped: Workspace, attempt: Path) -> Mapping[str, Path]:
        rows = []
        for row in manifest["sources"]:
            paper_root = attempt / "papers" / row["paper_id"]
            try:
                result = capture_public(row, row["locators"][0], source_root, paper_root,
                    title="Unreviewed source " + row["source_pdf_sha256"][:12], workspace=scoped)
                rows.append({"paper_id": row["paper_id"], "status": "source_packet_ready",
                             "source_packet_sha256": result["source_packet_sha256"],
                             "source_units": result["source_units"], "native_reader": result["native_reader"]})
            except PublicCaptureRefused as exc:
                native = None
                receipt_path = paper_root / "receipt.json"
                if receipt_path.exists():
                    native = json.loads(receipt_path.read_bytes()).get("native_reader")
                rows.append({"paper_id": row["paper_id"], "status": "refused", "reason": exc.code,
                             "native_reader": native})
        summary = {"version": "mosaic-public-capture-build/v1", "sources": rows,
                   "intake_refusals": manifest["refusals"],
                   "source_packets_ready": sum(row["status"] == "source_packet_ready" for row in rows),
                   "capture_refusals": sum(row["status"] == "refused" for row in rows),
                   "native_readers_ready": sum(row.get("native_reader") is not None for row in rows),
                   "model_calls": 0, "network_calls": 0, "scientifically_accepted": False,
                   "full_article_reader_ready": False, "native_reader_candidate_ready": any(row.get("native_reader") is not None for row in rows), "serving_artifacts_ready": False}
        scoped.write_canonical_json(attempt / "capture-summary.json", summary)
        # These are only files just created in this guarded attempt. Bind all
        # package objects/PDFs too, rather than trusting a pointer-only summary.
        return {p.relative_to(attempt).as_posix(): p for p in sorted(attempt.rglob("*"))
                if p.is_file() and p.name != "started.json"}

    captured = executor.execute("capture", {"manifest_sha256": digest(manifest), "mode": "native_only"},
                               dependencies={"intake": intake}, action=capture_action)
    summary_ref = captured["artifacts"]["capture-summary.json"]
    summary = json.loads((run_root / relative_path(summary_ref["path"])).read_bytes())
    report = executor.build_report(config, {"intake": intake, "capture": captured})
    return {"version": "mosaic-public-capture-command/v1", "capture_summary": summary,
            "stage_receipts": {"intake": intake, "capture": captured},
            "build_report_sha256": sha256(canonical_bytes(report)).hexdigest(),
            "checkpoint_status": "native_capture_prefix_only", "own_corpus_pipeline_ready": False,
            "scientifically_accepted": False}
