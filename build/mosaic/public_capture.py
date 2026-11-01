"""Isolated public PDF capture to a source packet.

This adapter intentionally bypasses ``Run`` and its development registry: one
caller supplies exactly one PDF path and an already registered public row.  It
uses the same register/capture/reconcile/inventory/source-policy primitives as
the current pipeline, but never reads private registries, sealed sets, recipes,
or source exclusions.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from mosaic.canonical import canonical_bytes
from mosaic.capture.coverage import coverage_summary, reconcile_pages
from mosaic.capture.pdf import capture_pdf
from mosaic.capture.register import SourceHashMismatch, register_source
from mosaic.inventory.figures import record_figures
from mosaic.inventory.studies import build_inventory, build_trees
from mosaic.package import PackageBuilder
from mosaic.tables.record import record_tables
from mosaic.vllm.source_policy import build_source_policy
from mosaic.workspace import Workspace
from mosaic.source_packets import snapshot
from mosaic.public_build import PublicBuildRefusal, _no_links, _source_root, relative_path
from mosaic.public_reader import build_public_reader


VERSION = "public-capture-source-packet/v1"
RECIPE_ID = "recipe-" + hashlib.sha256(VERSION.encode("utf-8")).hexdigest()
MAX_INPUT_TOKENS = 80_000


class PublicCaptureRefused(ValueError):
    """A safe, typed refusal from the public capture boundary."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _require_row(row: Mapping[str, Any]) -> tuple[str, str]:
    required = ("paper_id", "source_pdf_sha256", "byte_size", "locators", "source_status", "permission", "scientifically_accepted")
    if not all(key in row for key in required):
        raise PublicCaptureRefused("public_capture_registered_row_required")
    if not isinstance(row["source_pdf_sha256"], str):
        raise PublicCaptureRefused("public_capture_source_hash_invalid")
    digest = row["source_pdf_sha256"].lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise PublicCaptureRefused("public_capture_source_hash_invalid")
    if row["paper_id"] != "public-" + digest:
        raise PublicCaptureRefused("public_capture_paper_id_mismatch")
    if type(row["byte_size"]) is not int or row["byte_size"] < 1 or not isinstance(row["locators"], list) or not row["locators"]:
        raise PublicCaptureRefused("public_capture_registered_row_invalid")
    if row["source_status"] != "public_unreviewed_source" or row["permission"] not in ("user_declared_permitted", "fictional_fixture") or row["scientifically_accepted"] is not False:
        raise PublicCaptureRefused("public_capture_registration_state_invalid")
    return row["paper_id"], digest


def capture_public(
    row: Mapping[str, Any],
    locator: str,
    pdf_root: Path | str,
    output: Path | str,
    *,
    title: str,
    workspace: Workspace | None = None,
) -> dict[str, Any]:
    """Capture one explicitly registered public PDF into immutable artifacts.

    Hash comparison occurs before PyMuPDF opens the file.  ``output`` is a
    per-source output directory and cannot already exist.  On success it has a
    package, ``source.json``, ``source-binding.json``, ordered context, and a
    receipt.  Refusals raise ``PublicCaptureRefused`` and never invoke a model,
    OCR provider, directory scan, or private-policy reader.
    """
    work_id, expected = _require_row(row)
    if not isinstance(title, str) or not title.strip():
        raise PublicCaptureRefused("public_capture_unreviewed_title_required")
    if not isinstance(locator, str) or not locator.lower().endswith(".pdf"):
        raise PublicCaptureRefused("public_capture_pdf_locator_required")
    if locator not in row["locators"]:
        raise PublicCaptureRefused("public_capture_locator_not_registered")
    workspace = workspace or Workspace()
    try:
        source_root = _source_root(str(pdf_root), workspace.repo_root)
        source_path = source_root / relative_path(locator)
        _no_links(source_path, source_root)
    except PublicBuildRefusal as exc:
        raise PublicCaptureRefused(str(exc)) from exc
    if not source_path.is_file():
        raise PublicCaptureRefused("public_capture_source_missing")
    if source_path.stat().st_size > 32 * 1024 ** 2:
        raise PublicCaptureRefused("public_capture_large_pdf_separate_review")
    raw = source_path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise PublicCaptureRefused("public_capture_source_hash_mismatch")
    if len(raw) != row["byte_size"]:
        raise PublicCaptureRefused("public_capture_source_size_mismatch")
    if len(raw) > 32 * 1024 ** 2:
        raise PublicCaptureRefused("public_capture_large_pdf_separate_review")

    root = workspace.check(output)
    if root.exists():
        raise PublicCaptureRefused("public_capture_output_exists")
    workspace.mkdir(root)
    frozen_path = workspace.write_once(root / "source.pdf", raw)
    builder = PackageBuilder(work_id=work_id, recipe_id=RECIPE_ID, workspace=workspace, root=root / "packages")
    try:
        source_id, _ = register_source(
            builder, frozen_path, expected, role="main_article", source_label="main", portable_locator="source.pdf",
            relation_to_published="unresolved", external_evaluation="refused",
            identity_basis="catalog_id", title=title,
            version_evidence="Public caller supplied this exact SHA-256 for local processing.",
        )
        # Native capture and its local OCR decision tree are deterministic. OCR
        # remains disabled: this adapter must not introduce any provider egress.
        capture_pdf(builder, source_id, "main", frozen_path, renders=False, ocr=False)
        problems = reconcile_pages(builder.objects)
        if problems:
            raise PublicCaptureRefused("public_capture_page_reconciliation_failed")
        trees = build_trees(builder)
        source_regions: list[dict[str, Any]] = []
        # Record table regions so they remain visible to the policy.  Table
        # values are not interpreted; the measure-item exception is disabled in
        # this source-only public-capture stage.
        record_tables(builder, trees, interpret_tables=False, measure_items_only=False,
                      source_regions=source_regions, source_bytes={source_id: raw})
        record_figures(builder, trees, source_bytes={source_id: raw})
        build_inventory(builder, trees)
        policy = build_source_policy(builder, trees, source_regions)
        source, binding, ordered = snapshot(builder, policy, work_id, MAX_INPUT_TOKENS,
                                            source_regions=source_regions, source_bytes={source_id: raw})
    except SourceHashMismatch as exc:  # defensive: register_source must agree with preflight.
        raise PublicCaptureRefused("public_capture_source_hash_mismatch") from exc
    except PublicCaptureRefused:
        raise
    except ValueError as exc:
        if str(exc) != "missing_or_inconsistent_admitted_source":
            raise PublicCaptureRefused("public_capture_unreadable_or_incomplete") from exc
        # Preserve a complete native package and reader projection even though
        # no model-eligible body packet can be emitted.
        manifest_fields = {
            "recipe": {"recipe_id": RECIPE_ID, "recipe_sha256": RECIPE_ID.removeprefix("recipe-"),
                       "name": VERSION, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            "inputs": [{"role": "main_article", "sha256": expected, "path": "source.pdf", "bytes": len(raw)}],
            "coverage": coverage_summary(builder.objects),
            "stage_totals": {"register": "completed", "capture": "completed", "recover": "not_run_native_only_ocr_disabled", "inventory": "completed"},
            "extraction_scope": {"profile": "public_native_reader_only", "tables": "excluded", "notes": "No eligible model source packet; native reader retained."},
            "evidence_classes": {"deterministic": len(builder.objects), "replayed_historical_output": 0, "jev_advisory": 0},
            "status": "materialized", "permissions": {"external_evaluation": "refused", "redistribution": "not_granted_by_capture_unresolved"},
        }
        package_dir, manifest_id = builder.materialize(manifest_fields)
        reader = build_public_reader(package_dir, expected, root, workspace=workspace)
        workspace.write_canonical_json(root / "receipt.json", {
            "version": VERSION,
            "work_id": work_id,
            "status": "refused",
            "reason": "public_capture_no_eligible_source_units",
            "source_pdf_sha256": expected,
            "recovery": "not_run_native_only_ocr_disabled",
            "model_eligible_subset": "empty; this does not describe whole-article readability",
            "package_dir": package_dir.relative_to(root).as_posix(), "package_manifest_id": manifest_id,
            "native_reader": {"path": "native-reader.json", "sha256": hashlib.sha256(canonical_bytes(reader)).hexdigest(), "unit_count": len(reader["units"]), "scope": reader["scope"]},
            "scientifically_accepted": False,
        })
        raise PublicCaptureRefused("public_capture_no_eligible_source_units") from exc
    except Exception as exc:
        # Do not expose path or parser detail in a public receipt/error.
        raise PublicCaptureRefused("public_capture_unreadable_or_incomplete") from exc

    packet_sha = binding["source_packet_sha256"]
    if hashlib.sha256(canonical_bytes(source)).hexdigest() != packet_sha:
        raise PublicCaptureRefused("public_capture_source_packet_mismatch")
    manifest_fields = {
        "recipe": {"recipe_id": RECIPE_ID, "recipe_sha256": RECIPE_ID.removeprefix("recipe-"),
                   "name": VERSION, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "inputs": [{"role": "main_article", "sha256": expected, "path": "source.pdf", "bytes": len(raw)}],
        "coverage": coverage_summary(builder.objects),
        "stage_totals": {"register": "completed", "capture": "completed", "recover": "not_run_native_only_ocr_disabled", "inventory": "completed"},
        "extraction_scope": {"profile": "public_capture_source_only", "tables": "excluded",
                             "notes": "Native capture and inventory only; no model extraction and no table-value interpretation.",
                             "model_source_policy": policy},
        "evidence_classes": {"deterministic": len(builder.objects), "replayed_historical_output": 0, "jev_advisory": 0},
        "status": "materialized",
        "permissions": {"external_evaluation": "refused", "redistribution": "not_granted_by_capture_unresolved"},
    }
    package_dir, manifest_id = builder.materialize(manifest_fields)
    reader = build_public_reader(package_dir, expected, root, workspace=workspace)
    binding = {**binding, "public_capture_version": VERSION, "package_manifest_id": manifest_id,
               "table_regions": source_regions,
               "table_value_interpretation": "excluded"}
    report = {
        "version": VERSION,
        "work_id": work_id,
        "status": "source_packet_ready",
        "source_pdf_sha256": expected,
        "source_packet_sha256": packet_sha,
        "source_units": len(source["units"]),
        "omitted_nonbody_lanes": binding["omitted_nonbody_lanes"],
        "coverage": coverage_summary(builder.objects),
        "recovery": "not_run_native_only_ocr_disabled",
        "package_dir": package_dir.relative_to(root).as_posix(),
        "package_manifest_id": manifest_id,
        "native_reader": {"path": "native-reader.json", "sha256": hashlib.sha256(canonical_bytes(reader)).hexdigest(), "unit_count": len(reader["units"]), "scope": reader["scope"]},
        "model_calls": 0,
        "network_calls": 0,
        "scientifically_accepted": False,
    }
    upstream_input = {
        "paper_id": work_id,
        "title": title,
        "title_status": "caller_supplied_unreviewed_label_not_bibliographic_admission",
        "source_pdf_sha256": expected,
        "source": source,
    }
    workspace.write_canonical_json(root / "source.json", source)
    workspace.write_canonical_json(root / "source-binding.json", binding)
    workspace.write_canonical_json(root / "ordered-context.json", ordered)
    workspace.write_canonical_json(root / "prepare-upstream-input.json", upstream_input)
    workspace.write_canonical_json(root / "receipt.json", report)
    return report
