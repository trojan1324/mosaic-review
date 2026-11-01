"""One reproducible fictional build through capture, driver, index and interface.

Each action persists immutable stage artifacts. The replay is an engineering
test with manufactured responses; no model inference or qualification occurs.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import threading
from urllib.request import urlopen

from mosaic.canonical import canonical_bytes
from mosaic.public_build import StageExecutor, digest, relative_path
from mosaic.public_capture_build import capture_build
from mosaic.public_demo import prepare_demo, demo_config
from mosaic.workspace import Workspace


def _all_artifacts(attempt: Path) -> dict[str, Path]:
    return {p.relative_to(attempt).as_posix(): p for p in sorted(attempt.rglob("*"))
            if p.is_file() and p.name != "started.json"}


def build_demo(project_root: Path, *, workspace: Workspace, run_root: str = "run/demo", port: int = 8798) -> dict:
    fixture = prepare_demo(project_root, workspace=workspace)
    config = demo_config(run_root=run_root, port=port)
    captured = capture_build(config, project_root=project_root, workspace=workspace)
    receipts = captured["stage_receipts"]
    capture = receipts["capture"]
    root = project_root / relative_path(run_root)
    executor = StageExecutor(root, workspace=workspace, implementation_sha256=capture["implementation_sha256"])
    paper_id = "public-" + fixture["source_pdf_sha256"]
    reference = capture["artifacts"]["papers/" + paper_id + "/receipt.json"]
    capture_root = (root / relative_path(reference["path"])).parent
    cap = json.loads((capture_root / "receipt.json").read_bytes())
    native = json.loads((capture_root / "native-reader.json").read_bytes())
    source = json.loads((capture_root / "source.json").read_bytes())

    def inventory_action(ws, attempt):
        ws.write_canonical_json(attempt / "inventory.json", {
            "version": "mosaic-public-inventory-handoff/v1", "paper_id": paper_id,
            "native_reader_units": len(native["units"]), "model_eligible_units": len(source["units"]),
            "package_manifest_id": cap["package_manifest_id"], "source_packet_sha256": cap["source_packet_sha256"],
            "tables": "source images only; numeric cells are excluded from model evidence",
            "recovery": "native_only_ocr_disabled", "scientifically_accepted": False})
        return _all_artifacts(attempt)

    receipts["inventory"] = executor.execute("inventory", {"source_packet_sha256": cap["source_packet_sha256"]},
        dependencies={"capture": capture}, action=inventory_action)

    def extract_action(ws, attempt):
        from mosaic.public_runtime import run_extraction, fixture_replay_for_demo
        envelope = {"capture_receipt": cap, "source": source,
                    "source_binding": json.loads((capture_root / "source-binding.json").read_bytes()),
                    "prepare_upstream_input": json.loads((capture_root / "prepare-upstream-input.json").read_bytes()),
                    "runtime_authorization": {"kind": "fictional_fixture_local_only", "source_pdf_sha256": fixture["source_pdf_sha256"]}}
        code_root = Path(__file__).resolve().parent.parent
        profile = json.loads((code_root / "config/public_runtime_reference_v1.json").read_bytes())
        result = run_extraction(envelope, attempt / "runtime", workspace=ws, profile=profile,
                                replay=fixture_replay_for_demo(envelope))
        ws.write_canonical_json(attempt / "result-pointers.json", {
            key: Path(result[key]).relative_to(attempt).as_posix()
            for key in ("snapshot_path", "report_path", "receipt_path")})
        return _all_artifacts(attempt)

    receipts["extract"] = executor.execute("extract", {"source_packet_sha256": cap["source_packet_sha256"], "fixture": fixture},
        dependencies={"inventory": receipts["inventory"]}, action=extract_action)
    pointer_ref = receipts["extract"]["artifacts"]["result-pointers.json"]
    pointer_path = root / relative_path(pointer_ref["path"])
    pointers = json.loads(pointer_path.read_bytes())
    report_path = pointer_path.parent / relative_path(pointers["report_path"])
    snapshot_path = pointer_path.parent / relative_path(pointers["snapshot_path"])

    def verify_action(ws, attempt):
        # The actual runtime adapter has run the complete current validators and
        # generated this candidate-only snapshot. Preserve the whole object.
        snapshot = json.loads(snapshot_path.read_bytes())
        if snapshot.get("scientifically_accepted") is not False:
            raise ValueError("public_pipeline_snapshot_not_candidate")
        ws.copy_file(snapshot_path, attempt / "snapshot.json", expected_sha256=sha256(snapshot_path.read_bytes()).hexdigest())
        ws.write_canonical_json(attempt / "verification.json", {"scope": "mechanical_source_and_schema_checks",
            "semantic_accuracy": "not_evaluated", "human_qualification": "not_run", "scientifically_accepted": False})
        return _all_artifacts(attempt)

    receipts["verify"] = executor.execute("verify", {"snapshot_sha256": sha256(snapshot_path.read_bytes()).hexdigest()},
        dependencies={"extract": receipts["extract"]}, action=verify_action)

    def bibliography_action(ws, attempt):
        ws.write_canonical_json(attempt / "bibliography-disposition.json", {"paper_id": paper_id,
            "status": "withheld_identity_and_APA_unresolved", "admitted_identities": 0, "admitted_APA": 0,
            "reason": "This fictional source has no complete printed title/byline/journal/year or complete reference entry.",
            "external_lookups": 0, "scientifically_accepted": False})
        return _all_artifacts(attempt)

    receipts["bibliography"] = executor.execute("bibliography", {"pdf_sha256": fixture["source_pdf_sha256"]},
        dependencies={"capture": capture}, action=bibliography_action)

    def materialize_action(ws, attempt):
        from mosaic.public_serving import materialize_public
        materialize_public({"paper_id": paper_id, "source_pdf_path": capture_root / "source.pdf",
                            "source_pdf_sha256": fixture["source_pdf_sha256"], "native_reader": native,
                            "capture_receipt_path": capture_root / "receipt.json",
                            "capture_receipt_sha256": sha256((capture_root / "receipt.json").read_bytes()).hexdigest(),
                            "source_packet_path": capture_root / "source.json", "source_packet_sha256": cap["source_packet_sha256"]},
                           {"report_path": report_path, "report_sha256": sha256(report_path.read_bytes()).hexdigest(),
                            "snapshot_path": snapshot_path, "snapshot_sha256": sha256(snapshot_path.read_bytes()).hexdigest()},
                           attempt / "display", workspace=ws)
        return _all_artifacts(attempt)

    receipts["materialize"] = executor.execute("materialize", {"view": config["view"]},
        dependencies={"verify": receipts["verify"], "bibliography": receipts["bibliography"]}, action=materialize_action)
    bundle_path = root / relative_path(receipts["materialize"]["artifacts"]["display/bundle.json"]["path"])

    def serve_action(ws, attempt):
        from mosaic.public_serving import serve_public
        server = serve_public(bundle_path, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urlopen(f"http://127.0.0.1:{server.server_address[1]}/api/display/v1/status", timeout=15) as response:
                status = json.load(response)
            ws.write_canonical_json(attempt / "readiness.json", {"http_status": 200, "status_response": status,
                "scope": "loopback_API_readiness_only_browser_gate_separate", "scientifically_accepted": False})
        finally:
            server.shutdown()
            thread.join(10)
            server.server_close()
        return _all_artifacts(attempt)

    receipts["serve"] = executor.execute("serve", {"bundle_sha256": sha256(bundle_path.read_bytes()).hexdigest()},
        dependencies={"materialize": receipts["materialize"]}, action=serve_action)
    report = {"version": "mosaic-public-demo-build/v1", "fixture": fixture, "stages": receipts,
              "bundle": {"path": bundle_path.relative_to(root).as_posix(), "sha256": sha256(bundle_path.read_bytes()).hexdigest()},
              "demo_software_route_ready": True, "real_model_inference_run": False,
              "own_corpus_pipeline_ready": False, "scientifically_accepted": False,
              "optional_jev": "not_run_separate_advisory_sidecar", "publication_approved": False,
              "current_route_limits": ["fictional_replay_not_real_model_inference",
                                       "one_paper_fixture_identity_APA_measures_and_edges_withheld",
                                       "no_advisory_sidecar_recorded"],
              "remaining_gates": ["real_model_runtime_validation", "permitted_source_accuracy_evaluation",
                                  "release_rights_review"]}
    report_path = root / "reports" / (digest(report) + ".json")
    workspace.write_canonical_json(report_path, report)
    return {"build_report_path": str(report_path), "build_report_sha256": sha256(canonical_bytes(report)).hexdigest(),
            "bundle_path": str(bundle_path), "bundle_sha256": report["bundle"]["sha256"],
            "demo_software_route_ready": True, "own_corpus_pipeline_ready": False,
            "scientifically_accepted": False, "real_model_inference_run": False}
