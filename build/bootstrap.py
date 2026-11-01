"""Explicit capture, local extraction, corpus materialization and serving."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

from mosaic import LEGACY_ROOTS
from mosaic.public_build import PublicBuildRefusal, build_plan, load_config, preflight, register_sources, _no_links
from mosaic.workspace import Workspace

_SAFE_DEPENDENCY_MODULES = frozenset({
    "fitz", "pymupdf", "xgrammar", "torch", "transformers", "tokenizers", "jsonschema", "yaml",
    "mosaic.public_pipeline", "mosaic.public_literature_demo", "mosaic.public_model_inventory",
    "mosaic.public_execution", "mosaic.public_workflow", "mosaic.public_corpus",
    "mosaic.public_native_runtime", "mosaic.public_export", "mosaic.public_capture_build",
    "mosaic.public_capture", "mosaic.public_serving",
})
_SAFE_MISSING_CONFIG_KEYS = frozenset({
    "version", "profile", "sources", "run", "runtime", "view", "serve", "jev",
    "root", "files", "permission", "port", "enabled", "replay_only", "kind", "base_url",
    "model", "revision", "tokenizer_revision", "weights_sha256", "license", "columns",
    "default_screen", "sort", "driver_profile", "driver", "wall_clock_cap_seconds",
    "capture_root", "model_manifest", "weights_root", "base_url", "output",
})


def _safe_dependency_module(value: object) -> str | None:
    return value if isinstance(value, str) and value in _SAFE_DEPENDENCY_MODULES else None


def _safe_missing_config_key(value: object) -> str | None:
    return value if isinstance(value, str) and value in _SAFE_MISSING_CONFIG_KEYS else None


def _input_json(path: Path) -> dict:
    """Validate explicit configuration paths before opening their bytes."""
    if path.name.casefold().startswith(".env") or path.name.casefold().endswith(".private.json"):
        raise PublicBuildRefusal("public_credentials_file_refused")
    absolute = path.absolute()
    protected = {"sealed", "references", "diagnostic", "_holdback", "independent-reference", "jev_eval", "calibration", ".git"}
    if any(part.casefold() in protected or part.casefold().startswith(".env") for part in absolute.parts):
        raise PublicBuildRefusal("public_private_input_root_refused")
    if any(absolute.resolve().is_relative_to(legacy.resolve()) for legacy in LEGACY_ROOTS.values()):
        raise PublicBuildRefusal("public_legacy_input_root_refused")
    _no_links(absolute, Path(absolute.anchor))
    value = json.loads(absolute.read_bytes())
    if not isinstance(value, dict):
        raise PublicBuildRefusal("public_config_object_required")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "preflight", "ingest", "capture", "demo", "literature-demo", "verify-export", "prepare-local",
                                            "run-local", "materialize", "corpus", "serve", "inventory-model"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8798)
    parser.add_argument("--run-root", default="run/demo")
    parser.add_argument("--capture-root", type=Path, help="One capture-stage paper directory containing receipt.json and source.pdf.")
    parser.add_argument("--model-manifest", type=Path, help="JSON manifest with the complete local model file/hash roster.")
    parser.add_argument("--model-declaration", type=Path, help="Model/revision, tokenizer and declared runtime JSON for inventory-model.")
    parser.add_argument("--weights-root", type=Path, help="Local model directory whose files must exactly match the manifest.")
    parser.add_argument("--driver-profile", type=Path, help="JSON model identity, source permission declaration and finite run limits.")
    parser.add_argument("--base-url", help="Literal local HTTP endpoint; prepare-local only records it.")
    parser.add_argument("--output", type=Path, help="Immutable output directory within the workspace or OS temporary directory.")
    parser.add_argument("--prepared-root", type=Path, help="Frozen native preparation directory for run-local.")
    parser.add_argument("--manifest", type=Path, help="Source-bound corpus handoff JSON for materialize.")
    parser.add_argument("--reviews", type=Path, help="Optional source-field review JSON; never an automatic scientific qualification.")
    parser.add_argument("--bundle", type=Path, help="Hash-bound serving bundle for serve.")
    parser.add_argument("--assistant-config", type=Path, help="Optional query-time literature assistant configuration; credentials stay in the environment.")
    parser.add_argument("--live", action="store_true", help="Explicitly execute requests against the frozen local open-weights endpoint.")
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parent
    try:
        if args.assistant_config and not (args.command == "serve" or (args.serve and args.command in {"demo", "literature-demo", "corpus", "materialize"})):
            parser.error("--assistant-config requires a serving command")
        assistant_config = _input_json(args.assistant_config) if args.assistant_config else None
        if args.command == "literature-demo":
            from mosaic.public_literature_demo import build_literature_demo
            run_root = "run/literature-demo" if args.run_root == "run/demo" else args.run_root
            result = build_literature_demo(project_root, workspace=Workspace(), run_root=run_root, port=args.port)
            print(json.dumps(result, sort_keys=True), flush=True)
            if args.serve:
                _serve(Path(result["bundle_path"]), args.port, assistant_config=assistant_config)
            return 0
        if args.command == "inventory-model":
            if any(value is None for value in (args.model_declaration, args.weights_root, args.output)):
                parser.error("inventory-model requires --model-declaration, --weights-root and --output")
            from mosaic.public_model_inventory import inventory_local_model
            result = inventory_local_model(_input_json(args.model_declaration), args.weights_root.absolute(),
                args.output.absolute(), workspace=Workspace())
            print(json.dumps(result, sort_keys=True))
            return 0
        if args.command == "run-local":
            if not args.live or any(value is None for value in (args.prepared_root, args.capture_root, args.output)):
                parser.error("run-local requires --live, --prepared-root, --capture-root and --output")
            from mosaic.public_execution import run_local_program
            profile = _input_json(args.prepared_root / "driver-profile.json")
            result = run_local_program(args.prepared_root.absolute(), args.capture_root.absolute(), args.output.absolute(),
                workspace=Workspace(), deadline_epoch=time.time() + profile["driver"]["wall_clock_cap_seconds"],
                runtime_observation=None)
            print(json.dumps(result, sort_keys=True))
            return 0 if result.get("receipt", {}).get("native_execution_completed") is True else 2
        if args.command == "materialize":
            if args.manifest is None or args.output is None:
                parser.error("materialize requires --manifest and --output")
            from mosaic.public_workflow import load_corpus_manifest, check_serving
            from mosaic.public_corpus import materialize_public_corpus
            workspace = Workspace()
            sources = load_corpus_manifest(args.manifest.absolute(), project_root=project_root, workspace=workspace)
            result = materialize_public_corpus(sources, args.output.absolute(), workspace=workspace)
            check_serving(Path(result["bundle_path"]), workspace=workspace, output=args.output.absolute() / "readiness.json")
            print(json.dumps(result, sort_keys=True), flush=True)
            if args.serve:
                _serve(Path(result["bundle_path"]), args.port, assistant_config=assistant_config)
            return 0
        if args.command == "serve":
            if args.bundle is None:
                parser.error("serve requires --bundle")
            _serve(args.bundle.absolute(), args.port, assistant_config=assistant_config)
            return 0
        if args.command == "prepare-local":
            required = ("capture_root", "model_manifest", "weights_root", "driver_profile", "base_url", "output")
            if any(getattr(args, name) is None for name in required):
                parser.error("prepare-local requires --capture-root, --model-manifest, --weights-root, --driver-profile, --base-url and --output")
            from mosaic.public_native_runtime import load_capture_envelope, prepare_local_program
            workspace = Workspace()
            envelope = load_capture_envelope(args.capture_root.absolute(), workspace=workspace)
            result = prepare_local_program(
                envelope, _input_json(args.model_manifest), args.weights_root.absolute(),
                _input_json(args.driver_profile), args.base_url, args.output.absolute(), workspace=workspace)
            print(json.dumps(result, sort_keys=True))
            return 0
        if args.command == "verify-export":
            from mosaic.public_export import verify_export
            print(json.dumps(verify_export(project_root), sort_keys=True))
            return 0
        if args.command == "demo":
            from mosaic.public_pipeline import build_demo
            result = build_demo(project_root, workspace=Workspace(), run_root=args.run_root, port=args.port)
            print(json.dumps(result, sort_keys=True), flush=True)
            if args.serve:
                from mosaic.public_serving import serve_public
                server = serve_public(result["bundle_path"], port=args.port, assistant_config=assistant_config)
                print(f"MOSAIC fictional demo: http://127.0.0.1:{server.server_address[1]}/workspace.html", flush=True)
                try:
                    server.serve_forever()
                except KeyboardInterrupt:
                    pass
                finally:
                    server.server_close()
            return 0
        if args.config is None:
            parser.error("--config is required for this command")
        config = load_config(args.config, project_root=project_root)
        if args.command == "corpus":
            if any(value is None for value in (args.model_manifest, args.weights_root, args.driver_profile)):
                parser.error("corpus requires --model-manifest, --weights-root and --driver-profile")
            from mosaic.public_workflow import run_public_corpus
            result = run_public_corpus(config, project_root=project_root, workspace=Workspace(),
                model_manifest=_input_json(args.model_manifest), weights_root=args.weights_root.absolute(),
                driver_profile=_input_json(args.driver_profile),
                reviews=_input_json(args.reviews) if args.reviews else None, live=args.live)
            summary_keys = ("status", "build_report_path", "build_report_sha256", "prepared_source_count", "candidate_source_count",
                            "refused_source_count", "own_corpus_pipeline_ready", "real_model_inference_run", "scientifically_accepted",
                            "local_model_requests_completed")
            print(json.dumps({key: result[key] for key in summary_keys}, sort_keys=True), flush=True)
            if args.serve and result.get("materialization"):
                _serve(Path(result["materialization"]["bundle_path"]), args.port, assistant_config=assistant_config)
            return 0 if result["status"] in ("prepared_only", "candidate_corpus_served") else 2
        if args.command == "plan":
            result = build_plan(config)
        elif args.command == "preflight":
            result = preflight(config)
        elif args.command == "ingest":
            result = register_sources(config, project_root=project_root, workspace=Workspace())
        else:
            from mosaic.public_capture_build import capture_build
            result = capture_build(config, project_root=project_root, workspace=Workspace())
        print(json.dumps(result, sort_keys=True, ensure_ascii=False))
        if args.command == "preflight" and not result["python_312"]:
            return 2
        if args.command == "ingest" and not result["sources"]:
            return 2
        if args.command == "capture" and not result["capture_summary"]["source_packets_ready"]:
            return 2
        return 0
    except ImportError as exc:
        result = {"status": "refused", "reason": "public_dependency_unavailable",
                  "own_corpus_pipeline_ready": False}
        if (module := _safe_dependency_module(exc.name)) is not None:
            result["module"] = module
        print(json.dumps(result))
        return 2
    except KeyError as exc:
        result = {"status": "refused", "reason": "KeyError", "own_corpus_pipeline_ready": False}
        if exc.args and (key := _safe_missing_config_key(exc.args[0])) is not None:
            result["missing_key"] = key
        print(json.dumps(result))
        return 2
    except (PublicBuildRefusal, OSError, ValueError, TypeError, RuntimeError) as exc:
        # Typed public refusal codes explain repair without exposing arbitrary
        # exception content such as local paths, source text or provider replies.
        message = str(exc)
        reason = message if re.fullmatch(r"public_[a-z0-9_]+", message) else type(exc).__name__
        print(json.dumps({"status": "refused", "reason": reason,
                          "own_corpus_pipeline_ready": False}))
        return 2


def _serve(bundle: Path, port: int, *, assistant_config: dict | None = None) -> None:
    from mosaic.public_serving import serve_public
    server = serve_public(bundle, port=port, assistant_config=assistant_config)
    print(f"MOSAIC: http://127.0.0.1:{server.server_address[1]}/workspace.html", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
