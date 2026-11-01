"""Isolated public-build contracts; no inference, downloads or provider calls.

The private pipeline remains unchanged.  This module implements the portable
intake, view/build fingerprints and immutable stage receipts that a public
runner needs.  It deliberately does not label an unconnected runtime as ready.
See DESIGN_SPEC.md and BUILD_GUIDE.md for the exported architecture, commands, and remaining integration gates.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import re
import sys
import tomllib
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

from mosaic import LEGACY_ROOTS
from mosaic.canonical import canonical_bytes
from mosaic.workspace import Workspace, WorkspaceViolation


VERSION = "mosaic-public-build/v1"
HASH = re.compile(r"[0-9a-f]{64}\Z")
# These are an implementation DAG, not a claim that all stage handlers exist.
STAGES = {
    "intake": (), "capture": ("intake",), "inventory": ("capture",),
    "extract": ("inventory",), "verify": ("extract",),
    "bibliography": ("capture",), "materialize": ("verify", "bibliography"),
    "optional_jev": ("verify",), "serve": ("materialize",),
}
STAGE_HANDLERS = {
    "intake": "mosaic.public_build.register_sources",
    "capture": "mosaic.public_capture_build.capture_build",
    "inventory": "mosaic.public_capture.capture_public",
    "extract": "mosaic.public_execution.run_local_program",
    "verify": "mosaic.public_snapshot.corpus_report_to_public_snapshot",
    "bibliography": "mosaic.public_review.finalize_review",
    "materialize": "mosaic.public_corpus.materialize_public_corpus",
    "optional_jev": "mosaic.public_jev.assess_public_task",
    "serve": "mosaic.public_serving.serve_public",
}
STAGE_CONTRACTS = {
    "intake": {"purpose": "Bind explicitly selected PDFs to immutable local source identities.",
        "inputs": ["user source list", "permission declaration"], "outputs": ["input-manifest.json"],
        "check": "PDF signature, source bytes/hash, duplicate aliases and safe paths.",
        "limit": "Registration does not establish article identity, rights clearance or scientific quality."},
    "capture": {"purpose": "Preserve original text, page geometry and capture failures before interpretation.",
        "inputs": ["input manifest", "exact source PDF"], "outputs": ["source.json", "source-binding.json"],
        "check": "Source digest, original unit IDs, page locations and explicit unreadable regions.",
        "limit": "Parsed text alone does not establish reading order or complete capture."},
    "inventory": {"purpose": "Define eligible source regions and unresolved layout before model requests.",
        "inputs": ["source packet", "capture scope policy"], "outputs": ["region inventory", "source eligibility policy"],
        "check": "Declared coverage, table/figure exclusions and ambiguous boundaries.",
        "limit": "Layout inventory is distinct from later proposed study and analysis ownership."},
    "extract": {"purpose": "Use local open weights to propose findings, studies, measures, roles and selected facts.",
        "inputs": ["eligible source units", "pinned model and tokenizer", "frozen task definitions"],
        "outputs": ["candidate report", "request/response receipts", "unavailable-stage dispositions"],
        "check": "Responses refer only to offered source evidence; unsupported or missing answers remain unresolved.",
        "limit": "The selected schema is narrower than the full capture policy; it does not exhaust each article."},
    "verify": {"purpose": "Check candidate structure, literal witnesses, source binding and scoped constraints.",
        "inputs": ["candidate report", "source packet", "schemas and mechanical rules"],
        "outputs": ["verified candidate intermediate", "field refusals and coverage"],
        "check": "Identity/offset/hash replay and field-level failures; preserve the original proposal.",
        "limit": "Literal fidelity and software checks do not establish semantic ownership or scientific truth."},
    "bibliography": {"purpose": "Recover source-supported article metadata and separately process printed references.",
        "inputs": ["source packet", "identity/reference evidence"],
        "outputs": ["bibliography candidates", "complete APA entries or refusals", "typed citation edges"],
        "check": "Complete title/byline/journal/year; own-article DOI proof; reference provenance.",
        "limit": "A DOI hit is not source ownership; discovery recommendations are not actual citation edges."},
    "materialize": {"purpose": "Create versioned serving artifacts without rewriting extraction results.",
        "inputs": ["verified candidate intermediate", "bibliography", "view configuration"],
        "outputs": ["search.sqlite", "display bundle", "source-assets manifest", "citation index/manifest", "display masks"],
        "check": "Original IDs/text, source digests, separate reader admission and quality status survive projection.",
        "limit": "A clean display never promotes a candidate to scientifically accepted."},
    "optional_jev": {"purpose": "Record inspectable post-extraction advice in an independent optional branch.",
        "inputs": ["verified candidate intermediate", "explicit allowed scope", "replay or separately authorized provider"],
        "outputs": ["advisory annotations or assessment refusal"],
        "check": "Scope, record/replay provenance and budget; no extraction/acceptance control edge.",
        "limit": "Advice is not independent scientific validation; explicit public tasks and an optional recorded sidecar never control extraction or acceptance."},
    "serve": {"purpose": "Let a researcher find, inspect, compare, save and export source-bound evidence.",
        "inputs": ["all pinned serving artifacts", "approved frontend assets"],
        "outputs": ["loopback interface", "source openings", "saved provenance exports"],
        "check": "Browser-to-API-to-PDF, pagination, empty/refused states and saved evidence reconstruction.",
        "limit": "Usability and build reproducibility require separate evidence from scientific accuracy or research benefit."},
}
EVIDENCE_FIELDS = frozenset({
    "artifact_id", "source_pdf_sha256", "source_anchor", "quality_status",
    "provenance", "scientifically_accepted",
})
VIEW_FIELDS = frozenset({
    "title", "authors", "journal", "year", "construct", "measure",
    "distinct_article_count", "study", "design", "finding", "citation",
})
PRIVATE_INPUT_PARTS = frozenset({
    ".git", "sealed", "references", "diagnostic", "_holdback", "ledgers", "scratch",
})
WINDOWS_DEVICES = frozenset({"con", "prn", "aux", "nul", *[f"com{i}" for i in range(1, 10)],
                            *[f"lpt{i}" for i in range(1, 10)]})


class PublicBuildRefusal(ValueError):
    """A typed, non-secret reason suitable for a local build report."""


def digest(value: Any) -> str:
    return sha256(canonical_bytes(value)).hexdigest()


def _require(ok: bool, code: str) -> None:
    if not ok:
        raise PublicBuildRefusal(code)


def _keys(value: Any, allowed: set[str], required: set[str], code: str) -> None:
    _require(isinstance(value, dict) and required <= value.keys()
             and value.keys() <= allowed, code)


def relative_path(value: Any) -> Path:
    """Portable member paths: refuse traversal, drives and alternate streams."""
    _require(isinstance(value, str) and bool(value) and "\\" not in value
             and ":" not in value and "\x00" not in value,
             "public_relative_path_required")
    posix, windows = PurePosixPath(value), PureWindowsPath(value)
    _require(not posix.is_absolute() and not windows.is_absolute()
             and not windows.drive and all(p not in (".", "..", "") for p in value.split("/")),
             "public_path_escape_refused")
    _require(not any(p.endswith((".", " ")) for p in posix.parts),
             "public_ambiguous_path_refused")
    _require(not any(p.split(".")[0].casefold() in WINDOWS_DEVICES for p in posix.parts),
             "public_device_path_refused")
    return Path(*posix.parts)


def _inside(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def _no_links(path: Path, root: Path) -> None:
    _require(_inside(path, root), "public_path_escape_refused")
    current = path
    while True:
        is_junction = getattr(current, "is_junction", lambda: False)
        _require(not current.is_symlink() and not is_junction(), "public_link_refused")
        if current == root:
            break
        _require(current != current.parent, "public_path_escape_refused")
        current = current.parent


def _source_root(value: str, project_root: Path) -> Path:
    _require(isinstance(value, str) and bool(value) and not value.startswith(("\\\\", "//")),
             "public_local_source_root_required")
    raw = Path(value)
    path = (raw if raw.is_absolute() else project_root / relative_path(value)).absolute()
    # Do not discover or open private registries to determine public admission.
    private_parts = (path.relative_to(project_root.absolute()).parts
                     if path.is_relative_to(project_root.absolute()) else path.parts)
    _require(not any(part.casefold() in PRIVATE_INPUT_PARTS for part in private_parts),
             "public_private_input_root_refused")
    _require(not any(_inside(path, legacy) for legacy in LEGACY_ROOTS.values()),
             "public_legacy_input_root_refused")
    _no_links(path, path.anchor and Path(path.anchor) or project_root)
    return path.resolve()


def validate_view(value: Any) -> dict[str, Any]:
    _keys(value, {"columns", "default_screen", "sort"},
          {"columns", "default_screen", "sort"}, "public_view_keys_invalid")
    columns = value["columns"]
    _require(isinstance(columns, list) and bool(columns) and all(isinstance(x, str) for x in columns)
             and len(columns) == len(set(columns)) and set(columns) <= VIEW_FIELDS,
             "public_view_columns_invalid")
    _require(value["default_screen"] in ("start", "measures", "papers", "citations", "project"),
             "public_view_screen_invalid")
    _require(value["sort"] in ("distinct_article_count_desc", "year_desc", "title"),
             "public_view_sort_invalid")
    return {**deepcopy(value), "mandatory_evidence_fields": sorted(EVIDENCE_FIELDS)}


def validate_config(value: Any, *, project_root: Path) -> dict[str, Any]:
    """Strict public TOML surface. It carries declarations, never credentials.

    Paths are local. No directory crawl, implicit download, registry lookup,
    network request, or runtime launch is performed by configuration validation.
    """
    _keys(value, {"version", "profile", "sources", "run", "runtime", "view", "serve", "jev"},
          {"version", "profile", "sources", "run", "runtime", "view", "serve", "jev"},
          "public_config_keys_invalid")
    _require(value["version"] == VERSION and value["profile"] in ("demo", "corpus"),
             "public_profile_invalid")
    _keys(value["sources"], {"root", "files", "permission"}, {"root", "files", "permission"},
          "public_source_config_invalid")
    sources = value["sources"]
    _require(sources["permission"] in ("user_declared_permitted", "fictional_fixture"),
             "public_permission_declaration_required")
    _require(value["profile"] != "corpus" or sources["permission"] == "user_declared_permitted",
             "public_corpus_permission_required")
    _require(isinstance(sources["files"], list) and bool(sources["files"]), "public_explicit_pdf_list_required")
    paths = [relative_path(p) for p in sources["files"]]
    _require(all(p.suffix.casefold() == ".pdf" for p in paths), "public_pdf_only")
    folded = [p.as_posix().casefold() for p in paths]
    _require(len(folded) == len(set(folded)), "public_duplicate_source_locator")
    _source_root(sources["root"], project_root)
    _keys(value["run"], {"root"}, {"root"}, "public_run_config_invalid")
    run = relative_path(value["run"]["root"])
    _require(not any(p.casefold() in PRIVATE_INPUT_PARTS for p in run.parts),
             "public_private_run_root_refused")
    _keys(value["serve"], {"port"}, {"port"}, "public_serve_config_invalid")
    _require(type(value["serve"]["port"]) is int and 1024 <= value["serve"]["port"] <= 65535,
             "public_port_invalid")
    _keys(value["jev"], {"enabled", "replay_only"}, {"enabled", "replay_only"}, "public_jev_config_invalid")
    _require(type(value["jev"]["enabled"]) is bool and type(value["jev"]["replay_only"]) is bool,
             "public_jev_config_invalid")
    # This workflow accepts replayed advisory sidecars only. Explicit live
    # advisory sends use the separate public_jev sidecar command and cannot be
    # dependencies of extraction, verification, materialization, or serving.
    _require(not value["jev"]["enabled"] or value["jev"]["replay_only"],
             "public_live_advisory_workflow_refused")
    runtime = value["runtime"]
    if value["profile"] == "demo":
        _keys(runtime, {"kind"}, {"kind"}, "public_runtime_config_invalid")
        _require(runtime["kind"] == "recorded_replay", "public_demo_replay_required")
    else:
        fields = {"kind", "base_url", "model", "revision", "tokenizer_revision", "weights_sha256", "license"}
        _keys(runtime, fields, fields, "public_runtime_config_invalid")
        _require(runtime["kind"] == "local_open_weights", "public_open_weights_required")
        _require(all(isinstance(runtime[k], str) and runtime[k].strip() for k in fields),
                 "public_runtime_pin_required")
        _require(bool(HASH.fullmatch(runtime["weights_sha256"])), "public_weights_hash_required")
        _require(all(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", runtime[k])
                     for k in ("revision", "tokenizer_revision")), "public_immutable_revision_required")
        try:
            url = urlsplit(runtime["base_url"])
            valid = (url.scheme == "http" and url.hostname in ("127.0.0.1", "::1")
                     and url.port is not None and not url.username and not url.password
                     and url.path in ("", "/", "/v1") and not url.query and not url.fragment)
        except ValueError:
            valid = False
        _require(valid, "public_loopback_runtime_required")
    validate_view(value["view"])
    return deepcopy(value)


def load_config(path: Path, *, project_root: Path) -> dict[str, Any]:
    _require(not path.name.casefold().startswith(".env"), "public_credentials_file_refused")
    config_path = path.absolute()
    _source_root(str(config_path.parent), project_root)
    _no_links(config_path, config_path.parent)
    return validate_config(tomllib.loads(path.read_text(encoding="utf-8")), project_root=project_root)


def build_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    """Separate extraction identity from presentation so view changes are cheap."""
    extraction = {k: config[k] for k in ("profile", "sources", "runtime")}
    return {
        "version": VERSION, "config_sha256": digest(config),
        "extraction_config_sha256": digest(extraction),
        "view_sha256": digest(validate_view(config["view"])),
        "stages": [{"id": name, "requires": list(deps), **deepcopy(STAGE_CONTRACTS[name]),
                    "implemented_handler": name in STAGE_HANDLERS,
                    "handler": STAGE_HANDLERS.get(name),
                    "runtime_validation": "not_established_by_plan"} for name, deps in STAGES.items()],
        "scientifically_accepted": False, "own_corpus_pipeline_ready": False,
        "remaining_gates": ["capture_scope_and_recovery_expansion", "real_model_runtime_validation",
                            "source_review", "scientific_accuracy_evaluation", "publication_rights_review"],
    }


def preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    """Observe software locally, without probing a server or opening a source."""
    versions = {}
    for name in ("pymupdf", "jsonschema", "pyyaml", "xgrammar", "torch", "transformers", "tokenizers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"version": VERSION, "python": platform.python_version(), "os": platform.system(),
            "python_312": sys.version_info[:2] == (3, 12), "dependencies": versions,
            "profile": config["profile"], "model_runtime": "not_tested",
            "network_calls": 0, "weights_downloaded": False, "scientifically_accepted": False,
            "own_corpus_pipeline_ready": False}


class RunWorkspace(Workspace):
    """Narrow the existing Workspace further: writes only beneath this run."""
    def __init__(self, root: Path, *, workspace: Workspace):
        raw = root.absolute()
        anchor = workspace.repo_root.absolute() if raw.is_relative_to(workspace.repo_root.absolute()) else Path(raw.anchor)
        _no_links(raw, anchor)
        self.root = workspace.check(root)
        self.parent_workspace = workspace
        # Preserve the public project identity for read/config resolution; the
        # overridden check(), not these attributes, narrows every write.
        self.repo_root = workspace.repo_root
        self.temp_root = workspace.temp_root

    def check(self, path: Path | str) -> Path:
        raw = Path(path)
        target = self.parent_workspace.check(raw if raw.is_absolute() else self.root / raw)
        if not target.is_relative_to(self.root):
            raise WorkspaceViolation("public_run_write_escape_refused")
        _no_links(target, self.root)
        return target


def register_sources(config: Mapping[str, Any], *, project_root: Path,
                     workspace: Workspace) -> dict[str, Any]:
    """Register only explicitly named files; invalid files get dispositions.

    A declaration records the user's assertion of permission, not legal
    clearance. Identical bytes share one article identity and retain aliases.
    """
    validate_config(dict(config), project_root=project_root)
    root = _source_root(config["sources"]["root"], project_root)
    grouped: dict[str, Any] = {}
    refused = []
    for name in sorted(config["sources"]["files"]):
        try:
            path = root / relative_path(name)
            _no_links(path, root)
            _require(path.is_file(), "public_source_missing")
            _require(path.stat().st_size <= 32 * 1024 ** 2, "public_large_pdf_separate_review")
            data = path.read_bytes()
            _require(data.startswith(b"%PDF-"), "public_pdf_signature_invalid")
            source_hash = sha256(data).hexdigest()
            row = grouped.setdefault(source_hash, {"paper_id": "public-" + source_hash,
                "source_pdf_sha256": source_hash, "byte_size": len(data), "locators": [],
                "source_status": "public_unreviewed_source", "permission": config["sources"]["permission"],
                "scientifically_accepted": False})
            row["locators"].append(name)
        except (PublicBuildRefusal, OSError) as exc:
            refused.append({"locator": name, "status": "refused",
                            "reason": str(exc) if isinstance(exc, PublicBuildRefusal) else "public_source_unreadable"})
    result = {"version": "mosaic-public-input-manifest/v1", "sources": sorted(grouped.values(), key=lambda x: x["paper_id"]),
              "refusals": refused, "network_calls": 0, "registry_consulted": False,
              "scientifically_accepted": False}
    run_root = project_root / relative_path(config["run"]["root"])
    scoped = RunWorkspace(run_root, workspace=workspace)
    manifest_hash = digest(result)
    # Changed sources create a new immutable version; no prior manifest is replaced.
    scoped.write_canonical_json(run_root / "inputs" / manifest_hash / "input-manifest.json", result)
    return result


class StageExecutor:
    """Content-addressed stage receipts with verifiable resume and failures.

    A callback gets a Workspace restricted to one run and a unique attempt
    directory. It returns named output paths, all of which are digest-bound.
    Only successful receipts can satisfy a dependency. Optional Jev is not a
    dependency of extraction, verification, materialization, or serving.
    Callbacks for a given run require a single writer. Cross-process scheduling
    and simultaneous execution of the same stage are not implemented here.
    """
    def __init__(self, root: Path, *, workspace: Workspace, implementation_sha256: str):
        _require(bool(HASH.fullmatch(implementation_sha256)), "public_implementation_hash_required")
        self.workspace = RunWorkspace(root, workspace=workspace)
        self.root = self.workspace.root
        self.implementation_sha256 = implementation_sha256

    def _verify_receipt(self, receipt: Mapping[str, Any]) -> None:
        _require(receipt.get("version") == VERSION and receipt.get("status") == "succeeded"
                 and receipt.get("scientifically_accepted") is False, "public_stage_receipt_invalid")
        _require(receipt.get("stage") in STAGES and isinstance(receipt.get("fingerprint"), str)
                 and bool(HASH.fullmatch(receipt["fingerprint"])), "public_stage_receipt_invalid")
        identity = {"version": VERSION, "stage": receipt["stage"],
                    "inputs_sha256": receipt.get("inputs_sha256"),
                    "implementation_sha256": receipt.get("implementation_sha256"),
                    "dependencies": receipt.get("dependencies")}
        _require(digest(identity) == receipt["fingerprint"], "public_stage_receipt_fingerprint_changed")
        stored = self.workspace.check(self.root / "stages" / receipt["stage"] / receipt["fingerprint"] / "result.json")
        _require(stored.is_file() and canonical_bytes(receipt) == canonical_bytes(json.loads(stored.read_bytes())),
                 "public_stage_dependency_receipt_changed")
        _require(isinstance(receipt.get("artifacts"), dict) and bool(receipt["artifacts"]),
                 "public_stage_artifacts_required")
        parents = receipt.get("dependencies")
        _require(isinstance(parents, dict) and set(parents) == set(STAGES[receipt["stage"]]),
                 "public_stage_dependencies_invalid")
        for name, ref in parents.items():
            _require(isinstance(ref, dict) and isinstance(ref.get("fingerprint"), str)
                     and bool(HASH.fullmatch(ref["fingerprint"])), "public_dependency_reference_invalid")
            parent_path = self.workspace.check(self.root / "stages" / name / ref["fingerprint"] / "result.json")
            _require(parent_path.is_file(), "public_dependency_receipt_missing")
            parent = json.loads(parent_path.read_bytes())
            _require(parent.get("stage") == name and digest(parent) == ref.get("sha256"),
                     "public_dependency_receipt_changed")
            self._verify_receipt(parent)
        for artifact in receipt["artifacts"].values():
            target = self.workspace.check(relative_path(artifact["path"]))
            _require(target.is_file() and sha256(target.read_bytes()).hexdigest() == artifact["sha256"],
                     "public_stage_artifact_changed")

    def execute(self, stage: str, inputs: Mapping[str, Any], *,
                dependencies: Mapping[str, Mapping[str, Any]],
                action: Callable[[Workspace, Path], Mapping[str, Path]]) -> dict[str, Any]:
        _require(stage in STAGES and set(dependencies) == set(STAGES[stage]), "public_stage_dependencies_invalid")
        for name, receipt in dependencies.items():
            _require(receipt.get("stage") == name, "public_stage_dependency_identity_invalid")
            self._verify_receipt(receipt)
        identity = {"version": VERSION, "stage": stage, "inputs_sha256": digest(inputs),
                    "implementation_sha256": self.implementation_sha256,
                    "dependencies": {k: {"sha256": digest(v), "fingerprint": v["fingerprint"]}
                                     for k, v in dependencies.items()}}
        key = digest(identity)
        parent = self.workspace.check(self.root / "stages" / stage / key)
        result_path = parent / "result.json"
        if result_path.exists():
            receipt = json.loads(result_path.read_bytes())
            _require(receipt.get("stage") == stage and receipt.get("fingerprint") == key,
                     "public_stage_resume_identity_invalid")
            self._verify_receipt(receipt)
            return receipt
        self.workspace.mkdir(parent)
        # An interrupted attempt remains inspectable. Never overwrite it on retry.
        number = 1
        while (parent / f"attempt-{number}").exists():
            number += 1
        attempt = self.workspace.mkdir(parent / f"attempt-{number}")
        self.workspace.write_canonical_json(attempt / "started.json", {"version": VERSION,
            "stage": stage, "fingerprint": key, "attempt": number})
        try:
            outputs = action(self.workspace, attempt)
            _require(isinstance(outputs, Mapping) and bool(outputs), "public_stage_outputs_required")
            artifacts = {}
            for name, path in sorted(outputs.items()):
                _require(isinstance(name, str) and bool(name), "public_artifact_name_required")
                target = self.workspace.check(path)
                _require(target.is_file() and target.is_relative_to(attempt), "public_attempt_output_required")
                artifacts[name] = {"path": target.relative_to(self.root).as_posix(),
                                   "sha256": sha256(target.read_bytes()).hexdigest()}
            receipt = {**identity, "fingerprint": key, "status": "succeeded",
                       "attempt": number, "artifacts": artifacts, "scientifically_accepted": False}
            self.workspace.write_canonical_json(result_path, receipt)
            return receipt
        except BaseException as exc:
            # Avoid exception messages/tracebacks: they may contain private paths
            # or provider content. KeyboardInterrupt is resumable too.
            failure = {"version": VERSION,
                "stage": stage, "fingerprint": key, "status": "failed", "error_type": type(exc).__name__,
                "scientifically_accepted": False}
            if re.fullmatch(r"public_[a-z0-9_]+", str(exc)):
                failure["reason"] = str(exc)
            self.workspace.write_canonical_json(attempt / "failure.json", failure)
            raise

    def build_report(self, config: Mapping[str, Any], receipts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        """Describe a checkpoint without turning library callbacks into a release.

        Stage success means its named output bytes verify. It does not mean the
        public end-to-end handlers, runtime, release or scientific gates passed.
        Missing stages stay explicit, even if some other stage produced data.
        """
        _require(set(receipts) <= STAGES.keys(), "public_report_unknown_stage")
        for name, receipt in receipts.items():
            _require(receipt.get("stage") == name, "public_stage_dependency_identity_invalid")
            self._verify_receipt(receipt)
            _require(set(STAGES[name]) <= receipts.keys(), "public_report_dependency_missing")
            _require(receipt["dependencies"] == {parent: {"sha256": digest(receipts[parent]),
                     "fingerprint": receipts[parent]["fingerprint"]} for parent in STAGES[name]},
                     "public_report_dependency_lineage_mismatch")
        plan = build_plan(config)
        report = {"version": VERSION, "plan": plan,
            "implementation_sha256": self.implementation_sha256,
            "stages": [{"stage": name, "status": "succeeded" if name in receipts else "not_executed",
                        "receipt_sha256": digest(receipts[name]) if name in receipts else None}
                       for name in STAGES],
            "verified_artifacts": {name: deepcopy(receipt["artifacts"]) for name, receipt in receipts.items()},
            "checkpoint_status": "scaffolding_only", "own_corpus_pipeline_ready": False,
            "scientifically_accepted": False,
            "release_gates": {"cold_start": "pending", "runtime": "not_tested",
                              "rights": "pending", "export": "pending"}}
        self.workspace.write_canonical_json(self.root / "builds" / digest(report) / "build-report.json", report)
        return report
