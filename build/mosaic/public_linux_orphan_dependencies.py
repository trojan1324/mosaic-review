"""Fail-closed one-package repair for the fictional Linux public smoke.

The previous telemetry-upgrade repair remains immutable.  This module records
the whole installed dependency graph and removes only the orphan distribution
which caused that upgrade, after proving that the exact image's baseline has no
incoming dependency edge to it.  It is intentionally unusable as a general
package-cleanup mechanism.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import sha256
import base64
import csv
import io
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Callable, Mapping

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from mosaic.workspace import Workspace

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "public_linux_orphan_repair_v1.json"
IMAGE = "vllm/vllm-openai@sha256:6ee133356d49ed24cee03dd0cda3d3980279a11c107ce3e752c89e76540f41c9"
TARGET = "opentelemetry-exporter-otlp-common"
TARGET_VERSION = "0.66b1"
TARGET_WHEEL_SHA = "00ff8592c3a7cb729ff3fdc7ffa12372c243bdf2163e80c180994d0c7bd83ee9"
NCCL = 'torch 2.13.0+cu130 has requirement nvidia-nccl-cu13==2.29.7; platform_system == "Linux", but you have nvidia-nccl-cu13 2.30.7.'
OTEL = "opentelemetry-exporter-otlp-common 0.66b1 has requirement opentelemetry-sdk~=1.45.1, but you have opentelemetry-sdk 1.40.0."
PDF_NAME = "pymupdf"
PDF_WHEEL = "pymupdf-1.28.2-cp310-abi3-manylinux_2_28_x86_64.whl"
PDF_WHEEL_SHA = "397d6715c1f0df7548a92d0afd8ce370fc48fa47aeefac16be2bc04a16a8227f"
PDF_WHEEL_URL = "https://files.pythonhosted.org/packages/c7/06/dace3e27af26690cb20bead80dbac42941b0841eb689b8aabbd67dde16f0/" + PDF_WHEEL
ORPHAN_IMPORT_CODE = "from opentelemetry.sdk.trace import TracerProvider; from opentelemetry.sdk.metrics import MeterProvider; from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter; from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HTTP; from opentelemetry.exporter.prometheus import PrometheusMetricReader; print('orphan-repair-imports-ok')"
# These run inside the disposable public bundle.  They emit hashes and relative
# paths only; no distribution file contents are retained in the worker archive.
TARGET_RECORD_CODE = "import csv,hashlib,importlib.metadata as m,io,json,os; d=m.distribution('opentelemetry-exporter-otlp-common'); root=d.locate_file('.').resolve(); raw=d.read_text('RECORD'); out={};\nfor r in csv.reader(io.StringIO(raw)):\n p=r[0]; q=d.locate_file(p)\n if len(r)!=3 or not p or os.path.isabs(p) or '..' in p.replace('\\\\','/').split('/') or q.is_symlink() or not q.is_file() or not str(q.resolve()).startswith(str(root)+os.sep): raise RuntimeError('unsafe target RECORD')\n out[p.replace('\\\\','/')]=hashlib.sha256(q.read_bytes()).hexdigest()\nprint(json.dumps({'record':raw,'files':out,'root':str(root)},sort_keys=True,separators=(',',':')))"
TARGET_ABSENCE_CODE = "import json,sys; from pathlib import Path; root=Path(sys.argv[1]).resolve(); paths=json.loads(sys.argv[2]); bad=[]\nfor rel in paths:\n p=root.joinpath(*rel.split('/')); q=p.resolve()\n if not str(q).startswith(str(root)+__import__('os').sep) or p.exists() or p.is_symlink(): bad.append(rel)\nif bad: raise RuntimeError('target files remain: '+','.join(sorted(bad)))\nprint(json.dumps({'absent_paths':sorted(paths)},sort_keys=True,separators=(',',':')))"
# This intentionally does *not* hash every CUDA/Torch binary: on an 80 GB
# rental that is needlessly slow and makes a retained ZIP enormous.  Instead it
# preserves every distribution's complete RECORD path set and hashes its RECORD
# and METADATA files; the removed package's actual recorded files are hashed
# separately by TARGET_RECORD_CODE.  OpenTelemetry namespace files receive
# path/symlink guards and actual hashes; unrelated distributions retain only
# their RECORD/METADATA hashes and paths.
FILE_MANIFEST_CODE = (
    "import json; from mosaic.public_linux_orphan_dependencies import collect_installed_file_manifest; "
    "print(json.dumps(collect_installed_file_manifest(),sort_keys=True,separators=(',',':')))"
)
_OS_DISTRIBUTION_ROOTS = (Path('/usr/lib/python3/dist-packages'),)


def collect_installed_file_manifest(distributions=None) -> dict[str, Any]:
    """Preserve physical wheel and OS distributions, including shadowed versions.

    The pinned Ubuntu image contains 279 metadata locations for 275 distinct
    names. OS packages can lack wheel RECORD files, and four locations shadow
    newer pip installations. Their metadata remains in the before/after proof;
    they are never silently skipped or uninstalled. Unknown recordless roots,
    duplicate physical locations, and ambiguous telemetry ownership still fail.
    """
    import importlib.metadata as metadata

    rows = list(metadata.distributions() if distributions is None else distributions)
    out: dict[str, dict[str, Any]] = {}
    shared: dict[str, dict[str, str]] = {}
    locations: set[str] = set()
    for distribution in rows:
        name = _name(distribution.metadata['Name'])
        root = distribution.locate_file('.').resolve()
        location = Path(distribution._path)
        _need(not location.is_symlink() and location.exists()
              and location.resolve().is_relative_to(root), 'orphan_repair_metadata_location_invalid')
        physical = str(location.resolve())
        _need(physical not in locations, 'orphan_repair_duplicate_metadata_location')
        locations.add(physical)
        owner = name
        if owner in out:
            _need(not name.startswith('opentelemetry-'), 'orphan_repair_ambiguous_telemetry_distribution')
            owner = name + '@' + _hash(physical.encode())
        _need(owner not in out, 'orphan_repair_duplicate_metadata_location')
        raw = distribution.read_text('RECORD')
        # importlib also supports legacy .egg-info as one metadata file.
        meta = (distribution.read_text('METADATA') or distribution.read_text('PKG-INFO')
                or distribution.read_text(''))
        _need(isinstance(meta, str) and bool(meta), 'orphan_repair_distribution_metadata_missing')
        paths: list[str] = []
        checked: dict[str, str] = {}
        extra: dict[str, Any] = {'root': str(root), 'metadata_location': physical,
                                  'distribution_name': name, 'distribution_version': distribution.version}
        if raw is None:
            _need(root in _OS_DISTRIBUTION_ROOTS and not name.startswith('opentelemetry-'),
                  'orphan_repair_unrecognized_recordless_distribution')
            files = list(location.rglob('*')) if location.is_dir() else [location]
            legacy = {}
            for path in sorted(files):
                _need(not path.is_symlink() and path.resolve().is_relative_to(root),
                      'orphan_repair_unsafe_system_metadata')
                if path.is_file():
                    legacy[path.relative_to(root).as_posix()] = _hash(path.read_bytes())
            _need(bool(legacy), 'orphan_repair_system_metadata_empty')
            paths = sorted(legacy)
            extra.update({'inventory_kind': 'os-metadata-without-wheel-record',
                          'metadata_file_hashes': legacy})
        else:
            for row in csv.reader(io.StringIO(raw)):
                _need(len(row) == 3 and bool(row[0]), 'orphan_repair_malformed_distribution_record')
                path = row[0].replace('\\', '/')
                paths.append(path)
                if name.startswith('opentelemetry-'):
                    relative = _safe_rel(path)
                    actual = distribution.locate_file(str(relative))
                    _need(not actual.is_symlink() and actual.is_file() and actual.resolve().is_relative_to(root),
                          'orphan_repair_unsafe_telemetry_record')
                    checked[path] = _hash(actual.read_bytes())
            _need(len(paths) == len(set(paths)), 'orphan_repair_duplicate_distribution_path')
            extra['inventory_kind'] = 'wheel-record'
        out[owner] = {'record_sha256': _hash(raw.encode()) if raw is not None else None,
                      'metadata_sha256': _hash(meta.encode()), 'paths': sorted(paths), **extra}
        if name.startswith('opentelemetry-'):
            shared[name] = checked
    return {'distributions': out, 'shared_opentelemetry_files': shared}


class OrphanRepairRefusal(ValueError):
    pass


def _need(value: bool, code: str) -> None:
    if not value:
        raise OrphanRepairRefusal(code)


def _name(value: str) -> str:
    return canonicalize_name(value)


def _hash(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def parse_freeze(raw: bytes) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in raw.decode("utf-8", "strict").splitlines():
        parts = line.split(" @ ", 1) if " @ " in line else line.split("==", 1)
        _need(len(parts) == 2 and all(parts), "orphan_repair_freeze_invalid")
        name = _name(parts[0]); _need(name not in result, "orphan_repair_freeze_invalid")
        result[name] = parts[1]
    _need(bool(result), "orphan_repair_freeze_invalid")
    return result


@dataclass(frozen=True)
class OrphanRepairPlan:
    image: str
    target: str
    target_version: str
    target_wheel_sha256: str
    config_sha256: str


def load_plan(path: Path = CONFIG) -> OrphanRepairPlan:
    raw = Path(path).read_bytes()
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise OrphanRepairRefusal("orphan_repair_plan_invalid") from exc
    expected = {"version", "image", "remove_distribution", "source_provenance", "vendor_nccl_override"}
    _need(isinstance(data, dict) and set(data) == expected and data.get("version") == "mosaic-public-linux-orphan-repair/v1" and data.get("image") == IMAGE, "orphan_repair_plan_invalid")
    target, source, vendor = data.get("remove_distribution"), data.get("source_provenance"), data.get("vendor_nccl_override")
    _need(target == {"name": TARGET, "version": TARGET_VERSION, "wheel_sha256": TARGET_WHEEL_SHA}, "orphan_repair_target_invalid")
    _need(isinstance(source, dict) and source == {"dockerfile_sha256": "c62823feb562a4c3ab3050276c44c848b9354b11a21c077d29df114a07c7d911", "oci_config_sha256": "bd8a5d7a8774800c92d714c0c226b32c1863f929150f36efd610aebcd12a335d", "vllm_source_commit": "5df63a7203cdb9328d824a27b682e77c7b9928be"}, "orphan_repair_provenance_invalid")
    _need(vendor == {"torch": "2.13.0+cu130", "nccl": "2.30.7", "requires": "nvidia-nccl-cu13==2.29.7"}, "orphan_repair_vendor_override_invalid")
    return OrphanRepairPlan(IMAGE, TARGET, TARGET_VERSION, TARGET_WHEEL_SHA, _hash(raw))


def _environment(payload: object) -> dict[str, str]:
    keys = {"implementation_name", "implementation_version", "os_name", "platform_machine", "platform_release", "platform_system", "platform_version", "python_full_version", "platform_python_implementation", "python_version", "sys_platform"}
    _need(isinstance(payload, dict) and set(payload) == keys and all(isinstance(payload[k], str) for k in keys), "orphan_repair_environment_invalid")
    _need(payload["platform_system"] == "Linux" and payload["sys_platform"] == "linux" and payload["python_version"] == "3.12", "orphan_repair_environment_not_linux312")
    return dict(payload)


def _installed(metadata_raw: bytes, freeze: Mapping[str, str]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    try:
        payload = json.loads(metadata_raw); rows = payload["installed"]
    except (UnicodeDecodeError, ValueError, KeyError, TypeError) as exc:
        raise OrphanRepairRefusal("orphan_repair_metadata_invalid") from exc
    _need(isinstance(rows, list), "orphan_repair_metadata_invalid")
    found: dict[str, dict[str, Any]] = {}
    for row in rows:
        meta = row.get("metadata") if isinstance(row, dict) else None
        _need(isinstance(meta, dict) and isinstance(meta.get("name"), str) and isinstance(meta.get("version"), str) and isinstance(meta.get("requires_dist", []), list) and all(isinstance(item, str) for item in meta.get("requires_dist", [])), "orphan_repair_metadata_invalid")
        name = _name(meta["name"]); _need(name not in found, "orphan_repair_metadata_duplicate")
        found[name] = {"version": meta["version"], "requires": list(meta.get("requires_dist", [])), "direct_url": row.get("direct_url")}
    _need(set(found) == set(freeze), "orphan_repair_metadata_freeze_drift")
    for name, value in freeze.items():
        _need(found[name]["version"] == value or value.startswith(("file:", "http://", "https://")), "orphan_repair_metadata_freeze_drift")
    return found, _environment(payload.get("environment"))


def replay_graph(metadata_raw: bytes, freeze: Mapping[str, str], *, remove: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Evaluate every captured Linux requirement edge, never host markers."""
    installed, environment = _installed(metadata_raw, freeze)
    _need(remove <= {TARGET}, "orphan_repair_unapproved_projection")
    installed = {name: value for name, value in installed.items() if name not in remove}
    active: list[dict[str, Any]] = []; inactive: list[dict[str, Any]] = []; conflicts: list[dict[str, Any]] = []; invalid: list[dict[str, Any]] = []
    extras: set[tuple[str, str]] = set(); pending = deque((name, "") for name in sorted(installed)); seen: set[tuple[str, str]] = set()
    while pending:
        parent, extra = pending.popleft()
        if (parent, extra) in seen or parent not in installed: continue
        seen.add((parent, extra))
        for raw in installed[parent]["requires"]:
            base: dict[str, Any] = {"parent": parent, "extra_context": extra, "requirement": raw}
            try: req = Requirement(raw)
            except InvalidRequirement:
                invalid.append({**base, "reason": "malformed_requirement"}); continue
            child = _name(req.name); base.update(child=child, marker=str(req.marker) if req.marker else None, requested_extras=sorted(req.extras))
            marker_values = dict(environment); marker_values["extra"] = extra
            if req.marker is not None and not req.marker.evaluate(marker_values):
                inactive.append(base); continue
            active.append(base)
            for requested in req.extras:
                extras.add((child, requested)); pending.append((child, requested))
            observed = installed.get(child)
            if observed is None:
                conflicts.append({**base, "reason": "missing_distribution"}); continue
            if req.url:
                direct = observed.get("direct_url"); actual = direct.get("url") if isinstance(direct, dict) else None
                if actual is None: invalid.append({**base, "reason": "unverifiable_direct_url"})
                elif actual != req.url: conflicts.append({**base, "reason": "direct_url_mismatch"})
                continue
            try: valid = not req.specifier or Version(observed["version"]) in req.specifier
            except InvalidVersion: invalid.append({**base, "reason": "invalid_installed_version"}); continue
            if not valid: conflicts.append({**base, "reason": "version_constraint", "installed_version": observed["version"], "specifier": str(req.specifier)})
    incoming = [edge for edge in active + inactive if edge["child"] == TARGET]
    return {"environment": environment, "installed_count": len(installed), "removed": sorted(remove), "active_edge_count": len(active), "inactive_edge_count": len(inactive), "propagated_extras": [{"distribution": n, "extra": e} for n, e in sorted(extras)], "conflicts": conflicts, "invalid_or_unverifiable": invalid, "incoming_target_edges_all_markers": incoming, "replay_valid": not invalid}


def _lines(raw: bytes) -> set[str]:
    lines = [line.strip() for line in raw.decode("utf-8", "strict").splitlines() if line.strip()]
    _need(len(lines) == len(set(lines)), "orphan_repair_duplicate_diagnostic")
    return set(lines)


def _only_nccl(check: Mapping[str, Any], graph: Mapping[str, Any]) -> None:
    _need(check.get("returncode") == 1 and _lines(check["stdout"]) == {NCCL}, "orphan_repair_pip_check_not_nccl_only")
    conflicts = graph.get("conflicts")
    _need(isinstance(conflicts, list) and len(conflicts) == 1 and conflicts[0] .get("parent") == "torch" and conflicts[0].get("child") == "nvidia-nccl-cu13" and conflicts[0].get("reason") == "version_constraint" and conflicts[0].get("specifier") == "==2.29.7" and conflicts[0].get("installed_version") == "2.30.7" and not graph.get("invalid_or_unverifiable"), "orphan_repair_graph_not_nccl_only")


def _validate_pdf_metadata(metadata: bytes, freeze: Mapping[str, str]) -> None:
    installed, _ = _installed(metadata, freeze); row = installed.get(PDF_NAME)
    _need(isinstance(row, dict) and row.get("version") == "1.28.2", "orphan_repair_pdf_metadata_invalid")
    direct = row.get("direct_url"); url = direct.get("url") if isinstance(direct, dict) else None
    archive = direct.get("archive_info") if isinstance(direct, dict) else None
    observed_hash = archive.get("hash") if isinstance(archive, dict) else None
    _need(url == PDF_WHEEL_URL and observed_hash == "sha256=" + PDF_WHEEL_SHA and freeze.get(PDF_NAME) == PDF_WHEEL_URL + "#sha256=" + PDF_WHEEL_SHA, "orphan_repair_pdf_metadata_invalid")


def validate_baseline(*, freeze: Mapping[str, str], metadata: bytes, check: Mapping[str, Any], plan: OrphanRepairPlan) -> dict[str, Any]:
    _need(freeze.get(TARGET) == plan.target_version and freeze.get("opentelemetry-api") == "1.40.0" and freeze.get("opentelemetry-exporter-prometheus") == "0.61b0", "orphan_repair_baseline_versions_changed")
    graph = replay_graph(metadata, freeze)
    _need(check.get("returncode") == 1 and _lines(check["stdout"]) == {OTEL, NCCL}, "orphan_repair_baseline_check_unexpected")
    signatures = {(row.get("parent"), row.get("child"), row.get("reason"), row.get("specifier"), row.get("installed_version")) for row in graph["conflicts"]}
    _need(not graph["invalid_or_unverifiable"] and signatures == {(TARGET, "opentelemetry-sdk", "version_constraint", "~=1.45.1", "1.40.0"), ("torch", "nvidia-nccl-cu13", "version_constraint", "==2.29.7", "2.30.7")} and not graph["incoming_target_edges_all_markers"], "orphan_repair_target_not_orphan")
    projected = replay_graph(metadata, freeze, remove=frozenset({TARGET}))
    _need(projected["installed_count"] + 1 == graph["installed_count"], "orphan_repair_projection_delta_invalid")
    _only_nccl({"returncode": 1, "stdout": NCCL.encode()}, projected)
    return {"baseline_graph": graph, "projected_graph": projected}


def _safe_rel(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    _need(bool(value) and not path.is_absolute() and ".." not in path.parts and "." not in path.parts, "orphan_repair_record_path_unsafe")
    return path


def validate_record_ownership(record_raw: bytes, *, target: str = TARGET) -> dict[str, str]:
    """Validate a pip RECORD manifest supplied by the worker, not host paths."""
    try: rows = list(csv.reader(io.StringIO(record_raw.decode("utf-8", "strict"))))
    except (UnicodeDecodeError, csv.Error) as exc: raise OrphanRepairRefusal("orphan_repair_record_invalid") from exc
    result: dict[str, str] = {}
    blank_record_rows = 0
    for row in rows:
        _need(len(row) == 3 and row[0], "orphan_repair_record_invalid")
        path = _safe_rel(row[0]).as_posix()
        if not row[1] and not row[2]:
            blank_record_rows += 1; result[path] = ""; continue
        _need(row[1].startswith("sha256=") and row[2].isdigit(), "orphan_repair_record_invalid")
        digest = row[1][7:]
        try: decoded = base64.urlsafe_b64decode(digest + "=" * (-len(digest) % 4))
        except ValueError as exc: raise OrphanRepairRefusal("orphan_repair_record_hash_invalid") from exc
        _need(len(decoded) == 32 and path not in result, "orphan_repair_record_hash_invalid")
        result[path] = digest
    _need(bool(result) and blank_record_rows >= 1 and any(path.endswith(".dist-info/RECORD") for path in result), "orphan_repair_record_incomplete")
    return result


def validate_pre_removal_ownership(before: Mapping[str, Mapping[str, Any]], *, target_record: Mapping[str, str], target_file_hashes: Mapping[str, str]) -> None:
    _need(TARGET in before, "orphan_repair_target_manifest_delta_invalid")
    target_paths = set(target_record)
    _need(set(before[TARGET].get("paths", [])) == target_paths, "orphan_repair_target_manifest_drift")
    remaining_before = {path: owner for owner, files in before.items() if owner != TARGET for path in files.get("paths", [])}
    _need(not target_paths & set(remaining_before), "orphan_repair_file_ownership_collision")
    _need(set(target_file_hashes) == target_paths and all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in target_file_hashes.values()), "orphan_repair_target_file_hash_invalid")
    for path, digest in target_record.items():
        if digest:
            expected = base64.urlsafe_b64decode(digest + "=" * (-len(digest) % 4)).hex()
            _need(target_file_hashes[path] == expected, "orphan_repair_target_record_hash_mismatch")


def validate_file_manifests(before: Mapping[str, Mapping[str, Any]], after: Mapping[str, Mapping[str, Any]], *, target_record: Mapping[str, str], target_file_hashes: Mapping[str, str] | None = None) -> None:
    _need(TARGET in before and TARGET not in after, "orphan_repair_target_manifest_delta_invalid")
    target_paths = set(target_record)
    validate_pre_removal_ownership(before, target_record=target_record, target_file_hashes=target_file_hashes or {})
    _need(set(after) == set(before) - {TARGET}, "orphan_repair_unexpected_distribution_delta")
    for owner, files in after.items():
        _need(files == before[owner], "orphan_repair_remaining_file_drift")
    _need({name: version for name, version in before.items() if name != TARGET} == after, "orphan_repair_remaining_file_drift")


def _raw(output: Path, label: str, suffix: str) -> bytes:
    return (output / f"{label}.{suffix}.log").read_bytes()


def _command(run: Callable[..., dict], command: list[str], label: str, timeout: int, *, required: bool = True) -> dict[str, Any]:
    row = run(command, label, timeout, require_success=required)
    _need(isinstance(row, dict) and type(row.get("returncode")) is int, "orphan_repair_command_receipt_invalid")
    return row


def run_orphan_dependency_gate(*, run: Callable[..., dict], python: str, output: Path, workspace: Workspace, image: str, plan: OrphanRepairPlan | None = None) -> dict[str, Any]:
    plan = plan or load_plan(); _need(image == plan.image and _hash(CONFIG.read_bytes()) == plan.config_sha256, "orphan_repair_plan_drift")
    workspace.write_once(output / "dependency-repair-plan.json", CONFIG.read_bytes())
    _command(run, [python, "-m", "pip", "freeze", "--all"], "baseline-freeze", 30)
    _command(run, [python, "-m", "pip", "inspect", "--local"], "baseline-metadata", 30)
    baseline_check = _command(run, [python, "-m", "pip", "check"], "baseline-check", 60, required=False)
    _command(run, [python, "-c", TARGET_RECORD_CODE], "orphan-target-record", 30)
    before = parse_freeze(_raw(output, "baseline-freeze", "stdout")); meta = _raw(output, "baseline-metadata", "stdout")
    baseline = validate_baseline(freeze=before, metadata=meta, check={**baseline_check, "stdout": _raw(output, "baseline-check", "stdout")}, plan=plan)
    _command(run, [python, "-c", FILE_MANIFEST_CODE], "orphan-before-files", 120)
    manifests_before_raw = json.loads(_raw(output, "orphan-before-files", "stdout")); target_ownership = json.loads(_raw(output, "orphan-target-record", "stdout"))
    _need(isinstance(manifests_before_raw, dict) and isinstance(manifests_before_raw.get("distributions"), dict) and isinstance(manifests_before_raw.get("shared_opentelemetry_files"), dict), "orphan_repair_manifest_invalid")
    target_record = validate_record_ownership(str(target_ownership.get("record", "")).encode("utf-8")); target_files = target_ownership.get("files")
    _need(isinstance(target_files, dict), "orphan_repair_target_file_hash_invalid")
    validate_pre_removal_ownership(manifests_before_raw["distributions"], target_record=target_record, target_file_hashes=target_files)
    _command(run, [python, "-m", "pip", "uninstall", "--yes", TARGET], "orphan-uninstall", 60)
    target_root = target_ownership.get("root")
    _need(isinstance(target_root, str) and target_root, "orphan_repair_target_root_invalid")
    _command(run, [python, "-c", TARGET_ABSENCE_CODE, target_root, json.dumps(sorted(target_record))], "orphan-post-remove-absence", 30)
    _command(run, [python, "-m", "pip", "freeze", "--all"], "post-repair-freeze", 30)
    _command(run, [python, "-m", "pip", "inspect", "--local"], "post-repair-metadata", 30)
    post_check = _command(run, [python, "-m", "pip", "check"], "post-repair-check", 60, required=False)
    _command(run, [python, "-c", FILE_MANIFEST_CODE], "orphan-after-files", 120)
    after = parse_freeze(_raw(output, "post-repair-freeze", "stdout")); after_meta = _raw(output, "post-repair-metadata", "stdout")
    after_graph = replay_graph(after_meta, after); _only_nccl({**post_check, "stdout": _raw(output, "post-repair-check", "stdout")}, after_graph)
    _need({name: version for name, version in before.items() if name != TARGET} == after, "orphan_repair_distribution_delta_invalid")
    manifests_after_raw = json.loads(_raw(output, "orphan-after-files", "stdout")); _need(isinstance(manifests_after_raw, dict) and isinstance(manifests_after_raw.get("distributions"), dict) and isinstance(manifests_after_raw.get("shared_opentelemetry_files"), dict), "orphan_repair_manifest_invalid")
    validate_file_manifests(manifests_before_raw["distributions"], manifests_after_raw["distributions"], target_record=target_record, target_file_hashes=target_files)
    _need(manifests_after_raw["shared_opentelemetry_files"] == {name: files for name, files in manifests_before_raw["shared_opentelemetry_files"].items() if name != TARGET}, "orphan_repair_shared_namespace_drift")
    _command(run, [python, "-c", ORPHAN_IMPORT_CODE], "orphan-post-imports", 30)
    result = {"version": "mosaic-public-linux-orphan-gate/v1", "image": image, "plan_sha256": plan.config_sha256, "target": TARGET, "action": "remove_orphan_distribution", "baseline_freeze_sha256": _hash(_raw(output, "baseline-freeze", "stdout")), "baseline_metadata_sha256": _hash(meta), "post_repair_freeze_sha256": _hash(_raw(output, "post-repair-freeze", "stdout")), "post_repair_metadata_sha256": _hash(after_meta), "target_record_sha256": _hash(_raw(output, "orphan-target-record", "stdout")), "ownership_scope": "target actual files plus shared OpenTelemetry actual files; other distributions retain RECORD and METADATA hashes only", "baseline": baseline, "post_graph": after_graph, "scientifically_accepted": False}
    workspace.write_canonical_json(output / "dependency-gate.json", result); return result


def validate_orphan_pdf_stage(*, run: Callable[..., dict], python: str, output: Path, workspace: Workspace, before: Mapping[str, str], plan: OrphanRepairPlan | None = None) -> dict[str, Any]:
    plan = plan or load_plan(); _command(run, [python, "-m", "pip", "freeze", "--all"], "post-pdf-freeze", 30); _command(run, [python, "-m", "pip", "inspect", "--local"], "post-pdf-metadata", 30); check = _command(run, [python, "-m", "pip", "check"], "post-pdf-check", 60, required=False)
    installed = parse_freeze(_raw(output, "installed", "stdout")); after = parse_freeze(_raw(output, "post-pdf-freeze", "stdout")); _need(installed == after and {key for key in set(before)|set(after) if before.get(key)!=after.get(key)} <= {PDF_NAME}, "orphan_repair_pdf_delta_invalid")
    _need(after.get(PDF_NAME) in {"1.28.2", PDF_WHEEL_URL + "#sha256=" + PDF_WHEEL_SHA}, "orphan_repair_pdf_missing")
    pdf_metadata = _raw(output, "post-pdf-metadata", "stdout"); _validate_pdf_metadata(pdf_metadata, after); graph = replay_graph(pdf_metadata, after); _only_nccl({**check, "stdout": _raw(output, "post-pdf-check", "stdout")}, graph)
    result = {"changed_distributions": sorted({key for key in set(before)|set(after) if before.get(key)!=after.get(key)}), "post_pdf_freeze_sha256": _hash(_raw(output, "post-pdf-freeze", "stdout")), "post_pdf_metadata_sha256": _hash(_raw(output, "post-pdf-metadata", "stdout"))}
    workspace.write_canonical_json(output / "dependency-pdf-stage.json", result); return result


def verify_retained_orphan_dependency_gate(root: Path, *, expected_config_sha256: str) -> dict[str, Any]:
    """Replay retained v4 evidence without importing the Linux worker."""
    public_root = Path(root); output = public_root / "run" / "gpu-smoke"; gate = json.loads((output / "dependency-gate.json").read_bytes())
    _need(gate.get("version") == "mosaic-public-linux-orphan-gate/v1" and gate.get("plan_sha256") == expected_config_sha256 and gate.get("scientifically_accepted") is False, "orphan_repair_retained_gate_invalid")
    plan = load_plan(); _need(plan.config_sha256 == expected_config_sha256, "orphan_repair_retained_config_drift")
    outcome = json.loads((output / "outcome.json").read_bytes())
    _need(isinstance(outcome, dict) and outcome.get("dependency_gate_sha256") == _hash((output / "dependency-gate.json").read_bytes()) and outcome.get("dependency_config_sha256") == expected_config_sha256 and outcome.get("dependency_pdf_stage_sha256") == _hash((output / "dependency-pdf-stage.json").read_bytes()), "orphan_repair_retained_outcome_binding_invalid")
    labels = ["baseline-freeze", "baseline-metadata", "baseline-check", "orphan-target-record", "orphan-before-files", "orphan-uninstall", "orphan-post-remove-absence", "post-repair-freeze", "post-repair-metadata", "post-repair-check", "orphan-after-files", "orphan-post-imports", "install", "installed", "post-pdf-freeze", "post-pdf-metadata", "post-pdf-check"]
    commands = outcome.get("commands"); _need(isinstance(commands, list) and all(isinstance(row, dict) for row in commands), "orphan_repair_retained_commands_invalid")
    cursor = 0; receipts = {}
    for label in labels:
        receipt = json.loads((output / f"{label}.json").read_bytes()); command = receipt.get("command") if isinstance(receipt, dict) else None
        _need(isinstance(command, list) and all(isinstance(value, str) for value in command) and command and receipt.get("timed_out") is False and receipt.get("returncode") == (1 if label.endswith("-check") else 0), "orphan_repair_retained_command_invalid")
        matches = [i for i in range(cursor, len(commands)) if commands[i] == receipt]; _need(bool(matches), "orphan_repair_retained_command_order_invalid"); cursor = matches[0] + 1
        _need(receipt.get("stdout_sha256") == _hash(_raw(output, label, "stdout")) and receipt.get("stderr_sha256") == _hash(_raw(output, label, "stderr")), "orphan_repair_retained_command_hash_invalid"); receipts[label] = receipt
        tail = command[1:]
        if label.endswith("-freeze") or label == "installed": expected = ["-m", "pip", "freeze", "--all"]
        elif label.endswith("-metadata"): expected = ["-m", "pip", "inspect", "--local"]
        elif label.endswith("-check"): expected = ["-m", "pip", "check"]
        elif label == "orphan-target-record": expected = ["-c", TARGET_RECORD_CODE]
        elif label in {"orphan-before-files", "orphan-after-files"}: expected = ["-c", FILE_MANIFEST_CODE]
        elif label == "orphan-uninstall": expected = ["-m", "pip", "uninstall", "--yes", TARGET]
        elif label == "orphan-post-imports": expected = ["-c", ORPHAN_IMPORT_CODE]
        elif label == "install": expected = ["-m", "pip", "install", "--no-deps", "--require-hashes", "-r", tail[-1] if tail else ""]
        else: expected = ["-c", TARGET_ABSENCE_CODE, tail[2] if len(tail) == 4 else "", tail[3] if len(tail) == 4 else ""]
        _need(tail == expected, "orphan_repair_retained_command_semantics_invalid")
    _need(_raw(output, "orphan-post-imports", "stdout").strip() == b"orphan-repair-imports-ok", "orphan_repair_retained_import_invalid")
    # Re-run meaningful predicates from retained bytes, including bounded ownership.
    before = parse_freeze(_raw(output, "baseline-freeze", "stdout")); meta = _raw(output, "baseline-metadata", "stdout"); validate_baseline(freeze=before, metadata=meta, check={**receipts["baseline-check"], "stdout": _raw(output, "baseline-check", "stdout")}, plan=plan)
    after = parse_freeze(_raw(output, "post-repair-freeze", "stdout")); _need({name: value for name, value in before.items() if name != TARGET} == after, "orphan_repair_retained_freeze_delta_invalid"); graph = replay_graph(_raw(output, "post-repair-metadata", "stdout"), after); _only_nccl({**receipts["post-repair-check"], "stdout": _raw(output, "post-repair-check", "stdout")}, graph)
    ownership = json.loads(_raw(output, "orphan-target-record", "stdout")); record = validate_record_ownership(str(ownership.get("record", "")).encode()); files = ownership.get("files"); before_manifest = json.loads(_raw(output, "orphan-before-files", "stdout")); after_manifest = json.loads(_raw(output, "orphan-after-files", "stdout")); _need(isinstance(files, dict) and isinstance(before_manifest, dict) and isinstance(after_manifest, dict), "orphan_repair_retained_manifest_invalid"); validate_file_manifests(before_manifest["distributions"], after_manifest["distributions"], target_record=record, target_file_hashes=files); _need(after_manifest.get("shared_opentelemetry_files") == {name: paths for name, paths in before_manifest.get("shared_opentelemetry_files", {}).items() if name != TARGET}, "orphan_repair_retained_shared_namespace_invalid"); _need(receipts["orphan-post-remove-absence"]["command"][3:] == [ownership.get("root"), json.dumps(sorted(record))], "orphan_repair_retained_absence_command_invalid")
    absence = json.loads(_raw(output, "orphan-post-remove-absence", "stdout")); _need(isinstance(absence, dict) and absence.get("absent_paths") == sorted(record), "orphan_repair_retained_absence_invalid")
    post_pdf = parse_freeze(_raw(output, "post-pdf-freeze", "stdout")); _need(parse_freeze(_raw(output, "installed", "stdout")) == post_pdf and {name for name in set(after)|set(post_pdf) if after.get(name) != post_pdf.get(name)} <= {PDF_NAME} and post_pdf.get(PDF_NAME) == PDF_WHEEL_URL + "#sha256=" + PDF_WHEEL_SHA, "orphan_repair_retained_pdf_invalid"); pdf_metadata = _raw(output, "post-pdf-metadata", "stdout"); _validate_pdf_metadata(pdf_metadata, post_pdf); _only_nccl({**receipts["post-pdf-check"], "stdout": _raw(output, "post-pdf-check", "stdout")}, replay_graph(pdf_metadata, post_pdf))
    expected_lock = f"pymupdf @ {PDF_WHEEL_URL} --hash=sha256:{PDF_WHEEL_SHA}\n".encode(); _need((output / "requirements-linux-smoke.lock").read_bytes() == expected_lock and receipts["install"]["command"][-1].replace("\\", "/").rsplit("/", 1)[-1] == "requirements-linux-smoke.lock", "orphan_repair_retained_pdf_lock_invalid")
    declaration = json.loads((public_root / "run" / "smoke-runtime" / "declaration.json").read_bytes()); runtime = declaration.get("runtime", {}) if isinstance(declaration, dict) else {}; repair = runtime.get("dependency_repair", {}) if isinstance(runtime, dict) else {}; _need(runtime.get("wheels") == [{"name": PDF_WHEEL, "sha256": PDF_WHEEL_SHA}] and runtime.get("installed_inventory_sha256") == _hash(_raw(output, "installed", "stdout")) and repair.get("action") == "remove_orphan_distribution" and repair.get("removed_distribution") == TARGET and repair.get("plan_sha256") == expected_config_sha256, "orphan_repair_retained_declaration_invalid")
    _need(_hash(_raw(output, "baseline-freeze", "stdout")) == gate.get("baseline_freeze_sha256") and _hash(_raw(output, "post-repair-freeze", "stdout")) == gate.get("post_repair_freeze_sha256"), "orphan_repair_retained_hash_drift")
    return {"scientifically_accepted": False, "target": TARGET, "gate_sha256": _hash((output / "dependency-gate.json").read_bytes())}
