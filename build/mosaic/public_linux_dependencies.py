"""Fail-closed dependency gate for the fictional Linux public smoke."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from packaging.requirements import Requirement
from typing import Callable, Mapping

from mosaic.workspace import Workspace


class DependencyRefusal(ValueError):
    pass


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "public_linux_dependency_repair_v1.json"
BASE = {
    "opentelemetry-api": "1.40.0", "opentelemetry-exporter-otlp": "1.40.0",
    "opentelemetry-exporter-otlp-proto-common": "1.40.0",
    "opentelemetry-exporter-otlp-proto-grpc": "1.40.0",
    "opentelemetry-exporter-otlp-proto-http": "1.40.0",
    "opentelemetry-exporter-otlp-common": "0.66b1",
    "opentelemetry-exporter-prometheus": "0.61b0", "opentelemetry-proto": "1.40.0",
    "opentelemetry-sdk": "1.40.0", "opentelemetry-semantic-conventions": "0.61b0"}
NCCL = 'torch 2.13.0+cu130 has requirement nvidia-nccl-cu13==2.29.7; platform_system == "Linux", but you have nvidia-nccl-cu13 2.30.7.'
OTEL = 'opentelemetry-exporter-otlp-common 0.66b1 has requirement opentelemetry-sdk~=1.45.1, but you have opentelemetry-sdk 1.40.0.'
CLEAN = "No broken requirements found."
PDF_WHEEL = "pymupdf-1.28.2-cp310-abi3-manylinux_2_28_x86_64.whl"
PDF_WHEEL_SHA = "397d6715c1f0df7548a92d0afd8ce370fc48fa47aeefac16be2bc04a16a8227f"
PDF_WHEEL_URL = "https://files.pythonhosted.org/packages/c7/06/dace3e27af26690cb20bead80dbac42941b0841eb689b8aabbd67dde16f0/" + PDF_WHEEL
PDF_FREEZE_VALUES = {"1.28.2", PDF_WHEEL_URL + "#sha256=" + PDF_WHEEL_SHA}
OTEL_IMPORT_CODE = "from opentelemetry.sdk.trace import TracerProvider; from opentelemetry.sdk.metrics import MeterProvider; from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as G; from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as H; from opentelemetry.exporter.prometheus import PrometheusMetricReader; print('otel-imports-ok')"


def _need(ok: bool, code: str) -> None:
    if not ok:
        raise DependencyRefusal(code)


def _name(value: str) -> str:
    return value.lower().replace("_", "-")


def parse_freeze(raw: bytes) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in raw.decode("utf-8", "strict").splitlines():
        parts = line.split(" @ ", 1) if " @ " in line else line.split("==", 1)
        _need(len(parts) == 2 and all(parts), "public_linux_smoke_dependency_freeze_invalid")
        key = _name(parts[0]); _need(key not in result, "public_linux_smoke_dependency_freeze_invalid")
        result[key] = parts[1]
    _need(bool(result), "public_linux_smoke_dependency_freeze_invalid")
    return result


@dataclass(frozen=True)
class DependencyRepairPlan:
    image: str
    source_commit: str
    wheels: tuple[tuple[str, str, str], ...]
    config_sha256: str

    def lock_bytes(self) -> bytes:
        return "".join(f"{n}=={v} --hash=sha256:{h}\n" for n, v, h in self.wheels).encode()

    @property
    def versions(self) -> dict[str, str]:
        return {n: v for n, v, _ in self.wheels}


def load_plan(path: Path = CONFIG) -> DependencyRepairPlan:
    raw = path.read_bytes(); data = json.loads(raw); rows = data.get("repair_wheels") if isinstance(data, dict) else None
    _need(isinstance(data, dict) and data.get("version") == "mosaic-public-linux-dependency-repair/v1" and data.get("image") == "vllm/vllm-openai@sha256:6ee133356d49ed24cee03dd0cda3d3980279a11c107ce3e752c89e76540f41c9" and isinstance(rows, list) and len(rows) == 9,
          "public_linux_smoke_dependency_plan_invalid")
    provenance, vendor = data.get("source_provenance"), data.get("vendor_nccl_override")
    _need(isinstance(provenance, dict) and isinstance(vendor, dict)
          and provenance.get("dockerfile_sha256") == "c62823feb562a4c3ab3050276c44c848b9354b11a21c077d29df114a07c7d911"
          and provenance.get("dockerfile_url") == "https://raw.githubusercontent.com/vllm-project/vllm/5df63a7203cdb9328d824a27b682e77c7b9928be/docker/Dockerfile"
          and provenance.get("oci_config_sha256") == "bd8a5d7a8774800c92d714c0c226b32c1863f929150f36efd610aebcd12a335d"
          and data.get("vllm_source_commit") == "5df63a7203cdb9328d824a27b682e77c7b9928be"
          and vendor == {"torch": "2.13.0+cu130", "nccl": "2.30.7", "requires": "nvidia-nccl-cu13==2.29.7"},
          "public_linux_smoke_dependency_provenance_invalid")
    parsed = []
    for row in rows:
        _need(isinstance(row, dict) and all(isinstance(row.get(k), str) for k in ("name", "version", "sha256")) and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is not None,
              "public_linux_smoke_dependency_plan_invalid")
        parsed.append((_name(row["name"]), row["version"], row["sha256"]))
    _need({x[0] for x in parsed} == set(BASE) - {"opentelemetry-exporter-otlp-common"} and all(len(x[2]) == 64 for x in parsed),
          "public_linux_smoke_dependency_plan_invalid")
    return DependencyRepairPlan(data["image"], data["vllm_source_commit"], tuple(sorted(parsed)), sha256(raw).hexdigest())


def _lines(raw: bytes) -> set[str]:
    rows = [x.strip() for x in raw.decode("utf-8", "strict").splitlines() if x.strip()]
    _need(len(rows) == len(set(rows)), "public_linux_smoke_dependency_duplicate_diagnostic")
    return set(rows)


def validate_metadata(raw: bytes, freeze: Mapping[str, str]) -> None:
    try:
        payload = json.loads(raw); rows = payload["installed"]
    except (ValueError, KeyError, TypeError):
        raise DependencyRefusal("public_linux_smoke_dependency_metadata_invalid") from None
    _need(isinstance(rows, list), "public_linux_smoke_dependency_metadata_invalid")
    found: dict[str, tuple[str, list[str], object]] = {}
    for row in rows:
        meta = row.get("metadata") if isinstance(row, dict) else None
        if not isinstance(meta, dict) or not isinstance(meta.get("name"), str) or not isinstance(meta.get("version"), str):
            raise DependencyRefusal("public_linux_smoke_dependency_metadata_invalid")
        name, requires = _name(meta["name"]), meta.get("requires_dist", [])
        if name in found or not isinstance(requires, list) or any(not isinstance(x, str) for x in requires):
            raise DependencyRefusal("public_linux_smoke_dependency_metadata_invalid")
        found[name] = (meta["version"], requires, row.get("direct_url"))
    _need(set(found) == set(freeze), "public_linux_smoke_dependency_metadata_drift")
    for name, value in freeze.items():
        _need(name in found, "public_linux_smoke_dependency_metadata_drift")
        if value.startswith("file:"):
            direct = found[name][2]
            _need(isinstance(direct, dict) and direct.get("url") == value, "public_linux_smoke_dependency_metadata_drift")
        else:
            _need(found[name][0] == value, "public_linux_smoke_dependency_metadata_drift")
    requires = found.get("torch", ("", [], None))[1]
    try:
        parsed = [Requirement(value) for value in requires]
    except Exception:
        raise DependencyRefusal("public_linux_smoke_dependency_nccl_metadata_invalid") from None
    _need(any(_name(item.name) == "nvidia-nccl-cu13" and str(item.specifier) == "==2.29.7" and str(item.marker) == 'platform_system == "Linux"' for item in parsed), "public_linux_smoke_dependency_nccl_metadata_invalid")


def classify_baseline(freeze: Mapping[str, str], code: int, stdout: bytes, stderr: bytes = b"") -> str:
    _need(not stderr, "public_linux_smoke_dependency_check_stderr")
    _need(type(code) is int and code == 1 and all(freeze.get(k) == v for k, v in BASE.items()), "public_linux_smoke_dependency_baseline_drift")
    _need(freeze.get("torch") == "2.13.0+cu130" and freeze.get("nvidia-nccl-cu13") == "2.30.7",
          "public_linux_smoke_dependency_nccl_drift")
    _need(_lines(stdout) == {OTEL, NCCL}, "public_linux_smoke_dependency_unknown_conflict")
    return "repair_required"


def validate_post(freeze: Mapping[str, str], code: int, stdout: bytes, plan: DependencyRepairPlan,
                  *, before: Mapping[str, str] | None = None, stderr: bytes = b"") -> None:
    _need(not stderr, "public_linux_smoke_dependency_check_stderr")
    _need(all(freeze.get(k) == v for k, v in plan.versions.items()), "public_linux_smoke_dependency_repair_drift")
    _need(freeze.get("torch") == "2.13.0+cu130" and freeze.get("nvidia-nccl-cu13") == "2.30.7",
          "public_linux_smoke_dependency_nccl_drift")
    _need(type(code) is int and code == 1 and _lines(stdout) == {NCCL},
          "public_linux_smoke_dependency_unknown_post_conflict")
    if before is not None:
        changed = {key for key in set(before) | set(freeze) if before.get(key) != freeze.get(key)}
        _need(changed <= set(plan.versions), "public_linux_smoke_dependency_unexpected_mutation")


def _raw(output: Path, label: str, kind: str) -> bytes:
    return (output / f"{label}.{kind}.log").read_bytes()


def run_dependency_gate(*, run: Callable[..., dict], python: str, output: Path, workspace: Workspace,
                        image: str, plan: DependencyRepairPlan | None = None) -> dict:
    """Record complete baseline before installing the sole allowed repair."""
    plan = plan or load_plan(); _need(image == plan.image, "public_linux_smoke_dependency_image_drift")
    _need(sha256(CONFIG.read_bytes()).hexdigest() == plan.config_sha256, "public_linux_smoke_dependency_plan_drift")
    workspace.write_once(output / "dependency-repair-plan.json", CONFIG.read_bytes())
    run([python, "-m", "pip", "freeze", "--all"], "baseline-freeze", 30)
    run([python, "-m", "pip", "inspect", "--local"], "baseline-metadata", 30)
    check = run([python, "-m", "pip", "check"], "baseline-check", 60, require_success=False)
    baseline = parse_freeze(_raw(output, "baseline-freeze", "stdout"))
    validate_metadata(_raw(output, "baseline-metadata", "stdout"), baseline)
    mode = classify_baseline(baseline, check["returncode"], _raw(output, "baseline-check", "stdout"), _raw(output, "baseline-check", "stderr"))
    if mode == "repair_required":
        lock = workspace.write_once(output / "otel-repair.lock", plan.lock_bytes())
        run([python, "-m", "pip", "install", "--isolated", "--index-url", "https://pypi.org/simple",
             "--only-binary=:all:", "--no-cache-dir", "--no-deps", "--require-hashes", "-r", str(lock)], "otel-repair-install", 120)
    run([python, "-m", "pip", "freeze", "--all"], "post-repair-freeze", 30)
    run([python, "-m", "pip", "inspect", "--local"], "post-repair-metadata", 30)
    post = run([python, "-m", "pip", "check"], "post-repair-check", 60, require_success=False)
    after = parse_freeze(_raw(output, "post-repair-freeze", "stdout")); validate_metadata(_raw(output, "post-repair-metadata", "stdout"), after); validate_post(after, post["returncode"], _raw(output, "post-repair-check", "stdout"), plan, before=baseline, stderr=_raw(output, "post-repair-check", "stderr"))
    run([python, "-c", OTEL_IMPORT_CODE], "post-repair-imports", 30)
    result = {"version": "mosaic-public-linux-dependency-gate/v1", "image": image, "plan_sha256": plan.config_sha256,
              "vllm_source_commit": plan.source_commit, "baseline_mode": mode,
              "baseline_freeze_sha256": sha256(_raw(output, "baseline-freeze", "stdout")).hexdigest(),
              "baseline_metadata_sha256": sha256(_raw(output, "baseline-metadata", "stdout")).hexdigest(),
              "post_repair_freeze_sha256": sha256(_raw(output, "post-repair-freeze", "stdout")).hexdigest(),
              "post_repair_metadata_sha256": sha256(_raw(output, "post-repair-metadata", "stdout")).hexdigest(),
              "repair_lock_sha256": sha256(plan.lock_bytes()).hexdigest(),
              "allowed_vendor_exception": "exact source-bound vLLM NCCL 2.30.7 override only", "scientifically_accepted": False}
    workspace.write_canonical_json(output / "dependency-gate.json", result)
    return result


def validate_pdf_stage(*, run: Callable[..., dict], python: str, output: Path, workspace: Workspace,
                       before: Mapping[str, str], plan: DependencyRepairPlan | None = None) -> dict:
    plan = plan or load_plan(); run([python, "-m", "pip", "freeze", "--all"], "post-pdf-freeze", 30)
    check = run([python, "-m", "pip", "check"], "post-pdf-check", 60, require_success=False)
    after = parse_freeze(_raw(output, "post-pdf-freeze", "stdout")); changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
    _need(changed <= {"pymupdf"}, "public_linux_smoke_dependency_pdf_drift"); validate_post(after, check["returncode"], _raw(output, "post-pdf-check", "stdout"), plan, stderr=_raw(output, "post-pdf-check", "stderr"))
    _need(after.get("pymupdf") in PDF_FREEZE_VALUES, "public_linux_smoke_dependency_pdf_missing")
    result = {"changed_distributions": sorted(changed), "post_pdf_freeze_sha256": sha256(_raw(output, "post-pdf-freeze", "stdout")).hexdigest()}
    workspace.write_canonical_json(output / "dependency-pdf-stage.json", result); return result
