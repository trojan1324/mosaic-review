"""Adversarial refusal tests for the narrow Linux dependency repair."""
import pytest
import json

from mosaic.public_linux_dependencies import DependencyRefusal, BASE, classify_baseline, load_plan, parse_freeze, validate_post


def _freeze(extra=""):
    rows = [f"{name}=={version}" for name, version in BASE.items()]
    rows += ["torch==2.13.0+cu130", "nvidia-nccl-cu13==2.30.7"]
    return parse_freeze(("\n".join(rows) + extra + "\n").encode())


NCCL = "torch 2.13.0+cu130 has requirement nvidia-nccl-cu13==2.29.7; platform_system == \"Linux\", but you have nvidia-nccl-cu13 2.30.7."
OTEL = "opentelemetry-exporter-otlp-common 0.66b1 has requirement opentelemetry-sdk~=1.45.1, but you have opentelemetry-sdk 1.40.0."


def test_clean_exit_with_arbitrary_pip_text_refuses():
    with pytest.raises(DependencyRefusal):
        classify_baseline(_freeze(), 0, b"unexpected pip diagnostic\n")


def test_post_repair_unrelated_distribution_mutation_refuses():
    plan = load_plan(); before = _freeze(); after = _freeze("\nunrelated-package==9.9")
    after.update(plan.versions)
    with pytest.raises(DependencyRefusal, match="unexpected_mutation"):
        validate_post(after, 1, (NCCL + "\n").encode(), plan, before=before)


def test_duplicate_freeze_refuses():
    with pytest.raises(DependencyRefusal):
        parse_freeze(b"torch==1\ntorch==2\n")


def test_exact_known_baseline_is_repairable_not_clean():
    assert classify_baseline(_freeze(), 1, (OTEL + "\n" + NCCL + "\n").encode()) == "repair_required"


def test_plan_rejects_nonhex_wheel_digest(tmp_path):
    raw = __import__("json").loads(__import__("mosaic.public_linux_dependencies", fromlist=["CONFIG"]).CONFIG.read_text())
    raw["repair_wheels"][0]["sha256"] = "z" * 64
    path = tmp_path / "bad.json"; path.write_text(__import__("json").dumps(raw))
    with pytest.raises(DependencyRefusal):
        load_plan(path)


def test_retained_gate_missing_pdf_evidence_refuses_typed(tmp_path):
    from mosaic.public_linux_smoke import verify_retained_dependency_gate, PublicLinuxSmokeRefusal
    plan = load_plan()
    with pytest.raises(PublicLinuxSmokeRefusal):
        verify_retained_dependency_gate(tmp_path, expected_config_sha256=plan.config_sha256)


def test_retained_dependency_gate_healthy_fixture_passes(tmp_path):
    from test_public_linux_dependencies import create_retained_dependency_fixture
    from mosaic.public_linux_smoke import verify_retained_dependency_gate
    root, digest = create_retained_dependency_fixture(tmp_path), load_plan().config_sha256
    assert verify_retained_dependency_gate(root.parents[1], expected_config_sha256=digest)["baseline_mode"] == "repair_required"


@pytest.mark.parametrize("mutation", ["metadata", "import_returncode", "import_timedout", "empty_commands", "missing_pdf", "wrong_gate_image", "wrong_outcome_gate_hash"])
def test_retained_dependency_gate_refuses_tampered_receipts(tmp_path, mutation):
    from test_public_linux_dependencies import create_retained_dependency_fixture
    from mosaic.public_linux_smoke import verify_retained_dependency_gate
    from mosaic.public_linux_dependencies import DependencyRefusal
    from mosaic.public_linux_smoke import PublicLinuxSmokeRefusal
    root, digest = create_retained_dependency_fixture(tmp_path), load_plan().config_sha256; base = root
    assert verify_retained_dependency_gate(root.parents[1], expected_config_sha256=digest)["baseline_mode"] == "repair_required"
    if mutation == "metadata":
        path = base / "post-repair-metadata.stdout.log"; data = json.loads(path.read_text()); data["installed"][0]["metadata"]["version"] = "tampered"; path.write_text(json.dumps(data))
    elif mutation in {"import_returncode", "import_timedout", "empty_commands", "wrong_outcome_gate_hash"}:
        path = base / "outcome.json"; data = json.loads(path.read_text())
        if mutation == "empty_commands": data["commands"] = []
        elif mutation == "wrong_outcome_gate_hash": data["dependency_gate_sha256"] = "0" * 64
        else:
            import_receipt = json.loads((base / "post-repair-imports.json").read_text())
            receipt = next(x for x in data["commands"] if x == import_receipt)
            receipt["returncode" if mutation == "import_returncode" else "timed_out"] = 1 if mutation == "import_returncode" else True
            (base / "post-repair-imports.json").write_text(json.dumps(receipt))
        path.write_text(json.dumps(data))
    elif mutation == "missing_pdf":
        (base / "post-pdf-check.json").unlink()
    elif mutation == "wrong_gate_image":
        path = base / "dependency-gate.json"; data = json.loads(path.read_text()); data["image"] = "wrong"; path.write_text(json.dumps(data))
    with pytest.raises((DependencyRefusal, PublicLinuxSmokeRefusal)):
        verify_retained_dependency_gate(root.parents[1], expected_config_sha256=digest)
