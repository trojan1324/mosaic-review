"""Evidence-mock coverage for the bounded public runtime replay adapter."""
from __future__ import annotations

import json
import socket
from pathlib import Path
from threading import Lock
import time
from copy import deepcopy

import pymupdf
import pytest

from mosaic.canonical import canonical_bytes
from mosaic.public_capture import capture_public
from mosaic.public_runtime import PublicRuntimeRefusal, fixture_replay_for_demo, run_extraction
from mosaic.workspace import Workspace


pytestmark = pytest.mark.evidence_mock


def _pdf(path: Path) -> None:
    document = pymupdf.open()
    page = document.new_page(width=612, height=792)
    for y, text in ((72, "Study 1"), (110, "Abstract"), (145, "Fictional local fixture prose."),
                    (200, "Methods"), (235, "Fictional participants completed a printed scale."),
                    (290, "Results"), (325, "The fixture reports an effect.")):
        page.insert_text((72, y), text, fontsize=11)
    document.save(path)
    document.close()


def _profile() -> dict:
    return {"mode": "fictional_fixture_replay_only",
            "runtime": {"endpoint_policy": "no_endpoint_in_fixture_replay", "remote": False},
            "driver": {"candidate": "public-fixture-test", "run_id": "public-fixture-test-run",
                       "served_model_name": "fixture", "max_measure_stages_per_paper": 1,
                       "max_request_seconds": 5, "request_cap": 8, "output_token_cap": 49152,
                       "wall_clock_cap_seconds": 40, "article_request_cap": 8}}


class FixtureTransport:
    """Synthetic answers to real task schemas; no model or socket is used."""
    def __init__(self):
        self.requests = []
        self.lock = Lock()

    def request(self, method, url, headers, body, timeout):
        assert method == "POST" and url.startswith("http://127.0.0.1")
        wire = json.loads(body)
        schema = json.loads(wire["messages"][0]["content"].split("\nRESPONSE_SCHEMA:\n", 1)[1])
        with self.lock:
            self.requests.append(wire)
        properties = schema["properties"]
        if "findings" in properties:
            answer = {"article_kind": "original_empirical", "findings": [], "qualifications": []}
        elif "studies" in properties:
            answer = {"article_kind": "original_empirical", "studies": [{"identifier_sentence_ids": ["c001"],
                      "design": "uncertain", "design_sentence_ids": [], "population_sentence_ids": [],
                      "sample_size_sentence_ids": [], "measures": []}], "analyses": []}
        else:
            # The dependent study-fact schema is task-specific.  Empty slots
            # are an explicit synthetic unresolved selection, never a claim.
            answer = {key: {"condition": [], "manipulation_description": [], "analysed_n": [],
                      "excluded_n": [], "reliability": {}}
                      for key in schema["properties"] if key != "additionalProperties"}
            if "facts" in properties:
                answer = {"facts": []}
        response = {"model": "fixture", "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(answer)}}],
                    "usage": {"completion_tokens": 5, "prompt_tokens": 7}}
        return 200, {}, canonical_bytes(response)


def _packet(tmp_path: Path):
    workspace = Workspace(repo_root=tmp_path, temp_root=tmp_path)
    sources = workspace.mkdir(tmp_path / "sources")
    pdf = sources / "fixture.pdf"; _pdf(pdf)
    raw = pdf.read_bytes()
    digest = __import__("hashlib").sha256(raw).hexdigest()
    row = {"paper_id": "public-" + digest, "source_pdf_sha256": digest, "byte_size": len(raw),
           "locators": ["fixture.pdf"], "permission": "fictional_fixture",
           "source_status": "public_unreviewed_source", "scientifically_accepted": False}
    root = tmp_path / "capture"
    capture = capture_public(row, "fixture.pdf", sources, root, title="Unreviewed fictional fixture", workspace=workspace)
    read = lambda name: json.loads((root / name).read_bytes())
    return workspace, {"capture_receipt": capture, "source": read("source.json"),
                       "source_binding": read("source-binding.json"),
                       "prepare_upstream_input": read("prepare-upstream-input.json"),
                       "runtime_authorization": {"kind": "fictional_fixture_local_only",
                                                 "source_pdf_sha256": capture["source_pdf_sha256"]}}


def test_public_runtime_executes_actual_fixture_driver_and_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(socket, "socket", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no socket")))
    workspace, packet = _packet(tmp_path)
    transport = FixtureTransport()
    result = run_extraction(packet, tmp_path / "runtime", workspace=workspace, profile=_profile(),
                            replay={"kind": "recorded_fictional_fixture", "fixture_transport": transport,
                                    "fixture_count": lambda text: len(text) // 3})
    receipt = result["receipt"]
    assert len(transport.requests) == 3
    assert receipt["network_calls"] == "not_observed_by_adapter_fixture_transport"
    assert receipt["model_calls"] == "not_observed_by_adapter_fixture_transport"
    assert receipt["scientifically_accepted"] is False and receipt["live_runtime_supported"] is False
    snapshot = json.loads(Path(result["snapshot_path"]).read_bytes())
    assert snapshot["validation_status"] == "candidate_only"
    assert snapshot["study_fact_candidate"]["status"] == "materialized_source_bound_proposal"


def test_public_runtime_implicit_fixture_deadline_starts_after_preparation(tmp_path, monkeypatch):
    """Cold fictional preparation cannot exhaust the implicit execution allowance."""
    workspace, packet = _packet(tmp_path)
    transport = FixtureTransport()
    import scripts.complete_extraction_sequence as sequence
    import scripts.complete_extraction_runner as runner
    from types import SimpleNamespace
    now = [time.time()]
    monkeypatch.setattr("mosaic.public_runtime.time", SimpleNamespace(time=lambda: now[0]))

    def slow_prepare(*args, **kwargs):
        now[0] += 301.0
        return {"synthetic": "preparation-seam"}

    observed = {}
    def stop_at_driver(*args, deadline_epoch, **kwargs):
        observed["deadline_epoch"] = deadline_epoch
        raise RuntimeError("driver-seam-stop")
    monkeypatch.setattr(sequence, "prepare_upstream", slow_prepare)
    monkeypatch.setattr(runner, "prepare_program", lambda *args, **kwargs: {"synthetic": "program"})
    monkeypatch.setattr(runner, "run_program", stop_at_driver)
    with pytest.raises(RuntimeError, match="driver-seam-stop"):
        run_extraction(packet, tmp_path / "runtime", workspace=workspace, profile=_profile(),
                       replay={"kind": "recorded_fictional_fixture", "fixture_transport": transport,
                               "fixture_count": lambda text: len(text) // 3})
    assert observed["deadline_epoch"] == pytest.approx(now[0] + 300.0)


def test_public_runtime_explicit_deadline_is_not_extended_after_preparation(tmp_path, monkeypatch):
    workspace, packet = _packet(tmp_path)
    import scripts.complete_extraction_sequence as sequence
    import scripts.complete_extraction_runner as runner
    from types import SimpleNamespace
    now = [time.time()]
    monkeypatch.setattr("mosaic.public_runtime.time", SimpleNamespace(time=lambda: now[0]))

    def slow_prepare(*args, **kwargs):
        now[0] += 2.0
        return {"synthetic": "preparation-seam"}

    called = []
    monkeypatch.setattr(sequence, "prepare_upstream", slow_prepare)
    monkeypatch.setattr(runner, "prepare_program", lambda *args, **kwargs: {"synthetic": "program"})
    monkeypatch.setattr(runner, "run_program", lambda *args, **kwargs: called.append(True))
    with pytest.raises(PublicRuntimeRefusal, match="public_runtime_deadline_invalid"):
        run_extraction(packet, tmp_path / "runtime", workspace=workspace, profile=_profile(),
                       replay={"kind": "recorded_fictional_fixture", "fixture_transport": FixtureTransport(),
                               "fixture_count": len, "deadline_epoch": now[0] + 1.0})
    assert called == []


def test_public_runtime_refuses_mutated_source_or_nonfixture_mode(tmp_path):
    workspace, packet = _packet(tmp_path)
    packet["source"]["units"][0]["text"] = "changed"
    with pytest.raises(PublicRuntimeRefusal, match="source_packet_hash_mismatch"):
        run_extraction(packet, tmp_path / "runtime", workspace=workspace, profile=_profile(),
                       replay={"kind": "recorded_fictional_fixture", "fixture_transport": FixtureTransport(),
                               "fixture_count": len})
    workspace, packet = _packet(tmp_path / "second")
    profile = _profile(); profile["mode"] = "live"
    with pytest.raises(PublicRuntimeRefusal, match="live_mode_refused"):
        run_extraction(packet, tmp_path / "second" / "runtime", workspace=workspace, profile=profile,
                       replay={"kind": "recorded_fictional_fixture", "fixture_transport": FixtureTransport(),
                               "fixture_count": len})


@pytest.mark.parametrize("field,value", [("work_id", "public-" + "0" * 64),
                                          ("scientifically_accepted", True)])
def test_public_runtime_refuses_capture_identity_relabeling(tmp_path, field, value):
    workspace, packet = _packet(tmp_path)
    changed = deepcopy(packet); changed["capture_receipt"][field] = value
    with pytest.raises(PublicRuntimeRefusal, match="capture_identity_invalid"):
        run_extraction(changed, tmp_path / "runtime", workspace=workspace, profile=_profile(),
                       replay={"kind": "recorded_fictional_fixture", "fixture_transport": FixtureTransport(),
                               "fixture_count": len})


def test_public_runtime_refuses_packet_with_different_work_binding(tmp_path):
    workspace, packet = _packet(tmp_path)
    changed = deepcopy(packet); changed["source"]["work_id"] = "public-" + "0" * 64
    # Keep its old digest assertion, so a packet rewrite cannot become a valid binding.
    with pytest.raises(PublicRuntimeRefusal, match="source_packet_hash_mismatch"):
        run_extraction(changed, tmp_path / "runtime", workspace=workspace, profile=_profile(),
                       replay={"kind": "recorded_fictional_fixture", "fixture_transport": FixtureTransport(),
                               "fixture_count": len})


@pytest.mark.parametrize("key,value", [("request_cap", True), ("max_request_seconds", 0),
                                        ("wall_clock_cap_seconds", float("inf"))])
def test_public_runtime_refuses_malformed_driver_limits(tmp_path, key, value):
    workspace, packet = _packet(tmp_path)
    profile = _profile(); profile["driver"][key] = value
    with pytest.raises(PublicRuntimeRefusal, match="driver_bound_invalid"):
        run_extraction(packet, tmp_path / "runtime", workspace=workspace, profile=profile,
                       replay={"kind": "recorded_fictional_fixture", "fixture_transport": FixtureTransport(),
                               "fixture_count": len})


def test_public_runtime_refuses_rehashed_source_that_is_not_the_upstream_packet(tmp_path):
    workspace, packet = _packet(tmp_path)
    changed = deepcopy(packet)
    changed["source"]["units"][0]["text"] = "Other fixture heading\n"
    digest = __import__("hashlib").sha256(canonical_bytes(changed["source"])).hexdigest()
    changed["capture_receipt"]["source_packet_sha256"] = digest
    changed["source_binding"]["source_packet_sha256"] = digest
    with pytest.raises(PublicRuntimeRefusal, match="upstream_binding_mismatch"):
        run_extraction(changed, tmp_path / "runtime", workspace=workspace, profile=_profile(),
                       replay={"kind": "recorded_fictional_fixture", "fixture_transport": FixtureTransport(),
                               "fixture_count": len})


def test_hash_pinned_public_demo_replay_uses_empty_findings(tmp_path):
    from mosaic.public_demo import demo_pdf_bytes
    workspace = Workspace(repo_root=tmp_path, temp_root=tmp_path)
    sources = workspace.mkdir(tmp_path / "sources")
    raw = demo_pdf_bytes(); workspace.write_once(sources / "fictional-build.pdf", raw)
    digest = __import__("hashlib").sha256(raw).hexdigest()
    row = {"paper_id": "public-" + digest, "source_pdf_sha256": digest, "byte_size": len(raw),
           "locators": ["fictional-build.pdf"], "permission": "fictional_fixture",
           "source_status": "public_unreviewed_source", "scientifically_accepted": False}
    root = tmp_path / "capture"
    capture = capture_public(row, "fictional-build.pdf", sources, root, title="Unreviewed fictional fixture", workspace=workspace)
    read = lambda name: json.loads((root / name).read_bytes())
    packet = {"capture_receipt": capture, "source": read("source.json"), "source_binding": read("source-binding.json"),
              "prepare_upstream_input": read("prepare-upstream-input.json"),
              "runtime_authorization": {"kind": "fictional_fixture_local_only", "source_pdf_sha256": digest}}
    reference_profile = json.loads((Path(__file__).parents[1] / "config" / "public_runtime_reference_v1.json").read_bytes())
    result = run_extraction(packet, tmp_path / "runtime", workspace=workspace, profile=reference_profile,
                            replay=fixture_replay_for_demo(packet))
    report = json.loads(Path(result["report_path"]).read_bytes())
    assert report["original_report"]["findings"]["materialized"]["findings"] == []
    assert result["receipt"]["network_calls"] == 0
    assert result["receipt"]["model_calls"] == 0
