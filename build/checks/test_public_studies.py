from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

from mosaic.canonical import canonical_bytes

from mosaic import public_studies as module


PID = "public-" + "a" * 64
PDF = "a" * 64
PACKET = "b" * 64


def _snapshot(*, candidate=True, pages=(1,)):
    units = [{"unit_id": "u1", "text": "Study 1 reports a fictional measure.", "page": 1}]
    source = {"units": units, "units_sha256": sha256(canonical_bytes(units)).hexdigest(), "source_policy_sha256": "c" * 64}
    paper = {"paper_id": PID, "source_pdf_sha256": PDF}
    inventory = None
    if candidate:
        quote = {"quote": units[0]["text"], "segments": [{"unit_id": "u1", "start": 0, "end": len(units[0]["text"]), "text": units[0]["text"], "page": 1}], "pages": list(pages)}
        inventory = {"version": "mosaic-study-inspection/v1", "scientifically_accepted": False, "coverage_verified": False,
                     "paper": dict(paper), "source_identity": {"eligible_source_units_sha256": source["units_sha256"], "source_policy_sha256": source["source_policy_sha256"]},
                     "studies": [{"id": "study-0", "identifier_evidence": [quote], "measures": []}], "analyses": []}
    return {"version": "mosaic-public-snapshot/v1", "validation_status": "candidate_only", "scientifically_accepted": False,
            "paper": paper, "source": source, "source_packet_sha256": PACKET, "core_inventory_candidate": inventory}


def _handoff(tmp_path: Path, snapshot: dict) -> dict:
    path = tmp_path / "public-snapshot.json"; path.write_bytes(canonical_bytes(snapshot))
    return {"capture": {"paper_id": PID, "source_pdf_sha256": PDF, "source_packet_sha256": PACKET},
            "snapshot": {"snapshot_path": str(path), "snapshot_sha256": sha256(path.read_bytes()).hexdigest()}}


def test_real_handoff_path_binds_inventory_units_pages_and_public_paper_id(tmp_path: Path):
    manifest, files = module.build_from_public_handoffs([_handoff(tmp_path, _snapshot())], identities={})
    row = manifest["papers"][0]
    rendered = json.loads(files[row["data_path"]])
    assert row["paper_id"] == PID and row["study_records"] == 1
    assert rendered["studies"][0]["identifier_evidence"][0]["pages"] == [1]
    assert row["title"] == "Source identity awaiting review"


def test_rejects_arbitrary_direct_inventory_and_inline_snapshot(tmp_path: Path):
    try:
        module._build_public_studies([{"version": "mosaic-study-inspection/v1", "scientifically_accepted": False,
                                      "paper": {"paper_id": "arbitrary"}, "studies": [], "analyses": []}], identities={})
    except ValueError as exc:
        assert str(exc) == "public_study_direct_inventory_refused"
    else:
        raise AssertionError("arbitrary direct inventory accepted")
    try:
        module.build_from_public_handoffs([{"capture": {"paper_id": PID, "source_pdf_sha256": PDF, "source_packet_sha256": PACKET}, "snapshot": _snapshot()}], identities={})
    except ValueError as exc:
        assert str(exc) == "public_study_snapshot_reference_invalid"
    else:
        raise AssertionError("inline snapshot accepted")


def test_rejects_unvalidated_page_list_or_inventory_without_units(tmp_path: Path):
    bad_pages = _snapshot(pages=(2,))
    try:
        module.build_from_public_handoffs([_handoff(tmp_path, bad_pages)], identities={})
    except ValueError as exc:
        assert str(exc) == "public_study_source_witness_invalid"
    else:
        raise AssertionError("mismatched pages accepted")
    missing = _snapshot(); missing["source"].pop("units")
    try:
        module.build_from_public_handoffs([_handoff(tmp_path, missing)], identities={})
    except ValueError as exc:
        assert str(exc) == "public_study_source_units_missing"
    else:
        raise AssertionError("inventory without units accepted")


def test_missing_inventory_allows_empty_units_as_explicit_unavailable_candidate(tmp_path: Path):
    snapshot = _snapshot(candidate=False)
    snapshot["source"].pop("units")
    manifest, files = module.build_from_public_handoffs([_handoff(tmp_path, snapshot)], identities={})
    row = manifest["papers"][0]
    rendered = json.loads(files[row["data_path"]])
    assert row["study_records"] == row["analysis_records"] == 0
    assert rendered["limitations"] == ["Core study inventory was unavailable; no study records are displayed."]


def test_actual_public_handoff_fixture_satisfies_hardened_inventory_contract(tmp_path: Path):
    from test_public_corpus_pipeline import build_reviewed_fixture
    _, completed, source = build_reviewed_fixture(tmp_path / "generated-pdf")
    assert completed["status"] == "candidate_corpus_served"
    paper_id = source["capture"]["paper_id"]
    manifest, files = module.build_from_public_handoffs([source], identities={})
    rendered = json.loads(files[manifest["papers"][0]["data_path"]])
    assert rendered["paper"]["paper_id"] == paper_id
    assert rendered["studies"][0]["identifier_evidence"][0]["pages"]



def test_multi_unit_statement_uses_projectors_literal_newline_join(tmp_path: Path):
    snapshot = _snapshot()
    source = snapshot["source"]
    source["units"].append({"unit_id": "u2", "text": "Second source unit.", "page": 2})
    source["units_sha256"] = sha256(canonical_bytes(source["units"])).hexdigest()
    quote = snapshot["core_inventory_candidate"]["studies"][0]["identifier_evidence"][0]
    quote.update({"quote": "Study 1 reports a fictional measure.\nSecond source unit.",
                  "segments": [quote["segments"][0], {"unit_id": "u2", "start": 0, "end": 19,
                                                     "text": "Second source unit.", "page": 2}],
                  "pages": [1, 2]})
    snapshot["core_inventory_candidate"]["source_identity"]["eligible_source_units_sha256"] = source["units_sha256"]
    manifest, files = module.build_from_public_handoffs([_handoff(tmp_path, snapshot)], identities={})
    rendered = json.loads(files[manifest["papers"][0]["data_path"]])
    assert rendered["studies"][0]["identifier_evidence"][0]["quote"] == quote["quote"]
