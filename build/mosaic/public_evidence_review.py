"""Author literal reviews for public measures and citation edges.

This module packages an operator's exact native-reader selections.  It never
discovers a measure, interprets a response scale, resolves a cited work, or
claims rights to reproduce an item.  ``caller_attested_source_review`` is
deliberately distinct from human and agent-layout review in every receipt.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from mosaic.canonical import canonical_bytes
from mosaic.public_native_runtime import load_capture_envelope
from mosaic.public_serving import _native_reader
from mosaic.workspace import Workspace


VERSION = "mosaic-public-evidence-review/v1"
_ATTESTORS = frozenset({"caller_attested_source_review", "agent_source_layout_review", "human_source_review"})


class PublicEvidenceReviewRefusal(ValueError):
    """Typed refusal for an incomplete or nonliteral review selection."""


def _need(condition: bool, code: str) -> None:
    if not condition:
        raise PublicEvidenceReviewRefusal(code)


def _capture(capture_root: str, *, workspace: Workspace) -> tuple[dict, dict[str, dict]]:
    envelope = load_capture_envelope(capture_root, workspace=workspace)
    receipt = envelope["capture_receipt"]
    reader = _native_reader((workspace.check(capture_root) / "native-reader.json"), receipt["source_pdf_sha256"])
    return receipt, {unit["unit_id"]: unit for unit in reader["units"]}


def _attestation(value: Mapping[str, Any]) -> dict:
    _need(isinstance(value, Mapping) and set(value) == {"reviewer_type", "version"}, "public_evidence_attestation_invalid")
    reviewer_type, version = value.get("reviewer_type"), value.get("version")
    _need(reviewer_type in _ATTESTORS and isinstance(version, str) and version.strip(), "public_evidence_attestation_invalid")
    return {"reviewer_type": reviewer_type, "version": version.strip()}


def _selection(value: Mapping[str, Any], units: Mapping[str, Mapping[str, Any]], source_sha: str, *, lane: str | None = None) -> dict:
    _need(isinstance(value, Mapping) and set(value) == {"value", "unit_id", "start", "end"}, "public_evidence_selector_invalid")
    unit = units.get(value.get("unit_id"))
    start, end, text = value.get("start"), value.get("end"), value.get("value")
    _need(isinstance(unit, Mapping) and isinstance(text, str) and text and type(start) is int and type(end) is int
          and unit["start"] <= start < end <= unit["end"] and (lane is None or unit.get("lane") == lane),
          "public_evidence_selector_invalid")
    _need(unit["quote"][start-unit["start"]:end-unit["start"]] == text, "public_evidence_selector_not_literal")
    return {"unit_id": unit["unit_id"], "text": text, "start": start, "end": end,
            "source_pdf_sha256": source_sha, "quote_sha256": unit["quote_sha256"]}


def _receipt(status: str, attestation: Mapping[str, str], source_sha: str, payload: object) -> dict:
    receipt = {"source_pdf_sha256": source_sha, "reviewer_type": attestation["reviewer_type"],
               "version": attestation["version"], "reviewed_payload_sha256": sha256(canonical_bytes(payload)).hexdigest()}
    return {"status": status, "reviewer_type": attestation["reviewer_type"], "version": attestation["version"],
            "receipt": receipt, "receipt_sha256": sha256(canonical_bytes(receipt)).hexdigest()}


def review_measure(capture_root: str, *, label: Mapping[str, Any], items: list[Mapping[str, Any]],
                   attestation: Mapping[str, Any], workspace: Workspace) -> dict:
    """Return one source-reviewed measure candidate and verbatim item selections."""
    receipt, units = _capture(capture_root, workspace=workspace)
    actor = _attestation(attestation)
    label_witness = _selection(label, units, receipt["source_pdf_sha256"])
    _need(isinstance(items, list) and items, "public_evidence_items_required")
    item_witnesses = [_selection(item, units, receipt["source_pdf_sha256"]) for item in items]
    _need(len({(item["unit_id"], item["start"], item["end"]) for item in item_witnesses}) == len(item_witnesses),
          "public_evidence_duplicate_item_selector")
    payload = {"label": label_witness["text"].strip(), "witness": label_witness, "items": item_witnesses}
    return {"label": payload["label"], "literal_evidence": label_witness["text"], "witness": label_witness,
            "item_texts": item_witnesses, "review": _receipt("source_reviewed_meaningful_instrument", actor,
                                                                 receipt["source_pdf_sha256"], payload)}


def review_citation_edge(capture_root: str, *, literal_reference: Mapping[str, Any], source_paper_id: str,
                         target_paper_id: str, attestation: Mapping[str, Any], workspace: Workspace) -> dict:
    """Return one literal printed-reference candidate and an explicitly reviewed edge."""
    receipt, units = _capture(capture_root, workspace=workspace)
    _need(source_paper_id == receipt["work_id"] and isinstance(target_paper_id, str) and target_paper_id.startswith("public-"),
          "public_evidence_edge_identity_invalid")
    actor = _attestation(attestation)
    witness = _selection(literal_reference, units, receipt["source_pdf_sha256"], lane="references")
    reference_id = "reference:" + sha256(canonical_bytes([source_paper_id, witness])).hexdigest()
    payload = {"reference_id": reference_id, "source_paper_id": source_paper_id, "target_paper_id": target_paper_id}
    return {"literal_reference": witness["text"], "witness": witness, "target_paper_id": target_paper_id,
            "edge_status": "reviewed_edge", "edge_review": _receipt("source_reviewed_complete", actor,
                                                                          receipt["source_pdf_sha256"], payload)}


def _input(path: Path) -> dict:
    from mosaic.public_build import _no_links
    path = Workspace().check(path.absolute())
    if any(part.casefold() in {"sealed", "references", "diagnostic", "_holdback", "independent-reference", "jev_eval", "calibration", ".git"}
           or part.casefold().startswith(".env") or part.casefold().endswith(".private.json") for part in path.parts):
        raise PublicEvidenceReviewRefusal("public_evidence_private_input_refused")
    _no_links(path, Path(path.anchor))
    value = json.loads(path.read_bytes())
    _need(isinstance(value, dict), "public_evidence_input_invalid")
    return value


def _merge(existing: Mapping[str, Any] | None, paper_id: str, key: str, value: dict) -> dict:
    if existing is None:
        papers: dict[str, Any] = {}
    else:
        _need(isinstance(existing, Mapping) and set(existing) == {"version", "papers"}
              and existing.get("version") == "mosaic-public-corpus-reviews/v1" and isinstance(existing.get("papers"), Mapping),
              "public_evidence_existing_reviews_invalid")
        papers = {name: dict(record) for name, record in existing["papers"].items()}
    record = dict(papers.get(paper_id, {}))
    _need(record.keys() <= {"identity_review", "apa_review", "references", "measures", "advisory_sidecars"},
          "public_evidence_existing_reviews_invalid")
    values = list(record.get(key, []))
    _need(value not in values, "public_evidence_duplicate_review")
    values.append(value); record[key] = values; papers[paper_id] = record
    return {"version": "mosaic-public-corpus-reviews/v1", "papers": papers}


def main(argv: list[str] | None = None) -> int:
    """Write a new review envelope without changing an existing record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("measure", "edge"))
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True,
                        help="JSON selectors and an explicit attestation; all text must match the native reader exactly.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--existing-reviews", type=Path,
                        help="Existing immutable review envelope to copy and extend; it is never overwritten.")
    args = parser.parse_args(argv)
    workspace = Workspace()
    try:
        selection = _input(args.selection)
        receipt, _ = _capture(str(args.capture_root.absolute()), workspace=workspace)
        if args.command == "measure":
            _need(set(selection) == {"label", "items", "attestation"}, "public_evidence_measure_input_invalid")
            value = review_measure(str(args.capture_root.absolute()), label=selection["label"], items=selection["items"],
                                   attestation=selection["attestation"], workspace=workspace)
            key = "measures"
        else:
            _need(set(selection) == {"literal_reference", "target_paper_id", "attestation"}, "public_evidence_edge_input_invalid")
            value = review_citation_edge(str(args.capture_root.absolute()), literal_reference=selection["literal_reference"],
                                         source_paper_id=receipt["work_id"], target_paper_id=selection["target_paper_id"],
                                         attestation=selection["attestation"], workspace=workspace)
            key = "references"
        merged = _merge(_input(args.existing_reviews) if args.existing_reviews else None, receipt["work_id"], key, value)
        written = workspace.write_canonical_json(args.output.absolute(), merged)
        print(json.dumps({"status": "written", "path": str(written), "paper_id": receipt["work_id"],
                          "scientifically_accepted": False}, sort_keys=True))
        return 0
    except (OSError, ValueError, PublicEvidenceReviewRefusal) as exc:
        code = str(exc) if str(exc).startswith("public_") else "public_evidence_review_refused"
        print(json.dumps({"status": "refused", "reason": code, "scientifically_accepted": False}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
