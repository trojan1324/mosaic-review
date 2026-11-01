"""Explicit authoring of source-bound public display-review records.

This helper never infers bibliographic fields or asserts that a source was
reviewed.  An operator supplies every value, literal selector, and explicit
human/agent attestation; the helper merely verifies those selections against a
frozen public native reader and constructs the canonical receipts required by
the public corpus materializer.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from mosaic.canonical import canonical_bytes
from mosaic.public_native_runtime import PublicNativePreparationRefusal, load_capture_envelope
from mosaic.public_reader import is_own_article_metadata_unit
from mosaic.public_serving import _native_reader
from mosaic.workspace import ImmutableArtifactError, Workspace

VERSION = "mosaic-public-corpus-reviews/v1"
CATALOGUE_VERSION = "mosaic-public-review-catalogue/v1"
_IDENTITY_FIELDS = ("title", "authors", "journal", "year")
_APA_SUPPLEMENTS = frozenset({"volume", "issue", "pages_verbatim", "article_number", "doi"})
_ATTESTORS = frozenset({"human_source_review", "agent_source_layout_review"})


class PublicReviewRefusal(ValueError):
    """A deterministic public-review input refusal."""


def _need(condition: bool, code: str) -> None:
    if not condition:
        raise PublicReviewRefusal(code)


def _capture(capture_root: Path | str, *, workspace: Workspace) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = workspace.check(Path(capture_root).absolute())
    try:
        envelope = load_capture_envelope(root, workspace=workspace)
        receipt = envelope["capture_receipt"]
        native = json.loads((root / "native-reader.json").read_bytes())
        reader = _native_reader(native, receipt["source_pdf_sha256"])
    except (OSError, ValueError, PublicNativePreparationRefusal) as exc:
        code = str(exc)
        raise PublicReviewRefusal(code if code.startswith("public_") else "public_review_capture_invalid") from exc
    units = {unit["unit_id"]: dict(unit) for unit in reader["units"]}
    _need(bool(units), "public_review_native_units_missing")
    return dict(receipt), units


def native_unit_catalogue(capture_root: Path | str, *, workspace: Workspace) -> dict[str, Any]:
    """Expose reviewable native-reader units without changing capture evidence."""
    receipt, units = _capture(capture_root, workspace=workspace)
    return {"version": CATALOGUE_VERSION, "paper_id": receipt["work_id"],
            "source_pdf_sha256": receipt["source_pdf_sha256"], "scientifically_accepted": False,
            "units": [{key: unit.get(key) for key in ("unit_id", "page", "source_locator", "start", "end", "quote", "quote_sha256")}
                      for unit in sorted(units.values(), key=lambda row: (row["start"], row["unit_id"]))]}


def empty_review_draft(capture_root: Path | str, *, workspace: Workspace) -> dict[str, Any]:
    """Return a valid but deliberately unadmitted ``--reviews`` document."""
    receipt, _ = _capture(capture_root, workspace=workspace)
    return {"version": VERSION, "papers": {receipt["work_id"]: {}}}


def merge_reviews(reviews: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Combine immutable per-paper review records for one corpus invocation.

    ``finalize`` deliberately writes one source-bound record at a time.  A
    corpus can contain several explicitly selected PDFs, so its ``--reviews``
    input needs one immutable envelope with all of those records.  This helper
    only combines already-authored records: later corpus materialization still
    verifies every receipt and literal witness against its own capture.
    """
    _need(bool(reviews), "public_review_merge_inputs_required")
    papers: dict[str, Any] = {}
    for review in reviews:
        _need(isinstance(review, Mapping) and set(review) == {"version", "papers"}
              and review.get("version") == VERSION and isinstance(review.get("papers"), Mapping),
              "public_review_merge_input_invalid")
        for paper_id, record in review["papers"].items():
            _need(isinstance(paper_id, str) and paper_id.startswith("public-")
                  and isinstance(record, Mapping), "public_review_merge_input_invalid")
            _need(paper_id not in papers, "public_review_merge_duplicate_paper")
            papers[paper_id] = dict(record)
    return {"version": VERSION, "papers": papers}


def _attestation(value: object) -> dict[str, str]:
    _need(isinstance(value, Mapping) and set(value) == {"reviewer_type", "version"}, "public_review_attestation_invalid")
    reviewer_type, version = value.get("reviewer_type"), value.get("version")
    _need(reviewer_type in _ATTESTORS and isinstance(version, str) and version.strip(), "public_review_attestation_required")
    return {"reviewer_type": reviewer_type, "version": version.strip()}


def _selection(value: object, units: Mapping[str, Mapping[str, Any]], source_sha: str, *, own_article_metadata: bool = False) -> tuple[Any, dict[str, Any]]:
    _need(isinstance(value, Mapping) and set(value) == {"value", "unit_id", "start", "end"}, "public_review_selector_invalid")
    unit_id, start, end = value.get("unit_id"), value.get("start"), value.get("end")
    unit = units.get(unit_id) if isinstance(unit_id, str) else None
    _need(unit is not None and type(start) is int and type(end) is int and unit["start"] <= start < end <= unit["end"],
          "public_review_selector_invalid")
    _need(not own_article_metadata or is_own_article_metadata_unit(unit), "public_review_selector_not_own_article_metadata")
    text = unit["quote"][start - unit["start"]:end - unit["start"]]
    selected = value.get("value")
    _need(isinstance(selected, (str, int)) and not isinstance(selected, bool) and str(selected) == text and bool(text),
          "public_review_selector_not_literal")
    return selected, {"unit_id": unit_id, "text": text, "start": start, "end": end,
                      "source_pdf_sha256": source_sha, "quote_sha256": unit["quote_sha256"]}


def _nonoverlap(witnesses: list[Mapping[str, Any]]) -> None:
    grouped: dict[str, list[tuple[int, int]]] = {}
    for witness in witnesses:
        grouped.setdefault(witness["unit_id"], []).append((witness["start"], witness["end"]))
    for spans in grouped.values():
        spans.sort()
        _need(all(left[1] <= right[0] for left, right in zip(spans, spans[1:])), "public_review_selector_overlap")


def _receipt(status: str, attestation: Mapping[str, str], source_sha: str, payload: object) -> dict[str, Any]:
    receipt = {"source_pdf_sha256": source_sha, "reviewer_type": attestation["reviewer_type"],
               "version": attestation["version"], "reviewed_payload_sha256": sha256(canonical_bytes(payload)).hexdigest()}
    return {"status": status, "reviewer_type": attestation["reviewer_type"], "version": attestation["version"],
            "receipt": receipt, "receipt_sha256": sha256(canonical_bytes(receipt)).hexdigest()}


def finalize_review(capture_root: Path | str, *, identity: Mapping[str, Any], attestation: Mapping[str, Any],
                    apa: Mapping[str, Any] | None = None, workspace: Workspace) -> dict[str, Any]:
    """Construct a valid review only from explicit literal field selections."""
    receipt, units = _capture(capture_root, workspace=workspace)
    source_sha, paper_id = receipt["source_pdf_sha256"], receipt["work_id"]
    actor = _attestation(attestation)
    _need(isinstance(identity, Mapping) and set(identity) == set(_IDENTITY_FIELDS), "public_review_identity_fields_required")
    fields: dict[str, dict[str, Any]] = {}
    witnesses = []
    for name in _IDENTITY_FIELDS:
        selected, witness = _selection(identity[name], units, source_sha, own_article_metadata=True)
        if name == "year":
            _need(type(selected) is int and 1500 <= selected <= 2200, "public_review_year_invalid")
        else:
            _need(isinstance(selected, str) and selected.strip(), "public_review_identity_value_invalid")
        fields[name] = {"value": selected, "witness": witness}
        witnesses.append(witness)
    _nonoverlap(witnesses)
    identity_review = {**_receipt("source_reviewed_complete", actor, source_sha, fields), "fields": fields}
    paper: dict[str, Any] = {"identity_review": identity_review}
    if apa is not None:
        _need(isinstance(apa, Mapping) and set(apa) == {"authors_apa", "title_apa", "supplements"}, "public_review_apa_invalid")
        authors = apa.get("authors_apa")
        title_apa = apa.get("title_apa")
        raw_supplements = apa.get("supplements")
        # This binds an operator-curated rendering to the printed wording; it
        # intentionally does not determine whether its capitalization is APA-correct.
        _need(isinstance(authors, list) and authors and all(isinstance(item, str) and item.strip() for item in authors)
              and isinstance(title_apa, str) and title_apa.strip()
              and title_apa.casefold() == str(fields["title"]["value"]).casefold()
              and isinstance(raw_supplements, Mapping) and set(raw_supplements) <= _APA_SUPPLEMENTS,
              "public_review_apa_invalid")
        supplements: dict[str, dict[str, Any]] = {}
        supplement_witnesses = []
        for name, selection in raw_supplements.items():
            selected, witness = _selection(selection, units, source_sha, own_article_metadata=True)
            supplements[name] = {"value": selected, "witness": witness}
            supplement_witnesses.append(witness)
        _nonoverlap(supplement_witnesses)
        payload = {"authors_apa": list(authors), "title_apa": title_apa,
                   "byline_witness": dict(fields["authors"]["witness"]),
                   "supplements": supplements}
        paper["apa_review"] = {**_receipt("source_reviewed_apa_rendering", actor, source_sha, payload), **payload}
    return {"version": VERSION, "papers": {paper_id: paper}}


def write_review(path: Path | str, review: Mapping[str, Any], *, workspace: Workspace) -> Path:
    """Write an immutable canonical review/catalogue artifact."""
    try:
        return workspace.write_canonical_json(Path(path).absolute(), dict(review))
    except ImmutableArtifactError as exc:
        raise PublicReviewRefusal("public_review_output_immutable") from exc


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise PublicReviewRefusal("public_review_input_invalid") from exc
    _need(isinstance(value, dict), "public_review_input_invalid")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("catalogue", "draft", "finalize", "merge"))
    parser.add_argument("--capture-root", type=Path,
                        help="One captured paper root; required except when combining existing review files.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--attestation", type=Path)
    parser.add_argument("--apa", type=Path)
    parser.add_argument("--input", type=Path, action="append",
                        help="Immutable review JSON to combine; repeat for each paper when using merge.")
    args = parser.parse_args(argv)
    workspace = Workspace()
    try:
        if args.command in ("catalogue", "draft", "finalize") and args.capture_root is None:
            parser.error(f"{args.command} requires --capture-root")
        if args.command == "catalogue":
            value = native_unit_catalogue(args.capture_root, workspace=workspace)
        elif args.command == "draft":
            value = empty_review_draft(args.capture_root, workspace=workspace)
        elif args.command == "finalize":
            if args.identity is None or args.attestation is None:
                parser.error("finalize requires --identity and --attestation")
            value = finalize_review(args.capture_root, identity=_json(args.identity), attestation=_json(args.attestation),
                                    apa=_json(args.apa) if args.apa else None, workspace=workspace)
        else:
            if not args.input:
                parser.error("merge requires one or more --input review file")
            value = merge_reviews([_json(path) for path in args.input])
        written = write_review(args.output, value, workspace=workspace)
        print(json.dumps({"status": "written", "path": str(written), "scientifically_accepted": False}, sort_keys=True))
        return 0
    except PublicReviewRefusal as exc:
        print(json.dumps({"status": "refused", "reason": str(exc), "scientifically_accepted": False}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
