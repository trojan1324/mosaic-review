"""Explicit-source review authoring over the generated public PDF fixture."""
from copy import deepcopy
import json
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pymupdf
import pytest

from mosaic.public_corpus import materialize_public_corpus
from mosaic.public_review import (PublicReviewRefusal, empty_review_draft, finalize_review, merge_reviews,
                                  main as review_main, native_unit_catalogue, write_review)
from mosaic.public_workflow import run_public_corpus, source_handoff
from test_public_corpus_pipeline import build_reviewed_fixture
from test_public_execution import _Loopback
from test_public_workflow import _inputs
from mosaic.workspace import Workspace


def _selection(catalogue, text, value=None):
    unit = next(row for row in catalogue["units"] if text in row["quote"])
    start = unit["start"] + unit["quote"].index(text)
    return {"value": text if value is None else value, "unit_id": unit["unit_id"],
            "start": start, "end": start + len(text)}


def _identity(catalogue):
    return {"title": _selection(catalogue, "Study 1: Coordination in a Fictional Team"),
            "authors": _selection(catalogue, "Ruth Keller"),
            "journal": _selection(catalogue, "Fictional Methods Journal"),
            "year": _selection(catalogue, "2024", 2024)}


def test_explicit_attested_review_admits_existing_public_corpus_fixture(tmp_path):
    _, _, source = build_reviewed_fixture(tmp_path)
    workspace = Workspace(repo_root=tmp_path)
    capture = source["capture"]["capture_receipt_path"].parent
    catalogue = native_unit_catalogue(capture, workspace=workspace)
    draft = empty_review_draft(capture, workspace=workspace)
    paper_id = catalogue["paper_id"]
    assert draft["papers"][paper_id] == {}

    withheld = deepcopy(source)
    withheld.pop("identity_review"); withheld.pop("apa_review")
    withheld.update(draft["papers"][paper_id])
    result = materialize_public_corpus([withheld], tmp_path / "draft-view", workspace=workspace)
    assert result["admitted_identity_count"] == result["apa_count"] == 0

    review = finalize_review(capture, identity=_identity(catalogue),
                             attestation={"reviewer_type": "human_source_review", "version": "review-v1"},
                             apa={"authors_apa": ["Keller, R."], "title_apa": "Study 1: Coordination in a fictional team",
                                  "supplements": {"volume": _selection(catalogue, "12"),
                                                  "pages_verbatim": _selection(catalogue, "10-19")}},
                             workspace=workspace)
    review_path = write_review(tmp_path / "reviews.json", review, workspace=workspace)
    assert review_path.is_file()
    with pytest.raises(PublicReviewRefusal, match="output_immutable"):
        write_review(review_path, draft, workspace=workspace)
    admitted = deepcopy(withheld)
    admitted.update(review["papers"][paper_id])
    final = materialize_public_corpus([admitted], tmp_path / "reviewed-from-helper", workspace=workspace)
    assert final["admitted_identity_count"] == final["apa_count"] == 1
    with pytest.raises(PublicReviewRefusal, match="attestation_required"):
        finalize_review(capture, identity=_identity(catalogue),
                        attestation={"reviewer_type": "caller_attested_source_review", "version": "review-v1"},
                        workspace=workspace)


def test_review_refuses_missing_attestation_bad_selector_overlap_and_capture_drift(tmp_path):
    _, _, source = build_reviewed_fixture(tmp_path)
    workspace = Workspace(repo_root=tmp_path)
    capture = source["capture"]["capture_receipt_path"].parent
    catalogue = native_unit_catalogue(capture, workspace=workspace)
    identity = _identity(catalogue)
    with pytest.raises(PublicReviewRefusal, match="attestation_required"):
        finalize_review(capture, identity=identity,
                        attestation={"reviewer_type": "caller_attested_source_review", "version": "v1"}, workspace=workspace)
    overlapping = dict(identity)
    overlapping["journal"] = dict(identity["authors"])
    with pytest.raises(PublicReviewRefusal, match="selector_overlap"):
        finalize_review(capture, identity=overlapping,
                        attestation={"reviewer_type": "agent_source_layout_review", "version": "v1"}, workspace=workspace)
    bad = deepcopy(identity)
    bad["title"]["end"] += 1
    with pytest.raises(PublicReviewRefusal, match="selector_not_literal"):
        finalize_review(capture, identity=bad,
                        attestation={"reviewer_type": "agent_source_layout_review", "version": "v1"}, workspace=workspace)
    capture.joinpath("source.pdf").write_bytes(b"not the frozen PDF")
    with pytest.raises(PublicReviewRefusal, match="pdf_hash_mismatch"):
        native_unit_catalogue(capture, workspace=workspace)


def test_review_refuses_identity_fields_selected_from_literal_reference_lane(tmp_path):
    _, _, source = build_reviewed_fixture(tmp_path)
    workspace = Workspace(repo_root=tmp_path)
    capture = source["capture"]["capture_receipt_path"].parent
    catalogue = native_unit_catalogue(capture, workspace=workspace)
    reference = next(row for row in catalogue["units"] if "A fictional reference." in row["quote"])

    def from_reference(text, value=None):
        start = reference["start"] + reference["quote"].index(text)
        return {"value": text if value is None else value, "unit_id": reference["unit_id"],
                "start": start, "end": start + len(text)}

    cited_identity = {"title": from_reference("A fictional reference."),
                      "authors": from_reference("Keller, R."),
                      "journal": from_reference("Fictional Methods Journal"),
                      "year": from_reference("2024", 2024)}
    with pytest.raises(PublicReviewRefusal, match="not_own_article_metadata"):
        finalize_review(capture, identity=cited_identity,
                        attestation={"reviewer_type": "human_source_review", "version": "review-v1"}, workspace=workspace)


def test_merge_cli_drives_two_distinct_captures_to_reviewed_materialization(tmp_path):
    """Two source-bound reviews can be combined without copying either record."""
    config, kwargs = _inputs(tmp_path)
    workspace = kwargs["workspace"]

    def write_pdf(name, title, author, journal, year):
        with pymupdf.open() as document:
            page = document.new_page()
            for index, line in enumerate((title, author, journal, str(year), "Volume 12, pages 10-19",
                                          "Abstract", "A fictional source for the public review workflow.",
                                          "Methods", "Results", "References", "A fictional reference.")):
                page.insert_text((50, 50 + index * 35), line, fontsize=10)
            workspace.write_once(tmp_path / "sources" / name, document.tobytes())

    first = ("first.pdf", "Study 1: Coordination", "Ruth Keller", "Fictional Methods Journal", 2024)
    second = ("second.pdf", "Study 2: Cooperation", "Mina Shah", "Fictional Organization Review", 2025)
    write_pdf(*first); write_pdf(*second)
    config["sources"]["files"] = [first[0], second[0]]
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Loopback)
    thread = Thread(target=server.serve_forever); thread.start()
    config["runtime"]["base_url"] = f"http://127.0.0.1:{server.server_address[1]}/v1"
    try:
        run_public_corpus(config, **kwargs, live=True)
    finally:
        server.shutdown(); thread.join(); server.server_close()

    reviews, sources = [], []
    for _, title, author, journal, year in (first, second):
        capture = next(path.parent for path in (tmp_path / "run/corpus/stages/capture").rglob("native-reader.json")
                       if title in path.read_text(encoding="utf-8"))
        snapshot = next(path for path in (tmp_path / "run/corpus/native").rglob("public-snapshot.json")
                        if title in path.read_text(encoding="utf-8"))
        receipt = json.loads((snapshot.parent / "execution-receipt.json").read_bytes())
        report = snapshot.parent / "driver" / receipt["paper_report_path"]
        catalogue = native_unit_catalogue(capture, workspace=workspace)
        identity = {"title": _selection(catalogue, title), "authors": _selection(catalogue, author),
                    "journal": _selection(catalogue, journal), "year": _selection(catalogue, str(year), year)}
        review = finalize_review(capture, identity=identity,
                                 attestation={"reviewer_type": "agent_source_layout_review", "version": "fictional-test-v1"},
                                 apa={"authors_apa": [author.rsplit(" ", 1)[1] + ", " + author[0] + "."], "title_apa": title,
                                      "supplements": {"volume": _selection(catalogue, "12"),
                                                      "pages_verbatim": _selection(catalogue, "10-19")}},
                                 workspace=workspace)
        review_path = write_review(tmp_path / "reviews" / f"{year}.json", review, workspace=workspace)
        reviews.append(review_path)
        sources.append(source_handoff(capture, report, snapshot, workspace=workspace))

    merged_path = tmp_path / "reviews" / "corpus.json"
    assert review_main(["merge", "--input", str(reviews[0]), "--input", str(reviews[1]),
                        "--output", str(merged_path)]) == 0
    merged = json.loads(merged_path.read_bytes())
    assert len(merged["papers"]) == 2
    for source in sources:
        source.update(merged["papers"][source["capture"]["paper_id"]])
    result = materialize_public_corpus(sources, tmp_path / "two-reviewed", workspace=workspace)
    assert result["admitted_identity_count"] == result["apa_count"] == 2

    copied = deepcopy(sources)
    copied[1].update(merged["papers"][copied[0]["capture"]["paper_id"]])
    withheld = materialize_public_corpus(copied, tmp_path / "copied-review", workspace=workspace)
    assert withheld["admitted_identity_count"] == withheld["apa_count"] == 1
    queue = json.loads((tmp_path / "copied-review/identity-review-queue.json").read_bytes())
    assert any(row["reason"] == "identity_literal_witness_incomplete" for row in queue["items"])

    with pytest.raises(PublicReviewRefusal, match="merge_duplicate_paper"):
        merge_reviews([merged, merged])
    with pytest.raises(PublicReviewRefusal, match="merge_input_invalid"):
        merge_reviews([{"version": "wrong", "papers": {}}])
