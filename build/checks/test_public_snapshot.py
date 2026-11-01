from copy import deepcopy
import hashlib

import pytest

from mosaic.canonical import canonical_bytes
from mosaic.public_snapshot import PublicSnapshotRefused, corpus_report_to_public_snapshot
from scripts.complete_extraction_sequence import finalize_paper, prepare_measure_tasks, prepare_upstream
from scripts.study_fact_selections import finalize_with_study_facts, prepare as prepare_study_facts


RECIPE = {"vllm": {"served_model_name": "fixture"}}


def source():
    return {"units": [
        {"unit_id": "u1", "text": "Study 1 recruited 120 employees. We measured workload with three items.", "page": 1, "source_locator": "article:1"},
        {"unit_id": "u2", "text": "In Study 1, workload was the independent variable and fatigue was the dependent variable.", "page": 1, "source_locator": "article:1"},
        {"unit_id": "u3", "text": "Age was included as a control in the Study 1 fatigue model.", "page": 2, "source_locator": "article:2"},
    ], "source_policy_sha256": "b" * 64}


def plan():
    return prepare_upstream(RECIPE, {"paper_id": "public-synthetic", "title": "Public synthetic", "citation": "Synthetic", "source_pdf_sha256": "a" * 64, "source": source()})


def core():
    return {"article_kind": "original_empirical", "studies": [{"identifier_sentence_ids": ["c001"], "design": "observational", "design_sentence_ids": [], "population_sentence_ids": [], "sample_size_sentence_ids": [], "measures": [{"name": "workload", "name_sentence_id": "c001", "description_sentence_ids": [], "item_sentence_ids": []}]}], "analyses": [{"study_indices": [0], "evidence_sentence_ids": ["c002"], "roles": [{"role": "independent_variable", "variable_name": "workload", "name_sentence_id": "c002", "role_sentence_ids": ["c002"]}, {"role": "dependent_variable", "variable_name": "fatigue", "name_sentence_id": "c002", "role_sentence_ids": ["c002"]}]}]}


def findings():
    return {"article_kind": "original_empirical", "findings": [{"sentence_ids": ["c001"], "emphasis": "main", "attribution": "current_article_empirical"}], "qualifications": []}


def complete_report():
    p = plan()
    sequence = prepare_measure_tasks(p, core())
    answers = {stage["id"]: {"roles": {choice["choice_id"]: "unclear_or_redundant" for choice in stage["task"]["payload"]["choices"]}, "sample_witness": "not_stated", "items_elsewhere_witness": "not_stated"} for stage in sequence["measure_stages"] if stage["status"] == "awaiting_response"}
    return p, finalize_paper(p, findings_proposal=findings(), core_proposal=core(), measure_responses=answers)


def test_actual_successful_sequence_report_preserves_inventory_studies_and_measure_stages():
    p, report = complete_report()
    snapshot = corpus_report_to_public_snapshot(p, report)
    assert snapshot["core_inventory_candidate"] == report["core"]["materialized"]["inventory"]
    assert snapshot["studies_candidate"] == report["core"]["materialized"]["inventory"]["studies"]
    assert snapshot["measures_candidate"]["measure_stages"] == report["measures"]["measure_stages"]
    assert snapshot["source"]["units"] == source()["units"]
    assert snapshot["source_packet_candidate"] == p["source"]
    assert snapshot["source_packet_sha256"] == hashlib.sha256(canonical_bytes(p["source"])).hexdigest()
    assert snapshot["source_coverage"]["full_article_reader_coverage"] == "not_established"
    assert snapshot["validation_status"] == "candidate_only"


def test_current_runner_wrapper_validates_nested_digest_and_retains_wrapper_fields():
    p, original = complete_report()
    report = {"version": "complete-extraction-with-study-selections/v1", "original_report": original,
              "original_report_sha256": hashlib.sha256(canonical_bytes(original)).hexdigest(),
              "study_fact_stage": {"status": "unavailable", "reason": "study_fact_response_not_supplied"},
              "declared_unavailable_stages": [{"request_id": "paper-001-core", "reason": "declared", "declaration_sha256": "c" * 64}],
              "scientifically_accepted": False, "production_accepted": False, "complete_extraction_claimed": False}
    snapshot = corpus_report_to_public_snapshot(p, report)
    assert snapshot["candidate_report"] == report
    assert snapshot["study_fact_candidate"] == report["study_fact_stage"]
    assert any(row["field"] == "declared_unavailable_stage" for row in snapshot["field_dispositions"])
    bad = deepcopy(report); bad["original_report"]["paper"]["title"] = "changed"
    with pytest.raises(PublicSnapshotRefused, match="original_report_hash"):
        corpus_report_to_public_snapshot(p, bad)


def test_successful_real_study_fact_wrapper_retains_fact_and_source_witnesses():
    p, original = complete_report()
    task = prepare_study_facts(p, core())
    answer = {}
    for target in task["payload"]["targets"]:
        answer[target["study_id"]] = {"condition": [], "manipulation_description": [], "analysed_n": [], "excluded_n": [],
            "reliability": {measure["measure_id"]: [] for measure in target["measures"]}}
    target = task["payload"]["targets"][0]
    value = next(row for row in task["value_catalog"] if row["text"] == "120")
    answer[target["study_id"]]["analysed_n"] = [{"study_witness_id": target["identifier_sentence_ids"][0],
        "support_sentence_ids": [], "value_id": value["id"]}]
    report = finalize_with_study_facts(p, findings_proposal=findings(), core_proposal=core(),
        measure_responses={stage["id"]: {"roles": {choice["choice_id"]: "unclear_or_redundant" for choice in stage["task"]["payload"]["choices"]}, "sample_witness": "not_stated", "items_elsewhere_witness": "not_stated"} for stage in prepare_measure_tasks(p, core())["measure_stages"] if stage["status"] == "awaiting_response"},
        study_fact_task=task, study_fact_proposal=answer)
    assert report["study_fact_stage"]["status"] == "materialized_source_bound_proposal"
    snapshot = corpus_report_to_public_snapshot(p, report)
    assert snapshot["study_fact_candidate"]["materialized"]["facts"][0]["literal"] == "120"
    assert snapshot["study_fact_candidate"]["materialized"]["facts"][0]["evidence"]


def test_refuses_hash_claim_version_and_unknown_or_bad_witnesses():
    p, report = complete_report()
    bad_plan = deepcopy(p); bad_plan["source_identity"]["source_sha256"] = "0" * 64
    with pytest.raises(PublicSnapshotRefused, match="source_hash"):
        corpus_report_to_public_snapshot(bad_plan, report)
    bad_plan = deepcopy(p); bad_plan["source_identity"]["units_sha256"] = "0" * 64
    with pytest.raises(PublicSnapshotRefused, match="units_hash"):
        corpus_report_to_public_snapshot(bad_plan, report)
    bad_plan = deepcopy(p); bad_plan["source_identity"]["source_policy_sha256"] = "0" * 64
    with pytest.raises(PublicSnapshotRefused, match="source_policy_hash"):
        corpus_report_to_public_snapshot(bad_plan, report)
    bad = deepcopy(report); bad["complete_extraction_claimed"] = True
    with pytest.raises(PublicSnapshotRefused, match="claim"):
        corpus_report_to_public_snapshot(p, bad)
    bad = deepcopy(report); bad["version"] = "wrong"
    with pytest.raises(PublicSnapshotRefused, match="report_version"):
        corpus_report_to_public_snapshot(p, bad)
    bad = deepcopy(report); bad["unexpected"] = {}
    with pytest.raises(PublicSnapshotRefused, match="unknown_report"):
        corpus_report_to_public_snapshot(p, bad)
