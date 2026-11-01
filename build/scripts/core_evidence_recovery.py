"""Conservative field-local recovery for core-evidence proposals.

This is deliberately not a replacement for ``core_evidence_trial``.  It keeps
the frozen task/schema checks, but turns a bad name or unsupported field into a
quarantined field instead of rejecting an otherwise useful whole-paper proposal.
"""
from __future__ import annotations

from copy import deepcopy
import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from mosaic.canonical import canonical_bytes
from mosaic.workspace import Workspace
from scripts.core_evidence_trial import validate as validate_core_task
STRICT_VERSION = "core-evidence-v1"


RECOVERY_VERSION = "core-evidence-recovery-v2"


def _word_pattern(value: str) -> str:
    """Match a name as a standalone sequence, never a substring of another word."""
    before = r"(?<!\w)" if value and (value[0].isalnum() or value[0] == "_") else ""
    after = r"(?!\w)" if value and (value[-1].isalnum() or value[-1] == "_") else ""
    return before + re.escape(value) + after


def _normalised_pattern(value: str) -> str | None:
    """Permit only case/space/line-hyphen variation between the same name words."""
    tokens = re.findall(r"\S+", value)
    if not tokens:
        return None
    # A literal hyphen at a line break is a source-layout artifact; ordinary
    # punctuation is deliberately not normalised.
    join = r"(?:\s+|-\s*\n\s*)"
    core = join.join(re.escape(token) for token in tokens)
    before = r"(?<!\w)" if tokens[0][0].isalnum() or tokens[0][0] == "_" else ""
    after = r"(?!\w)" if tokens[-1][-1].isalnum() or tokens[-1][-1] == "_" else ""
    return before + core + after


def source_name_mapping(proposal_name: str, source_text: str) -> dict[str, Any] | None:
    """Return one source span for an exact or uniquely normalised name.

    Repeated exact occurrences preserve all matching spans. Normalized matches
    must be unique; this function does not choose between differently written
    source names. Matching text never establishes a semantic variable role.
    """
    if not isinstance(proposal_name, str) or not proposal_name:
        return None
    exact = list(re.finditer(_word_pattern(proposal_name), source_text))
    if exact:
        match = exact[0]
        return {"proposal_name": proposal_name, "source_name": match.group(0),
                "match_type": "exact", "source_span": {"start": match.start(), "end": match.end(), "text": match.group(0)},
                "all_exact_source_spans": [{"start": m.start(), "end": m.end(), "text": m.group(0)} for m in exact]}
    pattern = _normalised_pattern(proposal_name)
    if not pattern:
        return None
    normalised = list(re.finditer(pattern, source_text, flags=re.IGNORECASE))
    if len(normalised) != 1:
        return None
    match = normalised[0]
    return {"proposal_name": proposal_name, "source_name": match.group(0),
            "match_type": "case_insensitive_or_layout_normalised",
            "source_span": {"start": match.start(), "end": match.end(), "text": match.group(0)}}


def _evidence(table: dict[str, dict[str, Any]], ids: list[str]) -> list[dict[str, Any]]:
    return [deepcopy(table[sentence_id]) for sentence_id in ids]


def _validate_frozen_task(task: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Regenerate the statement table from its immutable source units."""
    return validate_core_task(task)


def _quarantine(result: dict[str, Any], path: str, original: Any, reason: str) -> None:
    result["quarantined_fields"].append({"path": path, "original_proposal": deepcopy(original), "reason": reason})


def _recover_name(result: dict[str, Any], table: dict[str, dict[str, Any]], name: str,
                  sentence_id: str, path: str) -> dict[str, Any] | None:
    mapping = source_name_mapping(name, table[sentence_id]["text"])
    if mapping is None:
        _quarantine(result, path, {"name": name, "name_sentence_id": sentence_id}, "name_not_exact_or_uniquely_normalisable_in_cited_sentence")
        return None
    return {"name": mapping["source_name"], "name_evidence": _evidence(table, [sentence_id]), "name_mapping": mapping}


def recover(proposal: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    """Recover independent valid fields while retaining source/schema validation.

    Roles retain their own role evidence and remain semantically unverified;
    matching a name never converts a proposed role into a verified one.
    """
    table = _validate_frozen_task(task)
    Draft202012Validator(task["schema"]).validate(proposal)
    result: dict[str, Any] = {
        "version": RECOVERY_VERSION,
        "strict_contract_version": STRICT_VERSION,
        "source_sha256": task["source_sha256"],
        "article_kind": proposal["article_kind"],
        "studies": [], "analyses": [], "quarantined_fields": [],
        "source_spans_verified": True, "exact_copy_by_construction": True,
        "semantic_labels_verified": False, "coverage_verified": False,
        "scientifically_accepted": False, "production_accepted": False,
        "recovery_is_not_promotion": True,
    }
    for study_index, study in enumerate(proposal["studies"]):
        out_study: dict[str, Any] = {
            "id": f"study-{study_index}", "status": "candidate_source_backed",
            "identifier_evidence": _evidence(table, study["identifier_sentence_ids"]),
            "population_evidence": _evidence(table, study["population_sentence_ids"]),
            "sample_size_evidence": _evidence(table, study["sample_size_sentence_ids"]),
            "measures": [], "semantic_labels_verified": False,
        }
        if study["design"] != "uncertain" and not study["design_sentence_ids"]:
            out_study.update({"design": "uncertain", "design_evidence": [], "design_status": "unresolved"})
            _quarantine(result, f"/studies/{study_index}/design", study["design"], "non_uncertain_design_without_source_witness")
        else:
            out_study.update({"design": study["design"], "design_evidence": _evidence(table, study["design_sentence_ids"]), "design_status": "proposed_source_witnessed" if study["design_sentence_ids"] else "unresolved"})
        for measure_index, measure in enumerate(study["measures"]):
            path = f"/studies/{study_index}/measures/{measure_index}"
            mapped = _recover_name(result, table, measure["name"], measure["name_sentence_id"], path + "/name")
            if mapped is None:
                continue
            if re.fullmatch(r"control\s+variables?", mapped["name"], flags=re.IGNORECASE):
                _quarantine(result, path, measure, "collective_control_label_is_not_a_measure")
                continue
            out_study["measures"].append({"id": f"study-{study_index}-measure-{measure_index}", **mapped,
                "description_evidence": _evidence(table, measure["description_sentence_ids"]),
                "item_evidence": _evidence(table, measure["item_sentence_ids"]),
                "status": "candidate_source_backed", "semantic_label_verified": False})
        result["studies"].append(out_study)
    for analysis_index, analysis in enumerate(proposal["analyses"]):
        valid_indices = [index for index in analysis["study_indices"] if index < len(result["studies"])]
        invalid_indices = [index for index in analysis["study_indices"] if index >= len(result["studies"])]
        if invalid_indices:
            _quarantine(result, f"/analyses/{analysis_index}/study_indices", analysis["study_indices"], "study_index_not_present_in_proposal")
        out_analysis: dict[str, Any] = {"id": f"analysis-{analysis_index}",
            "studies": [f"study-{index}" for index in valid_indices],
            "membership_status": "proposed" if valid_indices and not invalid_indices else "unresolved",
            "evidence": _evidence(table, analysis["evidence_sentence_ids"]), "roles": [],
            "status": "candidate_source_backed", "semantic_labels_verified": False}
        for role_index, role in enumerate(analysis["roles"]):
            path = f"/analyses/{analysis_index}/roles/{role_index}"
            mapped = _recover_name(result, table, role["variable_name"], role["name_sentence_id"], path + "/variable_name")
            if mapped is None:
                continue
            out_analysis["roles"].append({"id": f"analysis-{analysis_index}-role-{role_index}", "role": role["role"],
                "variable_name": mapped["name"], "name_evidence": mapped["name_evidence"], "name_mapping": mapped["name_mapping"],
                "role_evidence": _evidence(table, role["role_sentence_ids"]),
                "status": "candidate_role_unverified", "semantic_role_verified": False,
                "name_and_role_evidence_are_separate": role["name_sentence_id"] not in role["role_sentence_ids"]})
        result["analyses"].append(out_analysis)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    task_bytes = args.task.read_bytes()
    proposal_bytes = args.proposal.read_bytes()
    recovered = recover(json.loads(proposal_bytes), json.loads(task_bytes))
    recovered["input_task_sha256"] = hashlib.sha256(task_bytes).hexdigest()
    recovered["input_proposal_sha256"] = hashlib.sha256(proposal_bytes).hexdigest()
    Workspace(repo_root=Path(__file__).resolve().parents[1]).write_once(args.output, canonical_bytes(recovered))
    print(json.dumps({"output": str(args.output), "studies": len(recovered["studies"]), "analyses": len(recovered["analyses"]), "quarantined_fields": len(recovered["quarantined_fields"])}, sort_keys=True))


if __name__ == "__main__":
    main()
