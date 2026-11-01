"""Offline assembly for one source-bound, dependent extraction sequence.

This module deliberately does not run a model.  It prepares the two unchanged
whole-paper requests, accepts a later *new* core proposal, and only then makes
one target-only measure-field task per recovered measure.  A missing response
or an empty offered-choice set remains an explicit stage outcome; neither is
an absence claim about the article.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import re
from typing import Any, Mapping

from jsonschema import ValidationError
from mosaic.canonical import canonical_bytes
from scripts.core_evidence_trial import prepare as prepare_core
from scripts.core_linebreak_names import recover_linebreak_names
from scripts.core_declared_names import recover_declared_names
from scripts.core_rendered_names import recover_rendered_names, validate_layout_proof
from scripts.core_repeated_names import recover_repeated_names, VERSION as REPEATED_NAME_POLICY
from scripts.measure_catalog import build_catalog
from scripts.measure_field_grounded import materialize as materialize_measure
from scripts.measure_field_grounded import prepare_task as prepare_measure
from scripts.measure_field_grounded import VERSION as MEASURE_VERSION, FORMAT_VERSION
from scripts.measure_continuation_choices import VERSION as MEASURE_CONTINUATION_POLICY
from scripts.measure_continuation_boundaries import VERSION as CHECKED_CONTINUATION_POLICY, validate_boundary_proof
from scripts.measure_source_context import restore_context, source_context
from scripts.principal_statement_trial import materialize as materialize_findings
from scripts.principal_statement_trial import prepare_attributed
from scripts.study_inspection import build_inventory
from scripts.source_decoding import validate_source


VERSION = "complete-extraction-sequence/v1"
NAME_VERSION = "core-name-witnesses-v3"
DECLARED_NAME_VERSION = "core-name-witnesses-v4"
RENDERED_NAME_VERSION = "core-name-witnesses-v5"
_HASH = re.compile(r"[0-9a-f]{64}\Z")
SCHEMA_OMISSIONS = (
    "The selected core schema does not extract exclusions, reliability estimates, or experimental conditions.",
    "A missing study, analysis, measure, role, finding, or field is not evidence that it was absent from the source.",
)


def _digest(value: Any) -> str:
    return sha256(canonical_bytes(value)).hexdigest()


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def _paper_metadata(paper: Mapping[str, Any]) -> dict[str, str]:
    required = ("paper_id", "title", "source_pdf_sha256")
    _require(all(isinstance(paper.get(key), str) and paper[key].strip() for key in required),
             "complete_sequence_paper_metadata_required")
    _require(_HASH.fullmatch(paper["source_pdf_sha256"]) is not None,
             "complete_sequence_pdf_hash_required")
    return {key: paper.get(key, "") for key in ("paper_id", "title", "citation", "source_pdf_sha256")}


def _source(paper: Mapping[str, Any]) -> dict[str, Any]:
    source = paper.get("source")
    _require(isinstance(source, Mapping) and isinstance(source.get("units"), list),
             "complete_sequence_source_required")
    _require(isinstance(source.get("source_policy_sha256"), str)
             and _HASH.fullmatch(source["source_policy_sha256"]) is not None,
             "complete_sequence_source_policy_required")
    _require(source["units"], "complete_sequence_empty_source")
    validate_source(source, pdf_sha256=paper["source_pdf_sha256"])
    return dict(source)


def prepare_upstream(recipe: Mapping[str, Any], paper: Mapping[str, Any], *, measure_contract_version=MEASURE_VERSION,
                     name_recovery_version=NAME_VERSION, name_layout_proof=None, name_layout_proof_sha256=None,
                     repeated_name_policy=None, measure_continuation_policy=None,
                     measure_continuation_proof=None, measure_continuation_proof_sha256=None) -> dict[str, Any]:
    """Freeze the unchanged finding and core requests for one supplied source."""
    metadata, source = _paper_metadata(paper), _source(paper)
    _require(measure_contract_version in (MEASURE_VERSION, FORMAT_VERSION), 'complete_sequence_measure_version_invalid')
    _require(name_recovery_version in (NAME_VERSION, DECLARED_NAME_VERSION, RENDERED_NAME_VERSION), 'complete_sequence_name_version_invalid')
    _require(repeated_name_policy in (None, REPEATED_NAME_POLICY), 'complete_sequence_repeated_name_policy_invalid')
    _require(measure_continuation_policy in (None, MEASURE_CONTINUATION_POLICY, CHECKED_CONTINUATION_POLICY),
             'complete_sequence_measure_continuation_policy_invalid')
    if measure_continuation_policy == CHECKED_CONTINUATION_POLICY:
        validate_boundary_proof(measure_continuation_proof, source, pdf_sha256=metadata['source_pdf_sha256'],
                                proof_sha256=measure_continuation_proof_sha256)
    else:
        _require(measure_continuation_proof is None and measure_continuation_proof_sha256 is None,
                 'complete_sequence_unexpected_continuation_proof')
    if name_recovery_version == RENDERED_NAME_VERSION:
        validate_layout_proof(name_layout_proof, units=source['units'], pdf_sha256=metadata['source_pdf_sha256'],
                              proof_sha256=name_layout_proof_sha256)
    else:
        _require(name_layout_proof is None and name_layout_proof_sha256 is None,
                 'complete_sequence_unexpected_name_layout_proof')
    findings = prepare_attributed(recipe, source["units"])
    core = prepare_core(recipe, source["units"])
    supplied = paper.get("core_task")
    if supplied is not None:
        _require(supplied == core, "complete_sequence_core_task_changed")
    result = {
        "version": VERSION,
        "paper": metadata,
        "source": deepcopy(source),
        "source_identity": {"source_sha256": _digest(source), "units_sha256": _digest(source["units"]),
                            "source_policy_sha256": source["source_policy_sha256"]},
        "findings_task": findings,
        "core_task": core,
        "provider_calls": 0,
        "scientifically_accepted": False,
        "production_accepted": False,
    }
    if measure_contract_version != MEASURE_VERSION:
        result['measure_contract_version'] = measure_contract_version
        result['source_identity']['measure_contract_version'] = measure_contract_version
    if name_recovery_version != NAME_VERSION:
        result['name_recovery_version'] = name_recovery_version
        result['source_identity']['name_recovery_version'] = name_recovery_version
    if name_recovery_version == RENDERED_NAME_VERSION:
        result['name_layout_proof'] = deepcopy(name_layout_proof)
        result['name_layout_proof_sha256'] = name_layout_proof_sha256
        result['source_identity']['name_layout_proof_sha256'] = name_layout_proof_sha256
    if repeated_name_policy is not None:
        result['repeated_name_policy'] = repeated_name_policy
        result['source_identity']['repeated_name_policy'] = repeated_name_policy
    if measure_continuation_policy is not None:
        result['measure_continuation_policy'] = measure_continuation_policy
        result['source_identity']['measure_continuation_policy'] = measure_continuation_policy
    if measure_continuation_policy == CHECKED_CONTINUATION_POLICY:
        result['measure_continuation_proof'] = deepcopy(measure_continuation_proof)
        result['measure_continuation_proof_sha256'] = measure_continuation_proof_sha256
        result['source_identity']['measure_continuation_proof_sha256'] = measure_continuation_proof_sha256
    return result


def _validate_plan(plan: Mapping[str, Any]) -> None:
    _require(plan.get("version") == VERSION, "complete_sequence_version_invalid")
    rebuilt = prepare_upstream({"vllm": {"served_model_name": plan["core_task"]["body"]["model"]}},
                               {**plan["paper"], "source": plan["source"], "core_task": plan["core_task"]},
                               measure_contract_version=plan.get('measure_contract_version', MEASURE_VERSION),
                               name_recovery_version=plan.get('name_recovery_version', NAME_VERSION),
                               name_layout_proof=plan.get('name_layout_proof'),
                               name_layout_proof_sha256=plan.get('name_layout_proof_sha256'),
                               repeated_name_policy=plan.get('repeated_name_policy'),
                               measure_continuation_policy=plan.get('measure_continuation_policy'),
                               measure_continuation_proof=plan.get('measure_continuation_proof'),
                               measure_continuation_proof_sha256=plan.get('measure_continuation_proof_sha256'))
    _require(rebuilt["findings_task"] == plan["findings_task"], "complete_sequence_findings_task_changed")
    _require(rebuilt["source_identity"] == plan["source_identity"], "complete_sequence_source_identity_changed")


def materialize_core(plan: Mapping[str, Any], core_proposal: Mapping[str, Any]) -> dict[str, Any]:
    """Recover a new core proposal and make its v3 inventory, without promotion."""
    _validate_plan(plan)
    source = plan["source"]
    args = {'eligible_units': source['units'], 'source_policy_sha256': source['source_policy_sha256']}
    if plan.get('name_recovery_version') == RENDERED_NAME_VERSION:
        recovered = recover_rendered_names(dict(core_proposal), plan['core_task'], **args,
            pdf_sha256=plan['paper']['source_pdf_sha256'], proof=plan['name_layout_proof'],
            proof_sha256=plan['name_layout_proof_sha256'])
    else:
        recovery = recover_declared_names if plan.get('name_recovery_version') == DECLARED_NAME_VERSION else recover_linebreak_names
        recovered = recovery(dict(core_proposal), plan['core_task'], **args)
    if plan.get('repeated_name_policy') == REPEATED_NAME_POLICY:
        recovered = recover_repeated_names(dict(core_proposal), plan['core_task'], recovered)
    metadata = {**plan["paper"], "source_units_sha256": _digest(source["units"])}
    inventory = build_inventory(plan["core_task"], recovered, source, metadata=metadata,
                                original_proposal=dict(core_proposal))
    return {"version": VERSION, "core_proposal_sha256": _digest(core_proposal),
            "recovered": recovered, "inventory": inventory,
            "scientifically_accepted": False, "production_accepted": False,
            "limitations": ["Recovered labels and roles remain model proposals.",
                            "This is a new core inventory; it does not merge historical measure records."]}


def prepare_measure_tasks(plan: Mapping[str, Any], core_proposal: Mapping[str, Any]) -> dict[str, Any]:
    """Build dependent measure tasks after, and only after, the new core proposal."""
    core = materialize_core(plan, core_proposal)
    catalog = build_catalog([core["inventory"]])
    stages = []
    for version in catalog["versions"]:
        stage = {"id": plan["paper"]["paper_id"] + ":measure:" + version["id"],
                 "version_id": version["id"], "version": version,
                 "status": "not_prepared", "scientifically_accepted": False}
        if version["rights"]["display_items"] is not True:
            stage.update(status="rights_withheld", reason="measure_fields_not_offered")
        else:
            # Restore only whole units already cited by this measure.  The
            # grounded ownership task then gets its broader, source-bound
            # target context; it does not borrow another measure's evidence.
            try:
                context = source_context(version, plan["source"])
                expanded = restore_context(version, context)
                if expanded["rights"]["display_items"] is not True:
                    stage.update(status="rights_withheld", reason="rights_withheld_after_context_validation")
                else:
                    task = prepare_measure(expanded, mode="ownership",
                        contract_version=plan.get('measure_contract_version', MEASURE_VERSION),
                        continuation_policy=plan.get('measure_continuation_policy'),
                        source=plan['source'] if plan.get('measure_continuation_policy') else None,
                        continuation_proof=plan.get('measure_continuation_proof'),
                        continuation_proof_sha256=plan.get('measure_continuation_proof_sha256'))
                    stage["context"] = context
                    stage["expanded_version"] = expanded
                    stage["task"] = task
                    stage["task_sha256"] = _digest(task)
                    if not task["payload"]["choices"]:
                        stage.update(status="not_extracted", reason="no_offered_exact_field_candidate")
                    else:
                        stage["status"] = "awaiting_response"
            except ValueError as exc:
                stage.update(status="refused", reason="measure_context_or_task_refused",
                             error_type=type(exc).__name__)
        stages.append(stage)
    result = {"version": VERSION, "paper_id": plan["paper"]["paper_id"], "core": core,
            "measure_stages": stages, "counts": {"measure_versions": len(stages),
                "awaiting_response": sum(s["status"] == "awaiting_response" for s in stages),
                "not_extracted": sum(s["status"] == "not_extracted" for s in stages),
                "rights_withheld": sum(s["status"] == "rights_withheld" for s in stages)},
            "provider_calls": 0, "scientifically_accepted": False, "production_accepted": False}
    if plan.get('measure_continuation_policy'):
        result['continuation_source'] = deepcopy(plan['source'])
    if plan.get('measure_continuation_proof') is not None:
        result['continuation_proof'] = deepcopy(plan['measure_continuation_proof'])
    return result


def materialize_sequence(sequence: Mapping[str, Any], responses: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Bind only supplied answers to their same frozen child request.

    `responses` is keyed by the deterministic stage ID.  Unknown keys and a
    changed task are refusals.  Missing answers are preserved as unresolved.
    """
    _require(sequence.get("version") == VERSION, "complete_sequence_version_invalid")
    stages = []
    known = {stage["id"] for stage in sequence["measure_stages"]}
    _require(set(responses) <= known, "complete_sequence_unknown_measure_response")
    for stage in sequence["measure_stages"]:
        out = deepcopy(stage)
        answer = responses.get(stage["id"])
        if stage["status"] != "awaiting_response":
            if answer is not None:
                out.update(status="invalid_proposal", reason="response_not_expected_for_stage")
        elif answer is None:
            out.update(status="unresolved", reason="response_not_supplied")
        else:
            _require(_digest(stage["task"]) == stage["task_sha256"], "complete_sequence_measure_task_changed")
            try:
                result = materialize_measure(stage["task"], answer, stage["expanded_version"],
                    source=sequence.get('continuation_source') if stage['task'].get('continuation_policy') else None,
                    continuation_proof=sequence.get('continuation_proof'))
                out.update(status="materialized_source_bound_proposal", materialized=result)
            except (ValueError, KeyError, ValidationError) as exc:
                out.update(status="invalid_proposal", reason="measure_response_refused",
                           error_type=type(exc).__name__)
        stages.append(out)
    return {"version": VERSION, "paper_id": sequence["paper_id"],
            "core": deepcopy(sequence["core"]), "measure_stages": stages,
            "counts": {"materialized": sum(s["status"] == "materialized_source_bound_proposal" for s in stages),
                       "unresolved": sum(s["status"] == "unresolved" for s in stages),
                       "not_extracted": sum(s["status"] == "not_extracted" for s in stages),
                       "rights_withheld": sum(s["status"] == "rights_withheld" for s in stages),
                       "refused": sum(s["status"] == "refused" for s in stages),
                       "invalid_proposal": sum(s["status"] == "invalid_proposal" for s in stages)},
            "scientifically_accepted": False, "production_accepted": False,
            "limitations": ["No stage proves a complete scale or a verified variable role.",
                            "An unavailable or failed stage is retained, not converted into an absence claim."]}


def finalize_paper(plan: Mapping[str, Any], *, findings_proposal: Mapping[str, Any] | None,
                   core_proposal: Mapping[str, Any] | None,
                   measure_responses: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Make one whole-paper report while retaining unavailable/invalid stages.

    This is a report assembler, not a success condition.  It never replaces an
    unavailable response with an empty model answer and cannot claim that the
    selected thin sequence exhausts the paper's reportable information.
    """
    _validate_plan(plan)
    findings: dict[str, Any]
    if findings_proposal is None:
        findings = {"status": "unavailable", "reason": "findings_response_not_supplied"}
    else:
        try:
            findings = {"status": "materialized_source_bound_proposal",
                        "materialized": materialize_findings(findings_proposal, plan["findings_task"])}
        except (ValueError, KeyError, ValidationError) as exc:
            findings = {"status": "invalid_proposal", "reason": "findings_response_refused",
                        "error_type": type(exc).__name__}
    if core_proposal is None:
        core = {"status": "unavailable", "reason": "core_response_not_supplied"}
        measures = {"status": "not_started", "reason": "dependent_on_core_response"}
    else:
        try:
            sequence = prepare_measure_tasks(plan, core_proposal)
            core = {"status": "materialized_source_bound_proposal", "materialized": sequence["core"]}
            measures = materialize_sequence(sequence, measure_responses)
        except (ValueError, KeyError, ValidationError) as exc:
            core = {"status": "invalid_proposal", "reason": "core_response_refused",
                    "error_type": type(exc).__name__}
            measures = {"status": "not_started", "reason": "dependent_core_refused"}
    return {"version": VERSION, "paper": deepcopy(plan["paper"]), "findings": findings, "core": core,
            "measures": measures, "schema_omissions": list(SCHEMA_OMISSIONS),
            "scientifically_accepted": False, "production_accepted": False,
            "complete_extraction_claimed": False}
