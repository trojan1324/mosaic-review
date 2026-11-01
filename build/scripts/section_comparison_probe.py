"""Bounded, source-first section comparison driver for two approved inputs.

This driver only prepares or executes a finite overview plus three dependent
section tasks per structural study.  It never treats a completed request as a
scientific acceptance: every response remains source-bound, mechanically
audited, and development-only.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from jsonschema import Draft202012Validator, ValidationError

from mosaic import REPO_ROOT
sys.path.insert(0, str(REPO_ROOT))
from mosaic.canonical import canonical_bytes
from mosaic.recipe import resolve_recipe
from mosaic.vllm.client import VLLMClient
from mosaic.vllm.ledger import VLLMBudget, VLLMLedger
from mosaic.vllm.tokenizer import local_tokenizer
from mosaic.workspace import Workspace
from scripts.section_extraction_continuation import body_for
from scripts.section_extraction_contract import whole_proposal
from scripts.section_source_context import prepare_contexts
from scripts.selected_study_contract import materialize_selected
from scripts.structured_extraction_probe import CANDIDATE, overview_schema, present
from scripts.section_field_contract import (apply_measure_binding, measure_binding_packet,
    measure_binding_schema, materialize_results, result_units, typed_results_schema)
from scripts.section_field_probe import MEASURE_INSTRUCTION, RESULT_INSTRUCTION, make_body
from scripts.typed_statistic_choices import build_statistic_choices
from scripts.paragraph_results_contract import paragraph_results_schema, materialize_paragraph_results
from scripts.numbered_item_boundaries import normalize_numbered_items


ALLOWED_WORKS = frozenset({"KS-DOC-000102", "KS-DOC-000129"})
TASKS = ("measures", "measure_fields", "design", "results")
MAX_CONTEXT_TOKENS = 98_304


def stage_key(run_version: str, work_id: str, study_id: str, task: str) -> str:
    return "section-comparison-" + hashlib.sha256(
        canonical_bytes({"run_version": run_version, "work_id": work_id, "study_id": study_id, "task": task})
    ).hexdigest()[:20]


def study_stem(study_id):
    # Source identifiers can contain colons; paths must remain Windows-native.
    return 'study-' + hashlib.sha256(study_id.encode()).hexdigest()[:12]


def load_input(source_path: Path, binding_path: Path, ordered_path: Path, work_id: str):
    """Read one immutable source packet and prove its ordering is conservative."""
    if work_id not in ALLOWED_WORKS:
        raise ValueError("work_not_allowlisted")
    source = json.loads(source_path.read_bytes())
    binding = json.loads(binding_path.read_bytes())
    packet = json.loads(ordered_path.read_bytes())
    digest = hashlib.sha256(canonical_bytes(source)).hexdigest()
    if source.get("work_id") != work_id or binding.get("source_packet_sha256") != digest:
        raise ValueError("source_binding_identity_changed")
    units = [unit for study in packet.get("studies", []) for unit in study.get("units", [])]
    units += list(packet.get("unassigned_source_units", []))
    ids = [unit.get("unit_id") for unit in units]
    source_ids = [unit.get("unit_id") for unit in source.get("units", [])]
    if (len(ids) != len(set(ids)) or len(source_ids) != len(set(source_ids)) or set(ids) != set(source_ids)
            or {unit["unit_id"]: unit.get("text") for unit in units}
            != {unit["unit_id"]: unit.get("text") for unit in source["units"]}):
        raise ValueError("ordered_context_source_conservation_failed")
    by_id = {unit['unit_id']: unit for unit in units}
    if any(any(by_id[unit['unit_id']].get(key) != value for key, value in unit.items()) for unit in source['units']):
        raise ValueError('ordered_context_source_metadata_changed')
    if packet.get("coverage", {}).get("omitted_units") != 0 or packet.get("coverage", {}).get("duplicated_units") != 0:
        raise ValueError("ordered_context_reports_loss")
    return source, binding, packet


def overview_body(recipe, source, packet):
    """Build the low-reasoning overview under the same bounded native grammar."""
    import xgrammar as xgr

    units = source["units"]
    schema = overview_schema(units, packet["studies"])
    grammar = str(xgr.Grammar.from_json_schema(canonical_bytes(schema).decode(), max_whitespace_cnt=8))
    xgr.Grammar.from_ebnf(grammar)
    payload = {"task": "OVERVIEW", "ordered_source": [present(unit) for unit in units],
               "study_section_unit_ids": {study["study_id"]: [unit["unit_id"] for unit in study["units"]]
                                          for study in packet["studies"]}}
    instruction = ("Read only the supplied ordered source. Identify source-supported hypotheses and definitions from the "
                   "unassigned front matter; do not move units between studies. Return the exact schema with concise JSON.")
    return ({"model": recipe["vllm"]["served_model_name"], "temperature": 0, "seed": 0,
             "reasoning_effort": "low", "max_tokens": 4096,
             "messages": [{"role": "system", "content": instruction + "\nRESPONSE_SCHEMA:\n" + json.dumps(schema, sort_keys=True)},
                          {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
             "structured_outputs": {"grammar": grammar}}, payload, schema)


def planning_overview(packet):
    """A schema-valid empty overview for offline sizing only, never a model result."""
    return {"hypotheses": [], "definitions": [], "study_links": [], "boundary_concerns": []}


def planned_stage_count(packet) -> int:
    return 1 + len(packet["studies"]) * len(TASKS)


def budget_report(work_id: str, planned: int, ledger: VLLMLedger, budget: VLLMBudget):
    """Reserve all logical calls and as many safe retries as the ledger permits."""
    totals = ledger.totals()
    article = ledger.totals_for_article(work_id)
    limit = budget.article_limit(CANDIDATE, work_id)
    global_remaining = max(0, budget.request_cap - totals.requests)
    article_remaining = max(0, limit - article.requests)
    reserve = min(planned * 2, global_remaining, article_remaining)
    return {"logical_calls": planned, "reserved_attempts": reserve, "retry_headroom": max(0, reserve - planned),
            "ledger_requests_before": totals.requests,
            "article_requests_before": article.requests, "global_request_cap": budget.request_cap,
            "article_request_cap": limit, "global_remaining": global_remaining, "article_remaining": article_remaining,
            "global_within_cap": reserve >= planned and totals.requests + reserve <= budget.request_cap,
            "article_within_cap": reserve >= planned and article.requests + reserve <= limit,
            "unresolved": list(totals.unresolved)}


def execution_allowed(report):
    return not report["unresolved"] and report["global_within_cap"] and report["article_within_cap"]


def audit_response(response):
    """Classify one provider receipt without turning it into an accepted record."""
    if not isinstance(response, dict) or not isinstance(response.get("choices"), list) or len(response["choices"]) != 1:
        return "invalid_response", None
    choice = response["choices"][0]
    if choice.get("finish_reason") != "stop":
        return "incomplete_response", None
    try:
        return "parsed_unaccepted", json.loads(choice["message"]["content"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return "invalid_json", None


def dependent_tasks_after_measures(outcome):
    return ("design", "results") if outcome == "parsed_unaccepted" else ()


def results_roster_ready(measures, study_kind):
    """Do not give an empirical results task an empty or ambiguous role roster."""
    variables = measures.get("variables") if isinstance(measures, dict) else None
    if study_kind == "nonempirical_section":
        return True, "not_empirical"
    if not isinstance(variables, list) or not variables:
        return False, "empty_variable_inventory"
    ids = [row.get("id") for row in variables if isinstance(row, dict)]
    if len(ids) != len(variables) or not all(isinstance(value, str) and value for value in ids):
        return False, "invalid_variable_inventory"
    if len(ids) != len(set(ids)):
        return False, "duplicate_variable_ids"
    return True, "ready"


def finding_roles_complete(record):
    """A finding without analysis roles is incomplete even if source mechanics pass."""
    analyses = record.get("analyses", []) if isinstance(record, dict) else []
    return all(not analysis.get("findings") or bool(analysis.get("roles")) for analysis in analyses if isinstance(analysis, dict))


def _initial_record(study, measures):
    """Materialize only the measure inventory before its narrow field repair."""
    proposal = whole_proposal({"measures": measures}, study["study"]["study_id"], study["study"]["label"])
    normalized = normalize_numbered_items(proposal, study['units'])
    result = materialize_selected(normalized['record'], study["units"], study["choices"], [],
                                  [unit["unit_id"] for unit in study["study"]["units"]], study["reuse_links"])
    result['item_boundary_changes'] = normalized['changes']
    result['errors'] += normalized['errors']
    result['mechanical_valid'] = not result['errors']
    return result


def _add_design(base, study, design):
    """Copy independently materialized design facts into the field-bound measure base."""
    proposal = whole_proposal({"design": design}, base["study_id"], base["label"])
    selected = materialize_selected(proposal, study["units"], study["choices"], [],
                                    [unit["unit_id"] for unit in study["study"]["units"]], study["reuse_links"])
    result = deepcopy(base)
    result["design"] = selected["record"]["design"]
    result["gaps"] = list({canonical_bytes(g): g for g in [*base.get("gaps", []), *selected["record"].get("gaps", [])]}.values())
    return {"record": result, "manipulations": selected["manipulations"],
            "manipulation_copies": selected["manipulation_copies"], "errors": selected["errors"],
            "mechanical_valid": selected["mechanical_valid"], "semantic_accuracy_unassessed": True,
            "production_accepted": False}


def variable_evidence(base, context):
    """Provide each variable's own admitted paragraphs as context, without widening result citations."""
    by_id = {unit["unit_id"]: unit for unit in context["units"]}
    evidence = []
    for variable in base.get("variables", []):
        ids = list(variable.get("source_units", []))
        for measure in variable.get("measures", []):
            ids.extend(measure.get("source_units", []))
        ids = list(dict.fromkeys(unit_id for unit_id in ids if unit_id in by_id))
        evidence.append({"id": variable["id"], "name": variable["name"],
                         "source_unit_ids": ids, "source": [present(by_id[unit_id]) for unit_id in ids]})
    return evidence


def typed_results_body(recipe, base, context, hypotheses):
    units = result_units(context)
    choices = build_statistic_choices(units)
    payload = {"study_id": base["study_id"], "label": base["label"],
               "results_source": [present(unit) for unit in units],
               "variable_roster": [{"id": value["id"], "name": value["name"], "acquisition": value["acquisition"]}
                                   for value in base["variables"]],
               "variable_evidence": variable_evidence(base, context), "hypotheses": hypotheses,
               "statistic_choices": choices}
    schema = paragraph_results_schema(units, choices, [item["id"] for item in hypotheses], base["variables"],
                                      study_id=base['study_id'], label=base['label'])
    instruction = RESULT_INSTRUCTION.replace(
        'selected signed-estimate evidence (zero-based index in the expanded statistics list; CI expands into lower then upper)',
        'selected_estimate evidence with the exact choice_id of a coefficient or correlation selected for this finding') + (
        " Controls are covariates: retain their inclusion in later nested models when the supplied evidence explicitly says so. "
        "Distinguish an analysis role from acquisition. Never infer a covariate from a variable name alone. "
        "Variable evidence is context only; cite only result-source unit IDs in the response. "
        "Return one required object for EACH results paragraph ID. Each object may cite and select statistics only "
        "from that paragraph. A heading or paragraph without CORE findings has empty analyses. "
        "Do not copy a result from another paragraph just to fill an object. Context elsewhere can clarify variable "
        "names but cannot supply statistics or signs for a different model. For selected_estimate direction "
        "evidence use the exact statistic choice_id, never an array index.")
    return make_body(recipe, payload, schema, instruction), payload, schema, units, choices


def dry_run(source, packet, binding, work_id, run_version):
    rid, recipe = resolve_recipe(CANDIDATE)
    count = local_tokenizer(recipe["vllm"])
    overview, _, overview_schema_value = overview_body(recipe, source, packet)
    provisional, _, _ = prepare_contexts(packet, planning_overview(packet))
    stages = [{"stage": "overview", "task": "overview", "reasoning_effort": "low", "body": overview,
               "schema": overview_schema_value, "stage_key": stage_key(run_version, work_id, "article", "overview")}]
    for study in provisional:
        for task in TASKS:
            roster = []
            # Later stages depend on earlier model output.  Use the same bounded
            # source context for an upper-level offline sizing receipt; live
            # execution recomputes the exact field/result payload after validation.
            source_task = "measures" if task == "measure_fields" else task
            body, _, schema = body_for(source_task, study, [], recipe, roster)
            stages.append({"stage": f"{study['study']['study_id']}:{task}", "task": task,
                           "reasoning_effort": "medium", "body": body, "schema": schema,
                           "stage_key": stage_key(run_version, work_id, study["study"]["study_id"], task)})
    detail = []
    for stage in stages:
        estimate = count(json.dumps(stage["body"]["messages"], ensure_ascii=False)) + 256
        detail.append({key: stage[key] for key in ("stage", "task", "reasoning_effort", "stage_key")}
                      | {"prompt_estimate": estimate, "max_tokens": stage["body"]["max_tokens"],
                         "context_limit": MAX_CONTEXT_TOKENS, "within_context_limit": estimate + stage["body"]["max_tokens"] < MAX_CONTEXT_TOKENS,
                         "schema_sha256": hashlib.sha256(canonical_bytes(stage["schema"])).hexdigest()})
    budget = budget_report(work_id, len(stages), VLLMLedger(CANDIDATE), VLLMBudget.load())
    return {"kind": "section_comparison_dry_run", "run_version": run_version, "candidate": CANDIDATE,
            "recipe_id": rid, "work_id": work_id, "source_packet_sha256": binding["source_packet_sha256"],
            "source_policy_sha256": source.get("source_policy_sha256"), "input_kind": binding.get("kind", "reused_baseline_source"),
            "stages": detail, "largest_prompt_estimate": max(row["prompt_estimate"] for row in detail),
            "context_limit_violations": [row["stage"] for row in detail if not row["within_context_limit"]],
            "budget": budget, "execution_allowed": execution_allowed(budget) and all(row['within_context_limit'] for row in detail),
            "semantic_accuracy_unassessed": True, "production_accepted": False}


def write_dry_run(root: Path, report):
    Workspace().write_once(root / "dry-run.json", canonical_bytes(report))


def _sources_for(work_id):
    registry = json.loads((REPO_ROOT / "config" / "screen_sources_v1.json").read_bytes())
    return [source["sha256"] for source in registry["works"][work_id]["sources"]]


def run_live(root: Path, source, packet, binding, work_id, run_version, base_url):
    report = dry_run(source, packet, binding, work_id, run_version)
    if not report["execution_allowed"]:
        raise RuntimeError("cumulative_budget_preflight_refused")
    ws = Workspace()
    ws.write_once(root / "experiment-started.json", canonical_bytes(report))
    rid, recipe = resolve_recipe(CANDIDATE)
    count = local_tokenizer(recipe["vllm"])
    ledger = VLLMLedger(CANDIDATE)
    before = ledger.totals().requests
    client = VLLMClient(candidate=CANDIDATE, base_url=base_url, api_key=os.environ.get("VLLM_API_KEY"),
                        timeout=240, stream=True, live=True)
    stages = []
    complete_studies = set()
    overview_body_value, overview_payload, overview_schema_value = overview_body(recipe, source, packet)
    queue = [("overview", "article", "overview", overview_body_value, overview_payload, overview_schema_value, None)]

    def within_hard_cap():
        budget = VLLMBudget.load()
        totals = ledger.totals()
        article = ledger.totals_for_article(work_id)
        return totals.requests < budget.request_cap and article.requests < budget.article_limit(CANDIDATE, work_id)

    try:
        while queue:
            name, study_id, task, body, payload, schema, data = queue.pop(0)
            if count(json.dumps(body["messages"], ensure_ascii=False)) + body["max_tokens"] >= MAX_CONTEXT_TOKENS:
                raise RuntimeError("context_limit_refused")
            if not within_hard_cap():
                stages.append({"stage": name, "task": task, "accepted": False, "outcome": "runtime_budget_exhausted"})
                continue
            ws.write_once(root / f"{name}-request.json", canonical_bytes(body))
            started = time.monotonic()
            response = client.chat_completion(work_id=work_id, batch_id=stage_key(run_version, work_id, study_id, task),
                packet=payload, request_body=body, source_sha256s=_sources_for(work_id), text_provenance="native_article_text",
                remote=False, rights=["runpod_screen_auth"])
            ws.write_once(root / f"{name}-response.json", canonical_bytes(response))
            choice = response["choices"][0] if isinstance(response.get("choices"), list) and response["choices"] else {}
            outcome, parsed = audit_response(response)
            entry = {"stage": name, "task": task, "finish_reason": choice.get("finish_reason"),
                     "seconds": round(time.monotonic() - started, 2), "usage": response.get("usage"), "accepted": False}
            stages.append(entry)
            if parsed is None:
                entry["outcome"] = outcome
                continue
            ws.write_once(root / f"{name}-parsed.json", canonical_bytes(parsed))
            try:
                Draft202012Validator(schema).validate(parsed)
            except ValidationError:
                entry['outcome'] = 'schema_invalid'
                continue
            if task == "overview":
                prepared, accepted, routing = prepare_contexts(packet, parsed)
                ws.write_once(root / "routing-decisions.json", canonical_bytes(routing))
                for study in prepared:
                    body, payload, schema = body_for("measures", study, accepted["hypotheses"], recipe, [])
                    queue.append((f"{study_stem(study['study']['study_id'])}-measures", study["study"]["study_id"], "measures", body, payload, schema, {"study": study, "hypotheses": accepted["hypotheses"], "parts": {}}))
                entry["outcome"] = "overview_completed"
                continue
            state = data
            state["parts"][task] = parsed
            if task == "measures":
                roster_ok, roster_reason = results_roster_ready(parsed, 'study')
                entry['results_roster_gate'] = roster_reason
                if not roster_ok:
                    entry['outcome'] = 'incomplete_variable_inventory'
                    continue
                initial = _initial_record(state["study"], parsed)
                ws.write_once(root / f"{study_stem(study_id)}-measures-materialized.json", canonical_bytes(initial))
                entry["mechanical_valid"] = initial["mechanical_valid"]
                if not initial["mechanical_valid"]:
                    entry["outcome"] = "incomplete_measure_materialization"
                    continue
                state["base"] = initial["record"]
                binding_packet = measure_binding_packet(state["base"], state["study"])
                body = make_body(recipe, binding_packet, measure_binding_schema(binding_packet), MEASURE_INSTRUCTION)
                queue.append((f"{study_stem(study_id)}-measure-fields", study_id, "measure_fields", body,
                              binding_packet, measure_binding_schema(binding_packet), state))
                entry["outcome"] = "measures_completed_waiting_field_binding"
                continue
            if task == "measure_fields":
                binding_packet = measure_binding_packet(state["base"], state["study"])
                bound = apply_measure_binding(state["base"], parsed, binding_packet, state["study"])
                ws.write_once(root / f"{study_stem(study_id)}-measure-bound.json", canonical_bytes(bound))
                state["base"] = bound["record"]
                body, payload, design_schema = body_for("design", state["study"], state["hypotheses"], recipe, [])
                queue.append((f"{study_stem(study_id)}-design", study_id, "design", body, payload, design_schema, state))
                entry["outcome"] = "measure_fields_completed_waiting_design"
                continue
            if task == "design":
                designed = _add_design(state["base"], state["study"], parsed)
                ws.write_once(root / f"{study_stem(study_id)}-design-materialized.json", canonical_bytes(designed))
                if not designed["mechanical_valid"]:
                    entry["mechanical_valid"] = False
                    entry["outcome"] = "incomplete_design_materialization"
                    continue
                state["base"] = designed["record"]
                state["manipulations"] = designed["manipulations"]
                state['manipulation_copies'] = designed['manipulation_copies']
                body, payload, result_schema, _units, _choices = typed_results_body(recipe, state["base"], state["study"], state["hypotheses"])
                queue.append((f"{study_stem(study_id)}-results", study_id, "results", body, payload, result_schema, state))
                entry["outcome"] = "design_completed_waiting_results"
                continue
            if task == "results":
                units = result_units(state["study"])
                choices = build_statistic_choices(units)
                try:
                    materialized = materialize_paragraph_results(state["base"], parsed, units, choices,
                        [row["id"] for row in state["hypotheses"]], state["study"]["units"])
                except ValueError as exc:
                    entry['outcome'] = 'result_materialization_refused'
                    entry['reason'] = str(exc)
                    continue
                final_record = {**materialized, "manipulations": state.get("manipulations", []),
                                'manipulation_copies': state.get('manipulation_copies', [])}
                ws.write_once(root / f"{study_stem(study_id)}-final.json", canonical_bytes(final_record))
                entry["mechanical_valid"] = materialized["mechanical_valid"]
                entry["finding_roles_complete"] = finding_roles_complete(materialized["record"])
                if entry["mechanical_valid"] and entry["finding_roles_complete"]:
                    complete_studies.add(study_id)
                    entry["outcome"] = "mechanically_checked_not_accepted"
                else:
                    entry["outcome"] = "incomplete_findings_without_roles" if not entry["finding_roles_complete"] else "incomplete_result_materialization"
                continue
    finally:
        final = report | {"stages": stages, "inference_attempts": ledger.totals().requests - before,
                          "complete_studies": sorted(complete_studies),
                          "outcome": "complete_tasks_without_acceptance" if complete_studies and len(complete_studies) == len(packet['studies']) else "incomplete_tasks",
                          "production_accepted": False}
        ws.write_once(root / "probe-result.json", canonical_bytes(final))
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--ordered-context", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work-id", required=True, choices=sorted(ALLOWED_WORKS))
    parser.add_argument("--run-version", default="v1")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--base-url")
    args = parser.parse_args(argv)
    source, binding, packet = load_input(args.source, args.binding, args.ordered_context, args.work_id)
    report = dry_run(source, packet, binding, args.work_id, args.run_version)
    if not args.live:
        write_dry_run(args.out, report)
        print(json.dumps(report))
        return 0
    if not args.base_url:
        raise ValueError("live_base_url_required")
    final = run_live(args.out, source, packet, binding, args.work_id, args.run_version, args.base_url)
    print(json.dumps(final))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
