"""Source-verified presentation of candidate studies, measures and variable roles.

This adapter never infers a missing field or changes an extraction decision.
Missing evidence means unextracted here, not that the authors reported nothing.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

from mosaic.canonical import canonical_bytes
from mosaic.workspace import Workspace
from scripts.core_evidence_trial import validate
from scripts.source_decoding import unit_slice

VERSION = "mosaic-study-inspection/v1"
STATUS = "model_proposed_not_independently_reviewed"


def digest(value: Any) -> str:
    return sha256(canonical_bytes(value)).hexdigest()


def _key(unit: dict[str, Any]) -> tuple[str, int, str]:
    locator, page, text = unit.get("source_locator"), unit.get("page"), unit.get("text")
    if not isinstance(locator, str) or not locator or type(page) is not int or page < 1 or not isinstance(text, str):
        raise ValueError("source_locator_positive_page_and_text_required")
    return locator, page, sha256(text.encode("utf-8")).hexdigest()


def build_inventory(task: dict[str, Any], recovered: dict[str, Any], source: dict[str, Any], *,
                    metadata: dict[str, Any], original_proposal: dict[str, Any] | None = None) -> dict[str, Any]:
    """Bind displayed quotes to the original frozen task and current source policy."""
    table = validate(task)
    recovery_version = recovered.get("version")
    if recovery_version not in {"core-name-witnesses-v1", "core-name-witnesses-v2", "core-name-witnesses-v3", "core-name-witnesses-v4", "core-name-witnesses-v5"}:
        raise ValueError("supported_recovery_version_required")
    if 'repeated_name_policy' in recovered and recovery_version not in {
            'core-name-witnesses-v3', 'core-name-witnesses-v4', 'core-name-witnesses-v5'}:
        raise ValueError('repeated_name_base_version_invalid')
    if recovery_version != "core-name-witnesses-v1":
        # Later locators can change a proposed spelling. Require an exact replay
        # from the original proposal before exposing their source-bound fields.
        if original_proposal is None:
            raise ValueError("original_proposal_required_for_recovery_replay")
        if recovery_version == "core-name-witnesses-v5":
            from scripts.core_rendered_names import recover_rendered_names
            replay = recover_rendered_names(original_proposal, task, eligible_units=source['units'],
                source_policy_sha256=source['source_policy_sha256'], pdf_sha256=metadata.get('source_pdf_sha256'),
                proof=recovered['rendered_name_proof'], proof_sha256=recovered['rendered_name_proof_sha256'])
        elif recovery_version == "core-name-witnesses-v4":
            from scripts.core_declared_names import recover_declared_names
            replay = recover_declared_names(original_proposal, task, eligible_units=source["units"],
                                            source_policy_sha256=source["source_policy_sha256"])
        elif recovery_version == "core-name-witnesses-v3":
            from scripts.core_linebreak_names import recover_linebreak_names
            replay = recover_linebreak_names(original_proposal, task, eligible_units=source["units"],
                                              source_policy_sha256=source["source_policy_sha256"])
        else:
            from scripts.core_name_witnesses import recover_cited_names
            replay = recover_cited_names(original_proposal, task, eligible_units=source["units"],
                                         source_policy_sha256=source["source_policy_sha256"],
                                         allow_local_layout_match=True)
        if 'repeated_name_policy' in recovered:
            from scripts.core_repeated_names import recover_repeated_names, VERSION as repeated_version
            if recovered['repeated_name_policy'] != repeated_version:
                raise ValueError('repeated_name_policy_invalid')
            replay = recover_repeated_names(original_proposal, task, replay)
        if replay != recovered:
            raise ValueError("recovery_replay_mismatch")
    for flag in ("scientifically_accepted", "production_accepted", "semantic_labels_verified", "coverage_verified"):
        if recovered.get(flag) is not False:
            raise ValueError("unverified_recovery_required")
    if recovered.get("original_task_sha256") != digest(task):
        raise ValueError("original_task_hash_mismatch")
    allowed = source["units"]
    if recovered.get("eligible_source_units_sha256") != digest(allowed):
        raise ValueError("clean_source_hash_mismatch")
    if metadata.get("source_units_sha256") != digest(allowed):
        raise ValueError("paper_source_metadata_mismatch")
    policy_hash = source.get("source_policy_sha256")
    if not isinstance(policy_hash, str) or not re.fullmatch("[0-9a-f]{64}", policy_hash) or recovered.get("source_policy_sha256") != policy_hash:
        raise ValueError("source_policy_hash_mismatch")
    original = {u["unit_id"]: u for u in task["original_units"]}
    eligible = {_key(u) for u in allowed}
    if len(eligible) != len(allowed) or not eligible <= {_key(u) for u in original.values()}:
        raise ValueError("eligible_source_not_unique_exact_subset")
    paper_id = metadata.get("paper_id")
    if not isinstance(paper_id, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", paper_id):
        raise ValueError("safe_paper_id_required")
    title = metadata.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("paper_title_required")
    pdf_hash = metadata.get("source_pdf_sha256")
    if not isinstance(pdf_hash, str) or not re.fullmatch("[0-9a-f]{64}", pdf_hash):
        raise ValueError("source_pdf_hash_required")

    def quotes(items):
        if not isinstance(items, list):
            raise ValueError("evidence_list_required")
        result = []
        seen = set()
        for item in items:
            sid = item.get("id")
            if sid not in table or item != table[sid]:
                raise ValueError("evidence_statement_mismatch")
            if sid in seen:
                continue
            seen.add(sid)
            segments = []
            for segment in item["segments"]:
                unit = original[segment["unit_id"]]
                start, end = segment["start"], segment["end"]
                if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(unit["text"]) or unit["text"][start:end] != segment["text"]:
                    raise ValueError("evidence_offset_or_text_mismatch")
                if _key(unit) not in eligible:
                    raise ValueError("evidence_not_currently_eligible")
                segments.append({**unit_slice(unit, start, end),
                                 "page": unit["page"], "source_locator": unit["source_locator"],
                                 "unit_text_sha256": _key(unit)[2]})
            result.append({"statement_id": sid, "quote": item["text"], "segments": segments,
                           "pages": list(dict.fromkeys(s["page"] for s in segments))})
        return result

    def field(items, proposed_value=None):
        evidence = quotes(items)
        return {"proposed_value": proposed_value, "status": STATUS if evidence else "not_extracted",
                "evidence": evidence}

    def layout_evidence(record):
        mapping = record.get('name_mapping', {})
        if mapping.get('match_type') == 'repeated_identical_case_or_layout_spelling':
            return {'repeated_name_evidence': {'version': recovered['repeated_name_policy'],
                'mapping': deepcopy(mapping), 'scientifically_accepted': False, 'source_text_unchanged': True}}
        if mapping.get('match_type') != 'unique_reviewed_pdf_layout':
            return {}
        return {'name_layout_evidence': {'version': 'bound-name-layout-evidence/v1',
            'mapping': deepcopy(mapping), 'proof': deepcopy(recovered['rendered_name_proof']),
            'proof_sha256': recovered['rendered_name_proof_sha256'],
            'scientifically_accepted': False, 'source_text_unchanged': True}}

    studies = []
    for study in recovered["studies"]:
        studies.append({"id": study["id"], "status": STATUS,
            "identifier_evidence": quotes(study["identifier_evidence"]),
            "design": field(study["design_evidence"], study["design"]),
            "population": field(study["population_evidence"]), "sample": field(study["sample_size_evidence"]),
            "measures": [{"id": m["id"], "name": m["name"], "status": STATUS,
                          "name_evidence": quotes(m["name_evidence"]),
                          "description": field(m["description_evidence"]),
                          "items": field(m["item_evidence"]), **layout_evidence(m)} for m in study["measures"]]})
    study_ids = [s["id"] for s in studies]
    if len(set(study_ids)) != len(study_ids):
        raise ValueError("duplicate_study_id")
    analyses = []
    for analysis in recovered["analyses"]:
        if not set(analysis["studies"]) <= set(study_ids):
            raise ValueError("unknown_study_binding")
        analyses.append({"id": analysis["id"], "study_ids": list(analysis["studies"]), "status": STATUS,
            "evidence": quotes(analysis["evidence"]),
            "roles": [{"id": r["id"], "name": r["variable_name"], "role": r["role"], "status": STATUS,
                       "name_evidence": quotes(r["name_evidence"]), "role_evidence": quotes(r["role_evidence"]), **layout_evidence(r)}
                      for r in analysis["roles"]]})
    if len({a["id"] for a in analyses}) != len(analyses):
        raise ValueError("duplicate_analysis_id")
    return {"version": VERSION, "scientifically_accepted": False, "coverage_verified": False,
            "paper": {"paper_id": paper_id, "title": title, "citation": metadata.get("citation", ""),
                      "source_pdf_sha256": pdf_hash},
            "studies": studies, "analyses": analyses,
            "quarantine": {"remaining": len(recovered["quarantined_fields"]),
                           "original": len(recovered["original_quarantined_fields"])},
            "source_identity": {"original_task_sha256": digest(task), "recovered_sha256": digest(recovered),
                                "eligible_source_units_sha256": digest(allowed), "source_policy_sha256": policy_hash,
                                **({'name_layout_proof_sha256': recovered['rendered_name_proof_sha256'],
                                    'name_layout_field_ids': [r['id'] for kind,children in (('studies','measures'),('analyses','roles'))
                                        for parent in recovered[kind] for r in parent[children]
                                        if r.get('name_mapping', {}).get('match_type') == 'unique_reviewed_pdf_layout']}
                                   if recovery_version == 'core-name-witnesses-v5' else {}),
                                **({'repeated_name_policy': recovered['repeated_name_policy'],
                                    'repeated_name_field_ids': [r['id'] for kind,children in (('studies','measures'),('analyses','roles'))
                                        for parent in recovered[kind] for r in parent[children]
                                        if r.get('name_mapping', {}).get('match_type') == 'repeated_identical_case_or_layout_spelling']}
                                   if 'repeated_name_policy' in recovered else {})},
            "limitations": ["Names, roles, study boundaries and analysis assignments are model proposals, not independently reviewed facts.",
                            "Missing evidence is not extracted; it does not mean the paper did not report the information.",
                            "Study and analysis record counts may include duplicate or incorrectly grouped model proposals.",
                            "Original model inputs may contain text excluded by the corrected source policy; displayed witnesses pass the current filter."]}


def write_collection(inventories: list[dict[str, Any]], output: Path, *, workspace: Workspace,
                     template: Path | None = None) -> dict[str, Any]:
    """Write immutable per-paper data and a small search index next to the reader."""
    output = workspace.check(output)
    rows = []; seen = set()
    for inventory in inventories:
        if inventory.get("version") != VERSION or inventory.get("scientifically_accepted") is not False:
            raise ValueError("unverified_inventory_required")
        paper = inventory["paper"]; pid = paper["paper_id"]
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", pid) or pid in seen:
            raise ValueError("unique_safe_paper_id_required")
        seen.add(pid)
        pdf = output / "sources" / (pid + ".pdf")
        if not pdf.is_file() or sha256(pdf.read_bytes()).hexdigest() != paper["source_pdf_sha256"]:
            raise ValueError("preview_source_pdf_mismatch")
        raw = canonical_bytes(inventory)
        data_path = "study-data/" + pid + ".json"
        workspace.write_once(output / data_path, raw)
        rows.append({**paper, "data_path": data_path, "data_sha256": sha256(raw).hexdigest(),
                     "study_records": len(inventory["studies"]), "analysis_records": len(inventory["analyses"]),
                     "measure_names": [m["name"] for s in inventory["studies"] for m in s["measures"]],
                     "variable_names": [r["name"] for a in inventory["analyses"] for r in a["roles"]]})
    manifest = {"version": VERSION, "papers": rows, "scientifically_accepted": False, "coverage_verified": False}
    template = template or Path(__file__).resolve().parents[1] / "web" / "study-inspection.html"
    html = template.read_text(encoding="utf-8")
    html = html.replace("__SOURCE_TEXT_SCRIPT__", (Path(__file__).resolve().parents[1] / "web" / "source-text.js").read_text(encoding="utf-8"))
    payload = json.dumps(manifest, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    if html.count("__STUDY_MANIFEST__") != 1:
        raise ValueError("study_template_manifest_marker")
    workspace.write_once(output / "studies.html", html.replace("__STUDY_MANIFEST__", payload).encode("utf-8"))
    workspace.write_once(output / "study-manifest.json", canonical_bytes(manifest))
    return {"papers": len(rows), "study_records": sum(r["study_records"] for r in rows),
            "analysis_records": sum(r["analysis_records"] for r in rows),
            "measure_records": sum(len(r["measure_names"]) for r in rows),
            "role_records": sum(len(r["variable_names"]) for r in rows), "scientifically_accepted": False}
