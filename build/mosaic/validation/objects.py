"""Deterministic validators for every MOSAIC schema (audit M-32).

Each function returns a list of problems. They re-implement, in code, every
constraint the pinned decoder ignores (uniqueItems, numeric bounds,
patternProperties, format, if/then/else) and add the semantic rules JSON
Schema cannot express (ordered disjoint spans, text hashes, state/value
consistency, retry eligibility). Cross-object checks (references resolve,
witness text equals its text artifact slice) are in mosaic.validation.package.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Any, Callable

SHA = re.compile(r"^[0-9a-f]{64}$")
OBJREF = re.compile(r"^(manifest|source|page|text-region|table|figure-equation|study|assertion|relation|witness|disposition|validation-event|projection|dependency-edge)-[0-9a-f]{64}$")
LOCAL = re.compile(r"^[a-z_]+:[A-Za-z0-9._/\-]+$")
SCOPE_KEY = re.compile(r"^[a-z_]+_id$")
DECIMAL = re.compile(r"^-?[0-9]+(\.[0-9]+)?$")

STUDY_RELATIONS = {"replicates", "extends", "reuses_measure_from", "validates_measure_used_in",
                   "validates_manipulation_used_in", "alternative_analysis_of", "shares_materials_with",
                   "reuses_sample_from", "contrasts_with", "supports"}
SCOPE_REQUIRED = {
    "psychometric": ("instrument_version_id", "sample_id", "administration_id", "analysis_id"),
    "result": ("study_id", "sample_id", "analysis_id"),
    "sample": ("study_id", "sample_id"),
    "measure": ("study_id", "instrument_version_id"),
    "study_design": ("study_id",),
    "procedure": ("study_id",),
    "variable": ("study_id",),
    "work": ("work_id",),
}
NULLABLE_SCOPE = {"psychometric": {"analysis_id"}}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _unique(items: list, where: str, out: list[str]) -> None:
    seen = []
    for item in items:
        if item in seen:
            out.append(f"{where}: duplicate item (uniqueItems)")
            return
        seen.append(item)


def _timestamp(value: Any, where: str, out: list[str]) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        out.append(f"{where}: not a Z-suffixed UTC timestamp")
        return
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        out.append(f"{where}: not an ISO-8601 date-time")


def _envelope(doc: dict, expected_type: str, out: list[str]) -> None:
    if doc.get("type") != expected_type:
        out.append(f"type must be {expected_type}")
    if "id" in doc:
        out.append("object body must not carry its own id (C17)")
    _unique(doc.get("provenance_ids", []), "provenance_ids", out)
    rid = doc.get("recipe_id", "")
    if not (isinstance(rid, str) and rid.startswith("recipe-") and SHA.match(rid[7:])):
        out.append("recipe_id must be recipe-<sha256>")


def _spans(spans: list, out: list[str], where: str = "spans") -> None:
    last_end: dict[str, int] = {}
    _unique(spans, where, out)
    for i, span in enumerate(spans):
        s, e = span.get("start_codepoint"), span.get("end_codepoint")
        if not isinstance(s, int) or not isinstance(e, int) or s < 0 or e <= s:
            out.append(f"{where}[{i}]: need 0 <= start < end")
            continue
        art = span.get("text_artifact_id")
        if art in last_end and s < last_end[art]:
            out.append(f"{where}[{i}]: spans must be ordered and disjoint within an artifact")
        last_end[art] = e


def check_witness(doc: dict) -> list[str]:
    out: list[str] = []
    _envelope(doc, "witness", out)
    spans = doc.get("spans")
    kind = doc.get("witness_kind")
    if kind in ("text_span", "table_cell") and not spans:
        out.append("text_span/table_cell witnesses need at least one span")
    if spans:
        _spans(spans, out)
        if len(spans) > 1 and "join" not in doc:
            out.append("multi-span witness must declare join (M-08)")
    if "cell" in doc and kind != "table_cell":
        out.append("cell locator only on table_cell witnesses")
    if kind == "table_cell" and "cell" not in doc:
        out.append("table_cell witness needs the {table_id,row_index_or_path,column_id,header_path} locator (M-20)")
    text = doc.get("copied_text")
    if isinstance(text, str) and doc.get("copied_text_sha256") != sha256_text(text):
        out.append("copied_text_sha256 does not match copied_text")
    page = doc.get("page")
    if isinstance(page, dict):
        if page.get("display_ordinal") != page.get("physical_index", -2) + 1:
            out.append("display_ordinal must equal physical_index + 1")
        label = page.get("printed_label", {})
        if label.get("status") == "none" and label.get("value") is not None:
            out.append("printed label with status none must have value null")
        if label.get("status") in ("verified", "inferred") and label.get("value") in (None, ""):
            out.append("verified/inferred printed label needs a value")
    if page is None and doc.get("native_locator") is None and kind in ("text_span", "table_cell", "visual_region"):
        out.append("witness needs a page identity or a native locator")
    _unique(doc.get("context_witness_ids", []), "context_witness_ids", out)
    return out


def _value(value: dict, out: list[str]) -> None:
    kind = value.get("kind")
    if kind in ("integer", "decimal"):
        if "lexical" not in value or "value" not in value:
            out.append("numeric value needs lexical and typed value strings (FINAL §4.1)")
        elif not DECIMAL.match(value["value"]):
            out.append("typed numeric value must be a decimal string, never a float")
        if kind == "integer" and "value" in value and "." in value["value"]:
            out.append("integer value has a fractional part")
    if kind == "interval" and not ("lower" in value and "upper" in value):
        out.append("interval needs lower and upper")


def check_assertion(doc: dict) -> list[str]:
    out: list[str] = []
    _envelope(doc, "assertion", out)
    state = doc.get("state")
    value = doc.get("value")
    witnesses = doc.get("witness_ids", [])
    _unique(witnesses, "witness_ids", out)
    if state == "present":
        if not isinstance(value, dict):
            out.append("present requires a value")
        if not witnesses:
            out.append("present requires at least one witness")
    if state == "explicitly_absent" and not witnesses:
        out.append("explicitly_absent requires a witness citing the absence")
    if state == "not_found_in_reviewed_scope":
        if not doc.get("reviewed_scope"):
            out.append("not_found_in_reviewed_scope must enumerate the reviewed scope")
        if value is not None:
            out.append("not_found_in_reviewed_scope carries no value")
    if state == "ambiguous_or_conflicting" and len(doc.get("alternatives", [])) < 2:
        out.append("ambiguous_or_conflicting needs at least two alternatives with their own witnesses")
    if state in ("unavailable_or_illegible", "not_applicable", "not_processed") and not doc.get("reason"):
        out.append(f"{state} needs a typed reason")
    if isinstance(value, dict):
        _value(value, out)
    scope = doc.get("scope", {})
    for key, member in scope.items():
        if not SCOPE_KEY.match(key):
            out.append(f"scope key {key!r} must match ^[a-z_]+_id$ (patternProperties)")
        if member is not None and not (isinstance(member, str) and LOCAL.match(member)):
            out.append(f"scope member {key} is not a package-local id")
    family = doc.get("family")
    for key in SCOPE_REQUIRED.get(family, ()):
        if key not in scope:
            out.append(f"{family} scope requires {key} (M-15)")
        elif scope[key] is None and key not in NULLABLE_SCOPE.get(family, set()):
            out.append(f"{family} scope member {key} may not be null")
    if family in ("result", "psychometric", "interpretation") and "result_attribution" not in doc:
        out.append("result/psychometric/interpretation assertions need result_attribution (M-18)")
    cited = doc.get("cited_work")
    if cited and doc.get("result_attribution") == "author_result":
        out.append("a cited prior result is secondary_report; resolving the citation never promotes it (M-18)")
    if doc.get("interpretation_kind") == "derived":
        d = doc.get("derivation") or {}
        if not d.get("formula") or not d.get("input_witness_ids") or not d.get("assumptions"):
            out.append("derived value needs formula, input witness ids and assumptions (M-19)")
    tier = doc.get("acceptance_tier")
    if tier in ("accepted", "citable") and "acceptance_basis" not in doc:
        out.append("accepted records need acceptance_basis (M-29)")
    if tier in ("accepted", "citable") and doc.get("validation_status") != "accepted":
        out.append("accepted tier requires validation_status accepted")
    if doc.get("acceptance_basis") == "human_adjudication":
        adj = doc.get("adjudication") or {}
        confirmations = adj.get("gate_confirmations", {})
        missing = [g for g in ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8") if confirmations.get(g) is not True]
        if missing:
            out.append(f"human adjudication must confirm each of G1-G8 individually (M-28); missing {missing}")
        if not adj.get("consulted_witness_ids"):
            out.append("human adjudication must record consulted witnesses (M-28)")
        if "recorded_utc" in adj:
            _timestamp(adj["recorded_utc"], "adjudication.recorded_utc", out)
    if state == "present" and family in ("result", "psychometric") and isinstance(value, dict):
        if value.get("kind") in ("integer", "decimal") and not value.get("statistic"):
            out.append("per-value result assertions name their statistic (M-13)")
        if value.get("kind") == "paraphrase":
            out.append("a result/psychometric value must be a per-value lexical token, not a paraphrase bundle (M-13)")
    origin = doc.get("candidate_origin", {})
    if origin.get("backend") == "fixture" and origin.get("evidence_label") != "replayed historical output":
        out.append("fixture-backend candidates are labelled replayed historical output")
    _unique(doc.get("flags", []), "flags", out)
    return out


def check_relation(doc: dict) -> list[str]:
    out: list[str] = []
    _envelope(doc, "relation", out)
    if doc.get("from_id") == doc.get("to_id"):
        out.append("relation endpoints must differ")
    layer = doc.get("layer")
    rel = doc.get("relation")
    if layer == "study":
        if rel not in STUDY_RELATIONS:
            out.append(f"{rel} is not a study relation (see the §9.1 crosswalk)")
        if not doc.get("direction_basis"):
            out.append("study relations need direction_basis (M-16)")
        if not doc.get("witness_ids"):
            out.append("study relations need witnesses")
        for end in ("from_id", "to_id"):
            if not str(doc.get(end, "")).startswith("study:"):
                out.append(f"study relation {end} must be a study: id")
    if layer == "discovery":
        if doc.get("acceptance_tier") in ("accepted", "citable"):
            out.append("discovery links can never be accepted factual support")
        if doc.get("trust_tier", "discovery_hypothesis") != "discovery_hypothesis":
            out.append("discovery relations carry trust_tier discovery_hypothesis (M-38)")
    _unique(doc.get("witness_ids", []), "witness_ids", out)
    for key, member in (doc.get("scope") or {}).items():
        if not SCOPE_KEY.match(key):
            out.append(f"scope key {key!r} invalid")
    tier = doc.get("acceptance_tier")
    if tier in ("accepted", "citable") and "acceptance_basis" not in doc:
        out.append("accepted records need acceptance_basis (M-29)")
    if tier in ("accepted", "citable") and doc.get("validation_status") != "accepted":
        out.append("accepted tier requires validation_status accepted")
    if doc.get("acceptance_basis") == "human_adjudication":
        adj = doc.get("adjudication") or {}
        confirmations = adj.get("gate_confirmations", {})
        missing = [g for g in ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8") if confirmations.get(g) is not True]
        if missing:
            out.append(f"human adjudication must confirm each of G1-G8 individually (M-28); missing {missing}")
        if not adj.get("consulted_witness_ids"):
            out.append("human adjudication must record consulted witnesses (M-28)")
        if "recorded_utc" in adj:
            _timestamp(adj["recorded_utc"], "adjudication.recorded_utc", out)
    return out


def check_source(doc: dict) -> list[str]:
    out: list[str] = []
    _envelope(doc, "source", out)
    art = doc.get("artifact", {})
    if not SHA.match(str(art.get("sha256", ""))):
        out.append("artifact.sha256 invalid")
    if isinstance(art.get("bytes"), int) and art["bytes"] < 1:
        out.append("artifact.bytes must be >= 1")
    work = doc.get("work", {})
    if work.get("identity_basis") == "bibliographic_fallback_proposal" and not work.get("match_proposal_id"):
        out.append("bibliographic-fallback identity must carry a pending match proposal, never a merge (C-03)")
    works = [m.get("work_id") for m in doc.get("membership", [])]
    if work.get("work_id") not in works:
        out.append("work.work_id must appear among memberships")
    return out


def check_page(doc: dict) -> list[str]:
    out: list[str] = []
    _envelope(doc, "page", out)
    if doc.get("display_ordinal") != doc.get("physical_index", -2) + 1:
        out.append("display_ordinal must equal physical_index + 1")
    sig = doc.get("signals", {})
    ratio = sig.get("replacement_char_ratio")
    if isinstance(ratio, (int, float)) and not 0 <= ratio <= 1:
        out.append("replacement_char_ratio outside [0,1]")
    for key in ("char_count", "nonempty_span_count", "private_use_char_count", "image_count", "drawing_count", "runon_tokens"):
        if isinstance(sig.get(key), int) and sig[key] < 0:
            out.append(f"signals.{key} negative")
    if sig.get("zero_text") is True and sig.get("nonempty_span_count", 0) > 0:
        out.append("zero_text page cannot have non-empty spans")
    label = doc.get("printed_label", {})
    if label.get("status") == "none" and label.get("value") is not None:
        out.append("printed label with status none must be null")
    return out


def check_text_region(doc: dict) -> list[str]:
    out: list[str] = []
    _envelope(doc, "text-region", out)
    text = doc.get("text", "")
    if doc.get("text_sha256") != sha256_text(text):
        out.append("text_sha256 does not match text")
    n = len(text)
    indices = []
    for b in doc.get("blocks", []):
        s, e = b.get("start", -1), b.get("end", -1)
        if not (0 <= s <= e <= n):
            out.append(f"block {b.get('block_index')} range outside text")
        for ln in b.get("lines", []):
            if not (s <= ln.get("start", -1) <= ln.get("end", -1) <= e):
                out.append(f"line range outside block {b.get('block_index')}")
        indices.append(b.get("block_index"))
    _unique(indices, "block_index", out)
    order = doc.get("reading_order", [])
    _unique(order, "reading_order", out)
    if order and sorted(order) != sorted(indices):
        out.append("reading_order must be a permutation of block indices")
    flags = doc.get("glyph_flags", [])
    for f in flags:  # a witness is `recognized` where it overlaps a flag, so a flag must name characters of the text
        if not (0 <= f.get("start", -1) < f.get("end", -1) <= n):
            out.append(f"glyph flag [{f.get('start')}, {f.get('end')}) is empty or outside the text")
    if len({(f.get("start"), f.get("end"), f.get("reason")) for f in flags}) != len(flags):
        out.append("duplicate glyph flags")
    derivation = doc.get("derivation", {})
    if derivation.get("ocr_text_layer") and derivation.get("kind") != "native_raw":
        out.append("ocr_text_layer marks a native text layer taken as a scan's OCR layer (D-163); "
                   f"this derivation is {derivation.get('kind')!r}")
    return out


def check_study(doc: dict) -> list[str]:
    out: list[str] = []
    _envelope(doc, "study", out)
    if doc.get("inventory_status") == "ambiguous" and not doc.get("ambiguity"):
        out.append("ambiguous study boundary must record the ambiguity")
    regions = doc.get("boundary", {}).get("regions", [])
    if doc.get("inventory_status") != "ambiguous" and not regions:
        out.append("non-ambiguous study needs at least one boundary region")
    if regions:
        _spans(regions, out, "boundary.regions")
    return out


def check_disposition(doc: dict) -> list[str]:
    out: list[str] = []
    _envelope(doc, "disposition", out)
    d = doc.get("disposition")
    if d != "processed" and not doc.get("reason"):
        out.append(f"disposition {d} requires a reason")
    if d == "quarantined" and not doc.get("quarantine"):
        out.append("quarantined requires the quarantine record with content hash (C-04)")
    ocr = doc.get("ocr") or {}
    if ocr.get("status") == "ocr_succeeded" and not ocr.get("text_region_id"):
        out.append("ocr_succeeded must name the recognized text region")
    if ocr.get("status") in ("ocr_attempted_failed", "ocr_succeeded") and not ocr.get("engine"):
        out.append("attempted OCR must name its engine (N-60)")
    _unique(doc.get("signals", []), "signals", out)
    return out


def check_validation_event(doc: dict) -> list[str]:
    out: list[str] = []
    _envelope(doc, "validation-event", out)
    _timestamp(doc.get("recorded_utc"), "recorded_utc", out)
    if doc.get("event_kind") == "gate":
        for key in ("gate", "gate_order", "outcome", "checks"):
            if key not in doc:
                out.append(f"gate events need {key}")
        gate, order = doc.get("gate"), doc.get("gate_order")
        if gate and order and int(gate[1:]) != order:
            out.append("gate_order must equal the gate number (G1..G10 run in order)")
    return out


def check_table(doc: dict) -> list[str]:
    out: list[str] = []
    _envelope(doc, "table", out)
    status = doc.get("structure_status", "")
    if status.startswith("unsupported") and not doc.get("unsupported_reason"):
        out.append("unsupported table structure must say why (C-05)")
    cols = {c.get("column_id") for c in doc.get("columns", [])}
    for cell in doc.get("cells", []):
        pos = cell.get("position", {})
        if pos.get("column_id") not in cols:
            out.append(f"cell references unknown column {pos.get('column_id')}")
        for key in ("row_span", "col_span"):
            if key in cell and (not isinstance(cell[key], int) or cell[key] < 1):
                out.append(f"cell {key} must be >= 1")
    cont = doc.get("continuation", {})
    if cont.get("status") == "continues_table" and not cont.get("parent_table_id"):
        out.append("continuation link needs parent_table_id")
    return out


def check_dependency_edge(doc: dict) -> list[str]:
    out: list[str] = []
    _envelope(doc, "dependency-edge", out)
    if doc.get("from_ref") == doc.get("to_ref"):
        out.append("dependency edge cannot be a self-loop")
    return out


def check_manifest(doc: dict) -> list[str]:
    out: list[str] = []
    for tag, ids in (doc.get("objects") or {}).items():
        _unique(ids, f"objects.{tag}", out)
        for oid in ids:
            if not oid.startswith(tag + "-"):
                out.append(f"object {oid} listed under wrong type {tag}")
    if "created_utc" in doc:
        _timestamp(doc["created_utc"], "created_utc", out)
    return out


def check_span_catalog(doc: dict) -> list[str]:
    out: list[str] = []
    ids = [e.get("span_id") for e in doc.get("entries", [])]
    _unique(ids, "entries.span_id", out)
    for e in doc.get("entries", []):
        if not (0 <= e.get("start", -1) < e.get("end", -1)):
            out.append(f"{e.get('span_id')}: need start < end")
        m = re.match(r"^span:[0-9a-f]{12}@([0-9]+)-([0-9]+)$", str(e.get("span_id", "")))
        if m and (int(m.group(1)), int(m.group(2))) != (e.get("start"), e.get("end")):
            out.append(f"{e.get('span_id')}: id offsets disagree with start/end")
        if e.get("kind") == "unclassified" and e.get("needs_review") is not True:
            out.append(f"{e.get('span_id')}: unclassified spans force review (M-65)")
    return out


def check_stage_state(doc: dict) -> list[str]:
    from mosaic.stage_state import retry_eligible

    out: list[str] = []
    expected = retry_eligible(doc.get("state"), reconciled=bool(doc.get("reconciled")), attempt_number=doc.get("attempt_number", 1))
    if doc.get("retry_eligible") and not expected:
        out.append(f"state {doc.get('state')} is not retry-eligible (M-48/M-49)")
    n = doc.get("attempt_number")
    if isinstance(n, int) and not 1 <= n <= 2:
        out.append("attempt_number must be 1 or 2 (one retry maximum)")
    return out


def check_jev_call(doc: dict) -> list[str]:
    out: list[str] = []
    _timestamp(doc.get("timestamp_utc"), "timestamp_utc", out)
    if doc.get("row_kind") == "result" and doc.get("outcome") in (None, "pending"):
        out.append("result rows record a final outcome")
    if doc.get("zero_data_retention") is not True and doc.get("route") == "typesafe_compatible_systemone":
        out.append("every evaluation request sets zeroDataRetention (AUTH-1)")
    for key in ("input_tokens", "output_tokens", "latency_ms"):
        if isinstance(doc.get(key), int) and doc[key] < 0:
            out.append(f"{key} negative")
    n = doc.get("attempt")
    if isinstance(n, int) and not 1 <= n <= 2:
        out.append("attempt must be 1 or 2 (hard stop 4)")
    return out


def check_recipe(doc: dict) -> list[str]:
    out: list[str] = []
    t = doc.get("ocr", {}).get("tesseract", {})
    if t.get("available") and not t.get("version"):
        out.append("available Tesseract must record its version in the recipe digest (N-60)")
    return out


def check_projection(doc: dict) -> list[str]:
    out: list[str] = []
    _envelope(doc, "projection", out)
    if doc.get("mode") == "dry_run" and doc.get("status") == "applied":
        out.append("a dry run cannot be marked applied")
    return out


def check_figure_equation(doc: dict) -> list[str]:
    out: list[str] = []
    _envelope(doc, "figure-equation", out)
    b = doc.get("bbox") or []
    if len(b) == 4 and not (b[2] > b[0] and b[3] > b[1]):
        out.append("bbox must have x1 > x0 and y1 > y0")
    return out


CODE_VALIDATORS: dict[str, Callable[[dict], list[str]]] = {
    "witness": check_witness,
    "assertion": check_assertion,
    "relation": check_relation,
    "source": check_source,
    "page": check_page,
    "text-region": check_text_region,
    "study": check_study,
    "disposition": check_disposition,
    "validation-event": check_validation_event,
    "table": check_table,
    "dependency-edge": check_dependency_edge,
    "manifest": check_manifest,
    "span-catalog": check_span_catalog,
    "stage-state": check_stage_state,
    "jev-call": check_jev_call,
    "recipe": check_recipe,
    "projection": check_projection,
    "figure-equation": check_figure_equation,
}
