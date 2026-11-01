"""Ordered gates G1-G10 with named executing actors (FINAL v2 §8.1; audit M-27, M-16, M-17, M-31, N-05).

Each draft (assertion or relation) runs G1 -> G10 in order. A failure at G1,
G3 or G4 stops the unit (later gates are `not_run`); a scoped failure or flag
at G5-G8 leaves siblings untouched (salvage). Actors follow the v2 table
before any Jev family is qualified: code for G1-G6 and G8-G10, with human
review of every G5 flag, and a human for G7. G7 also runs a deterministic
pre-screen that rejects evidence which cannot support the claim at all (an
exact, correctly anchored but off-topic quotation passes G4 and fails here:
the Martin figure-ticks case, T22). No gate compares numbers with Jev.

The three standing contamination checks run inside G2/G5: page and box
membership of the witness (silent where there is no geometry), study-range
membership, and validation-sample anchor isolation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from mosaic import __version__
from mosaic.canonical import sha256_hex
from mosaic.candidates.q_adapter import Draft, typed_decimal
from mosaic.package import PackageBuilder, actor, utc_now
from mosaic.core_contract import cardinal_integer_at, cardinal_integer_value
from mosaic.eval.value_support import audit_value
from mosaic.validation import registry
from mosaic.validation.objects import NULLABLE_SCOPE, SCOPE_REQUIRED

GATES = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10"]
GATE_NAMES = {"G1": "source", "G2": "document", "G3": "schema", "G4": "exactness", "G5": "scientific_scope",
              "G6": "consistency", "G7": "semantic_support", "G8": "coverage", "G9": "qualification", "G10": "projection"}
STOPPING = {"G1", "G3", "G4"}
MAIN_KINDS = {"study", "substudy", "replication", "supplemental_study"}
VALIDATION_KINDS = {"pilot", "pretest", "validation_sample"}
VALIDATION_TEXT = re.compile(r"content[- ]validity|scale development|pre-?test|pilot (?:study|test)|validation (?:sample|study)", re.I)
STUDY_MENTION = re.compile(r"\b(?:Study|Experiment)\s+(\d+[a-z]?)\b", re.I)
# Predicates that hold several values at once (items of one measure, several claims); only
# single-valued predicates can contradict each other.
ITEM_SPAN_KINDS = {"whole_unit_minus_marker", "quoted_substring", "table_cell"}
MULTI_VALUED = {"item_wording", "anchor_wording", "summary_claim", "source_tension", "sequence_position", "analysis_role",
                "factor_level", "event", "condition_cell"}
DERIVATIVE = {"extends", "replicates", "reuses_measure_from", "reuses_sample_from", "adapted_from", "alternative_analysis_of"}
VALIDATING = {"validates_measure_used_in", "validates_manipulation_used_in"}
PASSIVE = re.compile(r"\b(?:is|was|were|are|been)\s+(?:extended|replicated|adapted|followed|mirrored)\b|\bfollowed by\b", re.I)
WORD = re.compile(r"[A-Za-z][A-Za-z\-']{2,}")
STOP = {"the", "and", "for", "with", "that", "this", "was", "were", "are", "from", "into", "not", "but", "their", "they",
        "its", "has", "have", "had", "which", "when", "than", "then", "also", "study", "studies", "participants", "measure",
        "measured", "item", "items", "reported", "report", "used", "using", "between", "each", "all", "any"}


def gate_actor(gate: str) -> dict[str, str]:
    if gate == "G7":
        return actor("human", "mosaic.gates.G7", "reviewer confirmation required; deterministic off-topic pre-screen")
    if gate == "G5":
        return actor("deterministic", "mosaic.gates.G5", "every flag goes to human review")
    return actor("deterministic", f"mosaic.gates.{gate}")


@dataclass
class GateResult:
    gate: str
    outcome: str  # pass | fail | flag | not_applicable | not_run
    checks: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GateContext:
    builder: PackageBuilder
    studies: dict[str, dict[str, Any]]            # local_id -> study object
    coverage_problems: list[str]
    qualified_recipes: set[str] = field(default_factory=set)
    adjudications: dict[str, dict[str, Any]] = field(default_factory=dict)  # draft key -> adjudication record
    rights_checked: bool = False
    drafts: list[Draft] = field(default_factory=list)
    catalogs: dict[str, dict[str, Any]] = field(default_factory=dict)

    def catalog(self, text_region_id: str) -> dict[str, Any] | None:
        if text_region_id not in self.catalogs:
            from mosaic.inventory.span_catalog import CatalogError, build_catalog
            try:
                self.catalogs[text_region_id] = build_catalog(text_region_id, self.objects[text_region_id])
            except CatalogError:
                self.catalogs[text_region_id] = None
        return self.catalogs[text_region_id]

    @property
    def objects(self) -> dict[str, dict[str, Any]]:
        return self.builder.objects


def _check(name: str, ok: bool | None, detail: str = "", flag: bool = False) -> dict[str, Any]:
    outcome = "not_applicable" if ok is None else ("pass" if ok else ("flag" if flag else "fail"))
    out = {"check": name, "outcome": outcome}
    if detail:
        out["detail"] = detail
    return out


def _combine(gate: str, checks: list[dict[str, Any]]) -> GateResult:
    outcomes = {c["outcome"] for c in checks}
    if "fail" in outcomes:
        return GateResult(gate, "fail", checks)
    if "flag" in outcomes:
        return GateResult(gate, "flag", checks)
    if outcomes and outcomes <= {"not_applicable"}:
        return GateResult(gate, "not_applicable", checks)
    return GateResult(gate, "pass", checks)


def _witness_ids(body: dict[str, Any]) -> list[str]:
    ids = list(body.get("witness_ids", []))
    for alt in body.get("alternatives", []) or []:
        ids.extend(w for w in alt["witness_ids"] if w not in ids)
    return ids


def _study_number(local: str) -> tuple[int | None, bool]:
    """(number, is_validation) for study:S2, study:S1-PRETEST, study:PRE1, study:PILOT."""
    label = local.split(":", 1)[1].upper()
    validation = any(k in label for k in ("PRE", "PILOT", "VALID"))
    m = re.search(r"(\d+)", label)
    return (int(m.group(1)) if m else None), validation


# ------------------------------------------------------------------------------------ G1..G10

def g1_source(ctx: GateContext, d: Draft) -> GateResult:
    objs = ctx.objects
    wids = _witness_ids(d.body)
    sources = {objs[w]["source_id"] for w in wids if w in objs} or {o for o, b in objs.items() if b["type"] == "source"}
    checks = []
    for sid in sorted(sources):
        src = objs[sid]
        checks.append(_check("hash_verified", src["artifact"].get("hash_verified_against") == src["artifact"]["sha256"], sid))
        missing = [p for p in ctx.coverage_problems if p.startswith(sid)]
        checks.append(_check("page_set_complete", not missing, "; ".join(missing)[:300]))
        checks.append(_check("processing_permitted", src["rights"]["local_processing"] == "allowed", sid))
    return _combine("G1", checks)


def g2_document(ctx: GateContext, d: Draft) -> GateResult:
    objs = ctx.objects
    checks = []
    for wid in _witness_ids(d.body):
        w = objs.get(wid)
        if w is None:
            checks.append(_check("witness_exists", False, wid))
            continue
        regions = [objs[s["text_artifact_id"]] for s in w.get("spans", [])]
        checks.append(_check("text_not_quarantined", all(r["status"] == "captured" for r in regions), wid))
        if w["transcription_state"] == "recognized":
            checks.append(_check("native_text", False, f"{wid}: OCR-recognized text needs visual confirmation", flag=True))
        first = regions[0] if regions else {}
        if w.get("page"):
            checks.append(_check("page_identity_matches_text", w["page"].get("page_id") == first.get("page_id"), wid))
            page = objs[w["page"]["page_id"]]
            width, height = page["geometry"]["width"], page["geometry"]["height"]
            boxes = w.get("geometry", [])
            if boxes:  # contamination check 1: page and box membership
                inside = all(-1 <= x0 <= x1 <= width + 1 and -1 <= y0 <= y1 <= height + 1 for x0, y0, x1, y1 in boxes)
                checks.append(_check("contamination_page_bbox_membership", inside, wid))
            else:
                checks.append(_check("contamination_page_bbox_membership", None, f"{wid}: no geometry (silent)"))
        else:
            checks.append(_check("contamination_page_bbox_membership", None, f"{wid}: non-PDF locator (silent)"))
    if not checks:
        checks.append(_check("witnesses", None, "no witness (missingness assertion)"))
    return _combine("G2", checks)


def g3_schema(ctx: GateContext, d: Draft) -> GateResult:
    errors = registry.validate(d.kind, d.body)
    checks = [_check("schema_and_code_validators", not errors, "; ".join(errors)[:400])]
    objs = ctx.objects
    for wid in _witness_ids(d.body):
        checks.append(_check("witness_reference", wid in objs and objs[wid]["type"] == "witness", wid))
    if d.kind == "assertion":
        study = d.body.get("scope", {}).get("study_id")
        if study:
            checks.append(_check("scope_study_in_inventory", study in ctx.studies, study, flag=True))
    else:
        for end in ("from_id", "to_id"):
            local = d.body[end]
            if local.startswith("study:"):
                checks.append(_check("endpoint_in_inventory", local in ctx.studies, local, flag=True))
    return _combine("G3", checks)


def _reslice(objs: dict[str, dict[str, Any]], w: dict[str, Any]) -> str:
    pieces = [objs[s["text_artifact_id"]]["text"][s["start_codepoint"]:s["end_codepoint"]] for s in w["spans"]]
    join = w.get("join")
    if len(pieces) == 1 or join == "none":
        return pieces[0]
    return {"adjacent_no_gap": "", "line_break": "\n"}.get(join, " […] ").join(pieces)


def g4_exactness(ctx: GateContext, d: Draft) -> GateResult:
    objs = ctx.objects
    body = d.body
    checks = []
    wids = _witness_ids(body)
    for wid in wids:
        w = objs[wid]
        again = _reslice(objs, w)
        checks.append(_check("copy_rederived_from_artifact", again == w["copied_text"]
                             and sha256_hex(again.encode("utf-8")) == w["copied_text_sha256"], wid))
    # Alternatives are separately asserted possibilities.  An interval must
    # be audited against its own witness set, never the union of alternatives.
    values = [(body.get("value"), list(body.get("witness_ids", [])))]
    values.extend((a.get("value"), list(a.get("witness_ids", []))) for a in body.get("alternatives", []) or [])
    texts = [objs[w]["copied_text"] for w in wids]
    for v, value_wids in values:
        if not isinstance(v, dict):
            continue
        if v.get("kind") in ("decimal", "integer"):
            lex = v.get("lexical", "")
            cardinal = cardinal_integer_value(lex) if body.get("predicate") == "n_items" and v.get("kind") == "integer" else None
            if cardinal is not None:
                def cardinal_in_own_witness(witness):
                    for span in witness["spans"]:
                        source_text = objs[span["text_artifact_id"]]["text"]
                        for match in re.finditer(re.escape(lex), source_text):
                            if (span["start_codepoint"] <= match.start() and match.end() <= span["end_codepoint"]
                                    and cardinal_integer_at(source_text, lex, match.start())):
                                return True
                    return False

                # Inspect the original region around the selected span: a
                # clipped witness cannot hide that ``seven`` was the tail of
                # ``twenty-seven``.
                ok = any(cardinal_in_own_witness(objs[wid]) for wid in wids) and cardinal == v.get("value")
            else:
                bounded = re.compile(rf"(?<![\d.]){re.escape(lex)}(?!\d)") if lex else None
                ok = bool(bounded) and any(bounded.search(t) for t in texts)
                try:
                    ok = ok and typed_decimal(lex) == v.get("value")
                except ValueError:
                    ok = False
            checks.append(_check("numeric_lexical_preserved", ok, f"lexical {lex!r} typed {v.get('value')!r}"))
        elif v.get("kind") == "interval":
            support = audit_value(objs, v, value_wids, body.get("predicate", ""))
            detail = ", ".join(support["reasons"])
            if support["outcome"] == "unsupported":
                checks.append(_check("interval_value_support", False, detail))
            elif support["outcome"] == "needs_review":
                checks.append(_check("interval_value_support", False, detail, flag=True))
            else:
                checks.append(_check("interval_value_support", True, detail))
        elif v.get("kind") == "text" and (body.get("predicate", "").endswith("_wording") or body.get("predicate") == "measure_label"):
            checks.append(_check("wording_equals_copied_text", v.get("lexical") in texts, body.get("subject_id", "")))
    if not checks:
        checks.append(_check("exactness", None, "no exact-copy claim"))
    return _combine("G4", checks)


def _inside(spans: list[dict[str, Any]], regions: list[dict[str, Any]]) -> bool:
    return all(any(r["text_artifact_id"] == s["text_artifact_id"] and r["start_codepoint"] <= s["start_codepoint"]
                   and s["end_codepoint"] <= r["end_codepoint"] for r in regions) for s in spans)


def direction_check(relation: str, from_id: str, to_id: str, witness_texts: list[str],
                    kinds: dict[str, str]) -> tuple[str, str]:
    """(outcome, detail) for a directed study relationship (M-16)."""
    if relation not in DERIVATIVE | VALIDATING:
        return "not_applicable", f"{relation} carries no temporal direction"
    fn, fval = _study_number(from_id)
    tn, tval = _study_number(to_id)
    fval = fval or kinds.get(from_id) in VALIDATION_KINDS
    tval = tval or kinds.get(to_id) in VALIDATION_KINDS
    if relation in VALIDATING:
        if tval and not fval:
            return "fail", f"{from_id} {relation} {to_id}: a main study cannot validate a pretest (relation_direction_reversal)"
        return ("pass", "validation study points at the study it validates") if fval else \
            ("flag", "validating endpoint is not identified as a pretest or validation sample")
    if fn is None or tn is None:
        return "flag", "study order not determinable from labels"
    from_label, to_label = from_id.split(":", 1)[1], to_id.split(":", 1)[1]
    stated = None  # (forward, sentence) from the first sentence that names both studies within one clause
    unclear = None  # a sentence naming both studies only across a comma
    for text in witness_texts:
        for sentence in re.split(r"(?<=[.;])\s+", " ".join(text.split())):
            positions = [(int(re.match(r"\d+", m.group(1)).group(0)), m.start()) for m in STUDY_MENTION.finditer(sentence)]
            fn_positions = [p for n, p in positions if n == fn]
            tn_positions = [p for n, p in positions if n == tn]
            if not (fn_positions and tn_positions):
                continue
            # Either study may also be named elsewhere in the sentence for an unrelated reason ("As
            # with Study 1, ... Study 3 extends Study 1's design"): the two mentions that state the
            # relation share a clause, so prefer a pair with no comma between them, then the closest.
            lo, hi = min(((min(fp, tp), max(fp, tp)) for fp in fn_positions for tp in tn_positions),
                         key=lambda pair: ("," in sentence[pair[0]:pair[1]], pair[1] - pair[0]))
            if "," in sentence[lo:hi]:
                # Only a lead-in or a relative clause joins them ("As in Study 1, Study 3 replicated
                # the manipulation"; "Study 1, which Study 3 extended, ..."); word order across the
                # comma does not say which study is the derivative, so this sentence is not read.
                unclear = unclear or sentence[:200]
                continue
            # Voice is read between the two mentions only, not from another clause of the sentence.
            forward = (lo in fn_positions) != bool(PASSIVE.search(sentence[lo:hi]))
            stated = (forward, sentence[:200])
            break
        if stated:
            break
    if fn < tn:
        if stated and stated[0]:
            return "pass", f"source states the earlier study as the derivative: {stated[1]}"
        return "fail", (f"{from_label} {relation} {to_label} points from an earlier study to a later one and no cited "
                        "sentence states it that way (relation_direction_reversal)")
    if stated is None:
        if unclear:
            return "flag", (f"order is consistent ({from_label} after {to_label}); the only cited sentence naming both "
                            f"studies joins them across a comma, so its direction is not read: {unclear}")
        return "flag", f"order is consistent ({from_label} after {to_label}) but no cited sentence names both studies"
    if stated[0]:
        return "pass", f"cited sentence names {from_label} first: {stated[1]}"
    return "fail", f"cited sentence reads the other way: {stated[1]} (relation_direction_reversal)"


def g5_scope(ctx: GateContext, d: Draft) -> GateResult:
    objs = ctx.objects
    body = d.body
    checks = []
    if d.kind == "relation":
        texts = [objs[w]["copied_text"] for w in body.get("witness_ids", []) if w in objs]
        if body["layer"] == "study":
            kinds = {k: v["kind"] for k, v in ctx.studies.items()}
            outcome, detail = direction_check(body["relation"], body["from_id"], body["to_id"], texts, kinds)
            checks.append({"check": "relation_direction", "outcome": outcome, "detail": detail})
        if body["relation"] == "derived_from":  # a variable cannot derive from itself through other variables (T12)
            edges: dict[str, set[str]] = {}
            for other in ctx.drafts:
                if other.kind == "relation" and other.body["relation"] == "derived_from":
                    edges.setdefault(other.body["from_id"], set()).add(other.body["to_id"])
            seen, frontier = set(), [body["to_id"]]
            while frontier and body["from_id"] not in seen:
                node = frontier.pop()
                if node not in seen:
                    seen.add(node)
                    frontier.extend(edges.get(node, ()))
            cyclic = body["from_id"] in seen
            checks.append(_check("derivation_acyclic", not cyclic,
                                 f"derived_from cycle through {body['from_id']}" if cyclic else ""))
        return _combine("G5", checks)
    scope = body.get("scope", {})
    study = ctx.studies.get(scope.get("study_id", ""))
    spans = [s for w in _witness_ids(body) for s in objs[w].get("spans", [])]
    if study is None:
        checks.append(_check("contamination_study_range_membership", False, f"{scope.get('study_id')} not in inventory", flag=True))
    else:
        if study["kind"] == "nonempirical_section":
            checks.append(_check("study_association_resolved", False,
                                 f"source section {study['local_id']}; empirical study association unresolved", flag=True))
    if study is not None and spans:
        regions = study["boundary"]["regions"]
        if study["kind"] == "paper_level":
            checks.append(_check("contamination_study_range_membership", None, "paper-level scope"))
        else:
            inside = _inside(spans, regions)
            checks.append(_check("contamination_study_range_membership", inside,
                                 f"witness outside {study['local_id']} regions" if not inside else study["local_id"], flag=True))
        if study["inventory_status"] == "ambiguous":
            checks.append(_check("study_boundary_resolved", False, study.get("ambiguity", "")[:300], flag=True))
    other = sorted({f"S{m.group(1).upper()}" for m in STUDY_MENTION.finditer(d.claim_text or "")}
                   - {scope.get("study_id", "").split(":", 1)[-1]})
    if other and study is not None and study["kind"] in MAIN_KINDS:
        checks.append(_check("claim_names_other_study", False, f"claim names {other} while scoped to {study['local_id']}", flag=True))
    # Contamination check 3: validation-sample anchor isolation.
    if body.get("predicate", "").endswith("_wording") and study is not None and study["kind"] in MAIN_KINDS:
        validation_regions = [r for s in ctx.studies.values() if s["kind"] in VALIDATION_KINDS for r in s["boundary"]["regions"]]
        in_validation = bool(validation_regions) and any(_inside([s], validation_regions) for s in spans)
        checks.append(_check("contamination_validation_anchor_isolation", not in_validation,
                             "wording taken from a pretest/validation region" if in_validation else ""))
        context = " ".join(objs[w]["copied_text"] for w in _witness_ids(body))
        unit_texts = []
        for s in spans:
            tr = objs[s["text_artifact_id"]]
            block = next((b for b in tr["blocks"] if b["start"] <= s["start_codepoint"] < max(b["end"], b["start"] + 1)), None)
            if block:
                unit_texts.append(tr["text"][block["start"]:block["end"]])
        if VALIDATION_TEXT.search(" ".join(unit_texts) + " " + context):
            checks.append(_check("contamination_validation_anchor_isolation_text", False,
                                 "source paragraph describes a pretest/validation task", flag=True))
    family = body.get("family")
    required = SCOPE_REQUIRED.get(family, ())
    if required:
        proposed = sorted(f for f in d.flags if f.startswith("scope_") and "proposed" in f)
        nullable = NULLABLE_SCOPE.get(family, set())
        missing = [k for k in required if k not in scope
                   or (not scope[k] and not (k in nullable and scope[k] is None))]
        # Adapters retain unknown identities as explicitly flagged proposal
        # slots so candidate objects remain inspectable. Their non-null IDs do
        # not resolve scientific scope, including for samples and psychometrics.
        complete = not missing and not proposed
        detail = f"scope proposed, not source-resolved: {proposed}" if proposed else (
            f"required scope members unresolved: {missing}" if missing else "")
        check = "complete_result_tuple" if family == "result" else "complete_family_scope"
        checks.append(_check(check, complete, detail, flag=True))
    if family == "result":
        if "statistic_not_confirmed_at_source" in d.flags:
            checks.append(_check("statistic_confirmed_at_source", False,
                                 "value located only by a bare-number match; the cited text does not show this "
                                 "number next to the claimed statistic", flag=True))
    if body.get("predicate") == "item_wording":
        # An item's wording is an item-like span of the source (a list item or standalone question, a quoted
        # string, a table cell); a substring of running prose is the authors' description, possibly a
        # paraphrase, and needs review before it is called verbatim (the Colby regression, FINAL v2 §8.3).
        for w in _witness_ids(body):
            for s in objs[w].get("spans", []):
                cat = ctx.catalog(s["text_artifact_id"])
                kinds = {e["kind"] for e in (cat or {}).get("entries", [])
                         if e["start"] == s["start_codepoint"] and e["end"] == s["end_codepoint"]}
                item_like = bool(kinds & ITEM_SPAN_KINDS)
                checks.append(_check("item_wording_is_item_span", item_like,
                                     "" if item_like else "wording is running text, not an item-like span: possible paraphrase",
                                     flag=True))
        checks.append(_check("rights_gate_restricted_instruments", True if ctx.rights_checked else False,
                             "" if ctx.rights_checked else "measure-catalog rights not checked; wording cannot be citable",
                             flag=True))
    roles = {objs[w]["attribution_role"] for w in _witness_ids(body)}
    if body.get("result_attribution") == "author_result" and roles & {"author_summary_of_prior_work", "quoted_participant",
                                                                     "quoted_practitioner", "quoted_historical_actor"}:
        checks.append(_check("attribution_consistent", False, f"author_result supported by {sorted(roles)}", flag=True))
    if not checks:
        checks.append(_check("scope", None, "nothing scoped to check"))
    return _combine("G5", checks)


def _typed(v: dict[str, Any] | None) -> str | None:
    if not isinstance(v, dict):
        return None
    try:
        return str(Decimal(v["value"])) if v.get("kind") in ("decimal", "integer") and "value" in v else v.get("lexical")
    except Exception:
        return v.get("lexical")


def same_scope(a: dict[str, Any], b: dict[str, Any]) -> bool | None:
    """True if two scopes name the same entities, False if any member differs, None if a member is
    unresolved on one side (then no conflict can be asserted either way)."""
    unknown = False
    for key in set(a) | set(b):
        x, y = a.get(key), b.get(key)
        if x and y and x != y:
            return False
        if not x or not y:
            unknown = True
    return None if unknown else True


def g6_consistency(ctx: GateContext, d: Draft) -> GateResult:
    body = d.body
    checks = []
    if d.kind == "assertion" and body.get("state") == "present" and body.get("predicate") not in MULTI_VALUED:
        mine = _typed(body.get("value"))
        for other in ctx.drafts:
            if other is d or other.kind != "assertion" or other.body.get("state") != "present":
                continue
            ob = other.body
            if (ob.get("subject_id"), ob.get("predicate")) != (body.get("subject_id"), body.get("predicate")):
                continue
            theirs = _typed(ob.get("value"))
            if theirs == mine:
                continue
            same = same_scope(body.get("scope", {}), ob.get("scope", {}))
            if same is False:
                checks.append(_check("same_scope_contradiction", True, f"{other.key}: different scope, not a conflict (M-15, M-17)"))
            elif same is True:
                checks.append(_check("same_scope_contradiction", False, f"{other.key}: {theirs} vs {mine} in the same scope; "
                                     "both kept as alternatives", flag=True))
            else:
                checks.append(_check("same_scope_contradiction", False, f"{other.key}: values differ but a scope member is "
                                     "unresolved; both kept, no conflict asserted (M-17)", flag=True))
    if any(f.startswith("source_tension:unresolved") for f in d.flags):
        checks.append(_check("source_tension_scope_check", False, "a candidate-reported discrepancy is kept as a proposal "
                             "until its model/analysis scope is checked (M-17)", flag=True))
    if not checks:
        checks.append(_check("consistency", True))
    return _combine("G6", checks)


def _tokens(text: str) -> set[str]:
    return {w.lower().strip("-'") for w in WORD.findall(text or "") if w.lower() not in STOP}


def g7_semantic(ctx: GateContext, d: Draft) -> GateResult:
    objs = ctx.objects
    body = d.body
    texts = [objs[w]["copied_text"] for w in _witness_ids(body) if w in objs]
    checks = []
    if texts:
        joined = " ".join(texts)
        claim = d.claim_text or ""
        if body.get("predicate", "").endswith("_wording"):
            letters = sum(c.isalpha() for c in joined)
            checks.append(_check("prescreen_wording_is_text", letters >= 3,
                                 "witness is numbers or tick labels, not wording" if letters < 3 else ""))
        else:
            want = _tokens(claim) | _tokens(body.get("subject_id", "").split("/")[-1].replace("-", " "))
            have = _tokens(joined)
            numeric_only = sum(c.isalpha() for c in joined) < 3
            if body.get("family") == "result":
                ok = True  # a result's number is matched in G4; what it measures is the reviewer's call
            else:
                ok = not numeric_only and (len(want) < 3 or bool(want & have))
            fragment = not ok and not numeric_only and any(not x.rstrip().endswith((".", "!", "?", ":", ";", '"', "”", ")", "]"))
                                                           for x in texts)
            checks.append(_check("prescreen_topical_support", ok,
                                 "" if ok else (f"cited text is a sentence fragment; supporting words may continue in the next "
                                                f"block ({sorted(want)[:6]})" if fragment
                                                else f"no content overlap between claim and evidence ({sorted(want)[:8]})"),
                                 flag=fragment))
    adj = ctx.adjudications.get(d.key)
    if adj is not None and adj.get("decision", "confirm") != "unsure":
        checks.append(_check("reviewer_confirmation", bool(adj["gate_confirmations"].get("G7")), adj["reviewer"]))
    elif adj is not None:
        checks.append(_check("reviewer_confirmation", False, f"reviewer unsure ({adj['reviewer']})", flag=True))
    else:
        checks.append(_check("reviewer_confirmation", False, "awaiting human review (G7 actor before qualification)", flag=True))
    return _combine("G7", checks)


def g8_coverage(ctx: GateContext, d: Draft) -> GateResult:
    checks = [_check("package_page_coverage", not ctx.coverage_problems, "; ".join(ctx.coverage_problems)[:300])]
    study = d.body.get("scope", {}).get("study_id") if d.kind == "assertion" else None
    if study:
        checks.append(_check("inventory_item_exists", study in ctx.studies, study, flag=True))
    return _combine("G8", checks)


def g9_qualification(ctx: GateContext, d: Draft) -> GateResult:
    rid = d.body["recipe_id"]
    ok = rid in ctx.qualified_recipes
    return _combine("G9", [_check("recipe_in_qualification_registry", ok,
                                  "" if ok else f"{rid[:19]}... not qualified: candidate/inspectable only", flag=True)])


def g10_projection(ctx: GateContext, d: Draft) -> GateResult:
    return GateResult("G10", "not_applicable", [_check("projection", None, "no projection at package build (§6.10 adapters)")])


RUNNERS = {"G1": g1_source, "G2": g2_document, "G3": g3_schema, "G4": g4_exactness, "G5": g5_scope,
           "G6": g6_consistency, "G7": g7_semantic, "G8": g8_coverage, "G9": g9_qualification, "G10": g10_projection}


def run_unit(ctx: GateContext, d: Draft) -> list[GateResult]:
    results: list[GateResult] = []
    stopped = False
    for gate in GATES:
        if stopped:
            results.append(GateResult(gate, "not_run", [_check("stopped", None, "an earlier gate failed")]))
            continue
        r = RUNNERS[gate](ctx, d)
        results.append(r)
        if r.outcome == "fail" and gate in STOPPING:
            stopped = True
    return results


def run_all(ctx: GateContext) -> dict[str, list[GateResult]]:
    return {d.key: run_unit(ctx, d) for d in ctx.drafts}


def gate_events(builder: PackageBuilder, results: dict[str, list[GateResult]], final_ids: dict[str, str]) -> list[str]:
    """One validation-event per gate, with a check row per unit, in gate order."""
    ids = []
    for order, gate in enumerate(GATES, start=1):
        checks, outcomes = [], []
        for key, rs in results.items():
            r = rs[order - 1]
            outcomes.append(r.outcome)
            for c in r.checks:
                row = {"check": c["check"], "outcome": c["outcome"] if c["outcome"] != "not_run" else "not_applicable",
                       "subject": final_ids.get(key, key)}
                if c.get("detail"):
                    row["detail"] = c["detail"][:500]
                checks.append(row)
        outcome = "fail" if "fail" in outcomes else "flag" if "flag" in outcomes else \
            "not_applicable" if outcomes and set(outcomes) <= {"not_applicable", "not_run"} else "pass"
        body = builder.envelope("validation-event", "recorded", gate_actor(gate), sorted(set(final_ids.values())))
        body.update(event_kind="gate", gate=gate, gate_order=order, actor=gate_actor(gate), subject_ids=sorted(set(final_ids.values())),
                    outcome=outcome, checks=checks, checker_version=f"mosaic.gates/{__version__}", recorded_utc=utc_now(),
                    detail={"gate_name": GATE_NAMES[gate], "units": len(results)})
        ids.append(builder.add(body))
    return ids
