"""Adapter from replayed Q candidates to package drafts (HANDOFF §6.5; audit M-13, M-16, M-18).

Adapted in spirit from Q/candidate_adapter.py (REUSE_LEDGER R020): every
evidence unit id must resolve, image-derived or unverified units are flagged,
and nothing is promoted. What is new:

* Exact wording (`verbatim_components`) is copied from the fresh capture and
  re-anchored to the span catalog, so a leading list marker ("1.") is not
  part of an item's wording.
* A finding's prose bundles several numbers ("B = 0.86 ... B = −0.53 ...").
  Each number becomes its own result assertion whose value is copied from the
  cited source text with its lexical form, statistic and comparator; a number
  the cited units do not contain is not accepted (M-13). The prose itself is
  kept as an extractor-inference assertion, never as a source value.
* Study relationships become directed `relation` drafts with a stated
  direction basis, checked by G5 (M-16).

Drafts are plain dicts; `mosaic.gates` evaluates them and `mosaic.tiers`
finalizes and adds them, because an object's id depends on its final tier.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

from mosaic.candidates import CandidateSet
from mosaic.canonical import canonical_bytes
from mosaic.eval.structure_validator import study_edges
from mosaic.inventory.q_copier import CopyError, UnitRef, copy_q_exact_text
from mosaic.inventory.span_catalog import LIST_MARKER, build_catalog
from mosaic.inventory.witnesses import WitnessError, text_witness
from mosaic.package import PackageBuilder, actor, utc_now
from mosaic.variable_roles import CORE_ROLES, ROLE_LABELS
from mosaic.core_contract import SCOPE_LABELS, field_definitions
from mosaic.validation.objects import SCOPE_REQUIRED, NULLABLE_SCOPE

STAT = r"(?:B|b|β|r|R2|R²|d|g|OR|F|t|z|χ2|χ²|p|N|n|M|SD|SE|α|ω|η2|η²)"
NUMBER = r"[−–-]?(?:\d+(?:\.\d+)?|\.\d+)"
CLAIM_VALUE = re.compile(rf"(?P<stat>{STAT})\s*(?P<cmp>=|<|>|≤|≥)\s*(?P<num>{NUMBER})")
CLAIM_CI = re.compile(rf"(?P<level>\d{{2}})%\s*CI\s*\[\s*(?P<lo>{NUMBER})\s*,\s*(?P<hi>{NUMBER})\s*\]")
COMPARATORS = {"=": "eq", "<": "lt", ">": "gt", "≤": "le", "≥": "ge"}
IMAGE_UNIT = re.compile(r"image|unverified")


@dataclass
class Draft:
    key: str
    kind: str  # assertion | relation
    body: dict[str, Any]
    claim_text: str = ""
    flags: list[str] = field(default_factory=list)


def typed_decimal(lexical: str) -> str:
    text = lexical.replace("−", "-").replace("–", "-")
    neg = text.startswith("-")
    text = text.lstrip("-")
    if text.startswith("."):
        text = "0" + text
    try:
        Decimal(text)
    except InvalidOperation:
        raise ValueError(f"not a decimal: {lexical}") from None
    return ("-" if neg else "") + text


def _number_pattern(typed: str) -> str:
    """Source lexical variants of a typed decimal: minus sign forms and an optional leading zero."""
    neg = typed.startswith("-")
    digits = typed.lstrip("-")
    body = re.escape(digits)
    if digits.startswith("0."):
        body = rf"0?{re.escape(digits[1:])}"
    return (r"[−–-]" if neg else r"(?<![−–-])") + rf"(?<![\d.]){body}(?![\d])"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48] or "x"


def _reference_component(text: str) -> str:
    """Encode opaque model IDs injectively into the package's local-ID alphabet.

    Underscore is reserved for UTF-8 byte escapes, so ``var:x``, ``var-x`` and
    a literal ``var_3ax`` cannot silently become the same variable. Historical
    replay bypasses this live-ID encoding; these identifiers are never display names.
    """
    return "".join(chr(byte) if (48 <= byte <= 57 or 65 <= byte <= 90 or 97 <= byte <= 122
                                  or chr(byte) in ".-/") else f"_{byte:02x}"
                   for byte in text.encode("utf-8")) or "_empty"


def _live_namespace(cset: CandidateSet) -> str:
    """Bind response-local identities to their batch and exact proposed content."""
    return hashlib.sha256(canonical_bytes({"batch": cset.batch_id, "payload": cset.payload})).hexdigest()


def _component_measure(comp: dict[str, Any], index: int) -> tuple[str, str | None, str | None]:
    """Return a stable local measure slot without turning an absent model label into a name.

    The live response schema permits an item component without ``measure``.  It is still useful
    evidence for the exact wording, but the adapter must not claim that its unnamed instrument is
    a measure called ``measure`` (or merge it with another unnamed component).  Keep every such
    component in its own explicitly unresolved local slot; the companion label assertion records
    the absence for review.
    """
    raw = comp.get("measure")
    if isinstance(raw, str) and raw.strip():
        return _slug(raw), raw, None
    identity = hashlib.sha256(canonical_bytes(comp)).hexdigest()[:16]
    return f"unresolved-{index + 1}-{identity}", None, "measure_name_unresolved"


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    s, e = start, end
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def _clause(claim: str, pos: int) -> str:
    start = max(claim.rfind(";", 0, pos), claim.rfind(". ", 0, pos)) + 1
    ends = [i for i in (claim.find(";", pos), claim.find(". ", pos)) if i >= 0]
    return claim[start:min(ends) if ends else len(claim)].strip()


class QCandidateAdapter:
    def __init__(self, builder: PackageBuilder, index: dict[str, UnitRef], q_units: dict[str, dict[str, Any]],
                 study_locals: set[str]):
        self.b, self.index, self.q_units, self.study_locals = builder, index, q_units, study_locals
        self._catalogs: dict[str, dict[str, Any]] = {}
        self._unit_witness: dict[str, str] = {}
        self._study_cache: dict[str, dict[str, Any]] = {}

    # --------------------------------------------------------------------------------- helpers
    def envelope(self, cset: CandidateSet, tag: str) -> dict[str, Any]:
        return self.b.envelope(tag, "materialized", cset.created_by, [])

    def unit_witness(self, uid: str) -> tuple[str | None, str | None]:
        """Whole-unit context witness for a cited unit, or (None, reason)."""
        if uid in self._unit_witness:
            return self._unit_witness[uid], None
        unit = self.q_units.get(uid)
        if unit is None:
            return None, f"unknown_evidence_unit:{uid}"
        if IMAGE_UNIT.search(uid) or IMAGE_UNIT.search(str(unit.get("kind", ""))):
            return None, f"evidence_unit_not_native:{uid}"
        ref = self.index.get(uid)
        if ref is None:
            return None, f"evidence_unit_not_in_capture:{uid}"
        text = self.b.objects[ref.text_region_id]["text"]
        s, e = ref.start, ref.end
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        wid = text_witness(self.b, [(ref.text_region_id, s, e)], attribution_role="author_report",
                           retrieval_transform=f"whole unit {uid} cited as evidence by a replayed candidate")
        self._unit_witness[uid] = wid
        return wid, None

    def _catalog(self, tr_id: str) -> dict[str, Any]:
        if tr_id not in self._catalogs:
            self._catalogs[tr_id] = build_catalog(tr_id, self.b.objects[tr_id])
        return self._catalogs[tr_id]

    def _witnesses(self, unit_ids: list[str], flags: list[str]) -> list[str]:
        out = []
        for uid in unit_ids:
            wid, problem = self.unit_witness(uid)
            if wid and wid not in out:
                out.append(wid)
            if problem:
                flags.append(problem)
        return out

    def _study(self, study_id: str, flags: list[str]) -> str:
        local = f"study:{study_id}"
        if local not in self.study_locals:
            flags.append(f"study_not_in_inventory:{local}")
        return local

    # ------------------------------------------------------------------------------ wording
    def verbatim(self, cset: CandidateSet, index: int, comp: dict[str, Any], study: str) -> Draft:
        flags: list[str] = []
        body = self.envelope(cset, "assertion")
        predicate = "item_wording" if comp.get("kind") == "item" else "anchor_wording"
        label = study.split(":", 1)[1]
        measure, _measure_label, measure_flag = _component_measure(comp, index)
        flags.append("scope_instrument_version_study_local")
        if measure_flag:
            flags.append(measure_flag)
        body.update(family="measure", subject_id=f"measure:{label}/{measure}", predicate=predicate,
                    scope={"study_id": study, "instrument_version_id": f"instrument:{label}/{measure}"},
                    interpretation_kind="reported",
                    validation_status="candidate", acceptance_tier="candidate",
                    candidate_origin=cset.origin(f"/verbatim_components/{index}", comp.get("exact_text")))
        try:
            copied = copy_q_exact_text(self.b, self.index, comp["unit_id"], comp["exact_text"], self.q_units,
                                       attribution_role="author_report")
        except (CopyError, WitnessError) as exc:
            body.update(state="not_processed", value=None, witness_ids=[], reason=f"exact copy refused: {exc}")
            return Draft(f"verbatim[{index}]", "assertion", body, f"{comp.get('measure', '')}: {comp.get('exact_text', '')}",
                         flags + [f"copy_failed:{exc}"])
        witness_id, text = copied.witness_id, copied.text
        if copied.located_by != "exact":
            flags.append(f"located_by_{copied.located_by}_match")
        # Re-anchor to the catalog: an item's wording excludes a standalone list marker.
        for entry in self._catalog(copied.text_region_id)["entries"]:
            if entry["kind"] == "whole_unit_minus_marker" and entry["end"] == copied.end and copied.start < entry["start"]:
                gap = self.b.objects[copied.text_region_id]["text"][copied.start:entry["start"]]
                if LIST_MARKER.match(gap + "x") or gap.strip().rstrip(".)").isdigit():
                    witness_id = text_witness(self.b, [(copied.text_region_id, entry["start"], entry["end"])],
                                              attribution_role="author_report", provenance_ids=[copied.witness_id],
                                              retrieval_transform=f"replayed selection re-anchored to catalog span "
                                                                  f"{entry['span_id']} (list marker excluded)")
                    text = self.b.objects[copied.text_region_id]["text"][entry["start"]:entry["end"]]
                    flags.append("list_marker_excluded")
                break
        body.update(state="present", value={"kind": "text", "lexical": text}, witness_ids=[witness_id])
        body["provenance_ids"] = sorted({copied.witness_id, witness_id})
        return Draft(f"verbatim[{index}]", "assertion", body, f"{comp.get('measure', '')}: {comp.get('exact_text', '')}", flags)

    # -------------------------------------------------------------------------- measure label
    def _study_object(self, study: str) -> dict[str, Any] | None:
        obj = self._study_cache.get(study)
        if obj is None:
            obj = next((o for o in self.b.objects.values() if o.get("type") == "study" and o.get("local_id") == study), None)
            if obj is not None:
                self._study_cache[study] = obj
        return obj

    def _label_candidates(self, region: dict[str, Any], lo: int, hi: int) -> list[tuple[int, int]]:
        """Trimmed (start, end) spans of every whole block or line inside [lo, hi) of one text-region."""
        text = region["text"]
        seen: set[tuple[int, int]] = set()
        out: list[tuple[int, int]] = []
        for block in region.get("blocks", []):
            spans = [(block["start"], block["end"])] + [(ln["start"], ln["end"]) for ln in block.get("lines", [])]
            for s, e in spans:
                ts, te = _trim(text, s, e)
                if ts >= te or ts < lo or te > hi:
                    continue
                if (ts, te) not in seen:
                    seen.add((ts, te))
                    out.append((ts, te))
        return out

    def _label_hits(self, scopes: list[tuple[str, int, int]], label: str) -> list[tuple[str, int, int]]:
        """Whole-unit (block/line) hits of `label` inside the given (text_region_id, lo, hi) scopes: a
        trimmed block or line, optionally with a trailing colon or period, equal to `label` case-sensitively
        -- exact match tried first, then whitespace-normalized (mirrors q_copier._whitespace_hits). A
        mid-sentence occurrence never counts, because only whole trimmed blocks/lines are candidates."""
        normalized = " ".join(label.split())
        exact_pat = re.compile(re.escape(label) + r"[:.]?")
        ws_pat = re.compile(re.escape(normalized) + r"[:.]?")
        exact_hits: list[tuple[str, int, int]] = []
        ws_hits: list[tuple[str, int, int]] = []
        seen: set[tuple[str, int, int]] = set()
        for region_id, lo, hi in scopes:
            region = self.b.objects.get(region_id)
            if region is None or region.get("type") != "text-region" or region.get("status") != "captured":
                continue
            for s, e in self._label_candidates(region, lo, hi):
                key = (region_id, s, e)
                if key in seen:
                    continue
                seen.add(key)
                cleaned = region["text"][s:e]
                if exact_pat.fullmatch(cleaned):
                    exact_hits.append(key)
                elif ws_pat.fullmatch(" ".join(cleaned.split())):
                    ws_hits.append(key)
        return exact_hits or ws_hits

    def _find_measure_label(self, label: str, unit_ids: list[str], study: str) -> list[tuple[str, int, int]]:
        """Whole-unit hits of `label`: first inside the measure's own cited evidence units, then (only if
        that finds nothing) inside the same study's boundary regions (the study object's boundary.regions,
        as G5's contamination check uses) -- never the whole document."""
        own = [(ref.text_region_id, ref.start, ref.end) for uid in unit_ids if (ref := self.index.get(uid)) is not None]
        hits = self._label_hits(own, label) if own else []
        if hits:
            return hits
        study_obj = self._study_object(study)
        if study_obj is None:
            return []
        boundary = [(r["text_artifact_id"], r["start_codepoint"], r["end_codepoint"])
                   for r in study_obj.get("boundary", {}).get("regions", [])]
        return self._label_hits(boundary, label)

    def measure_label(self, cset: CandidateSet, key: str, pointer: str, subject_id: str, instrument_version_id: str,
                      label: str | None, unit_ids: list[str], study: str) -> Draft:
        flags = ["scope_instrument_version_study_local"]
        body = self.envelope(cset, "assertion")
        body.update(family="measure", subject_id=subject_id, predicate="measure_label",
                    scope={"study_id": study, "instrument_version_id": instrument_version_id},
                    interpretation_kind="extractor_inference", validation_status="candidate", acceptance_tier="candidate",
                    candidate_origin=cset.origin(pointer, label))
        if not label:
            flags.append("measure_label_no_candidate_name")
            body.update(state="not_found_in_reviewed_scope", value=None, witness_ids=[],
                        reviewed_scope=unit_ids or ["(none cited)"])
            return Draft(key, "assertion", body, label or "", flags)
        hits = self._find_measure_label(label, unit_ids, study)
        if not hits:
            body.update(state="not_found_in_reviewed_scope", value=None, witness_ids=[],
                        reviewed_scope=unit_ids or ["(none cited)"])
            return Draft(key, "assertion", body, label, flags)
        if len(hits) == 1:
            tr, s, e = hits[0]
            text = self.b.objects[tr]["text"][s:e]
            wid = text_witness(self.b, [(tr, s, e)], attribution_role="author_report",
                               retrieval_transform=f"measure label matched a whole source block or line for {subject_id}")
            body.update(state="present", value={"kind": "text", "lexical": text}, witness_ids=[wid])
            return Draft(key, "assertion", body, label, flags)
        # More than one whole-unit hit: several distinct occurrences of the same label in scope are
        # ambiguous (D-017 refuse-rather-than-guess) rather than a same-text free pass.
        alternatives = []
        for tr, s, e in hits:
            text = self.b.objects[tr]["text"][s:e]
            try:
                wid = text_witness(self.b, [(tr, s, e)], attribution_role="author_report",
                                   retrieval_transform=f"measure label matched a whole source block or line for {subject_id}")
            except WitnessError:
                continue
            alternatives.append({"value": {"kind": "text", "lexical": text}, "witness_ids": [wid]})
        if len(alternatives) >= 2:
            body.update(state="ambiguous_or_conflicting", value=None, alternatives=alternatives,
                        witness_ids=sorted({w for a in alternatives for w in a["witness_ids"]}))
            flags.append("measure_label_ambiguous_match")
            return Draft(key, "assertion", body, label, flags)
        body.update(state="not_found_in_reviewed_scope", value=None, witness_ids=[],
                    reviewed_scope=unit_ids or ["(none cited)"])
        flags.append("measure_label_ambiguous_match_unresolved")
        return Draft(key, "assertion", body, label, flags)

    # ----------------------------------------------------------------------------- findings
    def _find_number(self, unit_ids: list[str], stat: str | None, cmp_sym: str | None,
                     typed: str) -> list[tuple[str, int, int, str, bool]]:
        """Occurrences (text_region_id, start, end, source lexical number, anchored) in the cited units.

        `anchored` is True only when the number was found next to the claimed statistic and
        comparator ("STAT cmp NUM"); a bare-digit fallback hit (no such pattern found) is marked
        False so callers do not bind the claimed statistic label to an unrelated number (audit finding 1).
        """
        pattern = _number_pattern(typed)
        hits: list[tuple[str, int, int, str, bool]] = []
        for uid in unit_ids:
            ref = self.index.get(uid)
            if ref is None:
                continue
            text = self.b.objects[ref.text_region_id]["text"]
            unit = text[ref.start:ref.end]
            full = re.compile(rf"(?<![A-Za-z]){re.escape(stat)}\s*{re.escape(cmp_sym)}\s*(?P<n>{pattern})", re.I) if stat and cmp_sym else None
            found = list(full.finditer(unit)) if full else []
            for m in found:
                hits.append((ref.text_region_id, ref.start + m.start(), ref.start + m.end(), m.group("n"), True))
            if not found:
                for m in re.finditer(pattern, unit):
                    hits.append((ref.text_region_id, ref.start + m.start(), ref.start + m.end(), m.group(0), False))
        return hits

    def finding(self, cset: CandidateSet, index: int, f: dict[str, Any], study: str) -> list[Draft]:
        drafts: list[Draft] = []
        flags: list[str] = []
        unit_ids = f.get("evidence_unit_ids", [])
        witnesses = self._witnesses(unit_ids, flags)
        topic = _slug(f.get("topic", f"finding-{index}"))
        subject = f"finding:{study.split(':', 1)[1]}/{topic}"
        claim = f.get("claim", "")
        body = self.envelope(cset, "assertion")
        body.update(family="interpretation", subject_id=subject, predicate="summary_claim", scope={"study_id": study},
                    interpretation_kind="extractor_inference", result_attribution="author_result",
                    validation_status="candidate", acceptance_tier="candidate",
                    candidate_origin=cset.origin(f"/findings/{index}", claim), witness_ids=witnesses)
        if f.get("status") == "unresolved":
            body.update(state="not_found_in_reviewed_scope", value=None, reviewed_scope=unit_ids or ["(none cited)"])
        else:
            body.update(state="present", value={"kind": "paraphrase", "lexical": claim})
        drafts.append(Draft(f"finding[{index}]", "assertion", body, claim, list(flags)))
        # One result assertion per number, copied from the cited source (M-13).
        values = [(m.start(), m.group("stat"), m.group("cmp"), m.group("num"), None) for m in CLAIM_VALUE.finditer(claim)]
        values += [(m.start(), "CI", None, None, m) for m in CLAIM_CI.finditer(claim)]
        for n, (pos, stat, cmp_sym, num, ci) in enumerate(sorted(values, key=lambda v: v[0])):
            # The candidate names neither sample nor analysis: both are proposed, flagged, and G5 keeps the
            # result tuple open until a reviewer resolves them.
            vflags: list[str] = ["scope_sample_proposed_default", "scope_analysis_proposed_from_topic"]
            label = study.split(":", 1)[1]
            rbody = self.envelope(cset, "assertion")
            rbody.update(family="result", subject_id=f"{subject}/v{n + 1}", predicate="reported_statistic",
                         scope={"study_id": study, "sample_id": f"sample:{label}/analyzed", "analysis_id": f"analysis:{label}/{topic}"},
                         interpretation_kind="reported", result_attribution="author_result",
                         validation_status="candidate", acceptance_tier="candidate",
                         candidate_origin=cset.origin(f"/findings/{index}", _clause(claim, pos)))
            if ci is not None:
                lo, hi = typed_decimal(ci.group("lo")), typed_decimal(ci.group("hi"))
                lo_hits = self._find_number(unit_ids, None, None, lo)
                hi_hits = self._find_number(unit_ids, None, None, hi)
                pair = [(a, b) for a in lo_hits for b in hi_hits if a[0] == b[0] and 0 < b[1] - a[2] <= 4]
                if len(pair) == 1:
                    (tr, s, _, lo_lex, _), (_, _, e, hi_lex, _) = pair[0]
                    wid = text_witness(self.b, [(tr, s, e)], attribution_role="author_result")
                    rbody.update(state="present", witness_ids=[wid],
                                 value={"kind": "interval", "lexical": self.b.objects[tr]["text"][s:e], "lower": lo, "upper": hi,
                                        "level": ci.group("level"), "statistic": "CI"})
                else:
                    rbody.update(state="not_found_in_reviewed_scope", value=None, witness_ids=[],
                                 reviewed_scope=unit_ids or ["(none cited)"])
                    vflags.append("interval_not_in_cited_source" if not pair else "ambiguous_numeric_occurrence")
            else:
                typed = typed_decimal(num)
                hits = self._find_number(unit_ids, stat, cmp_sym, typed)
                if len(hits) == 1:
                    tr, s, e, lex, anchored = hits[0]
                    wid = text_witness(self.b, [(tr, s, e)], attribution_role="author_result")
                    rbody.update(state="present", witness_ids=[wid], value={
                        "kind": "decimal" if "." in typed else "integer", "lexical": lex, "value": typed_decimal(lex),
                        "comparator": COMPARATORS[cmp_sym], "statistic": stat,
                        "sign": "negative" if typed.startswith("-") else ("zero" if Decimal(typed) == 0 else "positive")})
                    if not anchored:
                        # The number was located only by a bare-digit fallback: no "STAT cmp NUM" pattern
                        # matched anywhere in the cited units, so this hit is not anchored to the claimed
                        # statistic and may belong to a different one. Kept, but flagged rather than bound
                        # silently (audit finding 1; FINAL v2 M-13, §6.3).
                        vflags.append("statistic_not_confirmed_at_source")
                elif not hits:
                    rbody.update(state="not_found_in_reviewed_scope", value=None, witness_ids=[],
                                 reviewed_scope=unit_ids or ["(none cited)"])
                    vflags.append("numeric_value_not_in_cited_source")
                else:
                    alternatives = []
                    for tr, s, e, lex, _ in hits[:6]:
                        wid = text_witness(self.b, [(tr, s, e)], attribution_role="author_result")
                        alternatives.append({"value": {"kind": "decimal" if "." in typed else "integer", "lexical": lex,
                                                       "value": typed_decimal(lex), "statistic": stat}, "witness_ids": [wid]})
                    rbody.update(state="ambiguous_or_conflicting", value=None, alternatives=alternatives,
                                 witness_ids=sorted({w for a in alternatives for w in a["witness_ids"]}))
                    vflags.append("ambiguous_numeric_occurrence")
            drafts.append(Draft(f"finding[{index}].v{n + 1}", "assertion", rbody, _clause(claim, pos), vflags))
        return drafts

    # ------------------------------------------------------------------ variables, sequence, tensions
    def variable(self, cset: CandidateSet, index: int, v: dict[str, Any], study: str) -> Draft:
        flags: list[str] = []
        body = self.envelope(cset, "assertion")
        body.update(family="variable", subject_id=f"variable:{study.split(':', 1)[1]}/{_slug(v.get('name', str(index)))}",
                    predicate="analysis_role", scope={"study_id": study}, state="present",
                    value={"kind": "paraphrase", "lexical": v.get("analysis_role", "")},
                    witness_ids=self._witnesses(v.get("evidence_unit_ids", []), flags),
                    interpretation_kind="extractor_inference", validation_status="candidate", acceptance_tier="candidate",
                    candidate_origin=cset.origin(f"/derived_variables/{index}", v.get("definition")))
        return Draft(f"variable[{index}]", "assertion", body, f"{v.get('name', '')}: {v.get('definition', '')}", flags)

    def sequence(self, cset: CandidateSet, index: int, ev: dict[str, Any], study: str) -> Draft:
        flags: list[str] = []
        unit_ids = ev.get("evidence_unit_ids", [])
        body = self.envelope(cset, "assertion")
        body.update(family="procedure", subject_id=f"event:{study.split(':', 1)[1]}/{index + 1}", predicate="sequence_position",
                    scope={"study_id": study}, witness_ids=self._witnesses(unit_ids, flags),
                    interpretation_kind="extractor_inference", validation_status="candidate", acceptance_tier="candidate",
                    candidate_origin=cset.origin(f"/sequence/{index}", f"{ev.get('event', '')} | {ev.get('position', '')}"))
        if ev.get("order_status") == "unresolved":
            body.update(state="not_found_in_reviewed_scope", value=None, reviewed_scope=unit_ids or ["(none cited)"])
        else:
            body.update(state="present", value={"kind": "paraphrase", "lexical": f"{ev.get('event', '')}: {ev.get('position', '')}"})
        return Draft(f"sequence[{index}]", "assertion", body, ev.get("event", ""), flags)

    def tension(self, cset: CandidateSet, index: int, t: dict[str, Any], study: str) -> Draft:
        flags = [f"source_tension:{t.get('resolution_status', 'unknown')}"]
        body = self.envelope(cset, "assertion")
        body.update(family="interpretation", subject_id=f"tension:{study.split(':', 1)[1]}/{index + 1}", predicate="source_tension",
                    scope={"study_id": study}, state="present", value={"kind": "paraphrase", "lexical": t.get("description", "")},
                    witness_ids=self._witnesses(t.get("evidence_unit_ids", []), flags),
                    interpretation_kind="extractor_inference", result_attribution="system_hypothesis",
                    validation_status="candidate", acceptance_tier="candidate",
                    candidate_origin=cset.origin(f"/source_tensions/{index}", t.get("description")))
        return Draft(f"tension[{index}]", "assertion", body, t.get("description", ""), flags)

    def study_relationship(self, cset: CandidateSet, index: int, edge: dict[str, Any]) -> Draft:
        flags: list[str] = []
        body = self.envelope(cset, "relation")
        src, dst = self._study(edge["from_study_id"], flags), self._study(edge["to_study_id"], flags)
        relation = edge.get("mapped_relation") or edge["relation"]
        if relation != edge["relation"]:
            flags.append(f"relation_mapped:{edge['relation']}->{relation}")
        body.update(relation=relation, layer="study", from_id=src, to_id=dst,
                    direction_basis=f"replayed candidate states {edge['from_study_id']} {edge['relation']} {edge['to_study_id']}; "
                                    "checked against the source by G5",
                    witness_ids=self._witnesses(edge.get("evidence_unit_ids", []), flags),
                    validation_status="candidate", acceptance_tier="candidate",
                    candidate_origin=cset.origin(f"/study_relationships/{index}"))
        return Draft(f"study_relationship[{index}]", "relation", body, "", flags)

    # --------------------------------------------------------------- typed structure branches (M2)
    def _typed(self, cset: CandidateSet, key: str, pointer: str, family: str, subject: str, predicate: str,
               value: dict[str, Any], unit_ids: list[str], study: str, claim: str, extra_scope: dict[str, str | None] | None = None) -> Draft:
        flags: list[str] = []
        body = self.envelope(cset, "assertion")
        body.update(family=family, subject_id=subject, predicate=predicate, scope={"study_id": study, **(extra_scope or {})},
                    state="present", value=value, witness_ids=self._witnesses(unit_ids, flags),
                    interpretation_kind="extractor_inference", validation_status="candidate", acceptance_tier="candidate",
                    candidate_origin=cset.origin(pointer, claim))
        if not body["witness_ids"]:
            body.update(state="not_found_in_reviewed_scope", value=None, reviewed_scope=unit_ids or ["(none cited)"])
        return Draft(key, "assertion", body, claim, flags)

    def _relation(self, cset: CandidateSet, key: str, pointer: str, relation: str, src: str, dst: str,
                  unit_ids: list[str]) -> Draft:
        flags: list[str] = []
        body = self.envelope(cset, "relation")
        body.update(relation=relation, layer="scientific", from_id=src, to_id=dst, witness_ids=self._witnesses(unit_ids, flags),
                    validation_status="candidate", acceptance_tier="candidate", candidate_origin=cset.origin(pointer))
        return Draft(key, "relation", body, "", flags)

    def core_fact(self, cset: CandidateSet, index: int, fact: dict, study: str) -> Draft:
        """Keep one typed value and explicit source-proposed scope per CORE fact."""
        definition = field_definitions()[fact["field"]]
        scope = {}
        label = study.removeprefix("study:")
        for key in SCOPE_LABELS:
            name = fact["scope"][key]
            scope[key + "_id"] = (f"{key}:{label}/" + hashlib.sha256(name.strip().encode("utf-8")).hexdigest()[:20]
                                  if name is not None else None)
        unresolved = []
        for key in SCOPE_REQUIRED.get(definition["family"], ()):
            if key == "study_id" or scope.get(key) is not None or key in NULLABLE_SCOPE.get(definition["family"], set()):
                continue
            # The package contract requires a non-null local reference. This is
            # an explicitly unresolved proposed slot, never a named real sample.
            prefix = key.removesuffix("_id")
            scope[key] = f"{prefix}:{label}/unresolved-" + hashlib.sha256(
                (fact["field"] + "\0" + fact["subject"]).encode("utf-8")).hexdigest()[:20]
            unresolved.append(f"scope_{prefix}_proposed_unresolved")
        subject = f"{definition['family']}:{label}/" + hashlib.sha256(fact["subject"].strip().encode("utf-8")).hexdigest()[:20]
        representation = definition["representation"]
        if representation in ("integer", "number"):
            value = {"kind": "integer" if representation == "integer" else "decimal",
                     "lexical": fact["lexical"], "value": typed_decimal(fact["value"]),
                     "statistic": fact["statistic"], "comparator": fact["comparator"]}
        elif representation == "interval":
            value = {"kind": "interval", "lexical": fact["lexical"],
                     "lower": typed_decimal(fact["lower"]), "upper": typed_decimal(fact["upper"]),
                     "statistic": "CI"}
            if fact["level"] is not None:
                value["level"] = typed_decimal(fact["level"])
        else:
            value = {"kind": "text" if representation == "span" else "paraphrase",
                     "lexical": fact.get("exact_text", fact.get("lexical", ""))}
        description = fact["subject"] + "; " + "; ".join(f"{key}: {name}" for key, name in fact["scope"].items() if name is not None)
        draft = self._typed(cset, f"core_fact[{index}]", f"/core_facts/{index}", definition["family"],
                            subject, definition["predicate"], value, fact["evidence_unit_ids"], study, description, scope)
        if definition["family"] in ("result", "psychometric", "interpretation"):
            draft.body["result_attribution"] = fact["attribution"]
        draft.flags.append("scope_proposed_from_source_labels")
        draft.flags.extend(unresolved)
        if representation == "span":
            span = fact.get("resolved_span")
            try:
                if not span or fact.get("resolution_error"):
                    raise WitnessError("core_span_not_resolved")
                rid, start, end = span["text_artifact_id"], span["start_codepoint"], span["end_codepoint"]
                if self.b.objects[rid]["text"][start:end] != fact["exact_text"]:
                    raise WitnessError("core_span_text_mismatch")
                wid = text_witness(self.b, [(rid, start, end)], attribution_role="author_report",
                                   retrieval_transform="exact CORE wording selected from the offered span catalog")
                draft.body.update(witness_ids=[wid], state="present", interpretation_kind="reported")
                draft.body.pop("reviewed_scope", None)
            except (KeyError, WitnessError) as exc:
                draft.body.update(state="not_processed", value=None, witness_ids=[], reason=str(exc))
                draft.body.pop("reviewed_scope", None)
                draft.flags.append("core_wording_copy_refused")
        return draft

    def structure(self, cset: CandidateSet, study: str) -> list[Draft]:
        """Variables, factor levels, condition cells, allocation, events and measures of a reviewed structure
        candidate, each as a typed assertion or directed relation with its cited evidence."""
        p, label, out = cset.payload, study.split(":", 1)[1], []

        def local_variable_id(opaque_id: str) -> str:
            component = (f"batch-{_live_namespace(cset)}/{_reference_component(opaque_id)}"
                         if cset.backend == "vllm" else opaque_id)
            return f"variable:{label}/{component}"

        vloc = {v["variable_id"]: local_variable_id(v["variable_id"]) for v in p.get("variables", [])}
        vars_by_id = {v["variable_id"]: v for v in p.get("variables", [])}
        for i, v in enumerate(p.get("variables", [])):
            ev = v.get("evidence_unit_ids", [])
            out.append(self._typed(cset, f"variable[{i}].acquisition", f"/variables/{i}", "variable", vloc[v["variable_id"]],
                                   "acquisition", {"kind": "enum", "lexical": v["acquisition"]}, ev, study, v["name"]))
            if "role_assignments" in v:
                if v["acquisition"] == "unresolved":
                    out[-1].body.update(state="not_found_in_reviewed_scope", value=None, reviewed_scope=ev or ["(none cited)"])
                assignments = v["role_assignments"]
                if not assignments:
                    draft = self._typed(cset, f"variable[{i}].role.unresolved", f"/variables/{i}/role_assignments", "variable",
                                        vloc[v["variable_id"]], "analysis_role", {"kind": "enum", "lexical": ""}, ev, study, v["name"])
                    draft.body.update(state="not_found_in_reviewed_scope", value=None, reviewed_scope=ev or ["(none cited)"])
                    out.append(draft)
                for j, assignment in enumerate(assignments):
                    role = assignment["role"]
                    if role not in CORE_ROLES:
                        raise ValueError("unrecognized required variable role")
                    analysis = assignment.get("analysis")
                    if analysis is not None and (not isinstance(analysis, str) or not analysis.strip()):
                        raise ValueError("analysis label must be nonblank or null")
                    # Preserve distinct analyses even when their labels have the same slug.
                    analysis_id = (f"analysis:{label}/role-" + hashlib.sha256(analysis.strip().encode("utf-8")).hexdigest()[:20]
                                   if analysis is not None else None)
                    claim = f"{v['name']}: {ROLE_LABELS[role]}" + (f"; analysis: {analysis.strip()}" if analysis else "")
                    draft = self._typed(cset, f"variable[{i}].role[{j}]", f"/variables/{i}/role_assignments/{j}", "variable",
                                        vloc[v["variable_id"]], "analysis_role", {"kind": "enum", "lexical": role},
                                        assignment.get("evidence_unit_ids", []), study, claim,
                                        extra_scope={"analysis_id": analysis_id})
                    if analysis_id is None:
                        draft.flags.append("scope_analysis_unresolved")
                    else:
                        draft.flags.append("scope_analysis_proposed_from_label")
                    out.append(draft)
            else:  # historical Q fixtures retain their original representation
                for role in v.get("analysis_roles", []):
                    out.append(self._typed(cset, f"variable[{i}].role.{role}", f"/variables/{i}", "variable", vloc[v["variable_id"]],
                                           "analysis_role", {"kind": "enum", "lexical": role}, ev, study, v["name"]))
            for parent in v.get("based_on_variable_ids", []):
                if parent in vloc:
                    out.append(self._relation(cset, f"variable[{i}].derived_from.{parent}", f"/variables/{i}", "derived_from",
                                              vloc[v["variable_id"]], vloc[parent], ev))
        for i, f in enumerate(p.get("factor_levels", [])):
            for j, lvl in enumerate(f.get("levels", [])):
                out.append(self._typed(cset, f"factor_level[{i}][{j}]", f"/factor_levels/{i}/levels/{j}", "study_design",
                                       vloc.get(f["variable_id"], local_variable_id(f["variable_id"])), "factor_level",
                                       {"kind": "enum", "lexical": lvl["label"], "level": lvl["level_id"]},
                                       lvl.get("evidence_unit_ids", f.get("evidence_unit_ids", [])), study, lvl["label"]))
        for i, c in enumerate(p.get("condition_cells", [])):
            out.append(self._typed(cset, f"condition_cell[{i}]", f"/condition_cells/{i}", "study_design",
                                   f"condition:{label}/{_slug(c['condition_id'])}", "condition_cell",
                                   {"kind": "paraphrase", "lexical": c.get("description", c["condition_id"])},
                                   c.get("evidence_unit_ids", []), study, c.get("description", "")))
        if p.get("allocation"):
            a = p["allocation"]
            out.append(self._typed(cset, "allocation", "/allocation", "study_design", f"design:{label}/allocation", "allocation",
                                   {"kind": "paraphrase", "lexical": a.get("mechanism", "")}, a.get("evidence_unit_ids", []), study,
                                   f"assignment unit {a.get('assignment_unit')}; {a.get('exposures_per_participant')} exposures per participant"))
        events = {e["event_id"]: f"event:{label}/{_slug(e['event_id'])}" for e in p.get("events", [])}
        for i, e in enumerate(p.get("events", [])):
            out.append(self._typed(cset, f"event[{i}]", f"/events/{i}", "procedure", events[e["event_id"]], "event",
                                   {"kind": "paraphrase", "lexical": f"{e.get('event_type')}: {e.get('description', '')}"},
                                   e.get("evidence_unit_ids", []), study, e.get("description", "")))
        for i, edge in enumerate(p.get("event_edges", [])):
            if edge["from_event_id"] in events and edge["to_event_id"] in events:
                out.append(self._relation(cset, f"event_edge[{i}]", f"/event_edges/{i}", "precedes", events[edge["from_event_id"]],
                                          events[edge["to_event_id"]], edge.get("evidence_unit_ids", [])))
        for i, m in enumerate(p.get("measures", [])):
            mloc = _slug(m["measure_id"])
            out.append(self._typed(cset, f"measure[{i}].response_scale", f"/measures/{i}", "measure", f"measure:{label}/{mloc}",
                                   "response_scale", {"kind": "paraphrase", "lexical": m.get("response_scale") or ""},
                                   m.get("evidence_unit_ids", []), study, m.get("scoring") or "",
                                   {"instrument_version_id": f"instrument:{label}/{mloc}"}))
            mname = vars_by_id.get(m.get("variable_id"), {}).get("name")
            out.append(self.measure_label(cset, f"measure[{i}].label", f"/measures/{i}", f"measure:{label}/{mloc}",
                                          f"instrument:{label}/{mloc}", mname, m.get("evidence_unit_ids", []), study))
        for d in out:
            if d.body.get("family") == "measure":
                d.flags.append("scope_instrument_version_study_local")
        return out

    # ------------------------------------------------------------------------------ candidate notes
    def candidate_notes(self, cset: CandidateSet, study: str) -> list[str]:
        """One advisory `validation-event` per `critical_findings`/`uncertainties` entry of a reviewed
        structure candidate (amends D-021): the note's own field and text verbatim, its JSON pointer into
        the pinned fixture, the fixture's path and SHA-256, and the study it concerns. Never an assertion
        or relation, never scored by a gate, never read by a tier -- a reviewer's own record, unchanged."""
        p = cset.payload
        who = actor("llm_fixture_replay", "mosaic.candidates.q_adapter")
        ids = []
        for i, f in enumerate(p.get("critical_findings", [])):
            ev = self.b.envelope("validation-event", "recorded", who)
            ev.update(event_kind="candidate_note", actor=who, subject_ids=[study], recorded_utc=utc_now(),
                      detail={"note_kind": "critical_finding", "field": f.get("field"), "text": f["problem"],
                              "source_pointer": f"/critical_findings/{i}", "source_file": cset.source_file,
                              "source_sha256": cset.source_sha256, "study_local_id": study})
            ids.append(self.b.add(ev))
        for i, text in enumerate(p.get("uncertainties", [])):
            ev = self.b.envelope("validation-event", "recorded", who)
            ev.update(event_kind="candidate_note", actor=who, subject_ids=[study], recorded_utc=utc_now(),
                      detail={"note_kind": "uncertainty", "text": text, "source_pointer": f"/uncertainties/{i}",
                              "source_file": cset.source_file, "source_sha256": cset.source_sha256, "study_local_id": study})
            ids.append(self.b.add(ev))
        return ids

    # ----------------------------------------------------------------------------------- entry
    def adapt(self, cset: CandidateSet) -> list[Draft]:
        p = cset.payload
        flags: list[str] = []
        study = self._study(p["study_id"], flags) if "study_id" in p else "study:paper"
        drafts: list[Draft] = []
        for i, comp in enumerate(p.get("verbatim_components", [])):
            drafts.append(self.verbatim(cset, i, comp, study))
        if p.get("verbatim_components"):
            subj_label = study.split(":", 1)[1]
            groups: dict[str, dict[str, Any]] = {}
            for i, comp in enumerate(p["verbatim_components"]):
                mslug, measure_label, _measure_flag = _component_measure(comp, i)
                g = groups.setdefault(mslug, {"label": measure_label, "unit_ids": []})
                uid = comp.get("unit_id")
                if uid and uid not in g["unit_ids"]:
                    g["unit_ids"].append(uid)
            for mslug, g in groups.items():
                drafts.append(self.measure_label(cset, f"measure_label[{mslug}]", "/verbatim_components",
                                                  f"measure:{subj_label}/{mslug}", f"instrument:{subj_label}/{mslug}",
                                                  g["label"], g["unit_ids"], study))
        for i, fact in enumerate(p.get("core_facts", [])):
            drafts.append(self.core_fact(cset, i, fact, study))
        for i, f in enumerate(p.get("findings", [])):
            drafts.extend(self.finding(cset, i, f, study))
        for i, v in enumerate(p.get("derived_variables", [])):
            drafts.append(self.variable(cset, i, v, study))
        for i, ev in enumerate(p.get("sequence", [])):
            drafts.append(self.sequence(cset, i, ev, study))
        for i, t in enumerate(p.get("source_tensions", [])):
            drafts.append(self.tension(cset, i, t, study))
        if any(k in p for k in ("variables", "factor_levels", "condition_cells", "events", "measures")):
            drafts.extend(self.structure(cset, study))
        if any(k in p for k in ("critical_findings", "uncertainties")):
            self.candidate_notes(cset, study)
        for i, edge in enumerate(study_edges(p)):
            if edge["mapped_relation"] is None:
                flags.append(f"unmapped_study_relation:{edge['relation']}")
                continue
            drafts.append(self.study_relationship(cset, i, edge))
        for d in drafts:
            d.flags = sorted(set(d.flags + flags))
            if cset.backend == "vllm":
                d.key = f"batch-{_live_namespace(cset)}:{d.key}"
        return drafts
