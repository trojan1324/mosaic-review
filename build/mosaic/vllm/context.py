"""Context packets, span/unit catalogs and response-recovery helpers for the vLLM backend
(MODEL_SELECTION_PLAN.md S0; FINAL v2 §6.2 context packets, §6.3 enum-constrained span catalogs).

Two enum-constrained fields, built from two different structures (design §2):
`unit_id` (whole legacy units, from `live_unit_catalog`) and `span_id` (fine-grained
catalog entries, from `span_catalog.build_catalog`, scoped to one batch's blocks).
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

from mosaic.gates import _inside
from mosaic.inventory import span_catalog
from mosaic.inventory.q_copier import UnitRef, legacy_index
from mosaic.variable_roles import CORE_ROLES
from mosaic.core_contract import NUMBER, fact_schema

CHARS_PER_TOKEN_FALLBACK = 3.5
FINE_SPAN_OPTION_LIMIT = 128


class FineSpanOptionsRefused(ValueError):
    """An opt-in batch would expose too many new source options."""

    def __init__(self, batch_id: str, count: int):
        super().__init__(f"fine_span_options_refused:{batch_id}:{count}>{FINE_SPAN_OPTION_LIMIT}")
        self.batch_id = batch_id
        self.count = count


def _fine_span_ranges(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Conservative source-only fragments inside one admitted range."""
    cuts = [start]
    pairs = {"(": ")", "[": "]", "{": "}"}
    closing = set(pairs.values())
    stack: list[str] = []
    for at in range(start, end):
        char = text[at]
        if char in pairs:
            stack.append(pairs[char])
        elif char in closing:
            if not stack or char != stack.pop():
                return []
        elif not stack and char in ",;:":
            cuts.extend((at, at + 1))
    if stack:
        return []
    cuts.append(end)
    ranges: set[tuple[int, int]] = set()
    for left, right in zip(cuts, cuts[1:]):
        left, right = span_catalog._trim(text, left, right)
        if left >= right or text[left:right] in ",;:":
            continue
        piece = text[left:right]
        if piece[0] in "([{" and piece[-1:] in ")]}":
            continue  # parenthetical counts are context, not a standalone option
        ranges.add((left, right))  # retain a legitimate conjunction label whole
        conjunctions = []
        for match in re.finditer(r"\b(?:and|or)\b", piece, re.I):
            depth = 0
            for char in piece[:match.start()]:
                if char in pairs:
                    depth += 1
                elif char in closing:
                    depth -= 1
            if depth == 0:
                conjunctions.append((left + match.start(), left + match.end()))
        cursor = left
        for a, z in conjunctions:
            part = span_catalog._trim(text, cursor, a)
            if part[0] < part[1]:
                ranges.add(part)
            cursor = z
        part = span_catalog._trim(text, cursor, right)
        if part[0] < part[1]:
            ranges.add(part)
    return sorted((left, right) for left, right in ranges
                  if 3 <= right - left <= 160 and any(char.isalpha() for char in text[left:right]))


def _add_fine_span_options(catalog, region, blocks) -> set[str]:
    """Add review-required exact ranges and return only newly added IDs."""
    seen = {entry["span_id"] for entry in catalog["entries"]}
    new: set[str] = set()
    text, tr_id = region["text"], catalog["text_region_id"]
    for block in blocks:
        for left, right in _fine_span_ranges(text, block["start"], block["end"]):
            entry = span_catalog._entry(text, tr_id, "unclassified", left, right, block["block_index"])
            if entry["span_id"] not in seen:
                catalog["entries"].append(entry)
                seen.add(entry["span_id"])
                new.add(entry["span_id"])
    return new


def _check_fine_option_limit(batch_id: str, fine_ids: set[str], entries: list[dict[str, Any]],
                             block_indices: set[int] | list[int]) -> None:
    count = sum(entry["span_id"] in fine_ids and entry.get("block_index") in block_indices for entry in entries)
    if count > FINE_SPAN_OPTION_LIMIT:
        raise FineSpanOptionsRefused(batch_id, count)


def _add_sentence_options(catalog, region, blocks) -> set[str]:
    """Offer exact, review-required sentence ranges only inside admitted units.

    This is a model-input projection: capture and the historical inventory
    catalog remain unchanged. Boundaries are conservative conveniences, never
    evidence that a sentence is a fact or that its study attribution is right.
    """
    seen = {e["span_id"] for e in catalog["entries"]}
    added: set[str] = set()
    text, tr_id = region["text"], catalog["text_region_id"]
    abbreviation = re.compile(r"(?:\b(?:al|e\.g|i\.e|vs|Fig|Eq|Dr|Mr|Ms|Prof|No)|\b[A-Z])\.$", re.I)
    for block in blocks:
        start, end = block["start"], block["end"]
        unit = text[start:end]
        boundaries = [0]
        for match in re.finditer(r"[.!?][\"”]?\s+(?=[\"“]?[A-Z])", unit):
            before = unit[:match.start() + 1]
            if (abbreviation.search(before) or before.count("(") != before.count(")")
                    or before.count("[") != before.count("]")):
                continue
            boundaries.append(match.end())
        boundaries.append(len(unit))
        for a, b in zip(boundaries, boundaries[1:]):
            left, right = span_catalog._trim(text, start + a, start + b)
            if right - left < 12 or not any(c.isalpha() for c in text[left:right]):
                continue
            entry = span_catalog._entry(text, tr_id, "unclassified", left, right, block["block_index"])
            if entry["span_id"] not in seen:
                catalog["entries"].append(entry)
                seen.add(entry["span_id"])
                added.add(entry["span_id"])
    return added


def live_unit_catalog(objects: dict[str, dict[str, Any]], *, index=None) -> dict[str, dict[str, Any]]:
    """q_units-shaped {unit_id: {"kind": "text", "quote": <unit text>, "page": <1-based | None>}},
    built purely from legacy_index(objects). Drop-in replacement for the `q_units` dict literal
    Q's fixture file supplies, used only on the 'vllm' branch of stage_extract."""
    index = legacy_index(objects) if index is None else index
    out: dict[str, dict[str, Any]] = {}
    for uid, ref in index.items():
        text = objects[ref.text_region_id]["text"][ref.start:ref.end]
        out[uid] = {"kind": "text", "quote": text, "page": (ref.physical_index + 1) if ref.physical_index is not None else None}
    return out


def block_units(region: dict[str, Any]) -> dict[int, str]:
    """block_index -> legacy_unit_id for every block in one text-region that has one."""
    return {b["block_index"]: b["legacy_unit_id"] for b in region["blocks"] if b.get("legacy_unit_id")}


def study_unit_ids(objects: dict[str, dict[str, Any]], index: dict[str, UnitRef], study: dict[str, Any], *,
                   include_shared: bool = True) -> list[str]:
    """Unit ids whose UnitRef span lies inside study['boundary']['regions'], plus, when
    include_shared, ids inside study:paper's boundary. Sorted, deduplicated."""
    regions = list(study.get("boundary", {}).get("regions", []))
    if include_shared:
        paper = next((o for o in objects.values() if o.get("type") == "study" and o.get("local_id") == "study:paper"), None)
        if paper and paper is not study:
            regions += paper.get("boundary", {}).get("regions", [])
    out = set()
    for uid, ref in index.items():
        span = {"text_artifact_id": ref.text_region_id, "start_codepoint": ref.start, "end_codepoint": ref.end}
        if _inside([span], regions):
            out.add(uid)
    return sorted(out)


@dataclass(frozen=True)
class UnitBatch:
    batch_id: str
    study_local_id: str
    text_region_id: str
    block_indices: frozenset[int]
    catalog: dict[str, Any]  # span_catalog.build_catalog() output, unrestricted; scoping happens by
                             # filtering entries against block_indices at enum-build time
    unit_refs: dict[str, UnitRef] | None = None
    source_policy_sha256: str | None = None
    cell_field: str | None = None
    body_fragment: bool = False
    statement_span_ids: frozenset[str] = frozenset()


def batch_block_units(region, batch):
    if batch.cell_field or batch.body_fragment:
        return {0: next(iter(batch.unit_refs))}
    return {bi: uid for bi, uid in block_units(region).items()
            if batch.unit_refs is None or uid in batch.unit_refs}


def _estimate_tokens(text: str, tokenize: Callable[[str], int] | None) -> float:
    if tokenize is not None:
        try:
            return float(tokenize(text))
        except Exception:
            pass
    return len(text) / CHARS_PER_TOKEN_FALLBACK


def _study_blocks(objects: dict[str, dict[str, Any]], study: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """text_region_id -> blocks of that region that lie inside the study's own boundary (not shared)."""
    by_region: dict[str, list[dict[str, Any]]] = {}
    for r in study.get("boundary", {}).get("regions", []):
        tr_id = r["text_artifact_id"]
        region = objects.get(tr_id)
        if region is None or region.get("type") != "text-region":
            continue
        for block in region["blocks"]:
            if r["start_codepoint"] <= block["start"] and block["end"] <= r["end_codepoint"]:
                by_region.setdefault(tr_id, []).append(block)
    for blocks in by_region.values():
        blocks.sort(key=lambda b: b["block_index"])
    return by_region


def build_unit_batches(builder, study: dict[str, Any], *, max_input_tokens: float,
                       tokenize: Callable[[str], int] | None = None, source_policy=None,
                       sentence_options: bool = False, fine_span_options: bool = False) -> list[UnitBatch]:
    """Greedily groups a study's catalogued blocks into batches under max_input_tokens
    (measured by `tokenize`, falling back to a char/3.5 estimate), splitting only at block
    boundaries. Never crosses a study boundary (one batch, one study_local_id); a batch also
    never crosses a text-region boundary."""
    study_local = study["local_id"]
    batches: list[UnitBatch] = []
    index, digest = None, None
    if source_policy is not None:
        from mosaic.vllm.source_policy import policy_index, policy_sha256
        index = policy_index(builder.objects, source_policy)
        digest = policy_sha256(source_policy)
    for tr_id, blocks in sorted(_study_blocks(builder.objects, study).items()):
        region = builder.objects[tr_id]
        if region["status"] != "captured" or region["derivation"]["kind"] not in span_catalog.NATIVE_KINDS:
            continue  # only native text can be catalogued
        catalog = span_catalog.build_catalog(tr_id, region)
        if index is not None:
            blocks = [b for b in blocks if b.get("legacy_unit_id") in index]
            allowed = {b["block_index"] for b in blocks}
            catalog["entries"] = [e for e in catalog["entries"] if e["block_index"] in allowed]
        statement_ids = {e["span_id"] for e in catalog["entries"] if e["kind"] == "unclassified"}
        if sentence_options:
            statement_ids |= _add_sentence_options(catalog, region, blocks)
        fine_ids = _add_fine_span_options(catalog, region, blocks) if fine_span_options else set()
        refs = None if index is None else {b["legacy_unit_id"]: index[b["legacy_unit_id"]] for b in blocks}
        current: list[int] = []
        current_tokens = 0.0
        n = 0
        for block in blocks:
            btext = region["text"][block["start"]:block["end"]]
            btokens = _estimate_tokens(btext, tokenize)
            if current and current_tokens + btokens > max_input_tokens:
                n += 1
                batch_id = f"{study_local}#{tr_id.rsplit('-', 1)[1][:8]}#{n}"
                _check_fine_option_limit(batch_id, fine_ids, catalog["entries"], current)
                batches.append(UnitBatch(batch_id, study_local, tr_id, frozenset(current), catalog, refs, digest,
                                         statement_span_ids=frozenset(statement_ids)))
                current, current_tokens = [], 0.0
            current.append(block["block_index"])
            current_tokens += btokens
        if current:
            n += 1
            batch_id = f"{study_local}#{tr_id.rsplit('-', 1)[1][:8]}#{n}"
            _check_fine_option_limit(batch_id, fine_ids, catalog["entries"], current)
            batches.append(UnitBatch(batch_id, study_local, tr_id, frozenset(current), catalog, refs, digest,
                                     statement_span_ids=frozenset(statement_ids)))
    # An exact cell is its own request lane. No loading/stub/header/body block
    # is exposed as a side effect of offering its source offsets.
    for cell in source_policy["cells"] if source_policy is not None else []:
        span = {"text_artifact_id": cell["text_region_id"], "start_codepoint": cell["start"], "end_codepoint": cell["end"]}
        if not _inside([span], study.get("boundary", {}).get("regions", [])):
            continue
        tr_id = cell["text_region_id"]
        region = builder.objects[tr_id]
        catalog = span_catalog.build_catalog(tr_id, region)
        catalog["entries"] = [span_catalog._entry(region["text"], tr_id, "table_cell", cell["start"], cell["end"], 0)]
        uid = cell["unit_id"]
        batches.append(UnitBatch(f"{study_local}#cell#{cell['witness_id'].rsplit('-', 1)[1][:16]}", study_local,
                                 tr_id, frozenset({0}), catalog, {uid: index[uid]}, digest, cell["field"]))
    # A recovered range never borrows its withheld parent's ID, catalog or
    # neighbors. Keep original offsets for copying and PDF provenance.
    for row in source_policy.get("fragments", []) if source_policy is not None else []:
        span = {"text_artifact_id": row["text_region_id"], "start_codepoint": row["start"], "end_codepoint": row["end"]}
        if not _inside([span], study.get("boundary", {}).get("regions", [])):
            continue
        tr_id, uid = row["text_region_id"], row["unit_id"]
        region = builder.objects[tr_id]
        catalog = span_catalog.build_catalog(tr_id, region)
        catalog["entries"] = span_catalog.unit_entries(region["text"], tr_id,
                                                       {"start": row["start"], "end": row["end"], "block_index": 0})
        statement_ids = {e["span_id"] for e in catalog["entries"] if e["kind"] == "unclassified"}
        if sentence_options:
            statement_ids |= _add_sentence_options(catalog, region, [{"start": row["start"], "end": row["end"], "block_index": 0}])
        fine_ids = (_add_fine_span_options(catalog, region, [{"start": row["start"], "end": row["end"], "block_index": 0}])
                    if fine_span_options else set())
        _check_fine_option_limit(f"{study_local}#{uid}", fine_ids, catalog["entries"], {0})
        batches.append(UnitBatch(f"{study_local}#{uid}", study_local, tr_id, frozenset({0}),
                                 catalog, {uid: index[uid]}, digest, body_fragment=True,
                                 statement_span_ids=frozenset(statement_ids)))
    return batches


def context_packet(builder, batch: UnitBatch, study: dict[str, Any], *, already_proposed: set[str],
                   max_neighbor_chars: int = 2000,
                   interval_lexicals: list[str] | None = None) -> dict[str, Any]:
    """FINAL v2 §6.2 fields: work/source ids, study_id/kind, unit_ids (scoped to batch.block_indices
    via block_units), unit_pages, neighboring text (bounded, omitted_chars when cut), shared_unit_ids
    (study_unit_ids with include_shared), already_proposed candidate ids, reviewed_scope (every
    unit/span id this packet offers, echoed back for the coverage record)."""
    region = builder.objects[batch.text_region_id]
    index = legacy_index(builder.objects) if batch.unit_refs is None else batch.unit_refs
    bu = batch_block_units(region, batch)
    unit_ids = sorted({bu[bi] for bi in batch.block_indices if bi in bu})
    unit_pages: dict[str, int | None] = {}
    for uid in unit_ids:
        ref = index.get(uid)
        unit_pages[uid] = (ref.physical_index + 1) if ref and ref.physical_index is not None else None
    blocks_by_index = {b["block_index"]: b for b in region["blocks"]}
    if batch.cell_field or batch.body_fragment:
        ref = index[unit_ids[0]]
        blocks_by_index = {0: {"start": ref.start, "end": ref.end}}
    sel = sorted(batch.block_indices)
    first_block, last_block = blocks_by_index[sel[0]], blocks_by_index[sel[-1]]
    text = region["text"]
    before_start = max(0, first_block["start"] - max_neighbor_chars)
    after_end = min(len(text), last_block["end"] + max_neighbor_chars)
    neighbor = {"before": text[before_start:first_block["start"]], "after": text[last_block["end"]:after_end]}
    if before_start > 0:
        neighbor["omitted_chars_before"] = before_start
    if after_end < len(text):
        neighbor["omitted_chars_after"] = len(text) - after_end
    if batch.unit_refs is not None:
        before = [text[r.start:r.end] for r in sorted(index.values(), key=lambda r: r.start)
                  if r.end <= first_block["start"]]
        after = [text[r.start:r.end] for r in sorted(index.values(), key=lambda r: r.start)
                 if r.start >= last_block["end"]]
        # Join only eligible units in this study and region. Raw contiguous
        # region slices would reintroduce withheld tables and other studies.
        before, after = "\n[separate source unit]\n".join(before), "\n[separate source unit]\n".join(after)
        neighbor = {"before": before[-max_neighbor_chars:] if max_neighbor_chars else "",
                    "after": after[:max_neighbor_chars],
                    "omitted_chars_before": max(0, len(before) - max_neighbor_chars),
                    "omitted_chars_after": max(0, len(after) - max_neighbor_chars)}
    span_ids = sorted(e["span_id"] for e in batch.catalog["entries"] if e.get("block_index") in batch.block_indices)
    # IDs alone do not give the model evidence to read. Send the exact offered
    # text and the selectable span-to-text mapping, never the whole region.
    unit_catalog = [{"unit_id": uid, "text": text[index[uid].start:index[uid].end],
                     "start": index[uid].start, "end": index[uid].end,
                     "page": unit_pages[uid]} for uid in unit_ids]
    span_catalog_rows = []
    for entry in batch.catalog["entries"]:
        if entry["span_id"] not in span_ids:
            continue
        selected = span_catalog.select(batch.catalog, entry["span_id"], batch.text_region_id, region)
        span_catalog_rows.append({"span_id": selected.span_id, "unit_id": bu.get(entry["block_index"]),
                                  "text": selected.text, "kind": selected.kind,
                                  "start": selected.start, "end": selected.end,
                                  "text_sha256": selected.text_sha256})
    packet = {
        "work_id": builder.work_id, "source_package_id": builder.package_id,
        "study_id": study["local_id"], "study_kind": study.get("kind"),
        "text_region_id": batch.text_region_id, "batch_id": batch.batch_id,
        "unit_ids": unit_ids, "unit_pages": unit_pages, "span_ids": span_ids,
        "unit_catalog": unit_catalog, "span_catalog": span_catalog_rows,
        "neighboring_text": neighbor,
        "shared_unit_ids": study_unit_ids(builder.objects, index, study, include_shared=True),
        "already_proposed_candidate_ids": sorted(already_proposed),
        "reviewed_scope": sorted(set(unit_ids) | set(span_ids)),
    }
    if batch.source_policy_sha256:
        packet.update(source_policy_sha256=batch.source_policy_sha256,
                      source_lane="measure_table_wording" if batch.cell_field else "body_candidate")
    if interval_lexicals is not None:
        # This records precisely the source-only enum used by the response
        # schema, including line breaks and Unicode signs, in the packet hash.
        packet["interval_lexicals"] = list(interval_lexicals)
    return packet


def request_schema(batch: UnitBatch, unit_ids: list[str], *, max_items: int = 32,
                   core_only: bool = False,
                   interval_lexicals: list[str] | None = None) -> dict[str, Any]:
    """The QCandidateAdapter-payload-shaped response schema (design §5). span_id enum built from
    span_catalog.selection_schema, scoped to this batch's blocks; unit_id enum is `unit_ids`.
    Every array carries maxItems: max_items (re-checked in code, decoder-ignored per
    decoder_subset.py) and a model-supplied candidate_id string."""
    span_enum = span_catalog.selection_schema([batch.catalog], block_indices=set(batch.block_indices))
    statement_ids = [entry["span_id"] for entry in batch.catalog["entries"]
                     if entry["span_id"] in batch.statement_span_ids
                     and entry.get("block_index") in batch.block_indices]
    statement_span_enums = ({field: {"type": "string", "enum": statement_ids} for field in
                             ("manipulation", "qualitative_result", "hypothesis")}
                            if statement_ids and not batch.cell_field else None)
    unit_enum = {"type": "string", "enum": list(unit_ids)}
    with_cid: Callable[[dict], dict] = lambda props: {"type": "object", "properties": {**props, "candidate_id": {"type": "string"}}}
    evidence = {"evidence_unit_ids": {"type": "array", "items": unit_enum}}
    array = lambda item: {"type": "array", "maxItems": max_items, "items": item}
    role_assignment = {
        "type": "object", "required": ["role", "analysis", "evidence_unit_ids"],
        "properties": {"role": {"type": "string", "enum": list(CORE_ROLES)},
                       "analysis": {"type": ["string", "null"], "minLength": 1, "pattern": "\\S"},
                       "evidence_unit_ids": {"type": "array", "minItems": 1, "items": unit_enum}},
        "additionalProperties": False,
    }
    variable = with_cid({"variable_id": {"type": "string", "minLength": 1}, "name": {"type": "string", "minLength": 1},
                         "acquisition": {"type": "string", "enum": ["manipulated", "measured", "derived", "unresolved"]},
                         "role_assignments": array(role_assignment),
                         "based_on_variable_ids": {"type": "array", "items": {"type": "string"}},
                         "evidence_unit_ids": {"type": "array", "minItems": 1, "items": unit_enum}})
    variable.update(required=["candidate_id", "variable_id", "name", "acquisition", "role_assignments", "evidence_unit_ids"],
                    additionalProperties=False)
    schema = {
        "type": "object",
        "$defs": {"core_unit_id": unit_enum, "core_span_id": span_enum},
        "required": ["study_id"],
        "properties": {
            "study_id": {"type": "string", "enum": [batch.study_local_id, batch.study_local_id.removeprefix("study:")]},
            "core_facts": array(fact_schema({"$ref": "#/$defs/core_unit_id"}, {"$ref": "#/$defs/core_span_id"},
                                              interval_lexicals=interval_lexicals, span_enums=statement_span_enums)),
            "verbatim_components": array(with_cid({"span_id": span_enum, "measure": {"type": "string"}, "kind": {"type": "string"}})),
            "findings": array(with_cid({"topic": {"type": "string"}, "claim": {"type": "string"}, "status": {"type": "string"}, **evidence})),
            # New model output uses one scoped representation, including derived variables.
            # Historical Q replay still supports the old free-text branch in its adapter.
            "derived_variables": {"type": "array", "maxItems": 0},
            "sequence": array(with_cid({"event": {"type": "string"}, "position": {"type": "string"},
                                        "order_status": {"type": "string"}, **evidence})),
            "source_tensions": array(with_cid({"description": {"type": "string"}, "resolution_status": {"type": "string"}, **evidence})),
            "variables": array(variable),
            "factor_levels": array(with_cid({"variable_id": {"type": "string"}, "levels": {"type": "array"}, **evidence})),
            "condition_cells": array(with_cid({"condition_id": {"type": "string"}, "description": {"type": "string"}, **evidence})),
            "allocation": {"type": ["object", "null"]},
            "events": array(with_cid({"event_id": {"type": "string"}, "event_type": {"type": "string"},
                                      "description": {"type": "string"}, **evidence})),
            "event_edges": array(with_cid({"from_event_id": {"type": "string"}, "to_event_id": {"type": "string"}, **evidence})),
            "measures": array(with_cid({"measure_id": {"type": "string"}, "response_scale": {"type": "string"},
                                        "scoring": {"type": "string"}, **evidence})),
            "study_relationships": array(with_cid({"from_study_id": {"type": "string"}, "to_study_id": {"type": "string"},
                                                   "relation": {"type": "string"}, **evidence})),
            "continuation": {"type": "object", "properties": {"more": {"type": "boolean"}, "cursor": {"type": ["string", "null"]}}},
        },
        "additionalProperties": False,
    }
    if core_only:
        schema["properties"] = {k: schema["properties"][k] for k in ("study_id", "core_facts", "variables")}
        schema["required"] = ["study_id", "core_facts", "variables"]
    if batch.cell_field:
        facts = fact_schema({"$ref": "#/$defs/core_unit_id"}, {"$ref": "#/$defs/core_span_id"},
                            interval_lexicals=interval_lexicals, span_enums=statement_span_enums)
        facts["oneOf"] = [b for b in facts["oneOf"] if b["properties"]["field"]["const"] == batch.cell_field]
        schema["properties"] = {"study_id": schema["properties"]["study_id"], "core_facts": array(facts)}
        schema["required"] = ["study_id", "core_facts"]
    return schema


def coverage_record(offered_unit_ids: list[str], offered_span_ids: list[str], used_unit_ids: list[str],
                    used_span_ids: list[str]) -> dict[str, Any]:
    ou, os_ = set(offered_unit_ids), set(offered_span_ids)
    uu, us = set(used_unit_ids), set(used_span_ids)
    return {"offered": {"unit_ids": sorted(ou), "span_ids": sorted(os_)},
            "used": {"unit_ids": sorted(uu & ou), "span_ids": sorted(us & os_)},
            "unused": {"unit_ids": sorted(ou - uu), "span_ids": sorted(os_ - us)}}


def generation_schema(schema: dict) -> dict:
    """Preserve semantic validation while avoiding search-pattern decoder traps.

    JSON Schema's bare nonblank pattern is a search, whereas XGrammar places
    its regex between the string's quotes. That forces exactly one character.
    Generate ordinary nonempty strings; the original local schema still rejects
    blank labels. Older XGrammar also inserts a numeric top-level alternation
    without grouping, detaching the JSON quotes. Use an equivalent, grouped
    numeric expression on the wire. Local patterns, enums and boundaries stay
    intact; no numeric constraint is removed.
    """
    out = deepcopy(schema)

    def visit(node):
        if isinstance(node, dict):
            if node.get("pattern") == r"\S":
                del node["pattern"]
            elif node.get("pattern") == NUMBER:
                node["pattern"] = r"^[+−–-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?$"
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(out)
    return out


_LOOP = None


def looping_detected(raw_text: str, *, min_run: int = 40, min_repeats: int = 6) -> bool:
    """A run of at least `min_run` characters immediately repeated at least `min_repeats` times."""
    global _LOOP
    if _LOOP is None or _LOOP[0] != (min_run, min_repeats):
        _LOOP = ((min_run, min_repeats), re.compile(rf"(.{{{min_run},}}?)\1{{{min_repeats - 1},}}", re.DOTALL))
    return _LOOP[1].search(raw_text) is not None


def extract_json_object(raw_text: str) -> dict[str, Any] | None:
    """First-{-to-matching-last-} slice, respecting string literals; else None. Code-only
    leniency: relocates the object inside surrounding prose, never invents content."""
    start = raw_text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(raw_text)):
            ch = raw_text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = raw_text[start:i + 1]
                    try:
                        obj = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict):
                        return obj
                    break
        start = raw_text.find("{", start + 1)
    return None
