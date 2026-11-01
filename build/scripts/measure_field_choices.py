"""Individual, source-bound measure fields for a matched extraction experiment.

Candidate syntax supplies possible boundaries, never an assertion that a span
is an item or that an instrument is complete. The model selects roles; the
catalog still verifies every character against the original cited evidence.
The previous trial is unchanged for replay and comparison.
"""
from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Any, Mapping

from scripts.measure_field_trial import _candidate, _digest, _require, _sources
from scripts.measure_field_trial import prepare_task as baseline_task
from scripts.measure_catalog import attach_exact_measure_fields
from scripts.source_decoding import decoding_status


VERSION = "mosaic-measure-field-trial/v2"
_NUMBER = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)"
_COUNT = re.compile(rf"\b{_NUMBER}(?:\s+{_NUMBER}[- ]point(?:\s+Likert)?)?[-\s]+(?:items?|questions?|statements?|adjectives?)\b", re.I)
_QUOTES = re.compile(r'“([^”]{3,2400})”|"([^"\n]{3,2400})"|‘((?:[^’]|’(?=\w)){3,2400})’')
_MARKERS = re.compile(r"(?:^|\s)\((\d{1,2})\)\s*")
_DIRECT_QUESTION = re.compile(r"\b(?:How|What|When|Where|Why|Which|Who|Do|Does|Did|Are|Is|Have|Has|Would|Could|Can)\b[^?]{3,600}\?")
_SCALE = re.compile(rf"\b{_NUMBER}[- ]point(?:\s+Likert)?\s+(?:scale|questions?)\b|\bscale\s+ranging\s+from\b|\b[0-9]{{1,2}}\s*\([A-Za-z][^()\n]{{0,60}}\)", re.I)
_ENCODING = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ufffd]|[“\"]\d+[”\"]\s+5\s+[A-Za-z]")


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace(): start += 1
    while end > start and text[end - 1].isspace(): end -= 1
    return start, end


def choices(version: Mapping[str, Any]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    seen = set()
    for row in _sources(version):
        text = row["segment"]["text"]

        def add(start: int, end: int, rule: str, roles: list[str]) -> None:
            start, end = _trim(text, start, end)
            if end <= start: return
            segment = row["segment"]
            key = (segment["unit_id"], segment["source_locator"], segment["page"],
                   segment["start"] + start, segment["start"] + end, tuple(roles))
            if key in seen: return
            seen.add(key)
            choice = _candidate(row, start, end, rule)
            choice.update(boundary_rule=rule, role_allowlist=roles,
                          source_text_status=("visual_check_required" if _ENCODING.search(text[start:end]) else
                            decoding_status([segment])))
            found.append(choice)

        for match in _QUOTES.finditer(text):
            group = next(i for i in (1, 2, 3) if match.group(i) is not None)
            start, end = match.span(group)
            add(start, end, "quoted_span", ["items", "question_stems"])
            markers = list(_MARKERS.finditer(text[start:end]))
            numbered = (len(markers) >= 2 and markers[0].start() == 0
                        and [int(m.group(1)) for m in markers] == list(range(1, len(markers) + 1)))
            if numbered:
                for i, marker in enumerate(markers):
                    stop = start + markers[i + 1].start() if i + 1 < len(markers) else end
                    add(start + marker.end(), stop, "numbered_quote_member", ["items"])
            elif ";" in text[start:end]:
                member_start = start
                for separator in re.finditer(";", text[start:end]):
                    stop = start + separator.start()
                    add(member_start, stop, "semicolon_quote_member", ["items"])
                    member_start = stop + 1
                add(member_start, end, "semicolon_quote_member", ["items"])
            # A quoted lead-in is only a candidate. Do not synthesize a shared
            # stem by joining the author's preamble to respondent wording.
            colon = text.find(":", start, end)
            if colon >= 0:
                add(start, colon + 1, "quoted_lead_in", ["question_stems"])
        for match in _DIRECT_QUESTION.finditer(text):
            add(*match.span(), "direct_question", ["items"])
        if row['field'] == 'field_context_evidence':
            first_question = _DIRECT_QUESTION.search(text)
            if first_question and 0 < first_question.start() <= 1800:
                # A possible participant preamble, never automatically an
                # instruction: the model must reject author narration.
                add(0, first_question.start(), 'preceding_question_context', ['question_stems'])
        # Supply each complete scale-bearing sentence as one atomic choice.
        # This keeps endpoints together and avoids arbitrarily cutting an
        # anchor at a comma or returning just one endpoint of a scale.
        cuts = [0] + [m.end() for m in re.finditer(r'[.!?][”"]?\s+(?=[A-Z])', text)] + [len(text)]
        for start, end in zip(cuts, cuts[1:]):
            if _SCALE.search(text[start:end]):
                add(start, end, "scale_statement", ["response_scale"])
        for match in _COUNT.finditer(text):
            add(*match.span(), "literal_count_phrase", ["reported_item_count"])
    for number, choice in enumerate(found, 1):
        choice["choice_id"] = f"measure-choice-{number:03d}"
    return found


def prepare_task(version: Mapping[str, Any]) -> dict[str, Any]:
    task = baseline_task(version)
    offered = choices(version)
    task["version"] = VERSION
    task["payload"]["choices"] = offered
    for field in ("items", "question_stems", "response_scale"):
        ids = [c["choice_id"] for c in offered if field in c["role_allowlist"]]
        task["schema"]["properties"][field] = {"type": "array", "uniqueItems": True,
                                                "items": {"enum": ids} if ids else False}
        if not ids: task["schema"]["properties"][field]["maxItems"] = 0
    task["schema"]["properties"]["reported_item_count"]["enum"] = ["not_reported"] + [
        c["choice_id"] for c in offered if "reported_item_count" in c["role_allowlist"]]
    task["request"]["response_schema"] = task["schema"]
    task["request"]["messages"][0]["content"] += (
        " Boundary rules offer possibilities, not item labels. Prefer each complete individual member over a whole list; "
        "a semicolon may occur within one item, so split only when source context supports separate items. "
        "Author descriptions in third person and construct labels are not respondent wording. "
        "Select a scale_statement as a whole so its endpoints stay together. "
        "Do not select overlapping choices or both a whole list and its members. "
        "Keep original order. Encoding cautions do not authorize repairing or inventing text.")
    task["request"]["messages"][1]["content"] = json.dumps(task["payload"], ensure_ascii=False, sort_keys=True)
    return task


def materialize(task: Mapping[str, Any], response: Mapping[str, Any], version: Mapping[str, Any]) -> dict[str, Any]:
    from jsonschema import Draft202012Validator

    _require(task.get("version_sha256") == _digest(version), "measure_field_version_changed")
    _require(task == prepare_task(version), "measure_field_task_changed")
    Draft202012Validator(task["schema"]).validate(response)
    offered = {c["choice_id"]: c for c in task["payload"]["choices"]}
    occupied = []
    selected = {}
    for field in ("items", "question_stems", "response_scale"):
        rows = [offered[value] for value in response[field]]
        # Stable source order uses the original page/unit sequence represented
        # by the cited contexts, not lexical ordering of variable-width IDs.
        order = {}
        for src in _sources(version):
            s = src["segment"]
            order.setdefault((s["source_locator"], s["page"], s["unit_id"]), len(order))
        def position(c):
            s = c["segments"][0]
            return (s["page"], order[(s["source_locator"], s["page"], s["unit_id"])], s["start"])
        rows.sort(key=position)
        for choice in rows:
            s = choice["segments"][0]
            for old in occupied:
                same = all(s[k] == old[k] for k in ("unit_id", "page", "source_locator", "unit_text_sha256"))
                _require(not (same and s["start"] < old["end"] and old["start"] < s["end"]), "measure_field_overlap")
            occupied.append(s)
        selected[field] = [{k: c[k] for k in ("quote", "segments", "pages")} for c in rows]
    # Literal count evidence may share a phrase with a scale description. This
    # does not duplicate respondent items or establish a completeness claim.
    count_id = response["reported_item_count"]
    count = None if count_id == "not_reported" else {
        "value": offered[count_id]["quote"],
        "evidence": {k: offered[count_id][k] for k in ("quote", "segments", "pages")}}
    enriched = attach_exact_measure_fields(version, **selected, reported_item_count=count)
    enriched["item_disclosure"] = response["item_disclosure"]
    enriched["source_text_status"] = "visual_check_required" if any(
        _ENCODING.search(row["segment"]["text"]) for row in _sources(version)) else "indexed_text_only"
    return {"version": VERSION, "version_id": version["id"], "proposal": deepcopy(response), "measure": enriched,
            "scientifically_accepted": False, "coverage_verified": False,
            "interpretation": "Source-bound proposed fields. Item examples, count equality and syntax never establish a complete scale."}
