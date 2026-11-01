"""Read-only literal-support audit for model screening (D-215).

This is not a semantic scorer or an acceptance gate. A supported number can
still concern the wrong study, sample or hypothesis. Paraphrases, categories
and attribution always need adjudication. Reports contain identifiers and
reason codes, never source text. No file, provider or reference access occurs.
"""

from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from typing import Any

from mosaic.canonical import sha256_hex
from mosaic.core_contract import cardinal_integer_at, cardinal_integer_value
from mosaic.inventory.witnesses import NATIVE, OMISSION, WitnessError, _check_join

OUTCOMES = ("literal_supported", "unsupported", "needs_review", "not_asserted")
NUMBER = r"[−–+\-]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)(?:[eE][+\-]?\d+)?"
TOKEN = re.compile(rf"(?<![\w.−–+\-]){NUMBER}(?!\w|\.\d|,\d{{3}}(?!\d))")
COMPARATORS = {"=": "eq", "<": "lt", "≤": "le", "<=": "le", ">": "gt", "≥": "ge", ">=": "ge", "≈": "approximate", "~": "approximate"}
OPERATOR = r"(?:<=|>=|[=<>≤≥≈~])"
STATISTIC = r"(?:R²|R2|χ²|χ2|η²|η2|OR|SD|SE|B|b|β|r|d|g|F|t|z|p|N|n|M|α|ω)"
ANCHOR = re.compile(rf"(?<![\w])(?P<stat>{STATISTIC})(?:\s*\([^()\n]*\))?\s*(?P<op>{OPERATOR})\s*$")
OP_AT_END = re.compile(rf"(?P<op>{OPERATOR})\s*$")


def _decimal(value: Any) -> Decimal:
    if not isinstance(value, str) or not re.fullmatch(NUMBER, value):
        raise ValueError("invalid_decimal")
    try:
        result = Decimal(value.replace("−", "-").replace("–", "-").replace(",", ""))
    except InvalidOperation as exc:
        raise ValueError("invalid_decimal") from exc
    if not result.is_finite():
        raise ValueError("invalid_decimal")
    return result


def _result(errors=(), review=()) -> dict:
    return {"outcome": "unsupported" if errors else "needs_review" if review else "literal_supported",
            "reasons": sorted(set(errors) | set(review))}


def _witness(objects: dict, wid: str) -> tuple[str, list[tuple[str, int, int]], list[str]]:
    """Re-slice and check every span, including source binding and declared joins."""
    w = objects[wid]
    if w.get("type") != "witness" or not w.get("spans"):
        raise ValueError("invalid_witness")
    source = objects[w["source_id"]]
    if source.get("type") != "source" or w["artifact_sha256"] != source["artifact"]["sha256"]:
        raise ValueError("witness_source_mismatch")
    spans, snippets, review = [], [], []
    for span in w["spans"]:
        rid, start, end = span["text_artifact_id"], span["start_codepoint"], span["end_codepoint"]
        region = objects[rid]
        if (region.get("type") != "text-region" or region.get("status") != "captured"
                or region["derivation"]["kind"] not in NATIVE):
            raise ValueError("uncitable_text_region")
        if region["source_id"] != w["source_id"]:
            raise ValueError("witness_source_mismatch")
        text = region["text"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
            raise ValueError("invalid_span_bounds")
        spans.append((rid, start, end))
        snippets.append((text, start, end))
        if (w.get("transcription_state") != "native" or region["derivation"]["kind"] == "ocr_recognized"
                or region["derivation"].get("ocr_text_layer")
                or any(f["start"] < end and start < f["end"] for f in region.get("glyph_flags", []))):
            review.append("visual_confirmation_needed")
    first = objects[spans[0][0]]
    if first.get("page_id"):
        page = objects[first["page_id"]]
        if ((w.get("page") or {}).get("page_id") != first["page_id"]
                or page.get("type") != "page" or page["source_id"] != w["source_id"]):
            raise ValueError("witness_page_mismatch")
    elif w.get("page"):
        raise ValueError("witness_page_mismatch")
    join = w.get("join")
    if len(spans) > 1:
        if join not in ("none", "adjacent_no_gap", "line_break", "omitted_material_marker"):
            raise ValueError("invalid_join")
        _check_join(objects, spans, join)
    pieces = [text[start:end] for text, start, end in snippets]
    copy = pieces[0] if len(pieces) == 1 or join == "none" else {
        "adjacent_no_gap": "", "line_break": "\n", "omitted_material_marker": OMISSION}[join].join(pieces)
    if copy != w["copied_text"] or sha256_hex(copy.encode("utf-8")) != w["copied_text_sha256"]:
        raise ValueError("witness_copy_mismatch")
    return copy, snippets, review


def _numeric_context(value: dict, text: str, start: int) -> dict:
    # Only a directly adjacent label/operator can verify attributes. This bounded
    # look-behind never searches other sentences for a convenient statistic.
    prefix = text[max(0, start - 100):start]
    anchored, op = ANCHOR.search(prefix), OP_AT_END.search(prefix)
    errors, review = [], []
    if re.search(r"[−–+\-]\s+$", prefix):
        review.append("separated_sign_needs_review")
    if value.get("statistic"):
        if anchored:
            if value["statistic"] != anchored["stat"]:
                errors.append("statistic_mismatch")
        else:
            review.append("statistic_not_anchored")
    if value.get("comparator"):
        if op:
            if value["comparator"] != COMPARATORS[op["op"]]:
                errors.append("comparator_mismatch")
        else:
            review.append("comparator_not_anchored")
    return _result(errors, review)


def _numeric(value: dict, snippets: list, predicate: str = "") -> dict:
    lex = value.get("lexical", "")
    cardinal = cardinal_integer_value(lex) if value.get("kind") == "integer" and predicate == "n_items" else None
    try:
        number = Decimal(cardinal) if cardinal is not None else _decimal(lex)
        if number != _decimal(value.get("value")):
            return _result(["typed_value_mismatch"])
        if value["kind"] == "integer" and number != number.to_integral_value():
            return _result(["integer_value_not_integral"])
    except ValueError:
        return _result(["invalid_numeric_value"])
    sign = "negative" if number.is_signed() else "positive" if number > 0 else "zero"
    if value.get("sign") not in (None, sign, "unsigned" if not lex.startswith(("-", "−", "–", "+")) else sign):
        return _result(["sign_mismatch"])
    matches = []
    for text, start, end in snippets:
        if cardinal is not None:
            # Examine original text boundaries so a clipped witness cannot
            # erase a compound prefix, while retaining the match only when it
            # lies wholly inside this assertion's own witness.
            for occurrence in re.finditer(re.escape(lex), text):
                if (start <= occurrence.start() and occurrence.end() <= end
                        and cardinal_integer_at(text, lex, occurrence.start())):
                    matches.append(_numeric_context(value, text, occurrence.start()))
            continue
        # Match whole tokens in the ORIGINAL region, not an already clipped
        # quote: a span starting after a minus sign must not erase that sign.
        for m in TOKEN.finditer(text):
            if start <= m.start() and m.end() <= end and m.group() == lex:
                context = _numeric_context(value, text, m.start())
                if re.match(r"\s*%", text[m.end():]):
                    if context["outcome"] == "literal_supported":
                        context["outcome"] = "needs_review"
                    context["reasons"].append("source_percent_needs_review")
                matches.append(context)
    if not matches:
        return _result(["numeric_token_not_in_witness"])
    outcomes = {m["outcome"] for m in matches}
    if "literal_supported" in outcomes:
        return _result(review=["multiple_numeric_contexts"] if "unsupported" in outcomes else [])
    if "needs_review" in outcomes:
        return _result(review=sorted({r for m in matches for r in m["reasons"]}))
    return _result(sorted({r for m in matches for r in m["reasons"]}))


def _interval(value: dict, copies: list[str], snippets: list) -> dict:
    lex = value.get("lexical", "")
    if not lex or not any(lex in copy for copy in copies):
        return _result(["interval_not_in_witness"])
    pair = re.fullmatch(rf"\s*[\[(]?\s*(?P<lo>{NUMBER})\s*,\s*(?P<hi>{NUMBER})\s*[\])]?\s*", lex)
    if not pair:
        return _result(review=["interval_format_needs_review"])
    try:
        lo, hi = _decimal(value.get("lower")), _decimal(value.get("upper"))
        if lo != _decimal(pair["lo"]) or hi != _decimal(pair["hi"]) or lo > hi:
            return _result(["interval_bounds_mismatch"])
    except ValueError:
        return _result(["invalid_interval_value"])
    # Bind BOTH endpoints to this occurrence, not matching numbers elsewhere
    # among the witnesses. A clipped sign on an endpoint must also fail.
    locations, found = [], False
    for text, start, end in snippets:
        tokens = {(m.start(), m.end()) for m in TOKEN.finditer(text)}
        for occurrence in re.finditer(re.escape(lex), text):
            at = occurrence.start()
            if start <= at and occurrence.end() <= end:
                found = True
                if all((at + pair.start(k), at + pair.end(k)) in tokens for k in ("lo", "hi")):
                    locations.append((text, at))
    if not locations:
        return _result(["interval_endpoint_clipped"]) if found else _result(review=["interval_join_needs_review"])
    if not value.get("level") and not value.get("statistic"):
        return _result()
    labels = []
    for text, at in locations:
        label = re.search(rf"(?<![\w.−–+\-])(?:(?P<level>{NUMBER})\s*%\s*)?"
                          rf"(?:CI|confidence\s+interval)\s*(?::|=)?\s*[\[(]?\s*$",
                          text[max(0, at - 100):at], re.I)
        if label:
            labels.append(label)
    if not labels:
        return _result(review=["interval_level_or_label_not_anchored"])
    if value.get("statistic") not in (None, "CI"):
        return _result(review=["interval_statistic_needs_review"])
    try:
        if value.get("level") and not any(m["level"] is not None and
                                         _decimal(value["level"]) == _decimal(m["level"]) for m in labels):
            return _result(["interval_level_mismatch"])
    except ValueError:
        return _result(["invalid_interval_level"])
    return _result()


def audit_value(objects: dict, value: Any, witness_ids: list[str], predicate: str = "") -> dict:
    """Verify one value ONLY against its own witnesses; never borrow an alternative's."""
    if not isinstance(value, dict):
        return _result(["asserted_value_missing"])
    if not witness_ids:
        return _result(["asserted_witness_missing"])
    copies, snippets, review, errors = [], [], [], []
    for wid in witness_ids:
        try:
            copy, spans, flags = _witness(objects, wid)
            copies.append(copy)
            snippets.extend(spans)
            review.extend(flags)
        except (KeyError, TypeError, IndexError):
            errors.append("invalid_witness")
        except (ValueError, WitnessError) as exc:
            errors.append(str(exc))
    if errors:
        return _result(errors, review)
    kind = value.get("kind")
    if kind in ("integer", "decimal"):
        checked = _numeric(value, snippets, predicate)
        verified = {"kind", "lexical", "value", "sign", "statistic", "comparator"}
    elif kind == "interval":
        checked = _interval(value, copies, snippets)
        verified = {"kind", "lexical", "lower", "upper", "level", "statistic"}
    elif kind == "text" and (predicate.endswith("_wording") or predicate in ("measure_label", "anchor")):
        checked = _result() if value.get("lexical") in copies else _result(["wording_not_equal_to_witness"])
        verified = {"kind", "lexical"}
    else:
        checked, verified = _result(review=["semantic_adjudication_needed"]), set()
    if any(value.get(k) is not None for k in value.keys() - verified) and verified:
        review.append("additional_attributes_need_review")
    if checked["outcome"] == "unsupported":
        return _result(checked["reasons"], review)
    return _result(review=review + checked["reasons"])


def audit_assertion(objects: dict, body: dict) -> dict:
    state = body.get("state")
    if state in ("explicitly_absent", "not_found_in_reviewed_scope", "unavailable_or_illegible", "not_applicable", "not_processed"):
        return {"outcome": "not_asserted", "reasons": [], "values": []}
    if state not in ("present", "ambiguous_or_conflicting"):
        return {**_result(["invalid_assertion_state"]), "values": []}
    values = []
    if state == "present" or body.get("value") is not None:
        values.append(audit_value(objects, body.get("value"), body.get("witness_ids", []), body.get("predicate", "")))
    for alternative in body.get("alternatives") or []:
        values.append(audit_value(objects, alternative.get("value"), alternative.get("witness_ids", []), body.get("predicate", "")))
    if state == "ambiguous_or_conflicting" and len(body.get("alternatives") or []) < 2:
        values.append(_result(["conflicting_alternatives_missing"]))
    outcome = next(o for o in ("unsupported", "needs_review", "literal_supported") if any(v["outcome"] == o for v in values))
    return {"outcome": outcome, "reasons": sorted({r for v in values for r in v["reasons"]}), "values": values}


def audit_package(objects: dict[str, dict]) -> dict:
    """Audit assertions only; relation semantics and CORE eligibility are separate checks."""
    assertions = {oid: audit_assertion(objects, body) for oid, body in sorted(objects.items()) if body.get("type") == "assertion"}
    counts = Counter(row["outcome"] for row in assertions.values())
    return {"version": 1, "counts": {key: counts[key] for key in OUTCOMES}, "assertions": assertions}
