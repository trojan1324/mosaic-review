"""Opt-in choices for explicitly labelled effect and model-fit values.

This module leaves the existing printed-statistic menu byte-for-byte intact.
It adds a choice only when an effect label and its exact printed comparator and
value occur together in the same source unit.  It does not associate an
effect with a result, condition, or outcome.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mosaic.workspace import default_workspace
from scripts.reported_quantity_choices import build_quantity_choices
from scripts.typed_statistic_choices import _NUMBER


_SPACE = r"[ \t\r\n]*"
_REQUIRED_SPACE = r"[ \t\r\n]+"
_DEFINED_DISPERSION_RE = re.compile(r'(?<!\w)(?P<label>standard\s+(?:error\s*\[SE\]|deviation\s*\[SD\]))'
    + _SPACE + r'(?P<comparator><=|>=|<|>|=|≤|≥)' + _SPACE
    + r'(?P<value>'+_NUMBER+r')(?![\w]|\.\d)',re.I)
_COMPACT_MEAN_RE = re.compile(r'(?<!\w)(?P<label>M[A-Za-z][A-Za-z0-9_]*(?:[-\u00ad]\s*[A-Za-z][A-Za-z0-9_]*)*)'
    + _SPACE + r'(?P<comparator>=)' + _SPACE + r'(?P<value>'+_NUMBER+r')(?![\w]|\.\d)')
_MEAN_PAIR_BRIDGE = re.compile(_SPACE+r'(?:,'+_SPACE+r'SD'+_SPACE+r'='+_SPACE+_NUMBER+_SPACE+r')?vs\.?'+_SPACE)
_INDIRECT_WORD = r'(?:[-\u00ad]\s*)?'.join('indirect')
_AB_EFFECT_RE = re.compile(r'(?<!\w)'+_INDIRECT_WORD+r'\s+effect\s*:\s*(?P<label>ab)'
    + _SPACE+r'(?P<comparator>=)'+_SPACE+r'(?P<value>'+_NUMBER+r')(?![\w]|\.\d)',re.I)
_BRACKET_T_RE = re.compile(r'(?<!\w)(?P<label>t)' + _SPACE + r'\[' + _SPACE
    + r'(?P<df>' + _NUMBER + r')' + _SPACE + r'\]' + _SPACE
    + r'(?P<comparator><=|>=|<|>|=|≤|≥)' + _SPACE + r'(?P<value>' + _NUMBER + r')(?![\w]|\.\d)',re.I)
_GAMMA_RE = re.compile(r'(?<!\w)(?P<label>γ)' + _SPACE
    + r'(?P<comparator><=|>=|<|>|=|≤|≥)' + _SPACE + r'(?P<value>' + _NUMBER + r')(?![\w]|\.\d)')
_EFFECT_RE = re.compile(
    r"(?<![\w])(?P<label>indirect" + _SPACE + r"effect|BootSE)" + _SPACE
    + r"(?P<comparator><=|>=|<|>|=|≤|≥)" + _SPACE + r"(?P<value>" + _NUMBER + r")(?![\w]|\.\d)",
    re.IGNORECASE,
)
_R_SQUARED_RE = re.compile(
    r"(?<![\w])(?P<label>R[2²])" + _SPACE
    + r"(?P<comparator><=|>=|<|>|=|≤|≥)" + _SPACE + r"(?P<value>" + _NUMBER + r")(?![\w]|\.\d)",
    re.IGNORECASE,
)
_P_VALUE_RE = re.compile(
    r"(?<![\w])(?P<label>p)" + _SPACE
    + r"(?P<comparator><=|>=|<|>|=|≤|≥)" + _SPACE + r"(?P<value>" + _NUMBER + r")(?![\w]|\.\d)",
    re.IGNORECASE,
)
_CHANGE_RE = re.compile(
    r"(?<![\w])(?P<label>(?:increase|decrease)(?:d|s)?\s+by)\s+"
    + r"(?P<value>" + _NUMBER + r")(?![\w]|\.\d)", re.IGNORECASE,
)
_LABELED_COEFFICIENT_RE = re.compile(
    r"(?<![\w])(?P<label>B(?!ootSE\b)(?P<suffix>[a-z][A-Za-z0-9_]*"
    + r"(?:" + _SPACE + r"[×*]" + _SPACE + r"[A-Za-z][A-Za-z0-9_]*)*))"
    + _SPACE + r"(?P<comparator><=|>=|<|>|=|≤|≥)" + _SPACE
    + r"(?P<value>" + _NUMBER + r")(?![\w]|\.\d)",
)
_CONDITION_MEAN_RE = re.compile(
    r"(?<![\w])(?P<label>M(?:[A-Z][A-Za-z0-9_]*(?:" + _REQUIRED_SPACE + r"[A-Za-z][A-Za-z0-9_]*)*"
    + r"|[a-z][A-Za-z0-9_]*" + _REQUIRED_SPACE + r"[A-Za-z][A-Za-z0-9_]*))"
    + _SPACE + r"(?P<comparator><=|>=|<|>|=|≤|≥)" + _SPACE
    + r"(?P<value>" + _NUMBER + r")(?![\w]|\.\d)",
)
_BARE_CI_RE = re.compile(
    r"\[" + _SPACE + r"(?P<label>CI)" + _SPACE + r"(?:=|:)" + _SPACE
    + r"(?P<lower>" + _NUMBER + r")" + _SPACE + r"[,;]" + _SPACE
    + r"(?P<upper>" + _NUMBER + r")" + _SPACE + r"\]",
    re.IGNORECASE,
)
_COLON_CI_RE = re.compile(
    r'(?P<level>\d{1,3}%)' + _SPACE + r'(?P<label>CI|con(?:fi|ﬁ)dence' + _REQUIRED_SPACE
    + r'interval' + _SPACE + r'\[CI\])' + _SPACE + r':' + _SPACE
    + r'(?P<lower>' + _NUMBER + r')' + _SPACE + r',' + _SPACE
    + r'(?P<upper>' + _NUMBER + r')(?![\w]|\.\d)', re.I)
_DIFFERENCE_RE = re.compile(r'(?<![\w])(?P<label>difference)' + _SPACE
    + r'(?P<comparator>=)' + _SPACE + r'(?P<value>' + _NUMBER + r')(?![\w]|\.\d)',re.I)
_UNKNOWN_COLON_CI_RE = re.compile(r'(?<![\w])(?P<label>CI)' + _SPACE + ':' + _SPACE
    + r'(?P<lower>' + _NUMBER + ')' + _SPACE + ',' + _SPACE
    + r'(?P<upper>' + _NUMBER + r')(?![\w]|\.\d)',re.I)
_ESTIMATE_CI_PREFIX = re.compile(r'(?<![\w.])(?P<value>' + _NUMBER + r')'
    + _SPACE + r'(?:\[' + _SPACE + r'SE' + _SPACE + r'=' + _SPACE + _NUMBER + _SPACE + r'\])?'
    + _SPACE + r',' + _SPACE + r'$')
# Compact labels can resemble ordinary prose.  These common discourse and
# manuscript words cannot establish a statistic merely because they precede
# an equals sign; this is a conservative lexical guard, never a variable map.
_PROSE_LIKE_LABELS = {
    "because", "before", "being", "between", "both",
    "maturity", "mean", "means", "mediation", "method", "methods",
    "model", "models", "more", "most",
}


def _component(name: str, text: str, span: tuple[int, int]) -> dict[str, Any]:
    start, end = span
    return {"name": name, "start": start, "end": end, "text": text[start:end]}


def _covered(match: re.Match[str], choices: Sequence[Mapping[str, Any]], unit_id: str) -> bool:
    return any(choice.get("unit_id") == unit_id and choice.get("start") <= match.start()
               and match.end() <= choice.get("end") for choice in choices)


def _label_is_statistic(match: re.Match[str]) -> bool:
    return match.group("label").casefold() not in _PROSE_LIKE_LABELS


def _choice(unit_id: str, text: str, match: re.Match[str]) -> dict[str, Any]:
    label = match.group("label")
    statistic_type = "indirect_effect" if "indirect" in label.casefold() else "bootstrap_standard_error"
    return {
        "unit_id": unit_id,
        "start": match.start(),
        "end": match.end(),
        "text": text[match.start():match.end()],
        "statistic_type": statistic_type,
        "printed_label": label,
        "components": [
            _component("label", text, match.span("label")),
            _component("comparator", text, match.span("comparator")),
            _component("value", text, match.span("value")),
        ],
    }


def _labeled_choice(unit_id: str, text: str, match: re.Match[str], statistic_type: str) -> dict[str, Any]:
    return {
        "unit_id": unit_id,
        "start": match.start(),
        "end": match.end(),
        "text": text[match.start():match.end()],
        "statistic_type": statistic_type,
        "printed_label": match.group("label"),
        "components": [
            _component("label", text, match.span("label")),
            _component("comparator", text, match.span("comparator")),
            _component("value", text, match.span("value")),
        ],
    }


def _bare_ci_choice(unit_id: str, text: str, match: re.Match[str]) -> dict[str, Any]:
    return {
        "unit_id": unit_id,
        "start": match.start(),
        "end": match.end(),
        "text": text[match.start():match.end()],
        "statistic_type": "confidence_interval_unknown_level",
        "printed_label": match.group("label"),
        "components": [
            _component("label", text, match.span("label")),
            _component("lower", text, match.span("lower")),
            _component("upper", text, match.span("upper")),
        ],
    }


def build_named_effect_choices(units: Sequence[Mapping[str, Any]], *, spelled_coefficients: bool = False) -> list[dict[str, Any]]:
    """Add only unambiguously printed indirect-effect and BootSE values.

    Existing choices, including ``R2``/``R²`` and their separate p values,
    retain their original identifiers and contents.  The latter are already
    complete typed choices in the wrapped menu, so this function never creates
    an overlapping duplicate merely to relabel them.
    """
    base = build_quantity_choices(units)
    additions: list[dict[str, Any]] = []
    for unit in units:
        unit_id, text = unit["unit_id"], unit["text"]
        for match in _EFFECT_RE.finditer(text):
            if not _covered(match, [*base, *additions], unit_id):
                additions.append(_choice(unit_id, text, match))
    for index, choice in enumerate(additions, start=1):
        choice["choice_id"] = f"named{index:03d}"
    coefficients: list[dict[str, Any]] = []
    condition_means: list[dict[str, Any]] = []
    bare_intervals: list[dict[str, Any]] = []
    colon_intervals=[];differences=[];adjacent_estimates=[]
    for unit in units:
        unit_id, text = unit["unit_id"], unit["text"]
        for match in _LABELED_COEFFICIENT_RE.finditer(text):
            if _label_is_statistic(match) and not _covered(match, [*base, *additions, *coefficients], unit_id):
                coefficients.append(_labeled_choice(unit_id, text, match, "coefficient"))
        for match in _CONDITION_MEAN_RE.finditer(text):
            if _label_is_statistic(match) and not _covered(match, [*base, *additions, *condition_means], unit_id):
                condition_means.append(_labeled_choice(unit_id, text, match, "mean"))
        for match in _BARE_CI_RE.finditer(text):
            if not _covered(match, [*base, *additions, *bare_intervals], unit_id):
                bare_intervals.append(_bare_ci_choice(unit_id, text, match))
        for match in _COLON_CI_RE.finditer(text):
            if not _covered(match,base,unit_id):
                colon_intervals.append({'unit_id':unit_id,'start':match.start(),'end':match.end(),
                    'text':match.group(),'statistic_type':'confidence_interval','printed_label':match.group('label'),
                    'components':[_component(name,text,match.span(group)) for name,group in
                        [('confidence_level','level'),('lower','lower'),('upper','upper')]]})
            estimate=_ESTIMATE_CI_PREFIX.search(text[:match.start()])
            if estimate and not any(c['unit_id']==unit_id and c['start']<=estimate.start('value') and estimate.end('value')<=c['end'] for c in base+additions):
                start,end=estimate.span('value')
                adjacent_estimates.append({'unit_id':unit_id,'start':start,'end':end,'text':text[start:end],
                    'statistic_type':'estimate_adjacent_to_confidence_interval','printed_label':None,
                    'components':[_component('value',text,(start,end))]})
        for match in _DIFFERENCE_RE.finditer(text):
            if not _covered(match,base,unit_id):differences.append(_labeled_choice(unit_id,text,match,'reported_difference'))
    for prefix, candidates in (("coefficient", coefficients), ("condition_mean", condition_means), ("bare_ci", bare_intervals),
                              ('colon_ci',colon_intervals),('difference',differences),('ci_estimate',adjacent_estimates)):
        for index, choice in enumerate(candidates, start=1):
            choice["choice_id"] = f"{prefix}{index:03d}"
    unknown_colon=[]
    for unit in units:
        for match in _UNKNOWN_COLON_CI_RE.finditer(unit['text']):
            if not _covered(match,base+colon_intervals+bare_intervals,unit['unit_id']):
                value=_bare_ci_choice(unit['unit_id'],unit['text'],match)
                value['choice_id']=f'unknown_colon_ci{len(unknown_colon)+1:03d}'
                unknown_colon.append(value)
    changes=[]
    for unit in units:
        unit_id,text=unit['unit_id'],unit['text']
        for match in _CHANGE_RE.finditer(text):
            # A following printed unit belongs to the existing quantity choice;
            # do not add a second bare value for the same reported expression.
            start,end=match.span('value')
            if any(c['unit_id']==unit_id and c['start']<=start and end<=c['end'] for c in base+additions):
                continue
            changes.append({'unit_id':unit_id,'start':match.start(),'end':match.end(),
                'text':match.group(),'statistic_type':'reported_change','printed_label':match.group('label'),
                'components':[_component(k,text,match.span(k)) for k in ('label','value')],
                'choice_id':f'change{len(changes)+1:03d}'})
    order = {unit["unit_id"]: index for index, unit in enumerate(units)}
    bracket_tests=[]
    gamma_coefficients=[]
    paired_means=[];ab_effects=[];defined_dispersions=[]
    for unit in units:
        text=unit['text'];uid=unit['unit_id']
        for match in _DEFINED_DISPERSION_RE.finditer(text):
            if _covered(match,base,uid):continue
            kind='standard_error' if re.search(r'\[SE\]',match.group('label'),re.I) else 'standard_deviation'
            value=_labeled_choice(uid,text,match,kind)
            value['choice_id']=f'defined_dispersion{len(defined_dispersions)+1:03d}'
            defined_dispersions.append(value)
        mean_matches=list(_COMPACT_MEAN_RE.finditer(text))
        witnessed=set()
        for left,right in zip(mean_matches,mean_matches[1:]):
            if (_label_is_statistic(left) and _label_is_statistic(right)
                    and _MEAN_PAIR_BRIDGE.fullmatch(text[left.end():right.start()])):
                witnessed.update((left.start(),right.start()))
        for match in mean_matches:
            if match.start() in witnessed and not _covered(match,base+condition_means,uid):
                value=_labeled_choice(uid,text,match,'mean')
                value['choice_id']=f'paired_mean{len(paired_means)+1:03d}'
                paired_means.append(value)
        for match in _AB_EFFECT_RE.finditer(text):
            if not _covered(match,base+additions,uid):
                value=_labeled_choice(uid,text,match,'indirect_effect')
                value['choice_id']=f'ab_effect{len(ab_effects)+1:03d}'
                ab_effects.append(value)
        for match in _GAMMA_RE.finditer(unit['text']):
            if _covered(match,base,unit['unit_id']):continue
            value=_labeled_choice(unit['unit_id'],unit['text'],match,'coefficient')
            value['choice_id']=f'gamma{len(gamma_coefficients)+1:03d}'
            gamma_coefficients.append(value)
        for match in _BRACKET_T_RE.finditer(unit['text']):
            if _covered(match,base,unit['unit_id']):continue
            bracket_tests.append({'unit_id':unit['unit_id'],'start':match.start(),'end':match.end(),'text':match.group(),
                'statistic_type':'t_statistic','printed_label':match.group('label'),
                'components':[_component(k,unit['text'],match.span(k)) for k in ('df','comparator','value')],
                'choice_id':f'bracket_t{len(bracket_tests)+1:03d}'})
    spelled=[]
    if spelled_coefficients:
        pattern=re.compile(r'(?<![\w])(?P<label>coeff(?:icient)?\.?)' + _SPACE
            + r'(?P<comparator>=)' + _SPACE + r'(?P<value>' + _NUMBER + r')(?![\w]|\.\d)',re.I)
        existing=base+additions+coefficients
        for unit in units:
            for match in pattern.finditer(unit['text']):
                if _covered(match,existing,unit['unit_id']):continue
                choice=_labeled_choice(unit['unit_id'],unit['text'],match,'coefficient')
                choice['choice_id']=f'spelled_coefficient{len(spelled)+1:03d}'
                spelled.append(choice)
    return sorted([*spelled, *base, *additions, *coefficients, *condition_means, *bare_intervals, *colon_intervals, *differences, *adjacent_estimates, *changes,*unknown_colon,*bracket_tests,*gamma_coefficients,*paired_means,*ab_effects,*defined_dispersions], key=lambda choice: (
        order[choice["unit_id"]], choice["start"], choice["end"], choice["choice_id"]
    ))


def diagnose_named_effects(units: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count observed exact forms and existing menu coverage; make no accuracy claim."""
    base = build_quantity_choices(units)
    indirect = bootse = r_squared = model_p_pairs = added = coefficients = condition_means = bare_intervals = 0
    for unit in units:
        text, unit_id = unit["text"], unit["unit_id"]
        effects = list(_EFFECT_RE.finditer(text))
        indirect += sum("indirect" in match.group("label").casefold() for match in effects)
        bootse += sum(match.group("label").casefold() == "bootse" for match in effects)
        added += sum(not _covered(match, base, unit_id) for match in effects)
        coefficients += sum(_label_is_statistic(match) for match in _LABELED_COEFFICIENT_RE.finditer(text))
        condition_means += sum(_label_is_statistic(match) for match in _CONDITION_MEAN_RE.finditer(text))
        bare_intervals += len(list(_BARE_CI_RE.finditer(text)))
        r_matches = list(_R_SQUARED_RE.finditer(text))
        r_squared += len(r_matches)
        p_matches = list(_P_VALUE_RE.finditer(text))
        for r_match in r_matches:
            if any(0 <= p_match.start() - r_match.end() <= 16
                   and re.fullmatch(r"[\s,;\)\]\(]*", text[r_match.end():p_match.start()])
                   for p_match in p_matches):
                model_p_pairs += 1
    return {
        "indirect_effect_exact_forms": indirect,
        "bootse_exact_forms": bootse,
        "r_squared_exact_forms": r_squared,
        "r_squared_followed_by_exact_p": model_p_pairs,
        "new_named_effect_choices": added,
        "labeled_coefficient_exact_forms": coefficients,
        "condition_mean_exact_forms": condition_means,
        "bare_confidence_interval_exact_forms": bare_intervals,
        "existing_menu_choices": len(base),
    }


def _units_from_request(path: Path) -> list[dict[str, str]]:
    request = json.loads(path.read_text(encoding="utf-8"))
    messages = request.get("messages", [])
    payloads = [message.get("content") for message in messages if message.get("role") == "user"]
    if len(payloads) != 1 or not isinstance(payloads[0], str):
        raise ValueError("request_has_no_single_user_payload")
    payload = json.loads(payloads[0])
    source = payload.get("source")
    if not isinstance(source, list):
        raise ValueError("request_has_no_source")
    return [{"unit_id": row["unit_id"], "text": row["text"]} for row in source]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Count exact named-effect syntax in saved Results requests.")
    parser.add_argument("--request", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    units = [unit for request in args.request for unit in _units_from_request(request)]
    counts = diagnose_named_effects(units)
    default_workspace().write_json_pretty(args.output, {
        "purpose": "syntax diagnostic only; it does not assess semantic correctness or extraction accuracy",
        "request_count": len(args.request),
        "unit_count": len(units),
        "counts": counts,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
