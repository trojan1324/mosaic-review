"""Source-exact response-format candidates, opt-in after the grounded v4 trial.

These boundaries do not assign an instrument, fill a field or certify a scale.
Explicit respondent alternatives cannot be presented as questionnaire items.
"""
from copy import deepcopy
import re

from scripts.measure_field_choices import _NUMBER, _QUOTES, _ENCODING
from scripts.measure_field_trial import _sources, _candidate
from scripts.source_decoding import decoding_status

_FORMAT = re.compile(rf'(?<![\w\-–—−]){_NUMBER}[-\s]+point(?:\s+Likert)?\s+scales?\b', re.I)
_SELECT = re.compile(r'\beither\s+(?:select|choose)\s*$', re.I)
_OPTION = re.compile(r'^\s*(yes|no)\b', re.I)


def _source_key(choice):
    segment = choice['segments'][0]
    return tuple(segment[k] for k in ('unit_id', 'page', 'source_locator', 'unit_text_sha256'))


def refine_choices(version, previous):
    offered, extra = deepcopy(previous), []
    for row in _sources(version):
        text = row['segment']['text']
        pairs, quoted = [], list(_QUOTES.finditer(text))
        for a, b in zip(quoted, quoted[1:]):
            first = next(x for x in a.groups() if x is not None)
            second = next(x for x in b.groups() if x is not None)
            x, y = _OPTION.match(first), _OPTION.match(second)
            if not x or not y or {x[1].lower(), y[1].lower()} != {'yes', 'no'}:
                continue
            if not _SELECT.search(text[max(0, a.start() - 100):a.start()]):
                continue
            if not re.fullmatch(r'\s*,?\s*or\s*', text[a.end():b.start()], re.I):
                continue
            pair = _candidate(row, a.start(), b.end(), 'explicit_binary_options')
            pair.update(boundary_rule='explicit_binary_options', role_allowlist=['response_scale'])
            pairs.append(pair)
            extra.append(pair)
        for pair in pairs:
            span = pair['segments'][0]
            for choice in offered:
                segment = choice['segments'][0]
                if _source_key(choice) == _source_key(pair) and span['start'] <= segment['start'] < segment['end'] <= span['end']:
                    choice['role_allowlist'] = [role for role in choice['role_allowlist'] if role not in ('items', 'question_stems')]
                    choice['format_boundary'] = 'inside_explicit_response_options'
        for match in _FORMAT.finditer(text):
            choice = _candidate(row, match.start(), match.end(), 'stated_point_format')
            segment = choice['segments'][0]
            if any('response_scale' in old['role_allowlist'] and _source_key(old) == _source_key(choice)
                   and old['segments'][0]['start'] <= segment['start'] and old['segments'][0]['end'] >= segment['end'] for old in offered):
                continue
            choice.update(boundary_rule='stated_point_format', role_allowlist=['response_scale'])
            extra.append(choice)
    seen = {(*_source_key(c), c['segments'][0]['start'], c['segments'][0]['end']) for c in offered}
    for choice in extra:
        key = (*_source_key(choice), choice['segments'][0]['start'], choice['segments'][0]['end'])
        if key in seen:
            continue
        seen.add(key)
        choice['choice_id'] = f'measure-choice-{len(offered) + 1:03d}'
        choice['source_text_status'] = ('visual_check_required' if _ENCODING.search(choice['quote']) else
            decoding_status(choice['segments']))
        offered.append(choice)
    return offered
