"""Group clear adjacent Results block continuations without changing source bytes.

Physical PDF blocks are not necessarily prose paragraphs. This conservative
helper records adjacency/continuation evidence; it does not infer study scope.
"""
import re

_STATISTIC_LABEL = r'(?<!\w)(?:[pPbBβ]|SE|SD|M|t|F|r|R[2²]|η[2²])(?:\s*\([^()\n]+\))?'
_NUMERIC_START = r'[+\-\u2212]?\s*(?:\d|\.\d)'


def _recorded_boundary(left, right):
    """Opt-in boundary check; old frozen packets retain their historical replay."""
    a, b = left.get('source_context'), right.get('source_context')
    if a is None and b is None:
        return True
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    if a.get('version') != 'recorded-source-context/v1' or b.get('version') != a['version']:
        return False
    paths = left.get('section_paths')
    if not paths or not all(path and all(node.get('node_id') for node in path) for path in paths):
        return False
    if paths != right.get('section_paths') or not a.get('source_id') or a['source_id'] != b.get('source_id'):
        return False
    if a.get('reading_order_status') not in ('geometry_consistent', 'single_column') or b.get('reading_order_status') not in ('geometry_consistent', 'single_column'):
        return False
    page, next_page = a.get('physical_page_index'), b.get('physical_page_index')
    ordinal = a.get('admitted_ordinal')
    return (isinstance(page, int) and next_page in (page, page + 1)
            and isinstance(ordinal, int) and b.get('admitted_ordinal') == ordinal + 1)


def _comparison_continuation(left, right):
    if not left.get('source_context') or not _recorded_boundary(left, right):
        return False
    prior, current = left['text'].rstrip(), right['text'].lstrip()
    depth = prior.count('(') - prior.count(')')
    if depth <= 0 or not re.search(r'\bvs\.$', prior, re.I) or not re.match(r'[A-Za-z]', current):
        return False
    for character in current.split('\n\n', 1)[0]:
        depth += (character == '(') - (character == ')')
        if depth == 0:
            return True
    return False

def _open_statistic_continuation(left, right):
    """Recognize a witnessed unfinished statistic, never a nearby new finding."""
    if not left.get('section_paths') or left['section_paths'] != right.get('section_paths'):
        return False
    if left.get('page') is None or right.get('page') not in (left['page'], left['page'] + 1):
        return False
    prior, current = left['text'].rstrip(), right['text'].lstrip()
    depth = prior.count('(') - prior.count(')')
    if not prior.endswith(',') or depth <= 0:
        return False
    if not re.match(r'(?:BootSE|SE|SD|\d{1,3}%\s*CI)\s*(?:=|\[|:)', current):
        return False
    # Require closure of the outstanding parenthetical before a new paragraph,
    # rather than mistaking an inner test's closing parenthesis for that closure.
    for character in current.split('\n\n', 1)[0]:
        depth += (character == '(') - (character == ')')
        if depth == 0:
            return True
    return False


def group_result_blocks(units):
    groups = []
    ids = [u['unit_id'] for u in units]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate_result_unit')
    for unit in units:
        text = unit['text'].strip()
        linked = False
        if groups and text:
            previous = groups[-1]['units'][-1]
            prior = previous['text'].strip()
            # A complete sentence or a heading is not silently joined. Require
            # a broken prior sentence plus a visibly continuing next block.
            heading = len(prior.splitlines()) == 1 and re.search(r'\b(?:results|discussion)\b', prior, re.I)
            same_section = previous.get('section_paths') == unit.get('section_paths')
            continuation = bool(re.match(r'[a-z,;:)\]]', text))
            # A CI label at a column/page boundary ends in a colon, while
            # its continuation can start with a signed number. Neither is
            # ordinary sentence punctuation in this specific construction.
            interval_continuation = bool(
                re.search(r'\b(?:CI|confidence\s+interval)\s*:\s*$', prior, re.I)
                and re.match(_NUMERIC_START, text)
            )
            labelled_continuation = bool(
                (re.search(_STATISTIC_LABEL+r'\s*=\s*$', prior) and re.match(_NUMERIC_START, text))
                or (re.search(_STATISTIC_LABEL+r'\s*$', prior) and re.match(r'=\s*'+_NUMERIC_START, text))
            )
            prose_continuation = bool(prior and prior[-1] not in '.!?;:' and continuation)
            open_continuation = _open_statistic_continuation(previous, unit)
            comparison_continuation = _comparison_continuation(previous, unit)
            if not heading and same_section and _recorded_boundary(previous, unit) and (prose_continuation or interval_continuation or labelled_continuation or open_continuation or comparison_continuation):
                groups[-1]['units'].append(unit)
                groups[-1]['joins'].append({'previous_unit': previous['unit_id'], 'next_unit': unit['unit_id'],
                    'reason': ('adjacent_same_section_interval_continuation' if interval_continuation
                               else 'adjacent_same_section_labelled_statistic_continuation' if labelled_continuation
                               else 'adjacent_same_section_open_parenthetical_statistic_continuation' if open_continuation
                               else 'recorded_same_section_open_comparison_continuation' if comparison_continuation
                               else 'adjacent_same_section_broken_sentence_continuation')})
                linked = True
        if not linked:
            groups.append({'group_id': unit['unit_id'], 'units': [unit], 'joins': []})
    return groups
