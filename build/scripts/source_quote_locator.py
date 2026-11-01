"""Locate a quoted clause, returning original text and character offsets.

Whitespace and a hyphen at an alphabetic PDF line wrap may differ in the search
representation. An explicit option also permits a first-letter case change.
An optional alphabetic-ligature search preserves original glyphs and offsets.
The returned text is never reconstructed, recased or dehyphenated.
This proves a contiguous source location, not the semantics of the assertion.
"""
import re


_LIGATURES = dict(zip("ﬀﬁﬂﬃﬄﬅﬆ", ("ff", "fi", "fl", "ffi", "ffl", "st", "st")))


def _expand_ligatures(text):
    characters, offsets = [], []
    for index, char in enumerate(text):
        replacement = _LIGATURES.get(char, char)
        characters.extend(replacement)
        offsets.extend([index] * len(replacement))
    return ''.join(characters), offsets


def locate_quote(quote, source, *, allow_initial_case_change=False,
                 allow_alphabetic_ligatures=False):
    if allow_alphabetic_ligatures:
        expanded_quote, _ = _expand_ligatures(quote)
        expanded_source, offsets = _expand_ligatures(source)
        span = locate_quote(expanded_quote, expanded_source,
                            allow_initial_case_change=allow_initial_case_change)
        if span is None:
            return None
        start, end = span['start'], span['end']
        # A match may never start or end in the middle of one source glyph.
        if ((start and offsets[start] == offsets[start - 1])
                or (end < len(offsets) and offsets[end] == offsets[end - 1])):
            return None
        original_start, original_end = offsets[start], offsets[end - 1] + 1
        original_text = source[original_start:original_end]
        normalization = span['search_normalization']
        if original_text != span['text'] or expanded_quote != quote:
            normalization += '_and_alphabetic_ligatures'
        return {'start': original_start, 'end': original_end, 'text': original_text,
                'search_normalization': normalization}
    quote = re.sub(r'\s+', ' ', quote).strip()
    if not quote:
        return None
    parts = []
    for index, char in enumerate(quote):
        part = r'\s+' if char == ' ' else re.escape(char)
        if index == 0 and allow_initial_case_change and char.isascii() and char.isalpha():
            part = '(?i:' + part + ')'
        parts.append(part)
        following = quote[index + 1:index + 2]
        if char.isalpha() and following.isalpha():
            parts.append(r'(?:[-\u00ad][ \t]*(?:\r?\n[ \t]*)+)?')
        elif char in ('-', '\u00ad') and index and quote[index - 1].isalpha() and following.isalpha():
            parts.append(r'(?:[ \t]*(?:\r?\n[ \t]*)+)?')
    match = re.search(''.join(parts), source)
    if match is None:
        return None
    text = source[match.start():match.end()]
    comparison = re.sub(r'\s+', ' ', text).strip()
    initial_case_changed = comparison[:1] != quote[:1]
    if initial_case_changed:
        comparison = quote[:1] + comparison[1:]
    normalization = 'whitespace' if comparison == quote else 'whitespace_and_alphabetic_line_wrap'
    if initial_case_changed:
        normalization += '_and_initial_case'
    return {'start': match.start(), 'end': match.end(), 'text': text,
        'search_normalization': normalization}
