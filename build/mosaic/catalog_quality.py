"""Advisory checks on imported bibliography; never correct or merge identities."""
from datetime import date
import html
import re

VERSION = 'catalog-metadata-review/v1'


def metadata_issues(paper, *, as_of_year=None):
    year = date.today().year if as_of_year is None else as_of_year
    issues = []
    recorded = paper.get('year')
    if str(recorded).isdigit() and int(recorded) > year + 1:
        issues.append({'field': 'year', 'reason': 'future_year',
                       'message': 'The recorded publication year needs source verification.'})
    for field in ('title', 'authors', 'journal'):
        values = paper.get(field)
        values = values if isinstance(values, list) else [values]
        if any(isinstance(v, str) and html.unescape(v) != v for v in values):
            issues.append({'field': field, 'reason': 'encoded_display_text',
                           'message': 'Imported ' + field + ' text contains encoded display characters.'})
    doi = paper.get('doi')
    if isinstance(doi, str) and doi and (
            re.search(r'\s', doi)
            or re.search(r'(?:copyright|\(published|thissuggests|supponseptember)', doi, re.I)):
        issues.append({'field': 'doi', 'reason': 'possible_appended_text',
                       'message': 'The recorded DOI may include surrounding text; verify it against the source.'})
    return issues
