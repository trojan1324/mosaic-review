"""Heading roles for the deterministic study inventory (adapted from E/evoke/research_studio/documents.py).

Kept verbatim: _HEADING_ROLES, _STUDY_RE, _PILOT_STUDY_RE, _SECTION_NUMBER_RE,
_NAMED_NUMBERED_STUDY_RE, _STATISTICAL_TABLE_ROW_RE and the logic of
_heading_parts, _heading_role and _is_heading_text. Dropped: the rest of the
module (ingestion, footnote routing, table-free scope), which v2 replaces.
Changed: public names without the leading underscore; a `study_label` helper
that normalizes "Study 2", "Experiment 1a", "Pretest" for inventory IDs;
`split_section_number`, which also reads lettered ("C.3", "B."), Roman
("III.") and space-separated ("2.1 Participants") numbering, and an optional
`number` argument to `heading_role`/`is_heading_text` for a number printed on
its own line. `SECTION_NUMBER_RE` and `heading_parts` are unchanged, so the
page-level appendix split in capture reads headings as before.
Importing the original is forbidden (E's package installs the Command Center
guard on import, audit C-02).
"""

from __future__ import annotations

import re

HEADING_ROLES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^(abstract|summary)$", re.I), "abstract"),
    (re.compile(r"^(introduction|background)$", re.I), "introduction"),
    (re.compile(r"^(theory|theoretical (development|background|framework)|conceptual (development|framework)|literature review|hypotheses?|propositions?)$", re.I), "theory"),
    (re.compile(r"^(methods?|materials? and methods?|methodology|data|data and (methods|analysis)|empirical setting)$", re.I), "methods"),
    (re.compile(r"^(?:results?|findings?)(?:\s+for\s+(?:study|experiment)\s+[0-9]+[a-z]?)?$|^estimates(?:\s*:\s*[^.!?]{1,80})?$", re.I), "results"),
    (re.compile(r"^(discussion(?:\s+and\s+conclusion)?|general discussion|conclusion|conclusions|concluding remarks)$", re.I), "discussion"),
    (re.compile(r"^procedure$", re.I), "methods"),
    (re.compile(r"^(limitations?|future research|implications?)$", re.I), "discussion"),
    (re.compile(r"^(references|reference list|bibliography)$", re.I), "references"),
    (re.compile(r"^(?:(?:appendix|appendices)\b(?:\s+.*)?|supplement\b|supplementary\s+materials?\b(?:\s*[:—–-]\s*.*)?|supplemental\s+materials?\b(?:\s*[:—–-]\s*.*)?)$", re.I), "appendix"),
)
STUDY_RE = re.compile(
    r"^\s*(study|experiment)\s+([0-9]+[a-z]?|[ivxlcdm]+)(?=\s*(?:$|[:—–\-(])|\s+(?:methods?|results?)\s*$)",
    re.I,
)
PILOT_STUDY_RE = re.compile(
    r"^\s*(?:pilot(?:\s+study)?|pretest)(?:\s+([0-9]+[a-z]?|[ivxlcdm]+))?(?=\s*(?:$|[:—–\-(]))",
    re.I,
)
SECTION_NUMBER_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)(?:\.\s+|\s*\|\s*)(\S.*?)\s*$")
# Other section numbers: Arabic (parts of one or two digits, not starting at 0) followed by a space only
# ("2.1 Participants", "3 Study 1"), a letter with dotted parts ("C.3 Additional Robustness Checks"), and a
# single letter or a Roman numeral followed by a period or "|" ("B. Data", "III. Results"). A lone capital
# and a space ("A Theory of ...") and a year ("1998 was ...") are not numbers.
OTHER_SECTION_NUMBER_RE = re.compile(
    r"^\s*(?P<num>[1-9]\d?(?:\.\d{1,2})*(?=\s+[^\W\d_])|[A-Z](?:\.\d{1,2})+(?=\.?\s)|(?:[A-Z]|[IVXLCDM]+)(?=\.\s|\s*\|))"
    r"(?:\.?\s+|\s*\|\s*)(?P<title>\S.*?)\s*$"
)
NAMED_NUMBERED_STUDY_RE = re.compile(
    r"^(?:(?:vignette|field|laboratory|lab|online) experiment|(?:field|survey|laboratory|lab) study)$", re.I
)
STATISTICAL_TABLE_ROW_RE = re.compile(
    r"^\s*(?:[Ff]\s*\(\s*\d+\s*,\s*\d+\s*\)|(?:χ|chi(?:-square)?|[-−]?\s*2\s*LL)\b)",
    re.I,
)


def heading_parts(text: str) -> tuple[str, str | None]:
    match = SECTION_NUMBER_RE.match(text)
    if match:
        return match.group(2).strip(), match.group(1)
    return text.strip(), None


def split_section_number(text: str) -> tuple[str | None, str]:
    """(number, title) for any numbering `SECTION_NUMBER_RE` or `OTHER_SECTION_NUMBER_RE` reads; (None, text)."""
    match = SECTION_NUMBER_RE.match(text)
    if match:
        return match.group(1), match.group(2).strip()
    match = OTHER_SECTION_NUMBER_RE.match(text)
    if match:
        return match.group("num"), match.group("title").strip()
    return None, text.strip()


def heading_role(text: str, number: str | None = None) -> str:
    text, section_number = heading_parts(text)
    section_number = section_number or number
    if STUDY_RE.match(text) or PILOT_STUDY_RE.match(text):
        return "study"
    if section_number is not None and NAMED_NUMBERED_STUDY_RE.fullmatch(text):
        return "study"
    for pattern, role in HEADING_ROLES:
        if pattern.match(text.strip()):
            return role
    return "unclassified"


def is_heading_text(text: str, number: str | None = None) -> bool:
    value = text.strip()
    if not value or len(value) > 150 or value.endswith((".", ";", ",", "?", "!")):
        return False
    if STATISTICAL_TABLE_ROW_RE.match(value):
        return False
    return heading_role(value, number) != "unclassified"


def study_label(text: str) -> tuple[str, str] | None:
    """Return (kind, normalized label) for a study heading, e.g. ("study", "S2")."""
    value, _ = heading_parts(text.strip())
    m = STUDY_RE.match(value)
    if m:
        return "study", ("S" if m.group(1).lower() == "study" else "E") + m.group(2).upper()
    m = PILOT_STUDY_RE.match(value)
    if m:
        kind = "pretest" if value.lower().startswith("pretest") else "pilot"
        return kind, ("PRE" if kind == "pretest" else "PILOT") + (m.group(1) or "").upper()
    return None
