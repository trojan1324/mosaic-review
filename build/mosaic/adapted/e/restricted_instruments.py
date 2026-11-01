"""Instrument families whose items are never marked citable (adapted from E/evoke/restricted_instruments.py).

Kept unchanged from E (REUSE_LEDGER R024): Eric's 2026-08-03 decision to
withhold items of the commercially licensed families below, the family table,
whole-word matching for acronyms of six characters or fewer, the disambiguation
words for ambiguous acronyms, and fail-closed matching over the label and every
alias. Changed: `apply_display_rights` is dropped; the Package->EVOKE
projection records a citability decision per item instead of stripping text,
and it calls `restricted_holder` on the raw catalog row. E's served path runs
the check after `_project` has cut the aliases to six, which lets
"Emotional exhaustion scale" (its seventh alias names the Maslach Burnout
Inventory) through; checking the raw row closes that gap (FINAL v2 W69,
audit M-41). Also changed: the whole-word boundary no longer requires a
non-letter after a fragment that itself ends in a hyphen (`mbi-`, `bdi-`);
it was refusing every real hyphenated form (MBI-GS, BDI-II, ...) of the only
two families ever printed that way (D-056). Fragments ending in
some other non-letter (e.g. `pcl-5`) are unaffected and keep the boundary.
"""

from __future__ import annotations

import re
from typing import Any

RESTRICTED_FAMILIES: dict[str, str] = {
    # Mind Garden
    "maslach": "Mind Garden",
    "mbi-": "Mind Garden",
    "multifactor leadership": "Mind Garden",
    "mlq": "Mind Garden",
    "state-trait anxiety": "Mind Garden",
    "stai": "Mind Garden",
    # Pearson
    "mmpi": "Pearson",
    "beck depression": "Pearson",
    "bdi-": "Pearson",
    "wais": "Pearson",
    "wisc": "Pearson",
    "watson-glaser": "Pearson",
    # PAR
    "neo personality inventory": "PAR",
    "neo-pi": "PAR",
    "neo pi-r": "PAR",
    # The Myers-Briggs Company / CPP
    "myers-briggs": "The Myers-Briggs Company",
    "mbti": "The Myers-Briggs Company",
    "california psychological inventory": "The Myers-Briggs Company",
    "strong interest inventory": "The Myers-Briggs Company",
    "firo-b": "The Myers-Briggs Company",
    # MHS
    "eq-i": "MHS",
    "bar-on emotional": "MHS",
    "msceit": "MHS",
    "conners": "MHS",
    # Other
    "16pf": "PSI",
    "hogan personality": "Hogan Assessments",
    "wonderlic": "Wonderlic",
    "pcl-5": "US Department of Veterans Affairs (restricted use)",
}

_ACRONYM_MAX = 6

_DISAMBIGUATE: dict[str, tuple[str, ...]] = {
    "mlq": ("leadership", "transformational", "transactional"),
    "stai": ("anxiety", "spielberger"),
    "bdi-": ("depression", "beck"),
    "wais": ("intelligence", "wechsler"),
    "wisc": ("intelligence", "wechsler"),
}

# Printed forms that name the family without a disambiguation word (D-056): the Beck Depression
# Inventory's own editions and short forms. Any other "BDI-" acronym still needs "depression" or "beck".
_KNOWN_FORMS: dict[str, re.Pattern[str]] = {
    "bdi-": re.compile(r"(?<![a-z])bdi-(?:ii|ia|i|sf|fs|pc)(?![a-z])"),
}


def restricted_holder(measure: dict[str, Any]) -> str | None:
    """The rights holder if this measure belongs to a restricted family, else None."""
    haystack = " ".join(
        [str(measure.get("label") or "")]
        + [str(alias) for alias in measure.get("aliases") or []]
    ).casefold()
    for fragment, holder in RESTRICTED_FAMILIES.items():
        if len(fragment) <= _ACRONYM_MAX:
            # A fragment that itself ends in a hyphen (e.g. "mbi-") already supplies its own right boundary;
            # requiring the next character to not be a letter would refuse every real hyphenated form
            # (MBI-GS, BDI-II, ...), which is the only way these fragments are ever printed. Fragments that
            # end in some other non-letter (e.g. "pcl-5") are unaffected and keep the trailing boundary.
            trailing = "" if fragment.endswith("-") else r"(?![a-z])"
            if not re.search(rf"(?<![a-z]){re.escape(fragment)}{trailing}", haystack):
                continue
        elif fragment not in haystack:
            continue
        required = _DISAMBIGUATE.get(fragment)
        known = _KNOWN_FORMS.get(fragment)
        if required and not any(term in haystack for term in required) and not (known and known.search(haystack)):
            continue
        return holder
    return None
