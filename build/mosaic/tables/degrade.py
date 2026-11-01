"""Graceful-degradation constants for tables (M1); `mosaic.tables.record` applies them.

Graceful degradation for tables before M2 (FINAL v2 D.1 M1, T05 graceful branch; audit C-05).

Every table the section tree finds by its caption becomes a `table` object
with status `unsupported`, its caption witnessed, the study its caption names
(if any) recorded as the owner proposal, and no cells. A continued or
multi-part caption is marked `unsupported_multi_page`; everything else stays
`unresolved`. Each table also gets a `pending` region disposition, so coverage
shows that its values were not interpreted. The table's text is still
captured as ordinary blocks, but no value from an uninterpreted table can be
accepted: G5's study-range check and G7's human confirmation stand in the way,
and M2's table module owns interpretation (count_role, caption ownership,
value-to-condition).
"""

from __future__ import annotations

import re

from mosaic.inventory.section_tree import FLOAT_NUMBER, FLOAT_PREFIX, TABLE_WORD

# A table float's label, in the section tree's float-label grammar (FLOAT_LABEL, D-151) with the table words, so
# every table caption the tree detects is a table here: "Table 2", "TABLE V", "Table IIA", "Table EC1", "Online
# Appendix Table IIA" (D-206). The E1 benchmark keeps frozen copies of the old pattern (TABLE_CAPTION_V1, LABEL_V1).
TABLE_CAPTION = re.compile(rf"^(?P<prefix>(?:{FLOAT_PREFIX})?)(?:{TABLE_WORD})\s*(?P<label>{FLOAT_NUMBER})")
CONTINUED = re.compile(r"\(?\bcont(?:inued|\.|'d)\b\)?", re.I)
REASON = ("table interpretation is not reactivated before M2 (audit C-05): cells, headers and value-to-condition "
          "bindings are not extracted, and no value from this table is accepted")
