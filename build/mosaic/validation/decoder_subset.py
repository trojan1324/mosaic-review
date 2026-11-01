"""Model of what the pinned constrained decoder actually enforces (audit M-32).

The KB's decoder stack (lm-format-enforcer 0.11.3 behind vLLM, per the
2026-08 pilot audit and FINAL v2 §11.1) silently ignores `uniqueItems`,
`maxItems: 0`, numeric bounds, `patternProperties`, `format` and
`if`/`then`/`else`. `decoder_view` strips exactly those keywords so tests can
show a document the decoder would accept that the deterministic validator in
code still rejects. (The decoder also forces `additionalProperties: false`;
that makes it stricter, not weaker, so it is not modelled as a gap.)
"""

from __future__ import annotations

import copy
from typing import Any

IGNORED_KEYWORDS = ("uniqueItems", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                    "multipleOf", "patternProperties", "format", "if", "then", "else")


def decoder_view(schema: Any) -> Any:
    schema = copy.deepcopy(schema)

    def strip(node: Any) -> Any:
        if isinstance(node, dict):
            for key in IGNORED_KEYWORDS:
                node.pop(key, None)
            if node.get("maxItems") == 0:
                node.pop("maxItems")
            # patternProperties removed: keys it governed are no longer
            # constrained, and additionalProperties:false would now reject
            # every key, so the decoder model relaxes it where it depended on
            # patternProperties (the decoder emits free-form keys there).
            if node.get("additionalProperties") is False and "properties" not in node:
                node.pop("additionalProperties")
            for value in node.values():
                strip(value)
        elif isinstance(node, list):
            for item in node:
                strip(item)
        return node

    return strip(schema)
