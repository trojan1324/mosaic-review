"""Byte-exact serialization for constrained chat requests.

The normal form remains canonical JSON.  CORE extraction schemas get one
deliberate wire-only ordering adjustment: the discriminating ``field`` key is
emitted first within each CORE fact schema object's ``properties`` map.  Older
XGrammar versions use schema-property order to choose a ``oneOf`` branch, so
this lets the model choose the fact type before branch-specific fields.  The
parsed request remains identical.
"""

from __future__ import annotations

import json
from typing import Any

from mosaic.canonical import canonical_bytes


def _core_schema(body: dict[str, Any]) -> dict[str, Any] | None:
    response_format = body.get("response_format")
    if not isinstance(response_format, dict) or response_format.get("type") != "json_schema":
        return None
    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, dict):
        return None
    schema = json_schema.get("schema")
    if not isinstance(schema, dict):
        return None
    properties = schema.get("properties")
    return schema if isinstance(properties, dict) and "core_facts" in properties else None


def chat_request_bytes(body: dict[str, Any]) -> bytes:
    """Return the exact request bytes, with CORE discriminators first on the wire.

    Non-CORE requests return their canonical bytes unchanged.  The caller's
    body is never mutated; ledger identity is calculated from the returned
    bytes before an intent is appended.
    """
    canonical = canonical_bytes(body)
    ordered = json.loads(canonical)
    if _core_schema(ordered) is None:
        return canonical

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict) and "field" in properties:
                node["properties"] = {"field": properties["field"],
                                      **{key: value for key, value in properties.items() if key != "field"}}
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(_core_schema(ordered))
    wire = json.dumps(ordered, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    assert json.loads(wire) == json.loads(canonical)
    return wire
