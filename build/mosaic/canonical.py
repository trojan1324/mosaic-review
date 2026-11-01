"""Canonical JSON and content-addressed identifiers (binding C17).

Canonical form: keys sorted, UTF-8 (no ASCII escaping), no insignificant
whitespace, NaN/Infinity rejected, lone surrogates rejected. An object's
identifier is ``<type>-`` plus the SHA-256 of the canonical bytes of its body.
The body never contains its own ``id``; loaders derive it from the filename
and verify it.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

# One tag per FINAL §4.2 row (Sources/pages split into source and page) plus
# the dependency-graph artifact added for audit M-50.
OBJECT_TYPES = (
    "manifest",
    "source",
    "page",
    "text-region",
    "table",
    "figure-equation",
    "study",
    "assertion",
    "relation",
    "witness",
    "disposition",
    "validation-event",
    "projection",
    "dependency-edge",
)


class CanonicalJSONError(ValueError):
    pass


def _check(value: Any, path: str = "$") -> None:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise CanonicalJSONError(f"non-finite number at {path}")
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalJSONError(f"non-string key at {path}")
            _check(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _check(item, f"{path}[{i}]")
    elif value is None or isinstance(value, (str, bool, int)):
        return
    else:
        raise CanonicalJSONError(f"unsupported type {type(value).__name__} at {path}")


def canonical_bytes(obj: Any) -> bytes:
    _check(obj)
    try:
        text = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        return text.encode("utf-8")
    except (ValueError, UnicodeEncodeError) as exc:  # lone surrogates, NaN
        raise CanonicalJSONError(str(exc)) from exc


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def object_id(type_tag: str, body: dict) -> str:
    if type_tag not in OBJECT_TYPES:
        raise CanonicalJSONError(f"unknown object type tag: {type_tag}")
    if "id" in body:
        raise CanonicalJSONError("object body must not contain its own id")
    if body.get("type") != type_tag:
        raise CanonicalJSONError(f"body type {body.get('type')!r} does not match tag {type_tag!r}")
    return f"{type_tag}-{sha256_hex(canonical_bytes(body))}"


def parse_object_id(identifier: str) -> tuple[str, str]:
    for tag in sorted(OBJECT_TYPES, key=len, reverse=True):
        prefix = tag + "-"
        if identifier.startswith(prefix):
            digest = identifier[len(prefix):]
            if len(digest) == 64 and all(c in "0123456789abcdef" for c in digest):
                return tag, digest
    raise CanonicalJSONError(f"not a content identifier: {identifier}")


def loads_canonical(data: bytes) -> Any:
    """Parse canonical bytes and confirm they are already canonical."""

    def _reject_constant(name: str) -> Any:
        raise CanonicalJSONError(f"non-finite constant {name}")

    try:
        obj = json.loads(data.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:  # a BOM, bad UTF-8 or malformed JSON
        raise CanonicalJSONError(f"not canonical JSON: {exc}") from exc
    if canonical_bytes(obj) != data:
        raise CanonicalJSONError("bytes are not in canonical form")
    return obj
