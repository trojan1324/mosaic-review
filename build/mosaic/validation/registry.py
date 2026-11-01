"""Load schemas/ into a referencing Registry and expose validators by name."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Callable

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from mosaic import REPO_ROOT
from mosaic.validation import objects
from mosaic.validation.decoder_subset import decoder_view

SCHEMA_DIR = REPO_ROOT / "schemas"
BASE = "https://mosaic.local/schemas/"


@lru_cache(maxsize=1)
def _schemas() -> dict[str, dict]:
    out = {}
    for path in sorted(SCHEMA_DIR.glob("*.schema.json")):
        out[path.name[: -len(".schema.json")]] = json.loads(path.read_text(encoding="utf-8"))
    return out


def _registry(transform: Callable[[dict], dict] | None = None) -> Registry:
    resources = []
    for name, schema in _schemas().items():
        body = transform(schema) if transform else schema
        resources.append((BASE + f"{name}.schema.json", Resource(contents=body, specification=DRAFT202012)))
    return Registry().with_resources(resources)


@lru_cache(maxsize=None)
def schema_validator(name: str) -> Draft202012Validator:
    schema = _schemas()[name]
    return Draft202012Validator(schema, registry=_registry(), format_checker=FormatChecker())


@lru_cache(maxsize=None)
def decoder_validator(name: str) -> Draft202012Validator:
    """Validator with the keywords the pinned decoder ignores stripped out."""
    schema = decoder_view(_schemas()[name])
    return Draft202012Validator(schema, registry=_registry(decoder_view))


def code_validator(name: str) -> Callable[[dict], list[str]] | None:
    return objects.CODE_VALIDATORS.get(name)


def schema_names() -> list[str]:
    return list(_schemas())


def declared_type(name: str) -> str | None:
    return _schemas()[name].get("x-mosaic-type")


def validate(name: str, doc: dict) -> list[str]:
    """Full validation: JSON Schema plus the deterministic code validator."""
    errors = [f"schema: {e.message}" for e in schema_validator(name).iter_errors(doc)]
    check = code_validator(name)
    if check:
        errors += [f"code: {e}" for e in check(doc)]
    return errors


def schemas_digest() -> str:
    import hashlib

    h = hashlib.sha256()
    for path in sorted(SCHEMA_DIR.glob("*.schema.json")):
        h.update(path.name.encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


__all__ = ["schema_validator", "decoder_validator", "code_validator", "schema_names", "declared_type", "validate",
           "schemas_digest", "Path"]
