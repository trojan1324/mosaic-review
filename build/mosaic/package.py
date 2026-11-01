"""Content-addressed article packages (binding C17).

A package version is a directory packages/<work_id>/<package_version>/ holding
manifest.json and objects/<type>-<sha256>.json. Every object is validated
(JSON Schema plus the deterministic code validator) when it is added; its id
is derived from its canonical bytes; nothing already written is ever changed.
Renders and other regenerable sidecars live under packages/<work_id>/renders/
(git-ignored).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mosaic import REPO_ROOT, __version__
from mosaic.canonical import OBJECT_TYPES, canonical_bytes, loads_canonical, object_id, parse_object_id, sha256_hex
from mosaic.validation import registry
from mosaic.workspace import Workspace

PACKAGES_ROOT = REPO_ROOT / "packages"


class PackageValidationError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def actor(kind: str, module: str, name: str | None = None) -> dict[str, str]:
    out = {"kind": kind, "module": module, "version": __version__}
    if name:
        out["name"] = name
    return out


@dataclass
class PackageBuilder:
    work_id: str
    recipe_id: str
    workspace: Workspace = field(default_factory=Workspace)
    objects: dict[str, dict[str, Any]] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    root: Path = PACKAGES_ROOT

    @property
    def package_id(self) -> str:
        return f"package:{self.work_id}"

    def envelope(self, tag: str, status: str, created_by: dict[str, str], provenance_ids: list[str] | None = None) -> dict[str, Any]:
        return {
            "type": tag,
            "schema_version": f"mosaic.{tag}/1",
            "source_package_id": self.package_id,
            "created_by": created_by,
            "recipe_id": self.recipe_id,
            "provenance_ids": provenance_ids or [],
            "status": status,
        }

    def add(self, body: dict[str, Any]) -> str:
        tag = body.get("type")
        if tag not in OBJECT_TYPES or tag == "manifest":
            raise PackageValidationError(f"cannot add object of type {tag!r}")
        errors = registry.validate(tag, body)
        if errors:
            raise PackageValidationError(f"{tag} failed validation: {errors[:5]}")
        oid = object_id(tag, body)
        if oid not in self.objects:
            self.objects[oid] = body
            self.order.append(oid)
        return oid

    def get(self, oid: str) -> dict[str, Any]:
        return self.objects[oid]

    def of_type(self, tag: str) -> list[tuple[str, dict[str, Any]]]:
        return [(oid, self.objects[oid]) for oid in self.order if oid.startswith(tag + "-") and self.objects[oid]["type"] == tag]

    def renders_dir(self) -> Path:
        return self.root / self.work_id / "renders"

    def next_version(self) -> str:
        root = self.root / self.work_id
        existing = sorted(p.name for p in root.glob("v[0-9][0-9][0-9][0-9]") if p.is_dir()) if root.exists() else []
        return f"v{(int(existing[-1][1:]) + 1) if existing else 1:04d}"

    def materialize(self, manifest_fields: dict[str, Any], version: str | None = None) -> tuple[Path, str]:
        """Write objects and manifest; returns (package dir, manifest id)."""
        version = version or self.next_version()
        target = self.root / self.work_id / version
        if (target / "manifest.json").exists():
            raise PackageValidationError(f"package version {version} already exists; versions are immutable")
        grouped: dict[str, list[str]] = {}
        for oid in self.order:
            grouped.setdefault(parse_object_id(oid)[0], []).append(oid)
            self.workspace.write_once(target / "objects" / f"{oid}.json", canonical_bytes(self.objects[oid]))
        manifest = {
            "type": "manifest",
            "schema_version": "mosaic.manifest/1",
            "package_id": self.package_id,
            "work_id": self.work_id,
            "package_version": version,
            "objects": grouped,
            "created_utc": utc_now(),
            **manifest_fields,
        }
        errors = registry.validate("manifest", manifest)
        if errors:
            raise PackageValidationError(f"manifest failed validation: {errors[:5]}")
        data = canonical_bytes(manifest)
        self.workspace.write_once(target / "manifest.json", data)
        return target, f"manifest-{sha256_hex(data)}"


@dataclass
class LoadedPackage:
    root: Path
    manifest: dict[str, Any]
    manifest_id: str
    objects: dict[str, dict[str, Any]]

    def of_type(self, tag: str) -> list[tuple[str, dict[str, Any]]]:
        return [(oid, self.objects[oid]) for oid in self.manifest["objects"].get(tag, [])]


def load_package(root: Path, verify: bool = True) -> LoadedPackage:
    data = (root / "manifest.json").read_bytes()
    manifest = loads_canonical(data)
    objects: dict[str, dict[str, Any]] = {}
    for tag, ids in manifest["objects"].items():
        for oid in ids:
            raw = (root / "objects" / f"{oid}.json").read_bytes()
            body = loads_canonical(raw)
            if verify and object_id(tag, body) != oid:
                raise PackageValidationError(f"{oid}: content hash does not match its identifier")
            objects[oid] = body
    return LoadedPackage(root=root, manifest=manifest, manifest_id=f"manifest-{sha256_hex(data)}", objects=objects)


def latest_version(work_id: str, packages_root: Path = PACKAGES_ROOT) -> Path | None:
    root = packages_root / work_id
    if not root.exists():
        return None
    versions = sorted(p for p in root.glob("v[0-9][0-9][0-9][0-9]") if (p / "manifest.json").exists())
    return versions[-1] if versions else None


def dump_pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True)
