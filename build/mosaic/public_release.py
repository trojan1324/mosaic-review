"""Fail-closed inventory verification for a future sanitized public release.

This module validates a *listed* candidate export.  It never discovers files,
copies a repository, reads Git history, or opens excluded trees.  A passing
mechanics result means only that the listed bytes and paths are internally
consistent.  Publication remains blocked until rights, licence, dependency,
SBOM and NOTICE requirements are explicitly satisfied.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from mosaic.canonical import canonical_bytes
from mosaic.secrets_scan import scan_text
from mosaic.workspace import Workspace


INVENTORY_VERSION = "mosaic-public-release-inventory/v1"
RIGHTS_CLASSES = frozenset({"code", "documentation", "test", "fixture", "font", "model_metadata", "adapted_code"})
LICENSE_STATES = frozenset({"approved", "pending", "not_applicable"})
EXCLUDED_TOP_LEVEL = frozenset({".git", ".venv", ".venv-windows", "scratch", "output", "sealed", "references", "ledgers", "runs", "diagnostic", "_holdback", "assets", "data", "corpus", "bibliography"})
EXCLUDED_EXACT = frozenset({".env", "config/dev_sources.json", "config/e1_reference_manifest_v1.json", "config/p4_v1.json"})
WINDOWS_DEVICE_NAMES = frozenset({"con", "prn", "aux", "nul", *(f"com{number}" for number in range(1, 10)), *(f"lpt{number}" for number in range(1, 10))})
PRIVATE_ABSOLUTE = re.compile(r"(?:[A-Za-z]:[\\/](?:Users[\\/]|wsl-home[\\/])|\\\\wsl\$|fi[l]e://)", re.I)
SPDX_IDENTIFIERS = frozenset({"Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "CC-BY-4.0", "CC0-1.0", "GPL-3.0-only", "LGPL-3.0-only", "MIT", "MPL-2.0", "Unlicense"})
TOP_LEVEL_KEYS = frozenset({"version", "mode", "members", "dependency_closure", "release"})
MEMBER_KEYS = frozenset({"source_path", "public_path", "sha256", "rights_classification", "license_state"})
CLOSURE_KEYS = frozenset({"status", "reason"})
RELEASE_KEYS = frozenset({"spdx_license", "notice_member", "sbom_member"})


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_relative(value: object, *, field: str) -> str | None:
    if not isinstance(value, str) or not value or "\\" in value or value.startswith("/") or ":" in value:
        return None
    if any(ord(char) < 32 or char in '<>"|?*' for char in value):
        return None
    raw_parts = value.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        return None
    # Win32 normalizes these suffixes, so they can turn distinct manifest paths
    # into the same file on extraction.
    if any(part != part.rstrip(" .") for part in raw_parts):
        return None
    if any(part.split(".", 1)[0].casefold() in WINDOWS_DEVICE_NAMES for part in raw_parts):
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return None
    return path.as_posix()


def _excluded(path: str) -> bool:
    pieces = tuple(part.casefold() for part in PurePosixPath(path).parts)
    lowered = path.casefold()
    if not pieces or pieces[0] in EXCLUDED_TOP_LEVEL:
        return True
    if any(part.startswith(".env") or part in {"diagnostic", "_holdback"} for part in pieces):
        return True
    if lowered in EXCLUDED_EXACT:
        return True
    if pieces[0] == "config":
        # This foundation supports only the deliberately public example and
        # public-release inventory configuration.  It must not read scope,
        # source, freeze, reference, or protocol registries while auditing an
        # export candidate.
        return not (lowered in {"config/mosaic.public.example.toml", "config/mosaic.corpus.example.toml"}
                    or (len(pieces) == 2 and pieces[1].startswith("public_") and pieces[1].endswith(".json")))
    return pieces[0] == "docs" and (len(pieces) > 1 and pieces[1] in {"inputs", "human_tasks", "evidence"})


def _has_symlink_component(root: Path, relative: str) -> bool:
    current = root
    if current.is_symlink() or getattr(current, "is_junction", lambda: False)():
        return True
    for piece in PurePosixPath(relative).parts:
        current = current / piece
        if current.is_symlink() or getattr(current, "is_junction", lambda: False)():
            return True
    return False


def _finding(code: str, **details: object) -> dict[str, object]:
    return {"code": code, **details}


def verify_inventory(inventory: dict[str, Any], source_root: Path | str) -> dict[str, Any]:
    """Verify an explicit export inventory without copying or scanning unlisted files."""
    root = Path(source_root).resolve()
    findings: list[dict[str, object]] = []
    if not isinstance(inventory, dict):
        raise ValueError("public_release_inventory_not_object")
    unknown_top = sorted(set(inventory) - TOP_LEVEL_KEYS)
    if unknown_top:
        findings.append(_finding("inventory_unknown_fields", fields=unknown_top))
    if inventory.get("version") != INVENTORY_VERSION:
        findings.append(_finding("inventory_version_invalid"))
    mode = inventory.get("mode")
    if mode not in {"candidate", "release"}:
        findings.append(_finding("inventory_mode_invalid"))
    members = inventory.get("members")
    if not isinstance(members, list):
        findings.append(_finding("members_invalid"))
        members = []

    public_seen: dict[str, str] = {}
    source_seen: dict[str, str] = {}
    checked: list[dict[str, object]] = []
    mechanics_codes = {
        "inventory_version_invalid", "inventory_mode_invalid", "inventory_unknown_fields", "members_invalid", "members_empty", "member_not_object", "member_unknown_fields",
        "source_path_invalid", "public_path_invalid", "excluded_source_path", "excluded_public_path", "duplicate_source_path", "source_path_case_collision",
        "duplicate_public_path", "public_path_case_collision", "source_missing_or_not_file", "source_symlink_refused",
        "source_escape_refused", "sha256_missing_or_invalid", "sha256_mismatch", "secret_pattern_refused",
        "private_absolute_path_refused", "binary_or_unscannable_text_refused",
    }

    for index, member in enumerate(members):
        if not isinstance(member, dict):
            findings.append(_finding("member_not_object", index=index))
            continue
        unknown_member = sorted(set(member) - MEMBER_KEYS)
        if unknown_member:
            findings.append(_finding("member_unknown_fields", index=index, fields=unknown_member))
            continue
        source_path = _safe_relative(member.get("source_path"), field="source_path")
        public_path = _safe_relative(member.get("public_path"), field="public_path")
        if source_path is None:
            findings.append(_finding("source_path_invalid", index=index))
            continue
        if public_path is None:
            findings.append(_finding("public_path_invalid", source_path=source_path))
            continue
        if _excluded(source_path):
            findings.append(_finding("excluded_source_path", source_path=source_path))
            continue
        if _excluded(public_path):
            findings.append(_finding("excluded_public_path", public_path=public_path))
            continue
        collision = public_path.casefold()
        if collision in public_seen:
            code = "duplicate_public_path" if public_seen[collision] == public_path else "public_path_case_collision"
            findings.append(_finding(code, public_path=public_path, other_public_path=public_seen[collision]))
            continue
        public_seen[collision] = public_path
        source_collision = source_path.casefold()
        if source_collision in source_seen:
            code = "duplicate_source_path" if source_seen[source_collision] == source_path else "source_path_case_collision"
            findings.append(_finding(code, source_path=source_path, other_source_path=source_seen[source_collision]))
            continue
        source_seen[source_collision] = source_path
        expected = member.get("sha256")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            findings.append(_finding("sha256_missing_or_invalid", source_path=source_path))
            continue
        target = root / PurePosixPath(source_path)
        try:
            resolved = target.resolve(strict=True)
        except OSError:
            findings.append(_finding("source_missing_or_not_file", source_path=source_path))
            continue
        if _has_symlink_component(root, source_path):
            findings.append(_finding("source_symlink_refused", source_path=source_path))
            continue
        if not resolved.is_relative_to(root) or not resolved.is_file():
            findings.append(_finding("source_escape_refused", source_path=source_path))
            continue
        data = resolved.read_bytes()
        actual = _sha256(data)
        if actual != expected:
            findings.append(_finding("sha256_mismatch", source_path=source_path, expected=expected, actual=actual))
            continue
        rights = member.get("rights_classification")
        license_state = member.get("license_state")
        if rights not in RIGHTS_CLASSES:
            findings.append(_finding("rights_classification_unknown", source_path=source_path))
        if license_state not in LICENSE_STATES:
            findings.append(_finding("license_state_unknown", source_path=source_path))
        elif license_state != "approved":
            findings.append(_finding("license_state_not_approved", source_path=source_path, state=license_state))
        text: str | None
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            findings.append(_finding("binary_or_unscannable_text_refused", source_path=source_path))
            text = None
        if text is not None:
            secret_findings = scan_text(text, public_path)
            if secret_findings:
                findings.append(_finding("secret_pattern_refused", source_path=source_path, finding_count=len(secret_findings)))
            if PRIVATE_ABSOLUTE.search(text):
                findings.append(_finding("private_absolute_path_refused", source_path=source_path))
        checked.append({
            "source_path": source_path,
            "public_path": public_path,
            "sha256": actual,
            "size": len(data),
            "rights_classification": rights,
            "license_state": license_state,
        })

    closure = inventory.get("dependency_closure")
    if not isinstance(closure, dict) or closure.get("status") not in {"complete", "example_only"}:
        findings.append(_finding("dependency_closure_missing_or_invalid"))
    elif set(closure) - CLOSURE_KEYS:
        findings.append(_finding("dependency_closure_unknown_fields", fields=sorted(set(closure) - CLOSURE_KEYS)))
    elif closure["status"] != "complete":
        findings.append(_finding("dependency_closure_incomplete"))
    else:
        # This foundation does not resolve imports, package metadata, binaries or frontend assets.
        # A self-asserted complete flag is never dependency evidence.
        findings.append(_finding("dependency_closure_external_proof_not_implemented"))

    if not checked:
        findings.append(_finding("members_empty"))

    if mode == "release":
        release = inventory.get("release")
        if not isinstance(release, dict):
            findings.append(_finding("release_metadata_missing"))
            release = {}
        elif set(release) - RELEASE_KEYS:
            findings.append(_finding("release_unknown_fields", fields=sorted(set(release) - RELEASE_KEYS)))
        spdx = release.get("spdx_license")
        if not isinstance(spdx, str) or spdx not in SPDX_IDENTIFIERS:
            findings.append(_finding("release_spdx_missing_or_invalid"))
        public_paths = {entry["public_path"] for entry in checked}
        for field in ("notice_member", "sbom_member"):
            value = release.get(field)
            if not isinstance(value, str) or value not in public_paths:
                findings.append(_finding("release_required_member_missing", field=field))

    # This module does not inspect public Git history or independently establish
    # licences, rights, NOTICE/SBOM accuracy, or transitive dependencies.
    findings.append(_finding("publication_history_gate_not_implemented"))
    findings.sort(key=lambda value: json.dumps(value, sort_keys=True))
    mechanics_valid = not any(finding["code"] in mechanics_codes for finding in findings)
    return {
        "version": "mosaic-public-release-verification/v1",
        "inventory_sha256": _sha256(canonical_bytes(inventory)),
        "mode": mode,
        "checked_members": sorted(checked, key=lambda value: (str(value["public_path"]).casefold(), str(value["public_path"]))),
        "findings": findings,
        "package_mechanics_valid": mechanics_valid,
        "publication_status": "BLOCKED",
        "publication_publishable": False,
        "copied_files": 0,
        "history_scanned": False,
    }


def load_inventory(path: Path | str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("public_release_inventory_not_object")
    return value


def verify_inventory_file(path: Path | str, source_root: Path | str) -> dict[str, Any]:
    return verify_inventory(load_inventory(path), source_root)


def write_verification(report: dict[str, Any], output: Path | str, *, workspace: Workspace | None = None) -> Path:
    """Write a verification receipt only; this module intentionally never exports members."""
    return (workspace or Workspace()).write_canonical_json(output, report)
