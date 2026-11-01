"""Deterministic, explicitly allowlisted candidate export; never publication approval."""
from __future__ import annotations

from hashlib import sha256
import io
import json
from pathlib import Path
import zipfile

from mosaic.canonical import canonical_bytes
from mosaic.public_release import _safe_relative, _excluded, _has_symlink_component, PRIVATE_ABSOLUTE
from mosaic.secrets_scan import scan_text
from mosaic.workspace import Workspace


def export_candidate(spec: dict, source_root: Path, destination: Path, *, workspace: Workspace) -> dict:
    """Verify every listed source before writing a deterministic source-only ZIP.

    A caller supplies reviewed paths *and* expected hashes. There is no directory
    discovery, automatic dependency claim, Git history, private corpus or license
    inference. Fonts may be included only as individually pinned WOFF2 files.
    """
    if set(spec) != {"version", "members"} or spec["version"] != "public-export-list/v1":
        raise ValueError("public_export_spec_invalid")
    if not isinstance(spec["members"], list) or not spec["members"]:
        raise ValueError("public_export_members_missing")
    source_root = source_root.absolute()
    # An allowlist is scoped to this workspace, never authority to read another
    # directory by supplying its files' names and hashes.
    if source_root != workspace.repo_root.absolute():
        raise ValueError("public_export_source_root_refused")
    try:
        checked_root = workspace.check(source_root)
    except PermissionError:
        raise ValueError("public_export_source_root_refused") from None
    if checked_root != source_root or _has_symlink_component(source_root, ""):
        raise ValueError("public_export_source_root_refused")
    files, rows, seen = {}, [], set()
    for member in spec["members"]:
        if set(member) != {"source_path", "public_path", "sha256", "rights"}:
            raise ValueError("public_export_member_invalid")
        src = _safe_relative(member["source_path"], field="source_path")
        dst = _safe_relative(member["public_path"], field="public_path")
        if not src or not dst or _excluded(src) or _excluded(dst):
            raise ValueError("public_export_path_refused")
        if dst.casefold() in seen or dst == "EXPORT-MANIFEST.json":
            raise ValueError("public_export_duplicate_path")
        seen.add(dst.casefold())
        if _has_symlink_component(source_root, src):
            raise ValueError("public_export_link_refused")
        path = source_root / src
        if not path.is_file() or not path.resolve().is_relative_to(source_root.resolve()):
            raise ValueError("public_export_source_refused")
        data = path.read_bytes()
        if sha256(data).hexdigest() != member["sha256"]:
            raise ValueError("public_export_source_changed:" + src)
        if dst.endswith(".woff2"):
            if not data.startswith(b"wOF2") or len(data) > 2 * 1024 * 1024:
                raise ValueError("public_export_font_invalid")
        else:
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                raise ValueError("public_export_binary_refused:" + src) from None
            if scan_text(text, dst):
                raise ValueError("public_export_secret_refused:" + src)
            if PRIVATE_ABSOLUTE.search(text):
                raise ValueError("public_export_private_path_refused:" + src)
        if member["rights"] not in ("author_license_pending", "third_party_notice_review_pending"):
            raise ValueError("public_export_rights_state_invalid")
        files[dst] = data
        rows.append({"path": dst, "sha256": member["sha256"], "bytes": len(data),
                     "rights": member["rights"]})
    manifest = {"version": "mosaic-public-export/v1", "members": sorted(rows, key=lambda x: x["path"]),
                "source_list_sha256": sha256(canonical_bytes(spec)).hexdigest(),
                "private_git_history_included": False, "publication_approved": False,
                "license_status": "author_selection_and_third_party_review_pending"}
    files["EXPORT-MANIFEST.json"] = canonical_bytes(manifest)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 29, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    data = buffer.getvalue()
    workspace.write_once(destination, data)
    return {"version": "mosaic-public-export-receipt/v1", "archive_sha256": sha256(data).hexdigest(),
            "payload_files": len(rows), "manifest_sha256": sha256(files["EXPORT-MANIFEST.json"]).hexdigest(),
            "publication_approved": False, "cold_start_validated": False}


def verify_export(root: Path, *, reject_unlisted: bool = False) -> dict:
    """Verify immutable exported members; runtime files live outside this inventory."""
    manifest = json.loads((root / "EXPORT-MANIFEST.json").read_bytes())
    if manifest.get("version") != "mosaic-public-export/v1":
        raise ValueError("public_export_manifest_invalid")
    seen = set()
    for row in manifest["members"]:
        path = _safe_relative(row["path"], field="path")
        if not path or _excluded(path) or path.casefold() in seen or _has_symlink_component(root, path):
            raise ValueError("public_export_manifest_path_invalid")
        seen.add(path.casefold())
        data = (root / path).read_bytes()
        if len(data) != row["bytes"] or sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("public_export_member_changed:" + path)
    if reject_unlisted:
        # Runtime output and an explicitly supplied offline wheel directory are
        # separate from executable release members. Never ignore extra code in
        # the import or frontend trees.
        generated_roots = {".venv-public", "run", "example-data", "wheelhouse"}
        pending = [root]
        while pending:
            directory = pending.pop()
            for entry in directory.iterdir():
                relative = entry.relative_to(root)
                if relative.parts[0] in generated_roots or entry.name == "__pycache__":
                    continue
                if entry.is_symlink() or getattr(entry, "is_junction", lambda: False)():
                    raise ValueError("public_export_unlisted_link")
                if entry.is_dir():
                    pending.append(entry)
                elif relative.as_posix() != "EXPORT-MANIFEST.json" and relative.as_posix().casefold() not in seen:
                    raise ValueError("public_export_unlisted_file:" + relative.as_posix())
    return {"verified_members": len(seen), "publication_approved": False}
