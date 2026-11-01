"""Register a source artifact after verifying its hash (FINAL v2 §3.1; HANDOFF §2, §6.8)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pymupdf

from mosaic.package import PackageBuilder, actor

MEDIA = {".pdf": "application/pdf",
         ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
         ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
         ".txt": "text/plain", ".csv": "text/csv"}


class SourceHashMismatch(ValueError):
    pass


def register_source(
    builder: PackageBuilder,
    path: Path,
    expected_sha256: str,
    *,
    role: str,
    source_label: str,
    relation_to_published: str,
    external_evaluation: str,
    identity_basis: str = "catalog_id",
    doi: str | None = None,
    title: str | None = None,
    page_count: int | None = None,
    sensitive_data: str = "none_known",
    version_evidence: str | None = None,
    justification: str | None = None,
    portable_locator: str | None = None,
) -> tuple[str, bytes]:
    if portable_locator is not None:
        # Public exports bind source identity to content and a portable locator.
        # Existing private callers retain their historical acquisition path.
        from pathlib import PurePosixPath
        locator = PurePosixPath(portable_locator)
        if (not portable_locator or locator.is_absolute() or ".." in locator.parts
                or "\\" in portable_locator or ":" in portable_locator):
            raise ValueError("invalid_portable_source_locator")
    data = Path(path).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise SourceHashMismatch(f"{path}: sha256 {actual} != expected {expected_sha256}; refusing to process")
    if page_count is None and Path(path).suffix.lower() == ".pdf":
        # Counted independently of capture so coverage can reconcile every physical page (M-02).
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            page_count = doc.page_count
    body = builder.envelope("source", "registered", actor("deterministic", "mosaic.capture.register"))
    body.update(
        artifact={"sha256": actual, "bytes": len(data), "media_type": MEDIA[Path(path).suffix.lower()],
                  "original_name": Path(path).name, "acquisition_path": portable_locator if portable_locator is not None else str(path).replace("\\", "/"),
                  "hash_verified_against": expected_sha256, "page_count": page_count},
        membership=[{"work_id": builder.work_id, "role": role, "source_label": source_label,
                     **({"justification": justification} if justification else {})}],
        work={"work_id": builder.work_id, "identity_basis": identity_basis, "doi": doi, "title": title or "",
              "match_proposal_id": None},
        version={"relation_to_published": relation_to_published, **({"evidence": version_evidence} if version_evidence else {})},
        rights={"local_processing": "allowed", "external_evaluation": external_evaluation,
                "redistribution": "private_only", "sensitive_data": sensitive_data},
    )
    return builder.add(body), data
