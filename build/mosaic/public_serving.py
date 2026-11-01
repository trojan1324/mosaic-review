"""Public, local-only materialization and serving for one source-bound reader.

This module deliberately materializes no scientific findings.  It projects
literal native reader units, retains a whole candidate report separately, and
uses a source-free approved frontend template with local-only APIs.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, unquote, urlparse

from mosaic.canonical import canonical_bytes
from mosaic.public_build import relative_path
from mosaic.source_assets import load_assets, pdf_bytes
from mosaic.workspace import Workspace
from scripts.research_display_server_v1 import make_display_server, page_rows
from scripts.research_search_index import ResearchSearchIndex, build_index


VERSION = "mosaic-public-serving/v1"
FRONTEND_ROOT = Path(__file__).resolve().parent.parent / "public_frontend"
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_PROSE_LANES = frozenset({"body", "references", "appendix"})
_MASKED_LANES = frozenset({"front_matter", "front_or_preabstract", "abstract_heading", "section_heading",
                           "references_heading", "appendix_heading", "table_region_candidate"})
_NATIVE_LANES = _PROSE_LANES | _MASKED_LANES
_BUNDLE_ARTIFACTS = frozenset({"search.sqlite", "candidate-report.json", "bibliography.json",
                               "citations.json", "measures.json", "raw-passages.json",
                               "default-passages.json", "source-assets.json"})
_SYNTHETIC_BANNER = ("<strong>Synthetic demonstration.</strong> Every article, citation, passage, measure and "
                     "source PDF in this local package is fictional and is not real research evidence.")
_LOCAL_BANNER = ("<strong>Local candidate display.</strong> Source review controls display admission; "
                 "it does not establish scientific validity.")


class PublicServingRefusal(ValueError):
    """Typed, non-secret refusal for invalid public-serving inputs."""


def _replace_one(value: str | bytes, old: str | bytes, new: str | bytes, code: str) -> str | bytes:
    """Apply a reviewed frontend seam exactly once, never silently drift."""
    if value.count(old) != 1:
        raise PublicServingRefusal(code)
    return value.replace(old, new)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_hash(value: object, code: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise PublicServingRefusal(code)
    return value


def _read_json(value: object, code: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (str, Path)):
        try:
            parsed = json.loads(Path(value).read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            raise PublicServingRefusal(code) from exc
        if isinstance(parsed, dict):
            return parsed
    raise PublicServingRefusal(code)


def _resolve_input(workspace: Workspace, value: object, code: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise PublicServingRefusal(code)
    try:
        path = workspace.check(Path(value))
    except PermissionError as exc:
        raise PublicServingRefusal(code) from exc
    if not path.is_file():
        raise PublicServingRefusal(code)
    return path


def _linked_or_outside(root: Path, candidate: Path) -> bool:
    """Refuse links at every component rather than copying or serving through them."""
    try:
        root = root.absolute()
        candidate = candidate.absolute()
        relative = candidate.relative_to(root)
        if root.is_symlink() or getattr(root, "is_junction", lambda: False)():
            return True
        current = root
        for part in relative.parts:
            current = current / part
            if current.is_symlink() or getattr(current, "is_junction", lambda: False)():
                return True
        return not candidate.resolve(strict=True).is_relative_to(root.resolve(strict=True))
    except (OSError, ValueError):
        return True


def _native_reader(value: object, expected_pdf_sha: str) -> dict[str, Any]:
    reader = _read_json(value, "public_native_reader_invalid")
    if reader.get("version") != "public-native-reader/v1" or reader.get("scientifically_accepted") is not False:
        raise PublicServingRefusal("public_native_reader_invalid")
    if reader.get("source_pdf_sha256") != expected_pdf_sha or not isinstance(reader.get("package_manifest_id"), str):
        raise PublicServingRefusal("public_native_reader_binding_invalid")
    units = reader.get("units")
    if not isinstance(units, list) or not units:
        raise PublicServingRefusal("public_native_reader_units_invalid")
    seen: set[str] = set()
    for ordinal, unit in enumerate(units):
        if not isinstance(unit, dict) or unit.get("ordinal") != ordinal or not isinstance(unit.get("unit_id"), str):
            raise PublicServingRefusal("public_native_reader_units_invalid")
        if unit["unit_id"] in seen or not isinstance(unit.get("quote"), str) or not isinstance(unit.get("page"), int) or unit["page"] < 1:
            raise PublicServingRefusal("public_native_reader_units_invalid")
        if (unit.get("source_pdf_sha256") != expected_pdf_sha or not isinstance(unit.get("default_visible"), bool)
                or unit.get("lane") not in _NATIVE_LANES):
            raise PublicServingRefusal("public_native_reader_binding_invalid")
        if unit.get("quote_sha256") != hashlib.sha256(unit["quote"].encode("utf-8")).hexdigest():
            raise PublicServingRefusal("public_native_reader_quote_hash_mismatch")
        if not isinstance(unit.get("start"), int) or not isinstance(unit.get("end"), int) or unit["start"] < 0 or unit["end"] < unit["start"]:
            raise PublicServingRefusal("public_native_reader_offsets_invalid")
        seen.add(unit["unit_id"])
    mask = reader.get("default_mask")
    if not isinstance(mask, dict) or mask.get("status") not in {"candidate_literal_abstract_heading", "pending_abstract_boundary"}:
        raise PublicServingRefusal("public_native_reader_default_mask_invalid")
    if mask.get("status") == "pending_abstract_boundary":
        if mask.get("anchor_unit_id") is not None or any(unit["default_visible"] for unit in units):
            raise PublicServingRefusal("public_native_reader_default_visibility_invalid")
        return reader
    if not isinstance(mask.get("anchor_unit_id"), str):
        raise PublicServingRefusal("public_native_reader_default_mask_invalid")
    try:
        anchor = next(index for index, unit in enumerate(units) if unit["unit_id"] == mask["anchor_unit_id"])
    except StopIteration as exc:
        raise PublicServingRefusal("public_native_reader_default_anchor_invalid") from exc
    if units[anchor].get("lane") != "abstract_heading":
        raise PublicServingRefusal("public_native_reader_default_anchor_invalid")
    if any(unit.get("lane") == "abstract_heading" for unit in units[:anchor]):
        raise PublicServingRefusal("public_native_reader_default_anchor_not_earliest")
    for index, unit in enumerate(units):
        # The anchor itself and every preceding capture are never default reader
        # prose.  Heading/front/table rows are retained raw but cannot opt in via
        # an untrusted boolean.
        if (index <= anchor or unit["lane"] in _MASKED_LANES) and unit["default_visible"]:
            raise PublicServingRefusal("public_native_reader_default_visibility_invalid")
    return reader


def _candidate_only(value: object) -> bool:
    if isinstance(value, dict):
        return all(not (key in {"scientifically_accepted", "production_accepted", "qualified", "complete_extraction_claimed"} and item is not False)
                   and _candidate_only(item) for key, item in value.items())
    if isinstance(value, list):
        return all(_candidate_only(item) for item in value)
    return True


def _stage_envelope(capture: Mapping[str, Any], snapshot: Mapping[str, Any], *, workspace: Workspace,
                    paper_id: str, source_sha: str, package_manifest_id: str, reader: Mapping[str, Any]) -> tuple[Path, str]:
    """Verify the concrete public-capture/runtime handoff, never report aliases."""
    capture_receipt_path = _resolve_input(workspace, capture.get("capture_receipt_path"), "public_capture_receipt_missing")
    capture_receipt_sha = _require_hash(capture.get("capture_receipt_sha256"), "public_capture_receipt_hash_invalid")
    if _sha256(capture_receipt_path) != capture_receipt_sha:
        raise PublicServingRefusal("public_capture_receipt_hash_mismatch")
    receipt = _read_json(capture_receipt_path, "public_capture_receipt_invalid")
    if receipt.get("version") != "public-capture-source-packet/v1" or receipt.get("status") != "source_packet_ready" or receipt.get("scientifically_accepted") is not False:
        raise PublicServingRefusal("public_capture_receipt_invalid")
    if receipt.get("work_id") != paper_id or receipt.get("source_pdf_sha256") != source_sha or receipt.get("package_manifest_id") != package_manifest_id:
        raise PublicServingRefusal("public_capture_receipt_binding_invalid")
    native = receipt.get("native_reader")
    if not isinstance(native, dict) or native.get("sha256") != hashlib.sha256(canonical_bytes(reader)).hexdigest() or native.get("unit_count") != len(reader.get("units", [])) or native.get("scope") != reader.get("scope"):
        raise PublicServingRefusal("public_capture_native_reader_binding_invalid")
    source_path = _resolve_input(workspace, capture.get("source_packet_path"), "public_source_packet_missing")
    source_packet_sha = _require_hash(capture.get("source_packet_sha256"), "public_source_packet_hash_invalid")
    source = _read_json(source_path, "public_source_packet_invalid")
    if hashlib.sha256(canonical_bytes(source)).hexdigest() != source_packet_sha or receipt.get("source_packet_sha256") != source_packet_sha:
        raise PublicServingRefusal("public_source_packet_hash_mismatch")
    if source.get("work_id") != paper_id:
        raise PublicServingRefusal("public_source_packet_binding_invalid")
    snapshot_path = _resolve_input(workspace, snapshot.get("snapshot_path"), "public_snapshot_missing")
    snapshot_sha = _require_hash(snapshot.get("snapshot_sha256"), "public_snapshot_hash_invalid")
    if _sha256(snapshot_path) != snapshot_sha:
        raise PublicServingRefusal("public_snapshot_hash_mismatch")
    packet = _read_json(snapshot_path, "public_snapshot_invalid")
    if packet.get("version") != "mosaic-public-snapshot/v1" or packet.get("scientifically_accepted") is not False or packet.get("validation_status") != "candidate_only" or not _candidate_only(packet):
        raise PublicServingRefusal("public_snapshot_not_candidate_only")
    paper = packet.get("paper")
    if not isinstance(paper, dict) or paper.get("paper_id") != paper_id or paper.get("source_pdf_sha256") != source_sha:
        raise PublicServingRefusal("public_snapshot_paper_binding_invalid")
    if packet.get("source_packet_candidate") != source or packet.get("source_packet_sha256") != source_packet_sha:
        raise PublicServingRefusal("public_snapshot_source_packet_mismatch")
    source_meta = packet.get("source")
    if not isinstance(source_meta, dict) or source_meta.get("units_sha256") != hashlib.sha256(canonical_bytes(source.get("units"))).hexdigest():
        raise PublicServingRefusal("public_snapshot_source_units_mismatch")
    report_path = _resolve_input(workspace, snapshot.get("report_path"), "public_snapshot_report_missing")
    report_sha = _require_hash(snapshot.get("report_sha256"), "public_snapshot_report_hash_invalid")
    if _sha256(report_path) != report_sha:
        raise PublicServingRefusal("public_snapshot_report_hash_mismatch")
    report = _read_json(report_path, "public_snapshot_report_invalid")
    if packet.get("candidate_report") != report or not _candidate_only(report):
        raise PublicServingRefusal("public_snapshot_report_binding_invalid")
    return report_path, report_sha


def _copy_frontend(workspace: Workspace, output: Path, snapshot_id: str, *, paper_count: int = 1,
                   corpus_mode: bool = False, measure_versions: list[dict[str, Any]] | None = None,
                   study_manifest: dict[str, Any] | None = None, study_files: dict[str, bytes] | None = None) -> dict[str, str]:
    if not FRONTEND_ROOT.is_dir():
        raise PublicServingRefusal("public_frontend_assets_missing")
    if _linked_or_outside(FRONTEND_ROOT, FRONTEND_ROOT):
        raise PublicServingRefusal("public_frontend_link_refused")
    files: dict[str, str] = {}
    for source in sorted(FRONTEND_ROOT.rglob("*")):
        if not source.is_file():
            continue
        if _linked_or_outside(FRONTEND_ROOT, source):
            raise PublicServingRefusal("public_frontend_link_refused")
        relative = source.relative_to(FRONTEND_ROOT)
        target = output / "static" / relative
        if relative.as_posix() == "index.html":
            text = source.read_text(encoding="utf-8")
            manifest = json.dumps({"schema_version": "mosaic-platform-preview/v1", "mode": "read_only_search_api",
                                   "snapshot_id": snapshot_id, "scientifically_accepted": False,
                                   "coverage": {"paper_count": paper_count, "record_count": 0,
                                                "statement": "Source-bound public candidates; displayed identities require explicit source review."}},
                                  separators=(",", ":"))
            if text.count("__PUBLIC_MANIFEST__") != 1:
                raise PublicServingRefusal("public_frontend_manifest_seam_missing")
            text = text.replace("__PUBLIC_MANIFEST__", manifest)
            if corpus_mode:
                text = _replace_one(text, _SYNTHETIC_BANNER, _LOCAL_BANNER,
                                    "public_frontend_index_banner_seam_missing")
            workspace.write_once(target, text.encode("utf-8"))
        elif relative.as_posix() == "workspace.html":
            source_bytes = source.read_bytes()
            if source_bytes.count(b"__PUBLIC_PAPER_COUNT__") != 1:
                raise PublicServingRefusal("public_frontend_paper_count_seam_missing")
            body = source_bytes.replace(b"__PUBLIC_PAPER_COUNT__", str(paper_count).encode("ascii"))
            banner = (b"<body><p style=\"margin:0;padding:10px 24px;background:#f8f2da;font-weight:600\">"
                      + (b"Local candidate display &bull; source review controls display admission, not scientific validity"
                         if corpus_mode else b"Fictional build demonstration &bull; no real model inference or scientific validation")
                      + b"</p>")
            workspace.write_once(target, _replace_one(body, b"<body>", banner,
                                                        "public_frontend_workspace_body_seam_missing"))
        elif relative.as_posix() == "assets/research-display-v1.js":
            text = source.read_text(encoding="utf-8")
            original = "<a href=\"sources/${encodeURIComponent(r.paper_id)}.pdf${r.page>0?'#page='+r.page:''}\" target=\"_blank\" rel=\"noopener\">Open source PDF${r.page>0?' · page '+r.page:''}</a>"
            replacement = original + "<a href=\"source-viewer.html?paper=${encodeURIComponent(r.paper_id)}&page=${encodeURIComponent(r.page)}\">View source page</a>"
            if text.count(original) != 1:
                raise PublicServingRefusal("public_frontend_source_viewer_seam_missing")
            text = text.replace(original, replacement)
            original_rows = "  function pagination(total) {"
            reader_rows = """  function unreviewedReaderRows(rows) {
    return rows.map(r => `<button type=\"button\" class=\"paper\" data-title-paper=\"${esc(r.paper_id)}\"><strong>${esc(r.label || 'Unreviewed source packet')}</strong><span class=\"bibliography-authors\">No bibliographic identity is displayed.</span><span class=\"bibliography-journal\">Local source reader only</span><small>Browse literal captured passages →</small></button>`).join('');
  }
""" + original_rows
            if text.count(original_rows) != 1:
                raise PublicServingRefusal("public_frontend_reader_entry_seam_missing")
            text = text.replace(original_rows, reader_rows)
            text = _replace_one(text, "const bibliography = paperRows(rows);", "const bibliography = paperRows(rows)+unreviewedReaderRows(b.reader_entries||[]);", "public_frontend_reader_bibliography_seam_missing")
            action = "const sourceActions=S.paper?`<div class=\"actions\"><a href=\"sources/${encodeURIComponent(S.paper)}.pdf\" target=\"_blank\" rel=\"noopener\">Open source PDF</a><a href=\"citation-map.html?paper=${encodeURIComponent(S.paper)}\">Source-checked citations</a></div>`:'';"
            if corpus_mode:
                action = action.replace("</a></div>", "</a><a href=\"source-viewer.html?paper=${encodeURIComponent(S.paper)}&page=1\">View source page</a></div>")
            else:
                action = action.replace("</a></div>", "</a><a href=\"source-viewer.html?paper=${encodeURIComponent(S.paper)}&page=1\">View source page</a><span class=\"quiet\">Bibliographic identity unresolved.</span></div>")
            text = _replace_one(text, "const sourceActions=S.paper?`<div class=\"actions\"><a href=\"sources/${encodeURIComponent(S.paper)}.pdf\" target=\"_blank\" rel=\"noopener\">Open source PDF</a><a href=\"citation-map.html?paper=${encodeURIComponent(S.paper)}\">Source-checked citations</a></div>`:'';", action, "public_frontend_source_actions_seam_missing")
            workspace.write_once(target, text.encode("utf-8"))
        elif relative.as_posix() == "measures.html":
            text = source.read_text(encoding="utf-8")
            seam = r'(<script id="measure-data" type="application/json">)(.*?)(</script>)'
            matches = list(re.finditer(seam, text, flags=re.S))
            if len(matches) != 1:
                raise PublicServingRefusal("public_frontend_measure_data_seam_missing")
            try:
                data = json.loads(matches[0].group(2))
            except json.JSONDecodeError as exc:
                raise PublicServingRefusal("public_frontend_measure_data_invalid") from exc
            if (data.get("version") != "mosaic-measure-catalog/v1" or data.get("papers") != 0
                    or data.get("versions") != [] or data.get("scientifically_accepted") is not False):
                raise PublicServingRefusal("public_frontend_measure_data_not_empty")
            data["snapshot_id"] = snapshot_id
            data["api_snapshot_id"] = snapshot_id
            if corpus_mode:
                data["papers"] = paper_count
                data["versions"] = measure_versions or []
            text = text[:matches[0].start(2)] + json.dumps(data, separators=(",", ":")) + text[matches[0].end(2):]
            if corpus_mode:
                text = _replace_one(text, _SYNTHETIC_BANNER, _LOCAL_BANNER,
                                    "public_frontend_measure_banner_seam_missing")
                bibliography = {row["paper"]["paper_id"]: {"paper_id": row["paper"]["paper_id"], "title_verbatim": row["paper"]["title"], "authors_verbatim": [row["paper"]["authors"]], "journal_verbatim": row["paper"]["journal"], "publication_year": row["paper"]["year"]}
                                for row in measure_versions or []}
                seam = r'(<script id="measure-bibliography-v1" type="application/json">)(.*?)(</script>)'
                matches = list(re.finditer(seam, text, flags=re.S))
                if len(matches) != 1:
                    raise PublicServingRefusal("public_frontend_measure_bibliography_seam_missing")
                match = matches[0]
                text = text[:match.start(2)] + json.dumps(bibliography, separators=(",", ":")) + text[match.end(2):]
            workspace.write_once(target, text.encode("utf-8"))
        elif relative.as_posix() == "studies.html" and corpus_mode:
            text = source.read_text(encoding="utf-8")
            seam = r'(<script id="study-manifest" type="application/json">)(.*?)(</script>)'
            matches = list(re.finditer(seam, text, flags=re.S))
            if len(matches) != 1 or study_manifest is None:
                raise PublicServingRefusal("public_frontend_study_manifest_seam_missing")
            match = matches[0]
            if json.loads(match.group(2)).get("papers") != []:
                raise PublicServingRefusal("public_frontend_study_manifest_not_empty")
            payload = json.dumps(study_manifest, separators=(",", ":")).replace("<", "\\u003c")
            text = text[:match.start(2)] + payload + text[match.end(2):]
            banner = '<p style="margin:0;padding:10px 24px;background:#f8f2da;font-weight:600">Local candidate display &bull; source review does not establish scientific validity</p>'
            workspace.write_once(target, _replace_one(text, "<body>", "<body>" + banner, "public_frontend_study_body_seam_missing").encode("utf-8"))
        elif relative.as_posix() == "citation-map.html" and corpus_mode:
            text = source.read_text(encoding="utf-8")
            text = _replace_one(text, _SYNTHETIC_BANNER, _LOCAL_BANNER,
                                "public_frontend_citation_banner_seam_missing")
            text = _replace_one(text, "assets/citation-display-client-v1.js", "public-apa.js",
                                "public_frontend_citation_client_seam_missing")
            workspace.write_once(target, text.encode("utf-8"))
        elif relative.as_posix() == "citation-map.js":
            if corpus_mode:
                workspace.copy_file(source, target)
                files[relative.as_posix()] = _sha256(target)
                continue
            pending = """'use strict';
(() => {
  const status = document.querySelector('#citation-status');
  const results = document.querySelector('#citation-results');
  const form = document.querySelector('#citation-search');
  const message = 'Reference network is not built for this demonstration; retained source/reference passages remain available in Papers.';
  if (status) status.textContent = message;
  if (results) results.innerHTML = '<section class="cm-candidate"><p>' + message + '</p></section>';
  if (form) form.addEventListener('submit', event => { event.preventDefault(); if (status) status.textContent = message; });
})();
"""
            workspace.write_once(target, pending.encode("utf-8"))
        elif relative.suffix.casefold() == ".html":
            label = "Local candidate display &bull; source review does not establish scientific validity" if corpus_mode else "Fictional build demonstration &bull; no real model inference or scientific validation"
            banner = '<p style="margin:0;padding:10px 24px;background:#f8f2da;font-weight:600">' + label + '</p>'
            workspace.write_once(target, _replace_one(source.read_text(encoding="utf-8"), "<body>", "<body>" + banner,
                                                        "public_frontend_body_seam_missing").encode("utf-8"))
        else:
            workspace.copy_file(source, target)
        files[relative.as_posix()] = _sha256(target)
    for name, body in (study_files or {}).items():
        if not re.fullmatch(r"study-data/public-[0-9a-f]{64}\.json", name):
            raise PublicServingRefusal("public_frontend_study_path_invalid")
        target = output / "static" / relative_path(name)
        workspace.write_once(target, body)
        files[name] = _sha256(target)
    viewer = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Source page · MOSAIC</title><link rel="stylesheet" href="assets/fonts.css"><link rel="stylesheet" href="assets/tokens.css"><link rel="stylesheet" href="assets/components.css"><body><div class="rp-app"><header class="rp-band"><a class="rp-band__brand" href="workspace.html"><img src="assets/mosaic-lockup-reversed.svg" alt="MOSAIC"></a><nav class="rp-band__nav" aria-label="Main"><a class="rp-band__item" href="workspace.html">Start</a><a class="rp-band__item" href="index.html#papers">Papers</a><a class="rp-band__item" href="measures.html">Measures</a><a class="rp-band__item" href="citation-map.html">Citations</a></nav></header><main class="rp-content rp-stack rp-stack--lg"><a href="index.html#papers">← Return to passages</a><section class="rp-sign"><div class="rp-sign__arrow">↗</div><div class="rp-sign__cell rp-sign__cell--grow"><span class="rp-signlabel">Original source page</span><h1 class="rp-sign__title">Source image review</h1></div></section><p class="rp-lede">This is a locally rendered image of a hash-verified PDF page. It is not OCR or a reconstructed quotation.</p><p id="status" class="rp-caption" role="status">Loading source page…</p><p><a id="pdf" target="_blank" rel="noopener">Open original PDF bytes</a></p><img id="image" alt="Original source PDF page" style="display:block;max-width:100%;height:auto" hidden></main></div><script>(()=>{const q=new URL(location.href).searchParams,p=q.get('paper'),page=q.get('page'),s=document.querySelector('#status'),image=document.querySelector('#image');if(!p||!/^[1-9]\\d*$/.test(page||'')){s.textContent='A valid local paper and page are required.';return}document.querySelector('#pdf').href='sources/'+encodeURIComponent(p)+'.pdf#page='+page;const url='/api/display/v1/source-page?paper='+encodeURIComponent(p)+'&page='+encodeURIComponent(page);fetch(url).then(r=>{if(!r.ok)throw Error('Source page is unavailable.');return r.blob()}).then(blob=>{image.src=URL.createObjectURL(blob);image.hidden=false;s.textContent='Hash-verified local source page '+page+'.'}).catch(()=>s.textContent='Source page is unavailable.');})();</script></body></html>'''
    viewer = viewer.replace("</title>", "</title><style>.rp-band__brand img{width:164px;height:auto;display:block}.rp-band__nav{display:flex}.rp-band__item{display:flex;align-items:center;padding:0 16px}@media(max-width:720px){.rp-band__brand img{width:126px}.rp-band__item{padding:0 5px;font-size:12px}}</style>")
    old_nav = '<a class="rp-band__item" href="workspace.html">Start</a><a class="rp-band__item" href="index.html#papers">Papers</a><a class="rp-band__item" href="measures.html">Measures</a><a class="rp-band__item" href="citation-map.html">Citations</a>'
    new_nav = '<a class="rp-band__item" href="workspace.html">Start</a><a class="rp-band__item" href="measures.html">Measures</a><a class="rp-band__item" href="index.html#papers">Papers</a><a class="rp-band__item" href="measures.html#compare">Compare</a><a class="rp-band__item" href="measures.html#project">Project</a><a class="rp-band__item" href="citation-map.html">Citations</a>'
    viewer = viewer.replace(old_nav, new_nav)
    if not corpus_mode:
        viewer = viewer.replace("<body><div", "<body><p style=\"margin:0;padding:10px 24px;background:#f8f2da;font-weight:600\">Fictional build demonstration &bull; no real model inference or scientific validation</p><div")
    if corpus_mode:
        apa_script = output / "static/public-apa.js"
        workspace.write_text(apa_script, _corpus_citation_js())
        files["public-apa.js"] = _sha256(apa_script)
    viewer = viewer.replace('<a href="index.html#papers">← Return to passages</a>', '<a id="return" href="index.html#papers">← Return to passages</a>')
    viewer = viewer.replace("document.querySelector('#pdf').href=", "document.querySelector('#return').href='index.html?paper='+encodeURIComponent(p)+'#papers';document.querySelector('#pdf').href=")
    target = output / "static" / "source-viewer.html"
    workspace.write_once(target, viewer.encode("utf-8"))
    files["source-viewer.html"] = _sha256(target)
    return files




def _corpus_citation_js() -> str:
    return r'''(() => {
  'use strict';
  const $=s=>document.querySelector(s), esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  let page=1, request=0;
  const selected=new URL(location.href).searchParams.get('paper')||'';
  async function get(path){const r=await fetch(path);if(!r.ok)throw Error('Unavailable');return r.json();}
  async function load(){
    const current=++request,q=$('#citation-q').value.trim();
    const [a,c]=await Promise.all([get('/api/display/v1/apa?'+new URLSearchParams({q,paper:selected,page,page_size:25})),get('/api/citations/health')]);
    if(current!==request)return;
    const last=Math.max(1,Math.ceil(a.total/25));
    $('#formatted-citations').innerHTML=(a.items||[]).map(x=>'<article class="cm-row"><p>'+esc(x.citation_apa)+'</p></article>').join('')||'<p>No complete APA entries match this search.</p>';
    if(last>1){$('#formatted-citations').insertAdjacentHTML('beforeend','<p><button id="apa-prev" '+(page<=1?'disabled':'')+'>Previous</button> Page '+page+' of '+last+' <button id="apa-next" '+(page>=last?'disabled':'')+'>Next</button></p>');$('#apa-prev').onclick=()=>{page--;load();};$('#apa-next').onclick=()=>{page++;load();};}
    $('#citation-status').textContent=a.total+' matching complete APA entries; '+c.resolved_edge_count+' reviewed citation edges in this collection.';
    $('#citation-review-count').textContent=c.unresolved_reference_count+' retained printed references remain unresolved and outside the network.';
  }
  $('#formatted-citation-search').onsubmit=e=>{e.preventDefault();page=1;load().catch(failed);};
  function failed(){ $('#citation-status').textContent='The citation display is unavailable.'; }
  load().catch(failed);
})();'''


def _pending_page(title: str, detail: str) -> str:
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>{title} · MOSAIC</title><link rel="stylesheet" href="assets/fonts.css"><link rel="stylesheet" href="assets/tokens.css"><link rel="stylesheet" href="assets/components.css"><body><div class="rp-app"><header class="rp-band"><a class="rp-band__brand" href="workspace.html"><img src="assets/mosaic-lockup-reversed.svg" alt="MOSAIC"></a><nav class="rp-band__nav" aria-label="Main"><a class="rp-band__item" href="workspace.html">Start</a><a class="rp-band__item" href="index.html#papers">Papers</a><a class="rp-band__item" href="measures.html">Measures</a><a class="rp-band__item" href="citation-map.html">Citations</a></nav></header><main class="rp-hero"><div class="rp-stack rp-stack--lg"><span class="rp-voicelabel">Public candidate</span><h1 class="rp-display">{title}</h1><p class="rp-lede">{detail}</p></div></main></div></body></html>'''


def materialize_public(capture_receipt: Mapping[str, Any], snapshot: Mapping[str, Any], output: Path | str, *, workspace: Workspace) -> dict[str, Any]:
    """Materialize one public source into a hash-bound, unreviewed local bundle.

    ``capture_receipt`` must bind the PDF, native reader, public-capture
    receipt, and source packet. ``snapshot`` must bind the actual
    ``mosaic-public-snapshot/v1`` and raw candidate report. The report remains
    a whole artifact and is never expanded into a finding row.
    """
    output = workspace.check(output)
    if output.exists():
        raise PublicServingRefusal("public_serving_output_exists")
    source_sha = _require_hash(capture_receipt.get("source_pdf_sha256"), "public_source_sha_invalid")
    paper_id = "public-" + source_sha
    if capture_receipt.get("paper_id", paper_id) != paper_id:
        raise PublicServingRefusal("public_paper_id_mismatch")
    source_path = _resolve_input(workspace, capture_receipt.get("source_pdf_path"), "public_source_pdf_missing")
    if source_path.suffix.casefold() != ".pdf" or _sha256(source_path) != source_sha or not source_path.read_bytes().startswith(b"%PDF-"):
        raise PublicServingRefusal("public_source_pdf_hash_mismatch")
    native_value = capture_receipt.get("native_reader")
    if isinstance(native_value, (str, Path)):
        native_value = _resolve_input(workspace, native_value, "public_native_reader_missing")
    reader = _native_reader(native_value, source_sha)
    report_path, report_sha = _stage_envelope(capture_receipt, snapshot, workspace=workspace, paper_id=paper_id,
                                               source_sha=source_sha, package_manifest_id=reader["package_manifest_id"], reader=reader)
    workspace.mkdir(output)
    workspace.copy_file(source_path, output / "sources" / "source.pdf", expected_sha256=source_sha)
    workspace.copy_file(report_path, output / "candidate-report.json", expected_sha256=report_sha)
    raw_rows = []
    for unit in reader["units"]:
        raw_rows.append({"id": f"passage:{paper_id}:{unit['unit_id']}", "paper_id": paper_id, **unit,
                         "source_units_sha256": hashlib.sha256(canonical_bytes(reader["units"])).hexdigest(),
                         "package_manifest_id": reader["package_manifest_id"], "scientifically_accepted": False})
    index_input = {"records": [], "catalog": [], "source_passages": [{"paper_id": paper_id,
        "title": "Source identity awaiting review", "citation": None, "source_pdf_sha256": source_sha,
        "source_units_sha256": raw_rows[0]["source_units_sha256"],
        "passages": [{"unit_id": row["unit_id"], "text": row["quote"], "page": row["page"]} for row in raw_rows]}]}
    index_snapshot = build_index(index_input, output / "search.sqlite", workspace=workspace)
    # The generic index helper labels supplied source packets as admitted.  A
    # public caller's supplied source is instead explicitly unreviewed.
    conn = workspace.sqlite_connect(output / "search.sqlite")
    try:
        conn.execute("UPDATE papers SET source_inventory_status = 'public_unreviewed_source'")
        conn.execute("INSERT OR REPLACE INTO metadata VALUES (?, ?)", ("admitted_source_paper_count", "0"))
        conn.execute("INSERT OR REPLACE INTO metadata VALUES (?, ?)", ("public_unreviewed_source_count", "1"))
        conn.commit()
    finally:
        conn.close()
    index = ResearchSearchIndex(output / "search.sqlite")
    frontend = _copy_frontend(workspace, output, index.snapshot()["snapshot_id"])
    bibliography = {"version": "mosaic-public-bibliography/v1", "items": [], "unresolved_count": 1,
                    "reason": "Complete printed title/byline/journal/year has not been established."}
    citations = {"version": "mosaic-public-citations/v1", "items": [], "status": "pending_no_reference_graph"}
    measures = {"version": "mosaic-public-measures/v1", "items": [], "status": "pending_no_measure_materializer"}
    # The supplied unit boolean is provenance metadata only.  Default display
    # is derived from the earliest literal Abstract anchor and never exposes
    # pre-abstract units; standalone headings and table candidates remain raw
    # review material rather than reader prose.
    anchor_id = reader["default_mask"]["anchor_unit_id"]
    anchor_ordinal = next(row["ordinal"] for row in raw_rows if row["unit_id"] == anchor_id)
    hidden_lanes = {"abstract_heading", "section_heading", "references_heading", "appendix_heading", "table_region_candidate"}
    default = [row for row in raw_rows if row["ordinal"] > anchor_ordinal and row["default_visible"]
               and row.get("lane") not in hidden_lanes]
    for name, value in (("bibliography.json", bibliography), ("citations.json", citations), ("measures.json", measures),
                        ("raw-passages.json", {"items": raw_rows}), ("default-passages.json", {"items": default})):
        workspace.write_canonical_json(output / name, value)
    asset_manifest = {"version": "mosaic-source-assets/v1", "scientifically_accepted": False, "public_distribution_authorized": False,
                      "assets": [{"paper_id": paper_id, "source_pdf_sha256": source_sha,
                                  "availability": "local_file_present", "path": "sources/source.pdf"}]}
    workspace.write_canonical_json(output / "source-assets.json", asset_manifest)
    artifact_paths = ["search.sqlite", "candidate-report.json", "bibliography.json", "citations.json", "measures.json",
                      "raw-passages.json", "default-passages.json", "source-assets.json"]
    bundle = {"version": VERSION, "paper_id": paper_id, "source_pdf_sha256": source_sha,
              "native_reader_sha256": hashlib.sha256(canonical_bytes(reader)).hexdigest(), "report_sha256": report_sha,
              "artifacts": {name: _sha256(output / name) for name in artifact_paths}, "frontend": frontend,
              "allowed_source_pages": sorted({row["page"] for row in raw_rows}), "scientifically_accepted": False}
    workspace.write_canonical_json(output / "bundle.json", bundle)
    return {"bundle_path": str(output / "bundle.json"), "bundle_sha256": _sha256(output / "bundle.json"),
            "paper_id": paper_id, "snapshot_id": index.snapshot()["snapshot_id"], "scientifically_accepted": False}


def _verify_bundle(path: Path) -> tuple[dict[str, Any], Path]:
    path = Path(path).absolute()
    try:
        if (path.is_symlink() or getattr(path, "is_junction", lambda: False)()
                or path.resolve(strict=True) != path):
            raise PublicServingRefusal("public_bundle_link_refused")
    except OSError as exc:
        raise PublicServingRefusal("public_bundle_invalid") from exc
    bundle = _read_json(path, "public_bundle_invalid")
    if bundle.get("version") != VERSION or bundle.get("scientifically_accepted") is not False:
        raise PublicServingRefusal("public_bundle_invalid")
    root = path.parent
    if root.resolve(strict=True) != root:
        raise PublicServingRefusal("public_bundle_link_refused")
    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, dict) or not _BUNDLE_ARTIFACTS <= set(artifacts):
        raise PublicServingRefusal("public_bundle_artifact_set_invalid")
    for name, expected in artifacts.items():
        if not isinstance(expected, str) or not _HASH.fullmatch(expected):
            raise PublicServingRefusal("public_bundle_artifact_mismatch")
        try:
            actual = _sha256(root / relative_path(name))
        except (OSError, ValueError) as exc:
            raise PublicServingRefusal("public_bundle_artifact_mismatch") from exc
        if actual != expected:
            raise PublicServingRefusal("public_bundle_artifact_mismatch")
    expected_frontend = bundle.get("frontend")
    if not isinstance(expected_frontend, dict):
        raise PublicServingRefusal("public_bundle_frontend_invalid")
    static_root = root / "static"
    if _linked_or_outside(root, static_root):
        raise PublicServingRefusal("public_bundle_frontend_link_refused")
    actual_frontend = set()
    for item in static_root.rglob("*"):
        if _linked_or_outside(static_root, item):
            raise PublicServingRefusal("public_bundle_frontend_link_refused")
        if item.is_file():
            actual_frontend.add(item.relative_to(static_root).as_posix())
    if actual_frontend != set(expected_frontend):
        raise PublicServingRefusal("public_bundle_frontend_member_set_mismatch")
    for name, expected in expected_frontend.items():
        if not isinstance(name, str) or _sha256(root / "static" / relative_path(name)) != expected:
            raise PublicServingRefusal("public_bundle_frontend_mismatch")
    return bundle, root


def serve_public(bundle: Path | str, port: int = 0, *, assistant_config: dict | None = None,
                 workspace: Workspace | None = None):
    """Return a loopback-only HTTP server after full bundle verification."""
    bundle, root = _verify_bundle(Path(bundle))
    index = ResearchSearchIndex(root / "search.sqlite")
    assets = load_assets(root / "source-assets.json")
    # Refuse before binding a listening socket when the packaged PDF no longer
    # matches the identity that every reader unit asserts.  The page endpoint
    # repeats this check immediately before rasterization to catch later
    # tampering as well.
    # v1 initially carried a single paper_id.  Multi-source bundles retain the
    # same immutable artifact contract but make every asset binding explicit.
    if isinstance(bundle.get("paper_ids"), list):
        paper_ids = bundle["paper_ids"]
        expected_shas = bundle.get("source_pdf_sha256s")
        allowed_pages = bundle.get("allowed_source_pages")
        if (not paper_ids or not all(isinstance(item, str) for item in paper_ids)
                or not isinstance(expected_shas, dict) or set(expected_shas) != set(paper_ids)
                or not isinstance(allowed_pages, dict) or set(allowed_pages) != set(paper_ids)):
            raise PublicServingRefusal("public_multi_source_binding_invalid")
    else:
        paper_ids = [bundle.get("paper_id")]
        expected_shas = {paper_ids[0]: bundle.get("source_pdf_sha256")}
        allowed_pages = {paper_ids[0]: bundle.get("allowed_source_pages")}
    if not all(isinstance(item, str) and item for item in paper_ids):
        raise PublicServingRefusal("public_source_pdf_hash_mismatch")
    for paper_id in paper_ids:
        try:
            packaged_pdf = pdf_bytes(assets, paper_id, allowed_roots=[root])
        except ValueError as exc:
            raise PublicServingRefusal("public_source_pdf_hash_mismatch") from exc
        except (OSError, PermissionError) as exc:
            raise PublicServingRefusal("public_source_pdf_unavailable") from exc
        if hashlib.sha256(packaged_pdf).hexdigest() != expected_shas[paper_id]:
            raise PublicServingRefusal("public_source_pdf_hash_mismatch")
    raw = _read_json(root / "raw-passages.json", "public_raw_passages_invalid")["items"]
    default = _read_json(root / "default-passages.json", "public_default_passages_invalid")["items"]
    bibliography = _read_json(root / "bibliography.json", "public_bibliography_invalid")
    citations = _read_json(root / "citations.json", "public_citations_invalid")
    measures = _read_json(root / "measures.json", "public_measures_invalid")
    advisories = _read_json(root / "advisories.json", "public_advisories_invalid") if "advisories.json" in bundle["artifacts"] else {"items": [], "status": "not_supplied"}
    report = _read_json(root / "candidate-report.json", "public_report_invalid")
    from mosaic.public_literature import PublicLiteratureAssistant
    literature = PublicLiteratureAssistant(passages=default, bibliography=bibliography,
        snapshot_id=index.snapshot()["snapshot_id"], workspace=workspace or Workspace(),
        config=assistant_config)
    literature_token = secrets.token_urlsafe(32)
    from mosaic.public_citation_api import citation_routes
    reference_routes = citation_routes(bibliography, citations, source_ids=paper_ids)
    pages = {paper_id: set(allowed_pages[paper_id]) for paper_id in paper_ids}
    # The base display server normally reads static files per request.  Hold an
    # immutable verified snapshot instead, so a later local file replacement
    # cannot alter a running public preview.
    static_bytes = {}
    for name, expected in bundle["frontend"].items():
        data = (root / "static" / relative_path(name)).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise PublicServingRefusal("public_bundle_frontend_mismatch")
        static_bytes[name] = data
    def projected_rows(rows: list[dict[str, Any]], params: Mapping[str, str]) -> dict[str, Any]:
        paper = params.get("paper", "")
        terms = [term.casefold() for term in params.get("q", "").split() if term]
        try:
            page, size = int(params.get("page", "1")), int(params.get("page_size", "25"))
        except ValueError as exc:
            raise ValueError("invalid_public_passage_pagination") from exc
        if page < 1 or not 1 <= size <= 100:
            raise ValueError("invalid_public_passage_pagination")
        filtered = [row for row in rows if (not paper or row["paper_id"] == paper)
                    and all(term in row["quote"].casefold() for term in terms)]
        start = (page - 1) * size
        return {"passages": filtered[start:start + size], "total": len(filtered), "page": page,
                "page_size": size, "default_masked": rows is default}
    def public_bibliography(params: Mapping[str, str]) -> dict[str, Any]:
        items = []
        for row in bibliography.get("paper_items", bibliography.get("items", [])):
            item = dict(row)
            authors = row.get("authors_verbatim", row.get("authors", ""))
            item.update(title_verbatim=row.get("title_verbatim", row.get("title")),
                        authors_verbatim=authors if isinstance(authors, list) else [authors],
                        journal_verbatim=row.get("journal_verbatim", row.get("journal")),
                        publication_year=row.get("publication_year", row.get("year")))
            items.append(item)
        papers = {**bibliography, "items": items}
        result = page_rows(papers, params)
        admitted = {row.get("paper_id") for row in papers["items"]}
        result["reader_entries"] = [{"paper_id": paper_id, "label": "Unreviewed source packet",
                                     "source_pdf_sha256": expected_shas[paper_id],
                                     "identity_status": "bibliographic_identity_unresolved"}
                                    for paper_id in paper_ids if paper_id not in admitted]
        return result
    routes = {
        "/api/display/v1/bibliography": public_bibliography,
        "/api/display/v1/apa": lambda p: page_rows(bibliography, p),
        "/api/display/v1/citations": lambda p: page_rows(citations, p),
        "/api/display/v1/measures": lambda p: page_rows(measures, p),
        "/api/display/v1/advisories": lambda p: page_rows(advisories, p),
        "/api/display/v1/passages": lambda p: projected_rows(default, p),
        "/api/display/v1/raw-passages": lambda p: projected_rows(raw, p),
        "/api/display/v1/status": lambda p: {"bibliography": {"status": bibliography.get("status", "reviewed_identity_projection"), "count": len(bibliography.get("items", [])), "unresolved_count": bibliography.get("unresolved_count", 0)}, "citations": {"status": citations.get("status", "pending"), "count": len(citations.get("items", []))}, "measures": {"status": measures.get("status", "pending"), "count": len(measures.get("items", []))}, "advisories": {"status": advisories.get("status"), "count": len(advisories.get("items", [])), "may_not_gate_visibility": True}, "passages": {"default_count": len(default), "raw_count": len(raw), "scientifically_accepted": False}},
        "/api/display/v1/candidate-report": lambda p: report,
    }
    server = make_display_server(index, routes=routes, port=port, static_root=root / "static", source_assets=assets)
    parent = server.RequestHandlerClass
    class SourcePageHandler(parent):
        def do_POST(self):  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path not in {"/api/literature/preview", "/api/literature/draft"}:
                return super().do_POST()
            if (not self._local_request() or not hmac.compare_digest(
                    self.headers.get("X-Mosaic-Request", ""), literature_token)):
                self._json(403, {"error": "public_literature_local_request_required"}); return
            try:
                if parsed.query or self.headers.get("Transfer-Encoding"):
                    raise ValueError("public_literature_invalid_request")
                lengths = self.headers.get_all("Content-Length", [])
                if len(lengths) != 1:
                    raise ValueError("public_literature_invalid_request_size")
                length = int(lengths[0])
                if not 0 < length <= 50000 or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError("public_literature_invalid_request_size")
                self.connection.settimeout(10)
                body = json.loads(self.rfile.read(length))
                result = literature.preview(body) if parsed.path.endswith("/preview") else literature.draft(body)
                self._json(200, result)
            except (ValueError, RuntimeError) as exc:
                message = str(exc)
                code = message if re.fullmatch(r"public_[a-z0-9_]+", message) else "public_literature_request_refused"
                self._json(400, {"error": code})
            except Exception:
                self._json(503, {"error": "public_literature_temporarily_unavailable"})

        def _static(self, raw_path: str) -> None:
            raw_relative = "index.html" if raw_path == "/" else unquote(raw_path).lstrip("/")
            try:
                relative = relative_path(raw_relative).as_posix()
            except ValueError:
                self._json(404, {"error": "not_found"})
                return
            data = static_bytes.get(relative)
            if data is None:
                self._json(404, {"error": "not_found"})
                return
            import mimetypes
            mime = mimetypes.guess_type(relative)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/api/literature/status":
                if not self._local_request():
                    self._json(403, {"error": "public_literature_local_request_required"}); return
                if parsed.query:
                    self._json(400, {"error": "public_literature_invalid_query"}); return
                try:
                    self._json(200, {**literature.status(), "csrf_token": literature_token})
                except (ValueError, RuntimeError):
                    self._json(503, {"error": "public_literature_configuration_unavailable"})
                return
            if parsed.path.startswith("/api/citations/"):
                if not self._local_request():
                    self._json(403, {"error": "Local preview requests only."}); return
                route = reference_routes.get(parsed.path)
                if route is None:
                    self._json(404, {"error": "not_found"}); return
                try:
                    query = parse_qs(parsed.query, keep_blank_values=True)
                    if any(len(values) != 1 for values in query.values()):
                        raise ValueError("duplicate_query_parameter")
                    self._json(200, route({key: values[0] for key, values in query.items()}))
                except (ValueError, KeyError):
                    self._json(400, {"error": "invalid_citation_query"})
                return
            if parsed.path.startswith("/sources/"):
                if not self._local_request():
                    self._json(403, {"error": "Local preview requests only."}); return
                match = re.fullmatch(r"/sources/(public-[0-9a-f]{64})\.pdf", parsed.path)
                if match is None or match[1] not in expected_shas or parsed.query:
                    self._json(404, {"error": "source_pdf_not_allowlisted"}); return
                try:
                    data = pdf_bytes(assets, match[1], allowed_roots=[root])
                    if hashlib.sha256(data).hexdigest() != expected_shas[match[1]]:
                        raise ValueError("source_pdf_identity_mismatch")
                except (ValueError, OSError):
                    self._json(409, {"error": "source_pdf_unavailable_or_changed"}); return
                self.send_response(200); self.send_header("Content-Type", "application/pdf")
                self.send_header("Content-Length", str(len(data))); self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers(); self.wfile.write(data); return
            if parsed.path != "/api/display/v1/source-page":
                return super().do_GET()
            if not self._local_request():
                self._json(403, {"error": "Local preview requests only."}); return
            args = parse_qs(parsed.query, keep_blank_values=True)
            if set(args) != {"paper", "page"} or any(len(values) != 1 for values in args.values()):
                self._json(400, {"error": "invalid_source_page_query"}); return
            paper_id = args["paper"][0]
            if paper_id not in paper_ids:
                self._json(404, {"error": "source_page_not_allowlisted"}); return
            try: page = int(args["page"][0])
            except ValueError: self._json(400, {"error": "invalid_source_page"}); return
            if page not in pages[paper_id]:
                self._json(404, {"error": "source_page_not_allowlisted"}); return
            try:
                import pymupdf
                source = pdf_bytes(assets, paper_id, allowed_roots=[root])
                if hashlib.sha256(source).hexdigest() != expected_shas[paper_id]:
                    raise ValueError("source_pdf_identity_mismatch")
                doc = pymupdf.open(stream=source, filetype="pdf")
                try:
                    source_page = doc[page - 1]
                    estimated_pixels = int(source_page.rect.width * 1.5) * int(source_page.rect.height * 1.5)
                    if estimated_pixels > 24_000_000:
                        self._json(413, {"error": "source_page_image_too_large"}); return
                    pix = source_page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
                    if pix.width * pix.height > 24_000_000:
                        self._json(413, {"error": "source_page_image_too_large"}); return
                    png = pix.tobytes("png")
                    if len(png) > 16 * 1024 * 1024:
                        self._json(413, {"error": "source_page_image_too_large"}); return
                finally: doc.close()
            except (ValueError, OSError, IndexError):
                self._json(409, {"error": "source_page_unavailable_or_changed"}); return
            image_sha = hashlib.sha256(png).hexdigest()
            self.send_response(200); self.send_header("Content-Type", "image/png"); self.send_header("Content-Length", str(len(png)));
            self.send_header("Cache-Control", "no-store"); self.send_header("X-Content-Type-Options", "nosniff"); self.send_header("X-Source-PDF-SHA256", expected_shas[paper_id]); self.send_header("X-Source-Page", str(page)); self.send_header("X-Image-SHA256", image_sha); self.end_headers(); self.wfile.write(png)
    server.RequestHandlerClass = SourcePageHandler
    return server
