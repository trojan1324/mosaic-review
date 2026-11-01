"""Versioned conservative display routes over the unchanged private research API.

The original server still owns source identity checks, PDFs, exact-ID retrieval,
and the citation graph.  These additional GET routes expose review projections;
they never replace an extraction result or grant scientific acceptance.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys
from typing import Callable
from urllib.parse import parse_qs, urlparse

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.research_search_index import ResearchSearchIndex, make_server


PREFIX = "/api/display/v1/"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def make_display_server(index: ResearchSearchIndex, *, routes: dict[str, Callable],
                        port: int = 0, opening_images: dict | None = None, **kwargs):
    """Extend the existing handler without changing its frozen implementation."""
    if any(not key.startswith(PREFIX) for key in routes):
        raise ValueError("display_route_outside_namespace")
    server = make_server(index, port, **kwargs)
    original = server.RequestHandlerClass

    class DisplayHandler(original):
        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            if not parsed.path.startswith(PREFIX):
                return super().do_GET()
            if not self._local_request():
                self._json(403, {"error": "Local preview requests only."})
                return
            route = routes.get(parsed.path)
            image_route = parsed.path == PREFIX + "opening-image" and opening_images is not None
            if route is None and not image_route:
                self._json(404, {"error": "display_route_not_found"})
                return
            try:
                args = parse_qs(parsed.query, keep_blank_values=True)
                if any(len(values) != 1 for values in args.values()):
                    raise ValueError("duplicate_query_parameter")
                if image_route:
                    if set(args) != {"paper", "sha256"}:
                        raise ValueError("invalid_opening_image_query")
                    spec = opening_images.get(args["paper"][0])
                    if spec is None:
                        self._json(404, {"error": "opening_image_not_available"})
                        return
                    try:
                        raw = Path(spec["path"]).read_bytes()
                    except OSError:
                        self._json(409, {"error": "opening_image_unavailable"})
                        return
                    if sha256(raw).hexdigest() != spec["sha256"] or args["sha256"][0] != spec["sha256"]:
                        self._json(409, {"error": "opening_image_artifact_mismatch"})
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(raw)))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.send_header("X-Source-PDF-SHA256", spec["source_pdf_sha256"])
                    self.send_header("X-Image-SHA256", spec["sha256"])
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                value = route({key: values[0] for key, values in args.items()})
                self._json(200, {**value, "snapshot_id": index.snapshot()["snapshot_id"],
                                 "scientifically_accepted": False})
            except (ValueError, TypeError, KeyError, sqlite3.Error) as exc:
                self._json(400, {"error": str(exc)})

    server.RequestHandlerClass = DisplayHandler
    return server


def page_rows(payload: dict, args: dict, *, key="items") -> dict:
    """Stable bounded metadata browsing; missing fields never get guessed."""
    page, size = int(args.get("page", "1")), int(args.get("page_size", "25"))
    if page < 1 or not 1 <= size <= 100:
        raise ValueError("invalid_pagination")
    query = args.get("q", "").strip().casefold()
    paper = args.get("paper", "")
    rows = payload.get(key, [])
    if paper:
        rows = [row for row in rows if row.get("paper_id") == paper]
    if query:
        terms = query.split()
        rows = [row for row in rows if all(term in display_search_text(row).casefold()
                                          for term in terms)]
    return {**{k: v for k, v in payload.items() if k != key}, key: rows[(page-1)*size:page*size],
            "total": len(rows), "page": page, "page_size": size}


def display_search_text(row: dict) -> str:
    """Search the presented record, keeping audit paths out of user results."""
    if 'title_verbatim' in row:
        fields = ('paper_id', 'title_verbatim', 'authors_verbatim',
                  'journal_verbatim', 'publication_year')
        return json.dumps([row[key] for key in fields if key in row], ensure_ascii=False)
    if 'citation_apa' in row:
        return str(row['citation_apa'])
    if 'label' in row:
        return str(row['label'])
    return json.dumps(row, ensure_ascii=False)


def load_bundle(path: Path, index: ResearchSearchIndex, *, assets_path: Path | None = None) -> dict:
    """Bind every review projection to exact bytes and the unchanged source DB."""
    path = Path(path)
    data = json.loads(path.read_bytes())
    if data.get("version") != "mosaic-research-display-bundle/v1":
        raise ValueError("display_bundle_version_invalid")
    if data.get("source_index_sha256") != file_sha256(index.path):
        raise ValueError("display_bundle_source_index_mismatch")
    if data.get("snapshot_id") != index.snapshot()["snapshot_id"]:
        raise ValueError("display_bundle_snapshot_mismatch")
    if assets_path is not None and data.get('source_assets_sha256') != file_sha256(assets_path):
        raise ValueError('display_bundle_source_assets_mismatch')
    result = {}
    for name in ("bibliography", "citations", "measures"):
        spec = data["files"][name]
        target = (path.parent / spec["path"]).resolve()
        if not target.is_relative_to(path.parent.resolve()):
            raise ValueError("display_bundle_path_escape")
        raw = target.read_bytes()
        if sha256(raw).hexdigest() != spec["sha256"]:
            raise ValueError("display_bundle_artifact_mismatch")
        result[name] = json.loads(raw)
    return result


def reader_scope_ids(bibliography: dict) -> set[str]:
    """Bibliographic identity review does not authorize new passage display."""
    identities = {row['paper_id'] for row in bibliography['items']}
    explicit = bibliography.get('reader_scope_paper_ids')
    if explicit is None:
        return identities
    if (not isinstance(explicit, list) or not all(isinstance(item, str) for item in explicit)
            or len(set(explicit)) != len(explicit) or not set(explicit) <= identities):
        raise ValueError('display_reader_scope_invalid')
    return set(explicit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--static-root", required=True)
    parser.add_argument("--source-assets", required=True)
    parser.add_argument("--citation-index", required=True)
    parser.add_argument("--citation-manifest-sha256", required=True)
    parser.add_argument("--content-boundaries")
    parser.add_argument("--content-boundaries-sha256")
    parser.add_argument("--port", type=int, default=8790)
    args = parser.parse_args()
    if bool(args.content_boundaries) != bool(args.content_boundaries_sha256):
        parser.error('content boundaries require an independent manifest hash')
    index = ResearchSearchIndex(args.index)
    bundle = load_bundle(Path(args.bundle), index, assets_path=Path(args.source_assets))
    display_ids = None if args.content_boundaries else reader_scope_ids(bundle['bibliography'])
    from mosaic.citation_graph import CitationGraphIndex
    from mosaic.source_assets import load_assets
    assets = load_assets(args.source_assets)
    allowed_sources = {paper_id: row['source_pdf_sha256'] for paper_id, row in assets.items()}
    if args.content_boundaries:
        from mosaic.passage_display_v2 import ContentPassageDisplayIndex
        passages = ContentPassageDisplayIndex(args.index, allowed_sources=allowed_sources,
            boundaries_path=args.content_boundaries,
            expected_boundaries_sha256=args.content_boundaries_sha256,
            expected_source_assets_sha256=file_sha256(Path(args.source_assets)))
        display_ids = set(allowed_sources)
    else:
        from mosaic.passage_display_v1 import PassageDisplayIndex
        passages = PassageDisplayIndex(args.index, allowed_sources=allowed_sources)
    passage_status = passages.warm()
    citations = CitationGraphIndex(args.citation_index,
        expected_manifest_sha256=args.citation_manifest_sha256)
    routes = {PREFIX+name: (lambda params, name=name: page_rows(bundle[name], params))
              for name in ("bibliography", "citations", "measures")}
    routes[PREFIX+"status"] = lambda _: {
        "bibliography": {k: v for k, v in bundle["bibliography"].items() if k != "items"},
        "citations": {k: v for k, v in bundle["citations"].items() if k != "items"},
        "measures": {k: v for k, v in bundle["measures"].items() if k != "items"},
        "passages": passage_status}
    routes[PREFIX+"passages"] = lambda p: passages.display_page(
        paper_id=p.get("paper") or None, query=p.get("q", ""),
        page=int(p.get("page", "1")), page_size=int(p.get("page_size", "25")),
        scope_paper_ids=display_ids, complete_sentences_only=not bool(args.content_boundaries))
    server = make_display_server(index, routes=routes, port=args.port,
        opening_images=getattr(passages, "opening_images", None),
        static_root=args.static_root, source_assets=assets,
        citation_index=citations)
    print(f"Conservative research display: http://127.0.0.1:{args.port}/workspace.html", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        citations.close()


if __name__ == "__main__":
    main()
