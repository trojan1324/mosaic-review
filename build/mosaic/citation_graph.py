"""Offline, provenance-preserving citation graph primitives.

This module deliberately does not parse PDFs or resolve bibliography strings.
Callers provide immutable printed-reference records from an admitted capture.
An unresolved literal remains a literal node; it is never title-matched.
"""
from collections import Counter, defaultdict
import json
import sqlite3
import threading
import hashlib
from pathlib import Path
from mosaic.workspace import readonly_sqlite


RESOLUTION_STATUSES = {"resolved_exact", "unresolved", "ambiguous", "blocked"}


def _require(value, message):
    if not value:
        raise ValueError(message)


def validate_reference(record):
    """Validate one original-reference record without modifying it."""
    _require(isinstance(record, dict), "reference_record_object_required")
    for key in ("reference_id", "citing_work_id", "citing_source_sha256", "literal_reference", "provenance", "resolution"):
        _require(record.get(key), f"reference_{key}_required")
    resolution = record["resolution"]
    _require(isinstance(resolution, dict), "reference_resolution_object_required")
    status = resolution.get("status")
    _require(status in RESOLUTION_STATUSES, "reference_resolution_status_invalid")
    if status == "resolved_exact":
        _require(resolution.get("resolved_work_id"), "exact_resolution_target_required")
    elif resolution.get("resolved_work_id"):
        raise ValueError("unresolved_reference_cannot_have_target")
    provenance = record["provenance"]
    _require(isinstance(provenance, dict), "reference_provenance_required")
    printed = provenance.get("kind") == "printed_source_verified"
    legacy = provenance.get("kind") == "legacy_campaign_metadata"
    _require(printed or legacy, "reference_provenance_kind_invalid")
    if printed:
        _require(provenance.get("page") and provenance.get("unit_id"), "printed_reference_anchor_required")
    else:
        _require(provenance.get("file_sha256") and isinstance(provenance.get("record_id"), (str, int)),
                 "legacy_campaign_record_provenance_required")


def build_graph(records):
    """Build citation indexes from validated records.

    Only ``resolved_exact`` entries create work-to-work edges. All records are
    retained in ``literal_references`` so lossless inspection is possible.
    """
    records = list(records)
    seen = set()
    outgoing, incoming, literal_references = defaultdict(set), defaultdict(set), {}
    for record in records:
        validate_reference(record)
        rid = record["reference_id"]
        if rid in seen:
            raise ValueError("duplicate_reference_id")
        seen.add(rid)
        literal_references[rid] = record
        if record["resolution"]["status"] == "resolved_exact":
            source, target = record["citing_work_id"], record["resolution"]["resolved_work_id"]
            outgoing[source].add(target)
            incoming[target].add(source)
    return {
        "literal_references": literal_references,
        "outgoing": {k: sorted(v) for k, v in sorted(outgoing.items())},
        "incoming": {k: sorted(v) for k, v in sorted(incoming.items())},
    }


def bibliographic_coupling(graph, left, right):
    """Resolved references shared by two citing works."""
    return sorted(set(graph["outgoing"].get(left, ())) & set(graph["outgoing"].get(right, ())))


def co_citation(graph, left, right):
    """Works that cite both resolved targets."""
    return sorted(set(graph["incoming"].get(left, ())) & set(graph["incoming"].get(right, ())))


def bridge_candidates(graph, work_id):
    """Return explainable, non-edge recommendations through shared references.

    A recommendation is never inserted into the source graph and carries the
    exact shared resolved references that produced its score.
    """
    mine = set(graph["outgoing"].get(work_id, ()))
    scores = defaultdict(set)
    for reference in mine:
        for peer in graph["incoming"].get(reference, ()):
            if peer != work_id:
                scores[peer].add(reference)
    return [
        {"candidate_work_id": peer, "shared_resolved_references": sorted(shared), "score": len(shared),
         "recommendation_only": True}
        for peer, shared in sorted(scores.items(), key=lambda item: (-len(item[1]), item[0]))
    ]


SCHEMA = """
CREATE TABLE works (work_id TEXT PRIMARY KEY, title TEXT, doi TEXT, identity_status TEXT NOT NULL);
CREATE INDEX works_doi ON works(doi);
CREATE TABLE references_ (reference_id TEXT PRIMARY KEY, citing_work_id TEXT NOT NULL, literal_reference TEXT NOT NULL,
  resolution_status TEXT NOT NULL, target_work_id TEXT, edge_kind TEXT NOT NULL, provenance_json TEXT NOT NULL,
  resolution_basis TEXT, reference_match_verified INTEGER NOT NULL);
CREATE INDEX references_citing ON references_(citing_work_id, reference_id);
CREATE INDEX references_target ON references_(target_work_id, reference_id);
"""

SCHEMA_V2 = SCHEMA + """
ALTER TABLE references_ ADD COLUMN admitted_source_pdf_sha256 TEXT;
ALTER TABLE references_ ADD COLUMN legacy_source_id TEXT;
ALTER TABLE references_ ADD COLUMN citation_source_version_status TEXT;
CREATE TABLE graph_provenance (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def build_sqlite(records, works, output_path, workspace, provenance_root_digest=None):
    """Write a new immutable SQLite graph from already-admitted records."""
    path = workspace.check(output_path)
    if path.exists():
        raise FileExistsError("citation_graph_output_must_be_new")
    conn = workspace.sqlite_connect(path)
    try:
        conn.executescript(SCHEMA_V2 if provenance_root_digest else SCHEMA)
        work_ids = {w["work_id"] for w in works}
        conn.executemany("INSERT INTO works VALUES (?,?,?,?)", [
            (w["work_id"], w.get("title"), w.get("doi"), w.get("identity_status", "exact")) for w in works])
        for record in records:
            validate_reference(record)
            if record["citing_work_id"] not in work_ids:
                raise ValueError("citing_work_not_admitted")
            target = record["resolution"].get("resolved_work_id")
            if target and target not in work_ids:
                raise ValueError("resolved_target_not_admitted")
            fields = (
                record["reference_id"], record["citing_work_id"], record["literal_reference"],
                record["resolution"]["status"], target, record.get("edge_kind", record["provenance"]["kind"]),
                json.dumps(record["provenance"], sort_keys=True), record.get("resolution_basis"),
                1 if record.get("reference_match_verified", False) else 0)
            if provenance_root_digest:
                source_hash = record.get("admitted_source_pdf_sha256")
                if not isinstance(source_hash, str) or len(source_hash) != 64:
                    raise ValueError("citation_v2_source_pdf_sha256_required")
                fields += (source_hash, record.get("legacy_source_id"),
                           record.get("citation_source_version_status"))
            conn.execute("INSERT INTO references_ VALUES (" + ",".join("?" for _ in fields) + ")", fields)
        if provenance_root_digest:
            conn.execute("INSERT INTO graph_provenance VALUES (?,?)", ("provenance_root_sha256", provenance_root_digest))
            conn.execute("INSERT INTO graph_provenance VALUES (?,?)", ("schema_version", "citation-graph/v2"))
        conn.commit()
    finally:
        conn.close()


class CitationGraphIndex:
    """Read-only, parameterized query surface for an immutable graph index."""
    def __init__(self, path, expected_manifest_sha256=None):
        self.path = Path(path)
        self.conn = readonly_sqlite(self.path, immutable=True, check_same_thread=False)
        self.conn.execute("PRAGMA query_only=ON")
        self._lock = threading.RLock()
        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(references_)")}
        self._v2 = "admitted_source_pdf_sha256" in columns
        self._provenance_root = None
        if self._v2:
            self._verify_v2_snapshot(expected_manifest_sha256)

    def _verify_v2_snapshot(self, expected_manifest_sha256):
        """Fail closed unless a v2 graph has an independently supplied manifest pin."""
        if not isinstance(expected_manifest_sha256, str) or len(expected_manifest_sha256) != 64:
            self.conn.close()
            raise ValueError("citation_v2_manifest_pin_required")
        manifest_path = Path(str(self.path) + ".manifest.json")
        if not manifest_path.is_file():
            self.conn.close()
            raise ValueError("citation_v2_manifest_missing")
        raw = manifest_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_manifest_sha256:
            self.conn.close()
            raise ValueError("citation_v2_manifest_digest_changed")
        try:
            manifest = json.loads(raw)
        except json.JSONDecodeError as exc:
            self.conn.close()
            raise ValueError("citation_v2_manifest_invalid") from exc
        actual_graph = hashlib.sha256(self.path.read_bytes()).hexdigest()
        root = self.conn.execute("SELECT value FROM graph_provenance WHERE key='provenance_root_sha256'").fetchone()
        if (manifest.get("version") != "citation-graph-selected-shards/v2"
                or manifest.get("graph_sha256") != actual_graph
                or not root or manifest.get("provenance_manifest_sha256") != root[0]):
            self.conn.close()
            raise ValueError("citation_v2_snapshot_binding_invalid")
        self._provenance_root = root[0]

    def close(self):
        self.conn.close()

    def health(self):
        with self._lock:
            refs = self.conn.execute("SELECT count(*) FROM references_").fetchone()[0]
            resolved = self.conn.execute("SELECT count(*) FROM references_ WHERE target_work_id IS NOT NULL").fetchone()[0]
            works = self.conn.execute("SELECT count(*) FROM works").fetchone()[0]
            citing = self.conn.execute("SELECT count(DISTINCT citing_work_id) FROM references_").fetchone()[0]
            result = {"status": "ready", "source_record_count": refs, "resolved_edge_count": resolved,
                    "unresolved_reference_count": refs - resolved, "total_admitted_works":works,
                    "citing_work_count":citing, "coverage_scope":"admitted_citers_only",
                    "legacy_identity_label":"legacy_supplied_doi_identity_is_not_printed_reference_verification"}
            if self._v2:
                result.update({"provenance_status":"bound_v2", "provenance_root_sha256":self._provenance_root,
                               "citer_identity_scope":"work_level_metadata_not_exact_pdf_bibliography_verified"})
            else:
                result["provenance_status"] = "legacy_unbound_v1"
            return result

    def search(self, query, limit=20, cursor=0):
        with self._lock:
            limit = _limit(limit); cursor = int(cursor)
            like = "%" + _like(query) + "%"
            total=self.conn.execute("SELECT count(*) FROM works WHERE title LIKE ? ESCAPE '\\' OR doi LIKE ? ESCAPE '\\'",(like,like)).fetchone()[0]
            rows = self.conn.execute("SELECT work_id,title,doi,identity_status FROM works WHERE title LIKE ? ESCAPE '\\' OR doi LIKE ? ESCAPE '\\' ORDER BY work_id LIMIT ? OFFSET ?", (like, like, limit, cursor)).fetchall()
            return {"items": [_work(r) for r in rows], "total":total,"next_cursor": cursor + len(rows) if cursor + len(rows) < total else None}

    def search_references(self, query, limit=20, cursor=0):
        """Search retained literal references without inferring an identity edge."""
        with self._lock:
            limit = _limit(limit); cursor = _cursor(cursor)
            like = "%" + _like(query) + "%"
            source_columns = "r.admitted_source_pdf_sha256,r.legacy_source_id,r.citation_source_version_status" if self._v2 else "NULL,NULL,NULL"
            total = self.conn.execute("SELECT count(*) FROM references_ WHERE literal_reference LIKE ? ESCAPE '\\'", (like,)).fetchone()[0]
            rows = self.conn.execute(f"SELECT r.reference_id,r.literal_reference,r.resolution_status,r.target_work_id,r.edge_kind,r.provenance_json,r.resolution_basis,r.reference_match_verified,{source_columns},w.work_id,w.title,w.doi FROM references_ r LEFT JOIN works w ON w.work_id=r.citing_work_id WHERE r.literal_reference LIKE ? ESCAPE '\\' ORDER BY r.reference_id LIMIT ? OFFSET ?", (like, limit, cursor)).fetchall()
            items = []
            for row in rows:
                reference = _reference(row[:11])
                reference["citing_work"] = {"work_id": row[11], "title": row[12], "doi": row[13]}
                items.append(reference)
            return {"items": items, "total": total,
                    "next_cursor": cursor + len(items) if cursor + len(items) < total else None}

    def work(self, work_id, limit=20, cursor=0):
      with self._lock:
        limit = _limit(limit); cursor = int(cursor)
        row = self.conn.execute("SELECT work_id,title,doi,identity_status FROM works WHERE work_id=?", (work_id,)).fetchone()
        if not row: raise KeyError("unknown_work")
        ref_total=self.conn.execute("SELECT count(*) FROM references_ WHERE citing_work_id=?",(work_id,)).fetchone()[0]
        incoming_total=self.conn.execute("SELECT count(*) FROM references_ WHERE target_work_id=?",(work_id,)).fetchone()[0]
        source_columns = "admitted_source_pdf_sha256,legacy_source_id,citation_source_version_status" if self._v2 else "NULL,NULL,NULL"
        refs = self.conn.execute(f"SELECT reference_id,literal_reference,resolution_status,target_work_id,edge_kind,provenance_json,resolution_basis,reference_match_verified,{source_columns} FROM references_ WHERE citing_work_id=? ORDER BY reference_id LIMIT ? OFFSET ?", (work_id, limit, cursor)).fetchall()
        incoming_source_columns = "r.admitted_source_pdf_sha256,r.legacy_source_id,r.citation_source_version_status" if self._v2 else "NULL,NULL,NULL"
        incoming = self.conn.execute(f"SELECT r.citing_work_id,w.title,w.doi,r.reference_id,r.edge_kind,r.provenance_json,r.resolution_basis,r.reference_match_verified,{incoming_source_columns} FROM references_ r LEFT JOIN works w ON w.work_id=r.citing_work_id WHERE r.target_work_id=? ORDER BY r.reference_id LIMIT ? OFFSET ?", (work_id, limit, cursor)).fetchall()
        targets = [r[3] for r in refs if r[3]]
        peers = self.conn.execute("SELECT citing_work_id, count(DISTINCT target_work_id) FROM references_ WHERE target_work_id IN (SELECT target_work_id FROM references_ WHERE citing_work_id=? AND target_work_id IS NOT NULL) AND citing_work_id<>? GROUP BY citing_work_id ORDER BY 2 DESC,citing_work_id LIMIT ?", (work_id, work_id, limit)).fetchall()
        co, co_truncated = self._co_citation(work_id, limit)
        coupling=[self._peer(p,"shared_resolved_reference_count",work_id) for p in peers]
        return {"work": _work(row), "references": {"items": [self._reference_with_target(r) for r in refs], "total":ref_total,"next_cursor": cursor + len(refs) if cursor+len(refs)<ref_total else None}, "incoming": {"items": [{"work_id":r[0],"title":r[1],"doi":r[2],"reference_id":r[3],"edge_kind":r[4],"provenance":json.loads(r[5]),"resolution_basis":r[6],"reference_match_verified":bool(r[7]),"admitted_source_pdf_sha256":r[8],"legacy_source_id":r[9],"citation_source_version_status":r[10]} for r in incoming], "total":incoming_total,"next_cursor": cursor + len(incoming) if cursor+len(incoming)<incoming_total else None}, "bibliographic_coupling": coupling, "co_citation": co, "co_citation_scan_truncated":co_truncated, "co_citation_scan_cap":5000 if co_truncated else None, "bridge_candidates": [dict(x, recommendation_only=True, reason="shared_resolved_references") for x in coupling]}

    def _reference_with_target(self, row):
        result = _reference(row)
        if row[3]:
            target = self.conn.execute("SELECT title,doi FROM works WHERE work_id=?", (row[3],)).fetchone()
            result["target"] = {"work_id":row[3],"title":target[0],"doi":target[1]}
        return result

    def _peer(self, row, count_key, source_work_id):
        detail = self.conn.execute("SELECT title,doi FROM works WHERE work_id=?", (row[0],)).fetchone()
        shared=self.conn.execute("SELECT target_work_id FROM references_ WHERE citing_work_id=? AND target_work_id IN (SELECT target_work_id FROM references_ WHERE citing_work_id=?) GROUP BY target_work_id ORDER BY target_work_id",(row[0],source_work_id)).fetchall()
        refs=[]
        for (target,) in shared[:50]:
            meta=self.conn.execute("SELECT title,doi FROM works WHERE work_id=?",(target,)).fetchone(); refs.append({"work_id":target,"title":meta[0],"doi":meta[1]})
        return {"work_id":row[0],"title":detail[0] if detail else None,"doi":detail[1] if detail else None,count_key:row[1],"shared_resolved_references":refs,"shared_resolved_reference_total":len(shared),"shared_resolved_reference_cap":50}

    def _co_citation(self, work_id, limit):
        cap = 5000
        rows = self.conn.execute("SELECT other.target_work_id, other.citing_work_id FROM references_ chosen JOIN references_ other ON other.citing_work_id=chosen.citing_work_id WHERE chosen.target_work_id=? AND other.target_work_id IS NOT NULL AND other.target_work_id<>? ORDER BY other.target_work_id,other.citing_work_id LIMIT ?", (work_id,work_id,cap)).fetchall()
        grouped = defaultdict(set)
        for target,citer in rows: grouped[target].add(citer)
        result=[]
        for target,citers in sorted(grouped.items(),key=lambda x:(-len(x[1]),x[0]))[:limit]:
            detail=self.conn.execute("SELECT title,doi FROM works WHERE work_id=?",(target,)).fetchone()
            result.append({"work_id":target,"title":detail[0],"doi":detail[1],"shared_citing_work_ids":sorted(citers),"shared_citing_work_count":len(citers),"shared_citing_work_count_is_lower_bound":len(rows) == cap,"reason":"co_cited_by_same_admitted_citing_works"})
        return result, len(rows) == cap


def _limit(value):
    value = int(value)
    if not 1 <= value <= 100: raise ValueError("query_limit_invalid")
    return value


def _cursor(value):
    value = int(value)
    if value < 0: raise ValueError("query_cursor_invalid")
    return value


def _like(value):
    return str(value).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _work(row): return {"work_id":row[0],"title":row[1],"doi":row[2],"identity_status":row[3]}
def _reference(row): return {"reference_id":row[0],"literal_reference":row[1],"resolution_status":row[2],"target_work_id":row[3],"edge_kind":row[4],"provenance":json.loads(row[5]),"resolution_basis":row[6],"reference_match_verified":bool(row[7]),"admitted_source_pdf_sha256":row[8],"legacy_source_id":row[9],"citation_source_version_status":row[10]}
