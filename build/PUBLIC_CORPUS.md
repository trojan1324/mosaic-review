# Public corpus materialization

`mosaic.public_corpus.materialize_public_corpus` creates a local immutable
display bundle from explicitly supplied public candidate packets. It preserves
capture and snapshot artifacts; it does not run extraction, qualify a work, or
alter frozen results. The local browser is a candidate display, not a claim of
scientific acceptance.

## Review-bound display contract

### Authoring an explicit review

`mosaic.public_review` is a local authoring helper for an existing public capture.
It does not look up metadata, infer a review, or change the PDF/capture/snapshot.
It reads the full hash-validated native reader and offers four identity/APA commands:

- `catalogue` writes original unit IDs, pages, global offsets, literal quotes, and
  quote hashes for an operator to inspect.
- `draft` writes a valid `mosaic-public-corpus-reviews/v1` document containing an
  empty paper entry. It is intentionally unadmitted: passing it through
  `--reviews` does not create a Papers, APA, or measure row.
- `finalize` requires four explicit identity selections (`title`, `authors`,
  `journal`, `year`) and an attestation. Each selection is exactly
  `{ "value", "unit_id", "start", "end" }`; the value must equal the captured
  literal substring. Optional APA input supplies `authors_apa` plus literal
  selections for only `volume`, `issue`, `pages_verbatim`, `article_number`, or
  `doi`.
- `merge` combines one finalized immutable review file per paper. For example:

  ```powershell
  .venv-public/Scripts/python.exe -m mosaic.public_review merge --input run/reviews/paper-one.json --input run/reviews/paper-two.json --output run/reviews/corpus-reviews-v1.json
  ```

The accepted attestation types are `human_source_review` and
`agent_source_layout_review`. They record what the operator attests; the helper
does not verify a person's identity or make an agent review human. The latter type
must remain labelled as agent review. The helper rejects caller-only attestation,
PDF drift, missing/nonliteral/overlapping selections, unapproved APA fields, and
attempts to replace an immutable output file. It computes the canonical payload
and receipt hashes after those checks.

For literal measure items and citation edges, the separate
`.venv-public/Scripts/python.exe -m mosaic.public_evidence_review` helper offers
the `measure` and `edge` commands documented in `PUBLIC_BUILD_QUICKSTART.md`.
It merges an existing review packet without inferring a measure, citation target,
or semantic interpretation.

Identity fields and APA supplements must come from native `front_matter` or
`body` units. The helper and the materializer both reject reference, appendix,
table, heading and unknown lanes for this purpose. Those units remain available
for their own reference/appendix review. An eligible lane is only a prerequisite:
the reviewer must still establish that a value describes the containing article.

Each complete identity field currently needs one native unit. A title or full
byline split across units remains withheld; do not concatenate units or omit
authors to fit this format. A future ordered multi-span format needs its own
join rules and validation. APA author order and initials are a source-attested
rendering, not a mechanically verified conversion from the byline.

The resulting JSON has the top-level shape
`{"version":"mosaic-public-corpus-reviews/v1","papers":{"public-<pdf-sha256>":{...}}}`.
Use it with `bootstrap.py corpus --reviews`. It is distinct from the
`bootstrap.py materialize` manifest: materialization requires a
`mosaic-public-corpus-input/v1` `sources` list, with capture/report/snapshot
bindings and any review objects embedded in each source row.

Every source supplies the existing `capture` and `snapshot` handoffs. The
builder checks the source-PDF digest, exact snapshot digest, capture receipt,
native-reader binding, source packet, and candidate report before copying
anything. It copies verified PDFs, snapshots, reports, advisory provenance,
queues, and manifests, then hash-lists every copied file in `bundle.json`.

`identity_review` is the only route into **Papers**. It has status
`source_reviewed_complete`, a permitted reviewer type, versioned canonical
receipt, and literal witnesses for complete printed `title`, `authors`,
`journal`, and integer `year`. Each witness contains original `unit_id`, global
`start`/`end`, literal `text`, source-PDF hash, and captured-quote hash. A
complete Papers identity does not require APA rendering.

`apa_review` is a sibling of `identity_review`, never a replacement for it.
Its status is `source_reviewed_apa_rendering`; its receipt binds this exact
payload shape:

```json
{
  "authors_apa": ["Keller, R."],
  "byline_witness": {"unit_id":"u004","text":"Ruth Keller","start":71,"end":82,"source_pdf_sha256":"64-hex-PDF-digest","quote_sha256":"64-hex-quote-digest"},
  "supplements": {
    "volume": {"value":"12","witness":{"unit_id":"u006","text":"12","start":104,"end":106,"source_pdf_sha256":"64-hex-PDF-digest","quote_sha256":"64-hex-quote-digest"}},
    "pages_verbatim": {"value":"10-19","witness":{"unit_id":"u006","text":"10-19","start":114,"end":119,"source_pdf_sha256":"64-hex-PDF-digest","quote_sha256":"64-hex-quote-digest"}}
  }
}
```

The actual receipt has `source_pdf_sha256`, `reviewer_type`, `version`, and a
digest of that payload; `receipt_sha256` digests the receipt. APA authors may
be normalized but must be attested against the full printed byline. Only
`volume`, `issue`, `pages_verbatim`, `article_number`, and `doi` are allowed
supplements. Title, author, year, journal, and provenance overrides are
refused. Missing/invalid APA review leaves Papers visible but withholds APA.

`references` retain literal witnessed reference strings. A displayed edge needs
a known target and its own receipt; retained unresolved references are shown
separately. `measures` need literal evidence and a
`source_reviewed_meaningful_instrument` receipt. Symbols, units, placeholders,
filenames, and the shared exact generic-method vocabulary are withheld.
Frequency means distinct source papers, not repeated mentions.

## Optional advisory sidecars

Each `advisory_sidecars` item provides `task_path`, `sidecar_path`, and
`sidecar_sha256`. The builder replays existing local validation, requires
source/snapshot binding, copies both provenance files, and serves advice
separately. Advice is post-extraction only: it cannot modify extraction,
promote a candidate, or gate Papers, passages, citations, or measures.

## Local interface and APIs

Corpus mode keeps the approved MOSAIC blue interface. It renders Papers from
`/api/display/v1/bibliography`, APA from `/apa`, distinct-article measures from
`/measures`, reviewed edges plus unresolved-reference counts from `/citations`,
and optional advice from `/advisories`. The source reader uses `/passages` and
`/source-page` to return to the hash-verified local PDF image. Start shows
advice separately and labels it non-gating.

`serve_public(result["bundle_path"])` verifies declared artifacts and frontend
assets before serving local-only routes. No endpoint contacts a provider. Its
readiness check proves only that verified local routes responded; browser use,
coverage, model identity, rights, and scientific validity remain separate.

## Studies and reference-based discovery

The Studies view projects the existing core inventory from each hash-bound public
snapshot. Every quotation is checked against original unit offsets/text, page links
and source identity. An unavailable inventory produces an explicit empty view, never
a claim that the article contains no studies. Stage statuses and complete candidate
reports remain accessible. Study-fact and statistic wrappers are retained in full
reports; their fields are not silently promoted into the core study inventory.

The citation adapter ranks shared references by distinct reviewed target identities
and co-citation by distinct citing articles. Each recommendation exposes its shared
connections. Unresolved printed reference strings remain review material and do not
contribute edges or recommendation scores. These views do not establish scientific
agreement, independent validation or comprehensive coverage.
