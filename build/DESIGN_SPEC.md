# MOSAIC: from a permitted PDF to an inspectable research workspace

The artifact is the build method and executable software. Users provide permitted
PDFs, a local open-weights model and, optionally, credentials for query-time
assistance. No private corpus is an installation dependency. The fictional
examples demonstrate software behavior; they do not measure extraction accuracy.

## Follow the evidence

| Step | Input and action | Output and reason for the check |
| --- | --- | --- |
| 1. Register | Read only explicitly listed PDFs; verify bytes and permission declaration. | Content identities and input manifest. A file path is a locator, not evidence of article identity or rights clearance. |
| 2. Capture | Preserve native PDF text, page geometry and source bytes. | Original units and page links. Text and coordinates allow later inspection; ambiguous layout remains a limitation. |
| 3. Select scope | Identify source regions and preserve excluded material separately. | Model-eligible packet and reader mask. Front matter is not evidence of a finding; printed references remain available separately. |
| 4. Extract | A bounded local open-weights driver proposes structured findings, studies, measures and selected facts. | Candidate report, frozen requests and append-only outcomes. Schema-constrained selection points to original units rather than authorizing reconstructed quotations. |
| 5. Check | Validate schemas, source identities, literal witnesses and field relationships. | Candidate snapshot and explicit refusals. These checks establish mechanical constraints, not the truth of an interpretation. |
| 6. Review | Select literal metadata, reference and measure witnesses; record the reviewer type. | Separate display reviews. An agent attestation is labeled as such; it is never relabeled independent human validation. |
| 7. Materialize | Combine verified candidates and admitted display reviews. | Search index, Papers, source reader, measure versions and citation views. Missing identity or item evidence stays unresolved. |
| 8. Advise (optional) | Send an explicitly selected post-extraction task to Jev, or replay a retained sidecar. | Advisory evidence that cannot change extraction, visibility or qualification. See [PUBLIC_JEV.md](PUBLIC_JEV.md). |
| 9. Explore | Search, open the source page, inspect literal items and compare evidence. | Researcher-controlled navigation and saved work. Presentation choices can change without altering frozen source evidence. |
| 10. Synthesize (optional) | Preview eligible passages, then explicitly request a model draft. | Linked author-year citations and exact supporting spans. The user checks interpretation and coverage; valid links alone do not establish entailment. See [PUBLIC_LITERATURE.md](PUBLIC_LITERATURE.md). |

## Three different demonstrations

`python start.py --no-serve` installs the locked Windows environment and runs
native capture plus manufactured responses through the extraction driver. This
tests the end-to-end software route on one generated PDF.

`python start.py --literature-demo` builds a separate two-paper fictional reader
with complete printed identities and literal review witnesses. It demonstrates
literature retrieval and citation navigation. Its candidate reports are
manufactured fixtures, not model extractions. Preview needs no API key.

The own-corpus route is `bootstrap.py corpus` with explicit source, model and
driver inputs. Preparation makes no model request; `--live` is required to
execute against the local model server. See [CORPUS_WORKFLOW.md](CORPUS_WORKFLOW.md).
The public real-model route still requires an observed successful runtime test.
Neither fictional example closes that gap or demonstrates scientific accuracy.

## Contracts that stay fixed while the interface evolves

Each evidence record retains its source identity, literal text, page/unit
locator and validation status. Reviews and formatting are separate projections.
A model-proposed category is distinct from an original quotation and from an
independent adjudication. A printed-reference edge is distinct from a discovery
recommendation. Repeated builds verify and reuse completed immutable artifacts;
changed code or inputs create successors. Failed attempts remain inspectable.

Frontend columns, filters, default ordering and detail panels can be revised
without deciding the final research interface in advance. The evidence contract
is the stable part of this artifact. Public code and review tooling must support
user-owned corpora; the author's private corpus supplies a separate scale demo.

## Lessons carried into the build

| Observed problem | Design response | What remains to evaluate |
| --- | --- | --- |
| PDF headings and affiliations looked like findings. | Preserve raw capture; admit the reader's abstract/body/appendix projection separately. | Unusual layouts and missing Abstract headings. |
| Filenames and journal headers looked like article titles. | Require literal title, byline, journal and year witnesses. | Review effort and automatic metadata recovery. |
| Isolated words looked like measure items. | Require explicit literal item selections and preserve context and disclosure status. | Completeness and instrument rights. |
| Linked quotations could still support a wrong interpretation. | Bind exact supporting spans and retain generated drafts as drafts. | Semantic accuracy, causal and statistical interpretations. |
| Environment tests were mistaken for successful extraction. | Record installation, runtime, fixture and scientific evidence separately. | A successful real-model route and independent scientific evaluation. |
| Historical validation inputs disappeared. | Preserve the original pending gate and its hashes; never manufacture replacement historical evidence. | Prospective validation uses a separately named evaluation. |

## Scale as a testable claim

Larger collections can expose more comparable measures, contrasting findings and
reference connections. Collection size alone is not evidence of greater research
value. Evaluate fixed research questions on nested corpora, hold eligibility and
task rules constant, and record useful supported comparisons, errors, latency,
cost and review burden. A recording of the private installation illustrates
functionality and coverage; it does not reproduce private-corpus accuracy.

## Validation and release

Run the exported checks with their temporary directory inside this export:

```powershell
.venv-public/Scripts/python.exe -I -m pytest -q --basetemp .pytest-tmp-check
```

The outer GitHub package's `VALIDATION.json` identifies the tested archive and
the scope of each observation. Review [NOTICE.md](NOTICE.md) before distribution.
The candidate carries no author license selection or implicit rights grant.
Published article PDFs, private results, account credentials and model weights
are excluded from the public archive.
