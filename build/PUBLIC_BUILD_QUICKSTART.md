# MOSAIC public build quickstart

This is the runnable public artifact for the November 1 submission. It contains
software and a fictional fixture, never the project's private corpus, published
PDFs, model weights, credentials, or private advisory records.

## 1. Rebuild the fictional reference route

Use Windows x86_64 with CPython 3.12 in an extracted public package. The command
creates `.venv-public`, verifies the allowlisted export and the locked dependency
inventory, then runs the fictional source through the actual capture, driver
fixture, materialization, and local interface route.

```powershell
py -3.12 start.py --wheelhouse C:\path\to\locked-wheelhouse
```

Without `--wheelhouse`, installation reads public PyPI only for the declared
Python dependencies. It does not download model weights or send source text to a
model provider. Add `--no-serve` when a local browser should not stay open.

The served address is printed by the launcher. This route proves the packaged
software mechanics for the fictional fixture. It does not prove model behavior,
extraction accuracy, corpus coverage, or compatibility with a real Linux/GPU
runtime.

For a fully offline install, obtain the exact reviewed wheelhouse separately:

```powershell
py -3.12 start.py --wheelhouse C:\path\to\locked-wheelhouse --no-serve
```

## 2. Obtain explicit own-corpus commands

After installation, run the launcher in guidance mode:

```powershell
py -3.12 start.py --wheelhouse C:\path\to\locked-wheelhouse --print-next-steps
```

It prints JSON command arrays for the installed interpreter. The output keeps the
fictional route separate from an own-corpus route. It does not ingest a PDF,
contact a local model, or enable live inference.

## 3. Prepare an own-corpus run

Copy the examples named in `CORPUS_WORKFLOW.md`: the corpus TOML, model
declaration, and native driver profile. List only local PDFs you are permitted to
process. Inventory a fully staged local open-weights model, copy the returned
tree digest into the corpus configuration, and run the printed `plan`,
`preflight`, and no-`--live` `corpus` commands in that order.

The no-`--live` command captures PDFs and freezes source/model preparation. Its
`prepared_only` result is useful evidence, but it is not a successful model run.
`--live` is an explicit later command for a literal loopback endpoint. It uses no
hosted extraction fallback. Current package evidence covers the Windows
fictional route and CPU Linux preflight; observed GPU/model execution remains a
separate, uncompleted validation stage.

When a live own-corpus command completes candidate requests, its report sets
`local_model_requests_completed: true`. This records an observed local request
exchange; it does not establish which weights a server loaded, scientific
accuracy, or scientific acceptance.

## 4. Add research assistance only after an evidence index exists

For the companion fictional workspace, use `py -3.12 start.py --literature-demo`.
Open `/assistant.html` at the printed address and preview a question about
coordination. See `PUBLIC_LITERATURE.md` for the explicit server configuration
needed for generated drafts.

The literature-review interface retrieves only the admitted local evidence index,
shows the passages behind each response, and links parenthetical citations to
the relevant records. A model-generated draft is optional and must be explicitly
configured. It cannot promote a candidate, repair identity metadata, or replace
source review. If no assistant provider is configured, the evidence preview and
source navigation remain available.

To author reviewed item wording or a printed-reference edge from captured units,
use the native unit catalogue from `python -m mosaic.public_review catalogue`.
The evidence helper takes explicit literal selectors and creates the hashes:

```powershell
.venv-public/Scripts/python.exe -m mosaic.public_evidence_review measure --capture-root run/my-capture --selection config/my-item-selections.json --existing-reviews config/my-reviewed-identities.json --output run/reviews-with-items.json
.venv-public/Scripts/python.exe -m mosaic.public_evidence_review edge --capture-root run/my-capture --selection config/my-reference-selection.json --existing-reviews run/reviews-with-items.json --output run/reviews-with-edges.json
```

Each selector contains `value`, `unit_id`, `start` and `end` from the catalogue.
A measure input contains `label`, `items` and `attestation`; an edge input
contains `literal_reference`, `target_paper_id` and `attestation`. Attestation
contains `reviewer_type` and a review `version`. Select the type honestly:
`caller_attested_source_review`, `agent_source_layout_review` or
`human_source_review`. Pass the resulting envelope to the corpus command's
`--reviews` option. Literal matching does not determine whether a sentence is
an item or establish its reproduction rights; the reviewer must inspect it.

Run exported checks with `.venv-public/Scripts/python.exe -I -m pytest -q
--basetemp .pytest-tmp-check` (on one line). Native capture currently limits each
PDF to 32 MiB and the model packet to 80,000 tokens. Reader admission requires
a literal Abstract anchor; missing or uncertain boundaries remain unresolved.

## 5. Keep the evidence boundary visible

Use `BUILD_GUIDE.md` for the fictional route, `CORPUS_WORKFLOW.md` for the
own-corpus path, `PUBLIC_JEV.md` for optional post-extraction Jev assessment,
and `DESIGN_SPEC.md` for claims and limits. Preserve every run directory:
inputs, source capture, model preparation, responses, and display projections
are versioned artifacts. A later display or assistant change never rewrites
them.
