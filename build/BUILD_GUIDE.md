# Following one source through the build

The runnable reference profile is `py -3.12 start.py` on Windows x86_64 with Python 3.12. It exercises the actual processing code with a deliberately fictional source and manufactured response fixtures. The same evidence must never be described as a real-model accuracy result.

## Runnable fictional installation — implemented

From the exported package, use CPython 3.12 on Windows x86_64. For an offline,
reproducible installation, provide a separately obtained local wheelhouse:

```powershell
py -3.12 start.py --wheelhouse C:\path\to\locked-wheelhouse --no-serve
```

Without `--wheelhouse`, `start.py` deliberately uses the public PyPI index; that
is an installation-network choice, not a model or scientific-provider call. The
launcher verifies the export and exact locked dependency versions, then invokes
`bootstrap.py demo`. It produces a fictional fixture route only. It does not
accept an operator PDF, execute a local model, establish corpus coverage, or
create review records. The supplied-PDF and review routes are in
`CORPUS_WORKFLOW.md`.

| Step | What it does and why | Where to inspect it | What the check cannot establish |
| --- | --- | --- | --- |
| Install | Creates an isolated environment and installs each declared wheel under its exact version and SHA-256. | `start.py`, `requirements-public-windows.lock`, `config/public_wheel_inventory_v1.json` | This Windows CPU profile does not establish Linux GPU compatibility. |
| Register | Records only explicitly selected files, assigns `public-<PDF SHA-256>` identities and records the caller's permission declaration. | `mosaic/public_build.py`, `run/demo/inputs/` | A permission declaration is not a legal clearance or bibliographic identity. |
| Capture | Freezes source bytes, captures native page text and geometry, and retains a content-addressed package. | `mosaic/public_capture.py`, the capture stage's `source.pdf` and package | PDF text order may still need review; OCR is disabled in this profile. |
| Keep the full reader | Projects original native block IDs, exact wording, pages and offsets independently of what the model is allowed to read. Finds a candidate start only at a literal Abstract heading. | `mosaic/public_reader.py`, `native-reader.json` | No Abstract heading means an unresolved boundary, not an invented opening. Tables remain source-image material. |
| Prepare the model context | Retains the eligible body subset and its hash-bound inventory, separate from full-reader coverage. | `source.json`, `source-binding.json`, `mosaic/source_packets.py` | A shorter model packet is not proof that omitted regions contain no useful evidence. |
| Propose structured evidence | Runs current findings/core selection and any dependent measure/study-fact stages. The model normally selects offered source evidence; selected text is copied literally. The demonstration supplies manufactured responses through this real driver. | `mosaic/public_runtime.py`, `scripts/complete_extraction_runner.py`, extraction-stage request/response/ledger artifacts | The fixture demonstrates mechanics, not model reasoning. Missing facts remain missing; an empty field is not scientific absence. |
| Check and interpret | Validates response schemas, offered choices, source identity, literal witnesses and field constraints. Preserves base report, study-fact wrapper and field dispositions. | `mosaic/public_snapshot.py`, `public-snapshot.json`, complete candidate report | Correct bytes and schemas do not establish ownership, interpretation or scientific truth. Those need source-context and independent human evaluation. |
| Build the interface | Builds a derived search database, reader projection, source-assets manifest and approved frontend. Keeps the complete report separate from finding rows. | `mosaic/public_serving.py`, materialization-stage `display/bundle.json` | A searchable candidate is not qualified evidence. Incomplete identities/APA/measures remain withheld, not filled with guesses. |
| Verify and use | Verifies all serving artifacts, opens a loopback server, and checks readiness. Researchers inspect original source pages, preserve literal passages and make their own judgments. | `run/demo/reports/<digest>.json`, serving-stage `readiness.json` | HTTP readiness alone is not browser usability; browser and fresh-install evidence must be recorded separately. |

## What is immutable and what may change

The source PDF, capture package, original unit IDs/text, model inputs/responses, stage receipts and scientific statuses are versioned evidence. A later display mask or interface cannot rewrite them. Every stage binds its inputs, implementation, dependencies and outputs. A retry creates a new attempt; it does not erase a failed attempt. Repeating a successful build verifies the stored artifacts before reuse.

The frontend's eventual field selection remains open. New views can reuse these source/report contracts without rerunning extraction. Changing the scientific capture scope or introducing a new extracted fact family still requires a versioned method and corresponding tests. The current demonstration uses one fixed view configuration; a general view editor is not implemented.

## Refusals are part of the result

A wrong PDF hash, changed source packet, inconsistent model identity, missing dependency, out-of-scope transport or tampered artifact produces a refusal. Raw source files and earlier results survive. If model eligibility is empty, the native capture/reader can still survive. Source-bound bibliography, reference and measure builders keep incomplete candidates in separate review queues; exact witnesses alone do not establish a correct bibliographic role.

The private historical regression gate remains pending because its original recovery source is unavailable. No demonstration file substitutes for that evidence. This public package is a prospective engineering artifact.

## Assessment and validation

Jev is an optional post-extraction advisory branch. It does not accept evidence, change a tier, merge a construct or decide scientific qualification. Its public adapter takes an explicit hash-bound task and an installation's cumulative ledger/budget; a live call requires an explicit send action. Offline replay validates an existing sidecar without a key or network. The demonstration does not spend a provider budget or ship private advisory responses. See `PUBLIC_JEV.md`.

Scientific validation needs a separate, permitted real-source evaluation with frozen tasks, source-context judgments and independent review. The large private-corpus video can illustrate research use and scale, but cannot substitute for such a reproduction route. Current public claims should be limited to the software behaviors actually observed.

## Recovering from a failed build

Read the latest immutable failure artifact and correct the named prerequisite. Keep the prior run. Verified successful capture stages may be reused; failed attempts remain visible. Native execution has a stricter boundary: a prepared program binds one execution output, and an unknown send cannot be retried by selecting another output. Reconcile it before designing another experiment. Changing code, input bytes or configuration creates a new identity, not proof that a previous unknown outcome is safe to repeat. This build requires one writer per corpus run directory.

## Preparing a future local model run

Native preparation can be performed before a GPU is available. Start from a
public `capture` output directory containing `receipt.json`, `source.pdf`,
`source.json`, `source-binding.json` and `prepare-upstream-input.json`. Provide
an explicit complete local model manifest and model directory; fill the model
identity in `config/public_native_driver.example.json` to match the manifest.
The manifest contract is documented in `PUBLIC_RUNTIME.md` and implemented in
`mosaic/public_runtime_manifest.py`; it requires every model file's actual size
and SHA-256, the tokenizer identity, and separately labeled runtime/hardware
declarations. Do not copy the demonstration's model responses into this route.

```powershell
.venv-public\Scripts\python.exe bootstrap.py prepare-local --capture-root run/my-capture --model-manifest config/my-model-manifest.json --weights-root models/my-model --driver-profile config/public_native_driver.example.json --base-url http://127.0.0.1:8000/v1 --output run/native-prepared
```

This command checks the source/model bindings and freezes a native extraction
program. It performs no model execution and does not open the endpoint. Its
receipt distinguishes verified local bytes from caller declarations and leaves
Linux/GPU execution and scientific accuracy unvalidated. The paths above are
examples: use the actual capture-stage paper directory, not its parent run root.

The implemented preparation command is not a readiness result. A corpus command
without `--live` can report `prepared_only`; it still has no real model exchange,
browser assessment, coverage estimate, or scientific acceptance. A later local
HTTP readiness receipt verifies only the packaged loopback routes.

Only the explicit public export is intended for release review. Do not make a private checkout public. `EXPORT-MANIFEST.json` covers exported source bytes; dependency wheels, model terms, fonts and adapted code retain separate provenance and rights checks. Release approval and journal filing are separate author steps.

The native execution and multi-source view commands are linked in
`CORPUS_WORKFLOW.md`. They extend this preparation path; actual Linux model
execution remains a separate observed validation requirement.

## Reused implementation and public entry points

The extraction driver reuses tested helper functions from earlier experiments.
Some retained helper modules contain historical experiment identifiers; no
corresponding source files, private registries or experiment outputs are included.
Use `bootstrap.py` and the documented public modules. Private named-recipe
entry points are unavailable in this export and refuse explicitly. Public PDF
access requires the verified bundle's explicit root. Both model clients require
an explicit public source scope; private default policies remain in the private
repository and are not loaded by the public workflow.
