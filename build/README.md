# MOSAIC reproducible build candidate

MOSAIC turns explicitly selected PDFs into source-bound candidate evidence and an inspectable research interface. This candidate demonstrates the complete software route using a deliberately fictional PDF and manufactured test responses through the real extraction driver. It does **not** demonstrate a model extracting a real paper correctly.

Read [Following one source through the build](BUILD_GUIDE.md) for each step's inputs, checks, outputs and limits. [The corpus workflow](CORPUS_WORKFLOW.md) connects the local-model commands, reviewed corpus views and optional Jev sidecars.

Start with the [design specification](DESIGN_SPEC.md) for the complete evidence
flow, and the [command guide](PUBLIC_BUILD_QUICKSTART.md) for installation and
your own sources. The optional [literature assistant](PUBLIC_LITERATURE.md)
previews local evidence and can generate drafts with linked source citations.
The [feature scope](FEATURE_SCOPE.md) distinguishes this package from the
author's private demonstration and identifies the remaining release conditions.

The archive contains code, instructions, and fictional demonstration material.
It includes no published-article PDFs, page images, screenshots, or quotations
from the private corpus. Users supply their own permitted source files.
Software dependencies are installed separately. The pinned Linux GPU runtime
for live extraction is approximately 9.2 GB compressed; the selected model's files are
approximately 55.6 GB. Neither is bundled in the GitHub archive, and the Windows
fictional demonstration downloads neither the GPU image nor model weights.

## Start the demonstration

Reference environment: Windows x86_64, CPython 3.12. From this directory:

```powershell
py -3.12 start.py
```

This creates an isolated `.venv-public`, installs the exact hash-locked Python libraries from public PyPI, creates the fictional PDF, captures its native text, runs the actual driver with explicit fixture responses, builds serving artifacts and serves the approved interface on loopback port 8798. It downloads no model weights, sends no scientific text to a service and needs no API key. Stop the foreground command with Ctrl+C.

The launcher checks the installed library inventory and dependency compatibility
before running the build. If a reused environment contains other packages or
versions, it refuses; unpack into a fresh directory to retain the old environment
and establish a clean one.

For offline installation, prepare a wheel directory with the lock on the supported platform, then pass it explicitly:

```powershell
py -3.12 -m pip download --require-hashes --only-binary=:all: -r requirements-public-windows.lock --dest wheelhouse
py -3.12 start.py --wheelhouse wheelhouse
```

For a build without a persistent server, add `--no-serve`. In an environment where the dependencies are already installed, `.venv-public/Scripts/python.exe bootstrap.py demo` builds and verifies the artifacts; add `--serve` to launch the interface. Repeated execution verifies preserved artifacts. A source, implementation or view change receives its own version rather than modifying frozen results.

The pinned Windows wheel inventory contains 50 wheels totaling 201,455,250 bytes
(about 192.1 MiB before unpacking). One recorded fresh v32 Windows run took 620
seconds for the hash-locked installation plus fictional driver demonstration. That
is an observation of that machine and route, not a hardware or download-time promise.

To explore a separate two-paper fictional workspace with complete printed
metadata, APA entries, literal measure items and reference connections:

```powershell
py -3.12 start.py --literature-demo
```

Open `http://127.0.0.1:8798/assistant.html` to preview the fictional evidence.
This companion contains manufactured candidate reports and controlled review
fixtures. It does not demonstrate model extraction or scientific accuracy.
Model drafting is disabled until explicitly configured; preview needs no key.

## Check the build

Run checks with a writable temporary directory inside the exported package:

```powershell
.venv-public/Scripts/python.exe -I -m pytest -q --basetemp .pytest-tmp-check
```

The export's checks use fictional data and fake provider transports. They do
not send source text to a model, launch a GPU or evaluate scientific accuracy.

## What to inspect

The reader shows original passages after an explicit Abstract heading, exposes raw capture separately, and links to the exact source PDF and rendered source page. Printed references and appendices remain source material. Clean article identity, APA references, measures and citation links are withheld when their required evidence is missing. The demonstration intentionally contains unresolved bibliography; it must not invent authors or a journal to fill the screen.

The candidate report preserves model-proposal structure, literal source bindings, stage refusals and separate study-fact results. Mechanical verification means that the recorded inputs and outputs satisfy specific software constraints. It is not scientific qualification or independent human validation.

## Bring your own sources

Use `config/mosaic.public.example.toml` to declare only the PDFs you are permitted to process. `bootstrap.py plan`, `preflight`, `ingest` and `capture` accept `--config`. Capture is native-only and preserves the full reader even when no model-eligible text remains. OCR and automatic article identity/reference admission are unfinished public features.

`bootstrap.py prepare-local` checks a captured source, a complete local model
manifest and tokenizer, and a bounded driver profile. It freezes a native
extraction program without connecting to a model server. See the preparation
example in `BUILD_GUIDE.md`; the profile template is
`config/public_native_driver.example.json`.

`bootstrap.py inventory-model` generates the complete local file/hash manifest.
`bootstrap.py corpus` connects capture and native preparation; explicit `--live`
executes the frozen programs against an already running local model server and
builds the corpus views. `materialize` can rebuild reviewed views from existing
source/report artifacts, and `serve` reopens a verified bundle. See
[CORPUS_WORKFLOW.md](CORPUS_WORKFLOW.md) for inputs and recovery behavior.

The local open-weights runtime profile is described in [PUBLIC_RUNTIME.md](PUBLIC_RUNTIME.md). **Its Linux/GPU cold start has not been demonstrated by the Windows fixture run.** The fictional demonstration uses recorded fixtures; the separate native execution route never falls back to a hosted model. Optional [Jev advice](PUBLIC_JEV.md) requires an explicit task and send action, remains outside extraction, and can be replayed offline.

The current Linux worker uses the [single-package dependency repair](LINUX_DEPENDENCY_REPAIR.md).
It checks the complete installed dependency graph and file ownership before
removing one orphan package. A prior nine-package telemetry upgrade failed;
its results remain historical evidence. Local checks of this replacement do
not establish a successful Linux/model run.

Linux environment validation is a separate CPU-only step. Inside the pinned
image, `python3 -m mosaic.public_linux_environment preflight --output
/tmp/evidence/preflight-v1 --deadline-epoch <absolute-UTC-epoch>` checks the
installed dependency graph, package removal, telemetry imports, and the pinned
PDF-library installation and import. Use a new output directory and a deadline
at most 15 minutes away. `Dockerfile.runtime-preflight` provides an optional
container entrypoint; its build context must contain only this public export.
Retain Docker's exact image identity beside the command receipts. This step
allocates no GPU and downloads no model. Its success does not establish inference.

The subsequent GPU worker first requests one small fictional model response,
then runs the complete fictional article workflow. Both phases share the
existing request, output-token, and absolute-time limits. Failed stages retain
their evidence and stop before the next phase. See `PUBLIC_RUNTIME.md` for the
separate own-corpus commands and runtime limitations.

## Reproduction and release status

`EXPORT-MANIFEST.json` binds every exported source file. `requirements-public-windows.lock` binds each dependency wheel. The generated build report identifies source, implementation, capture, replay and serving artifacts. Consult the exact cold-start evidence supplied with the release; the existence of these files alone is not proof that a test passed.

This is a local review candidate. It contains no private Git history or corpus. Author code-license selection, adapted-code rights review and final redistribution review remain pending; do not treat this directory as an already licensed public release. Dependency and font notices are separately identified. No journal submission has been filed by this build.
