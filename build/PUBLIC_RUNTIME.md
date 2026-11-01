# Public runtime replay boundary

`mosaic.public_runtime.run_extraction(capture_receipt, output, *, workspace,
profile, replay)` is a narrow adapter over the current complete-driver route:
`prepare_upstream`, `prepare_program`, `run_program`, and the final
`finalize_with_study_facts` wrapper. It is not a CLI, inference server, or
public scientific-extraction service.

## What it accepts

The caller provides a path-free capture handoff dictionary containing the public-capture
receipt, `source.json`, `source-binding.json`, and
`prepare-upstream-input.json`. The adapter recomputes the canonical source
packet digest and requires agreement among the packet, binding, and upstream
input. It also requires an explicit `fictional_fixture_local_only`
authorization for the exact PDF hash. `PublicScope` permits only that work ID,
that single PDF hash, `fixture_replay` provenance, `remote=false`, and the
same fictional-fixture right.

The only implemented profile is `config/public_runtime_reference_v1.json`
(packaged beside this document in the export as a configuration artifact):
`fictional_fixture_replay_only`. Its `replay` argument supplies an in-memory,
recorded fixture transport and token counter. No endpoint, provider, model
weights, GPU, socket, or downloader is selected by this module. It records the
frozen program, phased intent/outcome ledger, wrapped candidate report, public
snapshot, and immutable runtime receipt through `Workspace`.

`fixture_replay_for_demo()` is the only canned convenience transport. It first
requires the exact deterministic `mosaic.public_demo` PDF SHA and its literal
`U001` `Study 1` source witness, then supplies manufactured fixture answers:
empty findings, one uncertain synthetic study, and empty study facts. It cannot
be applied to arbitrary PDFs, and those answers are labeled manufactured—not
recorded model output.

The current request builder really compiles every task with XGrammar:
`Grammar.from_json_schema`, then `Grammar.from_ebnf`, and embeds the schema in
the system prompt with `chat_template_kwargs.enable_thinking=false`. This is
the current grammar request shape, rather than the obsolete legacy
`candidate_batch`/`response_format` contract.

## Dependency and live-runtime status

The observed xgrammar 0.2.8 metadata requires `apache-tvm-ffi`, `pydantic`,
`torch`, `transformers`, `numpy`, and `typing-extensions`; Linux x86_64 also
declares `triton`. A CPU fixture replay therefore needs real compatible wheels
for that closure even though it needs no model weights or vLLM server. The
reference records the observed tokenizer version (0.23.2) and the existing
tokenizer artifact hash, but fixture replay injects its counter and does not
open a tokenizer cache.

The pinned model/revision identify the separately reviewed Linux experiment.
The CPU preflight can verify the image's dependency repair and selected imports;
its exact scope and terminal status are retained separately. It does not establish
accelerator/driver compatibility, a model-file manifest, inference, or a supported
end-to-end runtime. A live driver must be
co-located with its Linux loopback server; Windows may perform capture and
offline inspection only until a distinct transport is reviewed.

## Native preparation

`mosaic.public_native_runtime` now provides a separate preparation path. It
loads the explicit public capture directory, checks the frozen PDF and source
bindings, validates a complete caller-selected local model tree, checks the
tokenizer, and freezes a native program through the existing driver. It permits
only a literal `http://127.0.0.1` endpoint with an optional port and `/v1` path.
The public scope binds one work, one PDF hash, native text, local processing
and the caller's permission declaration. That declaration is not independent
rights clearance. No private source registry is used.

The public command is `bootstrap.py prepare-local`; run `--help` for its six
explicit input/output arguments. Copy and fill
`config/public_native_driver.example.json`; its model/revision placeholders
must agree with the local model manifest. The output freezes the program,
profile, manifest and preparation receipt. This command has no model execution
API, socket, download, provider call or GPU launch. Its status is
`prepared_only_execution_unvalidated`.

## Native loopback execution

`mosaic.public_execution.run_local_program(prepared_root, capture_root,
output, *, workspace, deadline_epoch, runtime_observation=None)` is the
separate execution boundary. It reopens the frozen public PDF/capture packet,
checks every prepared artifact hash, revalidates the model-file manifest and
local tokenizer, rebuilds the strict source scope, and accepts only the literal
prepared `127.0.0.1` endpoint. It then calls the current native
`run_program` with `live=True`; there is no hosted fallback, private scope, or
injected fixture transport in that route.

The preparation root receives one immutable output binding before a socket is
opened. A second output is refused; the same output can only return an existing
terminal receipt. The driver owns its new append-only ledger, finite request/
token/time/article caps, intent-before-send records, and unknown-outcome stop.
This adapter never retries an incomplete or unknown send.

The returned `report_path` is the exact wrapped per-paper report, rather than
the aggregate driver envelope. This lets `source_handoff` and the public
materializer retain the study-fact wrapper and its source witnesses. The receipt
also pins the aggregate report plus every driver and per-run-ledger artifact.

`deadline_epoch` is the caller's send-window deadline. The adapter adds the
driver's fixed 60-second cleanup reserve only when invoking `PhasedRun`; it
does not increase the frozen request or wall-clock budget. A served model must
appear in the literal loopback server's `/v1/models` response before any driver
request is created. Terminal reuse recomputes the original binding and hashes
for the intent, driver report, and snapshot.

The receipt records actual local protocol observations from `/health` and
`/v1/models`, plus the Python/OS and selected installed distribution versions.
Any supplied `runtime_observation` is retained only as a declaration. A mocked
loopback exchange is explicit software-test evidence and records
`real_model_inference_run=false`; a successful call does not certify model,
GPU, runtime, or scientific support. Reports and snapshots remain
candidate-only.

Actual Linux execution remains a bounded smoke experiment: the co-located vLLM
server, installed runtime/container, device compatibility and model behavior
must be observed and independently assessed. An existing successful smoke is
not a prerequisite for preparing or invoking this route. The private
`VLLMScope` is not reused.

`mosaic.public_runtime_manifest` now makes that prerequisite reviewable without
making it runnable. It validates an explicit, complete caller-selected model
tree (every relative file path, byte count, and SHA-256; no extras), tokenizer
identity, Linux runtime wheel lock/container digest, and declared hardware.
Its illustrative launch-plan renderer is explicitly `UNVALIDATED_DO_NOT_EXECUTE`, binds only
to `127.0.0.1`, and contains no execution or live-call API. Even a declared
smoke receipt leaves `supported_runtime=false`: independent Linux/GPU evidence
must be assessed outside this pure local-manifest preflight.

## Bounded Linux reference smoke

The exported `mosaic.public_linux_smoke` worker executes one generated fictional
PDF on a separately supplied Linux GPU. It does not rent hardware or read a
provider key. Its `run` command requires the exported root, a new model directory,
a new output directory under `run/`, and an absolute work deadline no more than
60 minutes away. The rental owner must independently enforce deletion, including
the five-minute cleanup reserve.

The worker pins the vLLM container digest and Qwen model revision in its source,
checks disk headroom and download CLI availability, records the unmodified base
dependency state, applies the narrowly pinned OpenTelemetry repair when needed,
and verifies the separately hash-locked PyMuPDF supplement. Raw diagnostics,
package mutations and the exact vendor NCCL exception remain inspectable;
unknown conflicts stop the worker before model download. See
[the dependency contract](LINUX_DEPENDENCY_REPAIR.md). The worker records
installed distributions and observed GPU/driver information. Before acquiring
weights it runs a bounded child process that imports vLLM, its API-server
entrypoint, and the public corpus workflow. A failed import stops before the
download; this phase does not start an engine or process a PDF. Acquisition has a
20-minute limit and preserves a 30-minute reserve;
the model inventory runs as a bounded child process. The server binds only to
`127.0.0.1`, and both `/health` and the exact served-model identity must pass
before the model probe begins. One fixed fictional READY probe has a one-request,
32-output-token, 60-second cap and retains its raw request and response. Only a
verified completed probe permits the public corpus route, whose caps are seven
requests, 49,120 output tokens and a 600-second send budget. The combined limit
remains eight requests and 49,152 output tokens, bounded by the outer deadline.
Its own server is terminated in `finally`.
Both success and failure retain an immutable outcome and a result archive;
timed-out child phases retain partial logs. The worker requires at least
121,172,064,868 free bytes before acquiring the pinned 55,586,032,434-byte file
roster. This space check is a conservative resource guard, not a measured runtime
guarantee or authority to rent hardware.

This is an executable experimental profile, not a portability guarantee. The
mocked execution checks establish orchestration and cleanup only. An actual
observed smoke, its failure stages, and separately verified rental deletion must
be reported from the experiment's receipts. No third-party PDF or private corpus
is part of this test. The generic corpus commands remain separate from this
fixed fictional experiment.

## Interpreting candidate results

The emitted report and snapshot are candidate-only. A materialized fixture
proposal means only that a synthetic answer passed the current mechanical,
source-binding contracts. It neither qualifies a paper nor assesses scientific
accuracy. Source units are model-eligible packet coverage, not a guarantee of
full-article reader coverage.
