# Public Jev advisory sidecars

Public Jev use is an optional, explicit post-extraction advisory. It is not part of extraction, qualification, validation, or reader visibility. A missing credential, refused task, unavailable service, or failed request leaves the public candidate and reader usable.

A `mosaic-public-jev-task/v1` task binds a `public-` id, completed candidate-snapshot path and SHA-256, exact source-packet and PDF SHA-256 values, caller-declared aliases/protected hashes, caller permission, source unit selections (`unit_id`, `start`, `end`), definitions, and questions. The runtime reconstructs quotations from the snapshot units and hashes each literal witness. It rejects declared protected aliases/hashes without consulting a private registry; this is not a claim that undeclared protected material has been independently cleared.

Callers either inject the installation's canonical-accounting `JevClient` or explicitly build one with `make_public_client(task, ledger_path=..., budget=...)`. That retains append-only cumulative-ledger, budget, idempotency, unknown-outcome, `require_secret`, and `zeroDataRetention: true` semantics. Both ledger and budget are installation-scoped and explicit; neither is reset per run nor loaded from a private default. No call occurs until `assess_public_task(...)` or the CLI's `--send` flag is used.

An assessment writes an immutable canonical `public-jev-advisory-<task-id>.json` sidecar through `Workspace`; it repeats the task and source bindings and records `caller_permission_is_not_independent_clearance`, `scientifically_accepted: false`, and `may_not_gate_visibility: true`. It never edits the snapshot or extraction results. Offline replay validates the task, snapshot, literal witnesses, and caller-provided sidecar SHA-256 without initializing a client, loading a key, opening a ledger, or making a network request.

```powershell
.venv-public\Scripts\python.exe -m mosaic.public_jev assess --task public-task.json --out public-sidecars --ledger public-ledgers\public-jev-calls.jsonl --usd-cap 200 --request-cap 20000 --send
.venv-public\Scripts\python.exe -m mosaic.public_jev replay --task public-task.json --sidecar public-sidecars\public-jev-advisory-public-example.json --sidecar-sha256 <recorded-sha256>
```

Choose the budget explicitly for the installation and retain the same ledger
across tasks; the example is not a provider-credit purchase or an automatic
allowance. Set `AI_GATEWAY_API_KEY` in the operator's environment. It is loaded
server-side through `require_secret` only at send time. Output paths are checked
before a request. A pre-existing sidecar requires explicit replay; verified
sidecar reuse does not call the provider again. The exported checks use a fake
transport and dummy key and do not establish live gateway compatibility.
