# Single-package Linux dependency repair

The pinned GPU image contains an unused `opentelemetry-exporter-otlp-common`
0.66b1 distribution alongside an older, internally compatible telemetry family.
The former package demands SDK 1.45.1; LMCache 0.5.6 caps the telemetry API at 1.40.0
and the Prometheus exporter at 0.61b0. Upgrading nine telemetry packages broke
those LMCache bounds in the observed DE-87 run. That failed archive is preserved.

The new worker proposes removing only the unused common exporter from the
original image, retaining the original 1.40/0.61 family. It cannot repair a
previously upgraded environment: an unexpected baseline stops before mutation.
The older `public_linux_dependency_repair_v1.json` and legacy verifier remain
for historical replay; they are not the current worker's preparation path.

## Ordered checks

| Stage | Required evidence | Refusal condition |
| --- | --- | --- |
| Observe the original image | Full package freeze, installed metadata, captured Linux marker environment and raw dependency diagnostics | Unknown package state, malformed metadata or any unexplained conflict |
| Check every dependency | All active Requires-Dist edges, propagated dependency extras, and every possible incoming edge to the removal target | Any incoming target requirement, unknown constraint, or projected conflict beyond the exact vendor NCCL mismatch |
| Prove file ownership | Every wheel RECORD path set; actual target and OpenTelemetry file hashes; RECORD/METADATA hashes and physical locations; byte hashes of OS package metadata without RECORD files | Shared target ownership, ambiguous telemetry installations, unsafe paths, unknown recordless locations or an incomplete target inventory |
| Remove one distribution | One explicit pip uninstall of the named target | A different removal, unexpected package/file change or retained target files |
| Verify the remaining environment | Full freeze/metadata/graph/check, unchanged retained file inventories, and SDK trace/metrics, OTLP gRPC/HTTP and Prometheus imports | New conflict, changed remaining package version or monitored file, or failed import |
| Add the PDF wheel | The existing hash-pinned PyMuPDF supplement, followed by complete checks | Drift outside that declared supplement |
| Permit model work | All prior checks and their retained receipts succeed | Missing or contradictory evidence; the model cannot start early |

The vendor's NCCL 2.30.7 override remains bound to the exact vLLM image/source
Dockerfile. Its raw failing pip diagnostic is retained. This exception never
applies to another dependency conflict. The repair does not upgrade telemetry,
Torch, NCCL, LMCache or the model.

The ownership check hashes actual files in the affected OpenTelemetry namespace.
For other wheel packages it checks complete RECORD inventories and METADATA,
without rereading every CUDA/Torch binary. OS packages without wheel RECORDs
retain their complete metadata-directory file hashes. It therefore establishes a bounded
file-integrity claim, not a complete byte-for-byte audit of the container.

The first actual CPU trial found 279 physical metadata locations for 275
distinct distribution names. Four older OS locations shadow newer pip packages,
and 16 OS locations have no wheel RECORD. The prior inventory assumed one
RECORD-bearing location per name and correctly stopped before uninstall, but
could not represent this real image. The successor preserves every physical
location, including shadowed versions. It permits recordless metadata only in
the observed `/usr/lib/python3/dist-packages` root, hashes those metadata files,
and still refuses duplicate physical locations, repeated wheel paths, ambiguous
telemetry installations, and unknown recordless roots. The dependency graph
continues to cover the 275 active names; physical inventories are a separate
before/after check. These changes require a new Linux trial, not a retrospective
pass for the failed export.

## What validation can establish

Retained Linux metadata covers 275 distributions. Independent and root replays
reproduce two original conflicts and three after the unsuccessful upgrade.
Projecting the one-package removal leaves 274 distributions and only the exact
NCCL mismatch. Originally requested application extras are unknown; the graph
propagates extras explicitly requested by active installed dependency edges.

The implementation has focused refusal/retained-evidence tests and an isolated
Windows installation/import trial. The exported CPU-only Linux preflight adds
an observed check in the pinned image: the complete repair sequence, PyMuPDF,
and imports of Torch, Tokenizers and XGrammar. A passed claim requires its
terminal receipt plus an external Docker observation binding the actual image
ID; the declared image string alone is insufficient. Consult the final
preparation checkpoint for the actual result.

Those CPU checks do not establish CUDA/NCCL initialization, vLLM server
compatibility, model acquisition or inference. The separately approved GPU
worker imports vLLM's API-server entrypoint and the public corpus workflow before
downloading weights, then requires one retained fictional response before the
full fictional route. The public Windows demonstration remains a distinct
software test with manufactured responses.

Every future attempt must keep its own baseline, mutation and post-PDF receipts,
hashes, result archive and independently verified deletion. Failed runs, older
exports, frozen science results and the 41 unavailable historical checks remain
unchanged. No hosted scientific extraction, Jev request, private-corpus upload,
publication or scientific qualification follows from this repair.
