# MOSAIC: review copy of the public build

MOSAIC turns explicitly selected PDFs into source-bound candidate evidence and an
inspectable research interface. This repository is the reproducible software
artifact for a stage-one submission. It contains code, documentation and a
generator for a fictional PDF. It contains no published articles, private corpus
data, source quotations, screenshots, model weights or credentials.

## Start here

- [What the build does](build/README.md)
- [Install and run](build/PUBLIC_BUILD_QUICKSTART.md)
- [Following one source through the build](build/BUILD_GUIDE.md)
- [Design specification](build/DESIGN_SPEC.md)
- [Public archive versus the private demonstration](build/FEATURE_SCOPE.md)
- [Validation annex: what was tested and what was not](VALIDATION.json)
- [Review terms](REVIEW_TERMS.md) and [third-party notices](build/NOTICE.md)

## Run the fictional route

Reference environment: Windows x86_64 with CPython 3.12. From the `build/`
directory:

```powershell
py -3.12 start.py --no-serve
```

The launcher creates `.venv-public`, installs hash-locked dependencies from
PyPI, checks the installed inventory, and runs the fictional PDF through native
capture, the extraction driver with predetermined fictional responses,
materialization and indexing. Omit `--no-serve` to keep the local interface
open at the printed loopback address. The route downloads no model weights,
calls no model provider and needs no API key. Use a fresh directory: the
launcher refuses an environment that contains other packages or versions.

## What the evidence shows

The documented Windows route passed from a bare CPython installation with a
newly created environment. The same archive passed a CPU-only Linux fictional
loopback route in a pinned container. A controlled GPU run loaded the
open-weights model and returned a readiness response, but made no extraction
request, so complete real-model extraction is not yet established. Forty-one
historical checks remain pending because their original inputs are unavailable.
None of these software checks measures extraction accuracy on real articles.

## Integrity

Everything under `build/` is byte-identical to the frozen v40 archive and is
bound by `build/EXPORT-MANIFEST.json`. This repository stores files without
line-ending conversion (see `.gitattributes`).

Core archive SHA-256: `f8bb7e1ada4e42fe2a7e1746861ad48c3b538f92cf0ea4faf253dd0ac26c5bad`
Core manifest SHA-256: `832a9109eee31858fc440bf76ccd625c976c4b1d78ff9000212d939d29393fed`
Validation annex SHA-256: `ecf8acaf4fd6d2b7123d743711201f547698976c1f6d84ff351e5e08472b9087`
