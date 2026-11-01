"""Jev through the Vercel AI Gateway, advisory only, behind eight hard stops (HANDOFF §6.6; FINAL v2 §12.5).

`ledger` holds the append-only call ledger and the cumulative caps, `scope`
the document-scope check against the frozen holdout, and `client` the
TypeSafe-compatible adapter. The stops, in the order the client applies them:
(8) document scope, (7) authorization beyond AUTH-1, (5) prompt/state hash,
(1) idempotency, (3) cumulative caps from the persisted ledger, (2) ledger
write before the call and after the response, and (4) one retry at most,
never after an unknown outcome. Stop (6), the secret scan, runs in the
pre-commit hook over the working tree, the ledgers and the staged diff.
"""
