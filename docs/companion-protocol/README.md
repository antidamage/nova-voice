# Companion protocol v1 — frozen envelopes

One JSON file per message type, each a valid instance of the corresponding
model in `src/nova_voice/companion/protocol.py`. They serve three purposes:

1. **Executable documentation.** These are what the wire actually looks like,
   not prose describing it.
2. **Server round-trip tests.** `tests/test_companion_fixtures.py` parses every
   file through the strict models and re-serialises it, so a fixture can never
   drift from the implementation without a test failing.
3. **Swift conformance fixtures.** The iOS package decodes these same files
   (NPT-905), which is how both sides are kept in agreement without a shared
   code generator.

## Conventions

- Field names are camelCase on the wire; the Python models declare them as
  aliases and are populated by alias.
- Timestamps are RFC 3339 with an explicit offset, always UTC (`Z`).
- Every job-related frame carries `jobId`, `attemptId` and, in the offer's
  envelope, the `idempotencyKey` and `inputRevision` that survive a retry or a
  local fallback.
- `traceId` is safe to log. Nothing else in a payload is.

## What must fail

Strictness is part of the contract, so the negative cases are fixtures too,
under `invalid/`. Each must be *rejected*:

- an unknown message type
- an unknown field on a known type
- a job result whose deadline fields are missing
- a frame over the size cap

If any of these ever parses, the edge has stopped being an edge.

## Result payloads

A `job_result`'s `result` object is validated separately, against the schema
named by its workload's `resultSchema` — see
`src/nova_voice/companion/workloads.py`. Those shapes come from
`model_json_schema()` on the same domain models the local passes already
constrain their sampler with, which is why field names there are snake_case
while the envelope around them is camelCase. That asymmetry is deliberate: the
envelope is a new contract, the result shapes are an existing one.
