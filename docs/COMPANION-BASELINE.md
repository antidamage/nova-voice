# Companion offload — pre-change baseline

Every claim here was read out of the source at the commit this document was
written against. Where a seam the roadmap assumed does **not** exist, that is
recorded as a correction rather than smoothed over — the whole value of this
document is that later tasks can rely on it without re-deriving it.

## What actually occupies llama.cpp's single slot

`llama-server` runs with `--parallel 1`: one request at a time. Exactly five
call sites reach it. This is the complete list — it was produced by tracing
every non-`interpretation/` caller of the `Interpreter` protocol, not by
reading the roadmap's table.

| # | Pass | Interpreter method | Caller | In a spoken turn? |
|---|---|---|---|---|
| 1 | Interpretation | `interpret` | `service.py` `handle()` | yes — hot path |
| 2 | Reply rendering | `render_response` | `service.py`, 3 call sites | yes — hot path, feeds TTS |
| 3 | Identity disclosure | `extract_self_profile_update` | `service.py` | yes — per turn |
| 4 | Objective confirmation | `confirm_objective` | `service.py` → `providers/nova/verify_loop.py` `llm_confirm` | yes — inside the verify loop |
| 5 | Reminder icon | `classify_icon` | `api.py` (dashboard proxy) | no — out of turn |

The remaining `Interpreter` attributes that appear in a naive grep
(`wake_words`, `personality`, `render_temperature`, `pronoun_instruction`,
`agent_name`, `web_access_enabled`, `web_answer_max_sentences`,
`long_response_probability`) are **settings properties, not model calls**. They
do not touch the slot.

### Correction: four roadmap workloads have no local implementation

The roadmap's default route table lists research synthesis, briefing
composition, automation drafting and a verification/review pass as routable
workloads. None of them is an LLM call today:

- **`ResearchManager`** (`research.py`) has no interpreter dependency at all.
  Its output comes from `_spoken_summary()` and `_citations()`, which are
  deterministic string assembly over `WebProvider` results.
- **`BriefingManager`** (`briefings.py`) likewise. `_agenda()` and
  `_conflicts()` are pure Python over calendar items.
- **`AutomationManager.draft()`** (`automation.py`) does not generate anything.
  It validates that the caller already supplied summary, trigger and proposed
  actions, and raises `AutomationLifecycleError` otherwise.
- **The verification/review pass is not a separate workload.** It *is*
  `confirm_objective`, invoked as `verify_loop`'s `llm_confirm` hook.

Consequence for NPT-003: a workload registry that "fails if a configured
workload lacks a handler" cannot list these four, because there is nothing to
fall back *to*. Offloading them (NPT-408/409) requires first building a local
LLM synthesis stage in those managers — new scope those tasks assumed already
existed. See the plan's M0 status note.

## Seams the companion work attaches to

### Interpreter

`interpretation/base.py` defines the protocol. `interpret` is the only
`@abstractmethod`; `extract_self_profile_update`, `classify_icon`,
`confirm_objective` and `render_response` have default implementations
returning `None`, which is why they are safe to route — every caller already
handles `None` as "this pass did not happen".

`interpretation/llama_cpp.py` is the local implementation. Its pinned opening
block builds `semanticTools` + `relevantState` + `selectedMemory` against
`--ctx-size 16384`; a 4,096-token companion budget cannot take it unmodified
(NPT-303).

### Capabilities

`capabilities/base.py` — `CapabilityProvider` (`manifest`/`execute`/`health`/
`close`), `CapabilityManifest` with
`execution_class: "iridium_local" | "household_lan_service"`, and `ToolPolicy`
(`risk`, `reversible`, `idempotent`, `parallel_safe`, `resource_templates`,
`requires_confirmation`, `cancellation`).

`capabilities/registry.py` — `CapabilityRegistry` validates every tool's JSON
Schema at registration, enforces that policies and tools match exactly, and
resolves `resource_templates` into concrete lock names. The allowlist in
`bootstrap.py` gates which provider ids may register at all.

### Execution and authority

`service.py::_execute_plan` executes a bounded action DAG with
provider-declared parallelism, materialises dependency failures as `blocked`
results, and binds cancellation per action. `HouseholdAuthority.authorize()`
maps a `SpeakerIdentity` to a role, then to base capabilities or a standing
delegation grant.

`service.py::execute_companion_action` is the companion's entry into that path
(added by this work): it validates through the registry, refuses `blocked`
risk, refuses anything needing confirmation, restricts speakerless background
jobs to low-risk actions, then defers to `_execute_plan`.

### Durable plans — the approval machinery NPT-406 assumes

It exists.

- `durable/models.py` — `PlanStepKind.QUESTION | APPROVAL | EVENT`,
  `PlanStepState.PAUSED`, and `ExecutionRecord` carrying `idempotency_key`
  (UNIQUE in SQLite), `lease_owner`, `lease_token`, `lease_expires_at`
  (validated timezone-aware).
- `durable/runner.py:146` pauses on those three step kinds;
  `resolve_step()` (line 323) resumes only a `PAUSED`/`WAITING` step of a
  permitted kind.

So durable approvals are an extension of working machinery, not new
infrastructure.

### Offline optimizers

`offline_optimizers.py` — `OfflineOptimizerPool` is explicitly
"recommendation-only, isolated from foreground execution" and reports
`"mode": "recommendation_only"`. NPT-409's constraint is an existing invariant
to preserve, not a new rule to add.

### Audio and satellites

`satellites/protocol.py` — NVAF v1. `SatelliteHello.client` is now
`linux-native | macos-native | browser | ios-native`. Frames are exactly
`BYTES_PER_FRAME` (640 = 20 ms of 16 kHz mono PCM16); `api.py` closes 1003 on
any other size.

Arbitration: `audio/election.py` (energy-envelope, ±10 frames) and
`audio/dedup.py` (6 s / 0.82 similarity) already resolve two satellites hearing
one utterance. A mobile election penalty (NPT-606) hooks in there.

### TLS

`cli.py` runs uvicorn with `ssl_cert_reqs=2` when a CA is configured, so the
listener *requires* a client certificate. But uvicorn publishes nothing about
that certificate into the ASGI scope — verified against uvicorn 0.46's
`wsproto_impl.py`, whose `handle_connect` builds a scope containing only
`{"websocket.http.response": {}}` under `extensions`.

This is why identity binding is application-level (NPT-102) rather than read
from a peer certificate.

## Dashboard seams

- `app/api/voice/satellites/route.ts` — the pattern a companion admin proxy
  follows: server-side mTLS via `data/nova-voice-tls/`, no upstream URL from
  the browser.
- `lib/voice-satellite-bridge.ts` — browsers cannot present a client
  certificate, so the dashboard relays browser satellites upstream under its
  own identity. Any identity binding must keep that path working: the hello
  carries a browser's id while the certificate says `nova-dashboard`.
- `lib/voice-satellite-reconnect.ts` assumes anything with the
  `voiceSatellite` capability is SSH-restartable. An iPhone is not (NPT-807).
- The dashboard has no `person.*` / `device_tracker.*` presence source today.

## Retained-for-comparison metrics (NPT-002)

Not captured. Measuring queue wait and TTFT under contention requires load on
the live household voice stack, which is not something to do unattended. The
harness and the exact quantities to capture are specified in
`docs/COMPANION-BASELINE-MEASUREMENT.md`; the run itself needs a quiet window
and the owner's go-ahead.
