# Nova Voice

The voice stack for Nova. Streaming speech recognition, a language model that
calls tools, and speech synthesis, wired together with session state,
multi-microphone arbitration and an authorization boundary in front of anything
that acts on the house. Every stage runs on household hardware.

It is a standalone service. It talks to [Nova HA Dashboard](https://github.com/antidamage/nova-ha-dashboard)
only through that project's documented HTTP/MCP interfaces — dashboard source is
never imported or bundled, so either side can be replaced independently.

## Where it fits

| Component | Interface |
|---|---|
| In-room microphone satellites (native, supervised) | Framed audio over mTLS |
| The dashboard's opt-in browser satellite | A dashboard-hosted bridge relays framed audio over mTLS |
| Nova HA Dashboard | Owns `GET /api/voice`; calls `POST /v1/settings/refresh` to push changes |
| Every open dashboard | Receives speaking-animation events during a reply |
| Home Assistant | Reached through the dashboard's tool surface, not directly |

## What it does

**Endpointing.** Audio-native cadence endpointing adds a bounded pause after
voice-activity detection when a turn sounds incomplete, so trailing off
mid-sentence doesn't submit half a request.

**Session state.** A wake word opens a conversation that stays open until 60
seconds of inactivity; the dashboard can change that timeout live. Follow-ups
need no second wake word.

**Arbitration.** Under the default household scope, a follow-up can land on a
different satellite than the one that heard the wake word. When several
satellites hear the same request, an election picks one to reply; the rest stay
silent while every dashboard still shows the speaking animation.

**Interruption classification.** While Nova is speaking, deterministic
classification separates true barge-in from backchannels, cross-talk, echo and
false triggers. Only true barge-in cancels playback.

**Commit ordering.** Final-buffer recognition is the authoritative transcript.
Interim hypotheses may prefetch read-only context and likely tool state, but no
tool call, memory write or speech commits until semantic endpointing finalizes
the turn.

**Authorization and verification.** Each turn carries an immutable `TurnTrace`
through interpretation, authorization, execution and verification. Verified
commands get a short spoken confirmation.

**Voice configuration.** Speaker, language, accent, pace, pitch, baseline mood
and emotion mirroring come from the dashboard's `GET /api/voice` contract and
apply live.

**Locality.** Audio, transcripts and model calls stay on the LAN. Raw audio lives
only in bounded memory and is never written to disk; development transcripts
expire after 24 hours.

## Install

```sh
cd nova-voice
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
nova-voice preflight
pytest
```

The checked-in defaults are development-safe by design: passive execution and
model-backed audio are **off**, shadow mode is on, transcripts expire after 24
hours, and raw audio is never written to disk. Nothing here silently enables
capture or execution.

Exercise the text and control path without audio:

```sh
nova-voice text "turn the lounge lights on" --wake
```

For supervised audio operation, use the units under `deploy/`. A live deployment
additionally needs physical microphones, model weights, TLS identities and the
Nova MCP token provisioned first.

## Layout

```
src/nova_voice/   replaceable audio, inference, interpretation, session,
                  capability-provider, satellite and retention adapters
skills/           compact instructions supplied to the deployed local LLM
config/           persona and satellite environment examples (no secrets)
docs/             contracts, architecture, model/VRAM evidence, rollout, tests
ops/              pinned model preparation, preflight, smoke-test scripts
```

An opt-in development microphone and response inspector is documented in
[`docs/DIAGNOSTICS.md`](docs/DIAGNOSTICS.md). It is served over the existing
authenticated endpoint and is separate from the dashboard's browser satellite.

## How a turn is processed

Every handled foreground turn carries an immutable `TurnTrace` through capture,
endpointing, context, interpretation, authorization, tool execution,
verification, rendering, speech and commit. Playback cancellation is independent
of provider-task cancellation: mutations finish and verify once side effects may
have begun, while read-only providers can opt into safe in-flight cancellation.

Final-buffer speech recognition is the authoritative transcript. During capture,
stable interim hypotheses may prefetch read-only room context and likely tool or
model state, but nothing commits — no tool call, no memory write, no speech —
until semantic endpointing finalizes the turn.

The synthesis service streams PCM in sentence and clause units, so playback can
be cancelled inside a unit or before the next one starts.

## Testing and evaluation

The automated suite (442 tests) passes. `nova_voice.evaluation` loads
path-confined mono PCM16 fixtures, compares pinned transcript/trace/monitor
outcomes and records replay latency. Its simulated household provider supports
controlled time, ordered events, delayed entity convergence, injected provider
failures, occupancy and same-timestamp concurrent speakers — which is what makes
a failure reproducible instead of anecdotal.

A SQLite evaluation registry grades outcome, policy, trace, latency, memory and
proactivity deterministically, calls a model grader only for inconclusive
structural metrics, and emits an explicit deployment-gate decision. Telemetry
carries only structure — revisions, timings, queue depths, stage and tool
outcomes, interruption classes, error types — never audio or transcripts.

### Not yet claimed as verified

The recorded household corpus, physical-microphone, latency, false-activation,
concurrent-residency and endurance gates are acceptance work in progress. Tier 0
acceptance ships a 24-hour monitor and a fail-closed gate covering model
stability, GPU headroom, queues, recognition progress during speech, unique
mutation IDs, latency percentiles, eleven corpus cases and real production
streaming — the code is deployed, but the milestone stays open until real
duration and corpus evidence passes.

## Private deployment reference

Hostnames, LAN addresses, account names and signing identities are deliberately
absent from this repository. Documentation refers to them as
`PRIVATEREF.md#<section>`; that file is git-ignored and lives only on household
machines. Copy your own values into a local `PRIVATEREF.md` when deploying.
