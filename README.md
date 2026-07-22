# Nova Voice

Nova Voice is the standalone, local voice runtime for the Nova household
dashboard. It runs on Iridium and talks to Nova only through its documented
HTTP/MCP interfaces; dashboard source is deliberately not imported or bundled.

Wake-word conversations retain user/agent history until 60 seconds of
inactivity by default; the dashboard can change that timeout live. Under the
default household arbitration scope, a follow-up can move between microphones
without another wake word. Verified dashboard
commands receive a short personality-aware confirmation; only the satellite that
heard the elected request plays it, while every dashboard continues to receive
the speaking animation.

Voice characteristics are owned by Nova's `GET /api/voice` contract. A dashboard
change calls `POST /v1/settings/refresh` on Iridium; the endpoint accepts no
settings payload, fetches the complete contract from Nova, and applies speaker,
language, accent, pace, pitch, baseline mood, and emotion mirroring live. The
same collection runs at service startup, while dashboard outages leave the
configured environment/persona defaults active.
Indium and Nocturnium are native, supervised audio satellites. The dashboard
also contains a supported, opt-in browser satellite: its own microphone can run
push-to-talk or always-on while the page is open, and a dashboard-hosted bridge
relays its framed audio to Iridium over mTLS. Browser capture remains subject to
HTTPS, user permission, page lifetime, and each device's Voice Agent switch.

## Development quick start

```sh
cd nova-voice
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
nova-voice preflight
pytest
```

The default development settings keep passive execution and model-backed audio
disabled. `nova-voice text "turn the lounge lights on" --wake` exercises the
text/control path once the local LLM and Nova endpoint are configured. Use the
deployment units under `deploy/` for supervised audio operation; the checked-in
defaults never silently enable capture or execution.

The checked-in runtime is intentionally development-safe: shadow mode is on,
transcripts are retained for at most 24 hours, and raw audio is never written
to disk. Physical microphones, model weights, TLS identities, and the Nova MCP
token must be provisioned before enabling a live deployment.

## Layout

- `src/nova_voice`: replaceable audio, inference, interpretation, session,
  capability-provider, satellite, and retention adapters
- `skills/`: compact instructions supplied to the deployed local LLM
- `config/`: persona and satellite environment examples (no secrets)
- `docs/`: contracts, architecture, model/VRAM evidence, rollout, and tests
- `ops/`: pinned model preparation, preflight, and smoke-test scripts

The opt-in development microphone and response inspector is documented in
[`docs/DIAGNOSTICS.md`](docs/DIAGNOSTICS.md). It is served by Nova Voice over
the existing authenticated endpoint and is separate from the dashboard browser
satellite.

## Current implementation limits

- Iridium keeps final-buffer STT as the authoritative transcript. During live
  capture, stable cache-aware interim hypotheses may prefetch read-only room
  context and likely-tool/LLM state; no tool, memory, or speech can commit until
  semantic endpointing finalizes the turn.
- The deployed vLLM-Omni TTS service streams PCM chunks for one finalized reply
  as sentence/clause synthesis units, and playback can be cancelled within a
  unit or before the next one begins.
- The 427-test automated suite passes. A manifest-driven PCM16 replay runner
  and fake-clock household simulator cover deterministic failure reproduction,
  but the complete recorded household corpus, physical-microphone,
  latency, false-activation, concurrent-residency, and endurance gates remain
  acceptance work rather than completed product guarantees.
- Every handled foreground turn carries an immutable `TurnTrace` through
  capture, endpointing, context, interpretation, authorization, tool execution,
  verification, rendering, speech, and commit. Playback cancellation is
  independent from provider-task cancellation; mutations finish and verify once
  side effects may have begun, while read-only providers may opt into safe
  in-flight cancellation.
- Audio-native cadence endpointing adds a bounded pause after base VAD when a
  turn appears incomplete. While Nova is speaking, deterministic classification
  separates true barge-in from backchannels, cross-talk, echo/noise, and false
  triggers; only true barge-in cancels playback. A 70 ms nonverbal listening
  cue may play once during an addressed extended pause and never represents
  task completion.

Evaluation code under `nova_voice.evaluation` loads path-confined mono PCM16
fixtures, compares pinned transcript/trace/monitor outcomes, and records replay
latency. Its simulated household provider supports controlled time, ordered
events, delayed entity convergence, injected provider failures, occupancy, and
same-timestamp concurrent speakers.

All inference remains local to the household LAN. Raw audio is held only in
bounded memory; development transcripts expire after 24 hours.

## Private deployment reference

Concrete household details (hostnames, LAN addresses, account names, signing
identities) are deliberately absent from this repository. Documentation refers
to them as `PRIVATEREF.md#<section>`; that file is git-ignored and lives only
on household machines. Copy your own values into a local `PRIVATEREF.md` when
deploying.
