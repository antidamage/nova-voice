# Nova Voice

Nova Voice is Nova's fully local, room-aware voice interface.  Its runtime
runs on Iridium.  It receives continuously framed microphone audio from native
satellites, determines whether a person is addressing Nova, interprets the
request, executes permitted household actions through documented providers,
and returns synthesized audio to the one satellite that captured the request.

`nova-voice` is a standalone product boundary.  It must not import dashboard
source, share dashboard components, add dashboard routes, or depend on a
dashboard page being open.  Nova is currently its first capability provider,
reached only through the dashboard's documented HTTP API and MCP interface.
Future providers join Nova Voice's provider registry without changing the
audio, interpretation, or session core.

## Implemented topology

```text
room microphones                     Iridium                    source satellite
Indium / Nocturnium --mTLS PCM--> satellite ingress --> VAD/election --> STT
       ^                                                              |
       |                                                              v
       +--- response PCM (source only) <--- TTS <--- LLM/session/action <---+
```

- Native satellites are supervised background processes, not browser
  microphones.  They capture and play audio whether or not the dashboard is
  open.
- The satellite transport is a mutually authenticated local-TLS WebSocket.
  It carries continuously framed 16 kHz mono PCM audio plus a versioned hello,
  health, timing, and playback-lifecycle control protocol.
- Central audio runtime applies VAD, speaker/room election, echo suppression,
  STT, session policy, interpretation, action execution, and TTS.  Individual
  implementations remain behind Nova Voice-owned interfaces.
- The response is rendered once and delivered only to the satellite whose
  microphone captured the request — the satellite that won source election
  and holds the turn's arbiter claim.  It is never fanned out to other
  satellites, even ones sharing the same `roomId`.

## Audio, rooms, and echo cancellation

Each satellite declares a stable identifier and `roomId` in its hello, but the
dashboard roster is the room authority: the voice server pulls per-satellite
room assignments with the voice settings (`satelliteRooms`) and the hello's
room is only a fallback for satellites the roster does not know.  Room IDs are
configuration, not inferred from hostnames.  Indium and Nocturnium are both
physically in the lounge, but `roomId` is used for election/arbiter scoping
and echo referencing only — it is not a playback fan-out list.  Whichever of
the two wins the utterance is the only one that speaks the response.

### One utterance, one handler

Satellites within earshot of each other hear the same speech, so the pipeline
enforces single-handling in three layers (`arbitration_scope`, default
`household`, treats every satellite as sharing the same air):

- **Source election** (`audio/election.py`): segments closing near-simultaneously
  are grouped per scope when their energy envelopes correlate *or* their
  capture intervals overlap by half the shorter segment; the highest
  SNR (+wake bonus) candidate wins the utterance.
- **Turn gate** (`audio/arbitration.py`): the elected satellite claims the
  scope for the duration of its turn, extended through response playback.
  Every other satellite's microphone is off (segments dropped,
  `segment_suppressed reason=turn_gate`) until the response finishes playing.
  Claims self-expire, are released on every early-drop path and on satellite
  disconnect, and are hard-capped, so the gate can never stick.
- **Transcript dedup** (`audio/dedup.py`): a post-STT safety net drops
  near-duplicate sequential transcripts inside a configurable window
  (same utterance heard twice); a longer rendering only upgrades the already
  displayed transcript line in place (`replacesId` on the dashboard transcript
  POST).  Unaddressed single-word fragments outside a conversation are dropped
  entirely (`ambient_min_words`).

### Response lock

Exactly one satellite plays each response — the one that won source election
and holds the turn's arbiter claim (`audio/arbitration.py`).  `RoomPlaybackRouter`
tracks every satellite connected to a room for diagnostics and echo scoping,
but `open_stream` only ever opens a stream to the winning satellite's own
connection; other satellites registered under the same `roomId` (e.g.
Nocturnium when Indium won) receive nothing, even though they are physically
co-located.  This closed a bug where every satellite sharing a room played the
response at once.  Satellites use a server-controlled preroll (currently
700 ms) before playback starts.

Because co-located satellites still share physical air, echo cancellation
stays scoped independently of who actually played:

- The server keeps the exact post-DSP response PCM as an echo reference for
  the whole arbitration scope (household by default), not only for the
  satellite that played it.  It compares incoming microphone frames with a
  20 ms energy envelope and also guards transcript matches.  This catches the
  playing satellite's own speaker leaking into a *different*, non-playing
  satellite's microphone in the same room, while still allowing a human
  barge-in.
- Indium is an input-only CoreAudio capture client.  It reports playback state
  and uses the shared server reference; macOS's native voice-processing/AEC is
  deliberately not claimed for that device.
- Nocturnium uses a PipeWire WebRTC echo-canceller graph.  Its virtual source
  `nova_voice_aec` is the default capture source and its virtual sink
  `nova_voice_aec_sink` is the default playback sink.  The capture process
  validates both defaults and fails closed if it would read the physical
  microphone directly.  It also sends the same playback lifecycle and common
  preroll used by Indium.
- `ops/configure-pipewire-aec.sh` creates and preserves this graph safely on
  repeat runs, including after its virtual nodes have become defaults.  The
  installer retains host-specific AEC targets instead of overwriting them.

The room echo guard is defence in depth, not a substitute for correctly routed
hardware AEC.  A satellite must never advertise AEC merely because a PipeWire
device exists; the process must actually capture the virtual AEC source.

## Runtime and interpretation

The central pipeline is selected through explicit interfaces and deployment
configuration:

- A pinned Pipecat processor prototype may be retained only if it meets the
  local latency and complexity gate.
- The intended LLM is official Qwen3.5-4B, locally converted Q4_K_M, run as a
  single resident llama.cpp instance in non-thinking, JSON-Schema/GBNF
  constrained mode.
- STT candidates are English Nemotron Streaming 0.6B, Nemotron 3.5 `en-GB`,
  and Parakeet TDT 0.6B v3.  The selected engine is measured on Iridium rather
  than chosen from published model-card figures.
- TTS candidates are Qwen3-TTS 12 Hz 1.7B/0.6B CustomVoice and current/original
  Chatterbox variants.  Exactly one measured winner is deployed at a time.
- One Python process shares a CUDA context between selected STT and TTS; the
  LLM is a separate llama.cpp process.  There is no runtime model swapping or
  silent startup downgrade.

Audio is continuously sent to Iridium.  Central VAD frames speech segments;
each final transcript receives structured interpretation.  A transcript-first,
editable wake list recognises `beemo`, common ASR spellings, and optional
greeting prefixes.  Direct requests or explicit desired states may act without
the wake word, while ambient speech has a much higher execution threshold.
Statements about what a person intends to do themselves never become actions.

The session engine tracks active conversation state, transcript confidence,
room/satellite provenance, and barge-in.  The wake-opened conversation window
is dashboard-tunable (`conversationIdleSeconds`, default 60 s, applied live to
both the conversation tracker and the goal-session clock) and is shared
household-wide under the default arbitration scope, so a follow-up elected on
another microphone continues the same exchange.  Abandonment phrases end a
conversation only when they are the whole utterance (courtesy tokens and the
wake word aside) — sentences merely containing a cue keep the window open.
Successful dashboard commands are acknowledged with a single spoken word
rendered by the LLM in the configured personality ("Done."), with the fixed
template as its deterministic fallback; failures keep a short informative
sentence.  Input emotion is estimated before
response rendering; persona configuration may shape wording and a bounded TTS
style instruction but cannot weaken execution, verification, privacy, or
safety policy.

## Actions and provider boundary

LLM output is a validated structured interpretation, not an authority to act.
One utterance may produce a bounded ordered `actions[]`.  Each action is
resolved against Nova Voice contracts, policy checked, sent through the
dashboard HTTP/MCP provider, and verified before success is spoken.  Discovery,
health, and administrative operations use the dashboard MCP endpoint.  The
dashboard's REST routes need not themselves be semantically versioned: Nova
Voice owns a versioned compatibility contract around external providers.

## Satellites and operation

Indium uses a macOS LaunchAgent so capture stays in the logged-in user's
audio/privacy context.  Nocturnium (address: see `PRIVATEREF.md#1.2`) uses a hardened systemd
user service for its kiosk user's PipeWire graph; a restricted system service
is appropriate only for direct ALSA capture.  Their processes and default
capture policy are always on.  Dashboard foreground state may be supplied as
interpretation context but never controls capture or process lifetime.

The health and diagnostics surfaces expose registered satellite pipelines,
room membership, protocol/AEC capability, playback state, provider readiness,
and selected model status.  Deployment is coordinated through the repository's
Voice/Dashboard chain; `deploy-nova-stack.ps1` installs the Iridium runtime and
does not restart unrelated LLM/TTS services unless required.

## Privacy and non-negotiables

- Inference and audio remain on the household LAN; there is no cloud fallback.
- Raw audio is not persisted by default.  Development transcripts and derived
  conversation text expire within 24 hours.
- Every response is delivered to exactly the one satellite that won the
  turn (the response lock), and is provided as an echo/cancellation
  reference to every microphone in the arbitration scope so a co-located,
  non-playing satellite is not confused by the playing satellite's audio.
- A response must be cancellable on its source satellite even when that
  satellite's native AEC is unavailable; server echo protection remains
  enabled.
- Persona may complain briefly, but cannot weaken execution verification,
  privacy, action policy, or safety behavior.
- Physical satellite acceptance and endurance remain gated by the measurements
  in [docs/PLAN.md](docs/PLAN.md); model candidates are not product guarantees
  until they pass Iridium's concurrent-residency, latency, schema, and local
  listening tests.

## Supporting documents

- [docs/PLAN.md](docs/PLAN.md): phases, deliverables, risks, and acceptance gates
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): component and data-flow detail
- [docs/MODELS.md](docs/MODELS.md): model research and VRAM budget
- [docs/CONTRACTS.md](docs/CONTRACTS.md): stable internal and LLM-facing contracts
- [docs/TESTING.md](docs/TESTING.md): command, ambient, latency, AEC, and retention tests
- [docs/DIAGNOSTICS.md](docs/DIAGNOSTICS.md): opt-in microphone/STT/LLM/TTS test page

The `skills/` directory holds compact instructions injected into the deployed
LLM.  `config/persona.example.yaml` keeps persona behavior and voice style in
configuration rather than application logic.
