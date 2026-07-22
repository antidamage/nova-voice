# Architecture and behavior

## System map

```text
Nocturnium service / Indium LaunchAgent / Iridium mic / dashboard browser
        | native mTLS WebSocket       | browser WSS via dashboard mTLS bridge
        +-----------------------------+----------------+
                               v
                    Nova Voice on Iridium
  edge/capture gate -> central VAD/wake -> final-buffer STT -> prosody -> interpretation
                                                   |
                                      session + policy engine
                                                   |
                         semantic capability tools -> verifier
                                                   |
                                  persona reply -> tone mapper -> TTS
                                                   |
                          elected input satellite speaker only
                                                   |
                         speaking-state event -> every dashboard client

                    capability-provider boundary
                          | Nova adapter today
                          | future providers later
                          v
              dashboard REST/MCP (external protocol only)
```

Iridium is the only inference host. Satellites capture/play audio and report
room/device metadata; they do not host alternative STT, TTS, or LLM models.
The legacy Home Assistant Assist/Wyoming service has been retired and is no
longer a supported fallback.

The opt-in development diagnostics client is a Nova Voice-owned page served by
the same mTLS API. It uses explicit push-to-record browser capture and feeds a
bounded PCM utterance into the exact resident STT -> interpretation -> policy ->
persona -> TTS path. It is not a dashboard component, an always-on microphone,
or an alternate inference pipeline, and it is unavailable outside development
shadow mode.

The supported dashboard browser satellite is a separate production path. Its
AudioWorklet emits the same framed PCM protocol through a same-origin WSS
connection; a dashboard-hosted bridge relays bytes to Iridium with the
dashboard's mTLS client identity. Push-to-talk sends `begin_turn`; page-bound
always-on mode uses normal wake gating. It depends on a secure context,
microphone permission, an open page, and the per-device Voice Agent setting.

## Repository and ownership boundary

Nova Voice owns its inference runtime, services, schemas, configuration, health
endpoint, metrics, native satellite protocol, and deployment. It never imports
dashboard source. The dashboard remains an external capability provider over
REST/MCP, and separately owns the optional browser capture client and its dumb
mTLS relay. Those browser-edge components do not contain inference, policy, or
tool execution logic.

The only permitted knowledge crossing that boundary is:

- documented REST/MCP operations wrapped by a Nova Voice-owned, versioned
  compatibility contract and conformance fixtures
- authentication and endpoint configuration supplied at deployment
- runtime state and action results returned by those interfaces

Contract fixtures live under Nova Voice and are tested against a fake provider
plus a read-only dashboard integration test. A dashboard release is not
required to add, deploy, disable, or repair the native voice runtime.
Browser-satellite changes are coordinated across the Voice/Dashboard deployment
chain. Killing Nova Voice must leave the dashboard, Home Assistant, and touch
control unchanged.

The core depends on a generic `CapabilityProvider` interface. The initial
`NovaDashboardProvider` is one adapter; later local-AI, media, household, or
other task providers register manifests, schemas, policy metadata, and compact
LLM skills without entering the audio/session packages.

Provider manifests declare execution locality. All local-AI capabilities are
hard-routed to Iridium; satellites are audio I/O clients and cannot become tool
execution hosts. LAN-service providers may call an explicitly configured local
endpoint, but no provider gets implicit shell or network access.

## Pipeline processors

Prototype Pipecat for frame ordering, interruption semantics, and replaceable
processors, pinning the exact tested release. Transport, sessions, policy, and
model/provider adapters remain behind Nova Voice-owned interfaces. Keep Pipecat
only if the custom local STT/TTS adapters meet latency and maintenance gates; it
must be replaceable without changing domain contracts. Capability-specific
processors remain outside the core.

1. `SatelliteIdentityProcessor`: authenticates and attaches device/room metadata.
2. `AudioConditioningProcessor`: resamples to 16 kHz mono for STT/wake and emits
   a parallel playback-quality stream.
3. `ActivityProcessor`: server VAD, utterance framing, pre-roll, audio-cadence
   endpointing, bounded pause extension, and one-shot listening-cue signal.
4. `WakeProcessor`: transcript-first matching against the dashboard-managed
   accepted-word list, with an optional acoustic model when one has been provisioned.
5. `SttService`: the selected Nemotron adapter. Cache-aware live hypotheses feed
   a stable-prefix gate for read-only prefetch; a final batch decode remains
   authoritative.
6. `ProsodyProcessor`: deterministic, baseline-relative acoustic features.
7. `DeduplicationProcessor`: elects one of overlapping household streams.
8. `InterpretationProcessor`: calls the single LLM with bounded structured context.
9. `SessionPolicyProcessor`: applies passive/active thresholds and goal state.
10. `CapabilityToolProcessor`: validates a bounded ordered action plan, routes,
    executes, and verifies each action through a provider; Nova alias resolution
    lives only in its provider adapter.
11. `PersonaRenderer`: renders verified results, bounds complaint/style behavior,
    and compiles tone. Common results are deterministic; complex results may use
    the same resident LLM after execution.
12. `StreamingTtsService`: sentence/clause output with mid-unit and inter-unit
    barge-in cancellation.
13. `RetentionObserver`: writes TTL data; `StructuralTelemetry` independently
    records content-free revision, timing, queue, policy/tool, interruption,
    memory, proactivity, and error evidence in memory and optional JSONL.
14. `EvaluationHarness`: manifest-pinned PCM16 replay plus an isolated
    fake-clock household capability provider. It is not registered in the live
    provider registry and cannot reach household services.
15. `EvaluationRegistry`: stores version-pinned scenarios and structural grader
    outcomes in SQLite. Deployment eligibility requires a latest passing run for
    every registered exact-pin scenario plus successful pinned failure replays.

These processors are coordinated by `ForegroundTurnStateMachine`. Its immutable
snapshots contain hashed input/context revisions rather than transcript text,
ordered stage records and timings, the deterministic policy decision, tool and
verification journals, response revisions, cancellation decisions, and a
terminal status. The audio runtime owns capture/endpoint/speak; the service owns
contextualize through render; commit closes the trace. The completed trace is
attached to `HandleResult` and emitted with the redacted `response_completed`
monitor event.

Speech and task cancellation are separate controls. `playback_cancel` stops the
source satellite promptly without cancelling provider work. An explicit task
cancellation request may skip queued actions; during a provider call it cancels
only tools whose manifest declares `cancellation: "anytime"`. Mutating tools
default to `before_side_effects`, so an in-flight mutation completes its
verification and later queued actions are cancelled.

## Activation and execution policy

Wake word is optional for low-risk smart-home requests, not optional for every
possible action.

### Passive mode

All multiword VAD-delimited human speech is transcribed and evaluated by the
structured interpretation pass. The legacy narrow-vocabulary prefilter is off
by default so device aliases, names, and imperfect transcripts cannot suppress
a command before intent classification. Speech may update a short-lived room
context buffer, but it acts only when all are true:

- speech act is `directive` or `desired_state`
- action is reversible and allowlisted
- target and desired state are unambiguous
- execution confidence meets the passive threshold
- utterance is not assistant playback/echo, media duplication, or quoted speech

An interpretation that returns an executable speech act, `decision=execute`,
and a bounded action plan is deterministically treated as addressed even without
a wake word. The independent address score cannot contradict and veto that
classification; the higher passive interpretation-confidence threshold still
applies.

Observations, self-intentions, third-person plans, quoted commands, and uncertain
pronouns do not act. Passive context expires quickly and never becomes durable memory.

### Active conversation

Starts when a wake-word turn is accepted. For 60 seconds by default after each
usable turn, accepted follow-up speech is addressed to the agent without a
repeated wake word. The timeout is dashboard-tunable and applied live. Under
the default household arbitration scope, microphones share the conversation
and its user/assistant history; changing to room-scoped arbitration separates
them. This context never overrides tool validation or supplies a missing
high-impact confirmation.

The persona, system context, rounded household-local time (minute precision with
AM/PM), and weather snapshot are established at conversation start. The model
may use time or weather later when it answers the request or is a meaningful
addition, but it does not refresh or force those facts into every turn.

Direct `shut up`, `be quiet`, and `stop talking` utterances cancel queued speaker
audio and end the room conversation. Quoted or third-party uses of those phrases
do not trigger cancellation.

### Risk classes

- v1 low risk: lights, ordinary switches, climate within configured bounds,
  reversible timers, and local dashboard tasks
- confirmation: ambiguous target/value, unusual duration, destructive task edit
- not exposed in v1: locks, alarm/security, purchases, external messages, account
  changes, shell access, dashboard/host restart, and arbitrary Home Assistant services

## Speech-act interpretation

The LLM receives the transcript, confidence, room, active goal, compact relevant
state, wake status, prosody summary, and the bounded user/assistant history for
an active conversation. It never receives raw Home Assistant state or raw audio.

Required speech acts:

- `directive`: an instruction/request to Nova, including polite questions
- `desired_state`: user explicitly wants a household state to be true
- `self_intention`: user says they plan/need/have to do something themselves
- `observation`: reports current or general state
- `question`: asks for information without requesting a state change
- `third_party`: describes another person's action/need
- `quoted_or_media`: quoted, read, or likely playback speech
- `social`: conversation without a tool goal
- `unclear`: insufficient evidence

Deterministic policy consumes the label; the model cannot authorize its own
unsupported tool or lower a threshold.

The same structured result contains zero to four semantic actions. Every action
has an order and dependency list. Policy can remove or downgrade actions but
cannot invent one. Independent low-risk actions may execute concurrently only
when the provider marks them safe; otherwise execution is ordered. The first
interpretation result contains response style, not an unverified success claim.

## Emotion and matching tone

Emotion inference is a separate logical step inside the single structured LLM
pass. It fuses words with acoustic evidence. Initial output is:

```json
{
  "label": "angry",
  "confidence": 0.78,
  "intensity": 0.72,
  "evidence": ["lexical", "high_energy", "fast_rate"]
}
```

The tone mapper converts only label/intensity/persona into an allowlisted TTS
instruction. Example mappings:

| Input label | TTS direction |
| --- | --- |
| calm | calm, warm, steady pace |
| grumpy | mildly irritated, dry, restrained |
| angry | forceful and angry, clear rather than shouted |
| excited | energetic, bright, quicker pace |
| bored | flat, low energy, slower pace |
| sad | subdued, gentle, slower pace |
| anxious | tense but intelligible, slightly quick |
| neutral | natural conversational delivery |

The selected persona can add a bounded modifier. Mirroring strength is
configurable, with matching enabled by default. The tool action is resolved
before persona or tone rendering so style cannot alter behavior.

## Conversation state machine

```text
PASSIVE
  | wake or accepted direct command
  v
LISTENING -> INTERPRETING -> ACTING -> RESPONDING
                    |           |          |
                    | unclear   | failed   | barge-in
                    v           v          v
                 CLARIFYING   FOLLOW_UP  LISTENING
                    |           |
                    +-----+-----+
                          v
             ALL ACTIONS VERIFIED -> GOAL SATISFIED -> LISTENING
                                                       |
                                             60 seconds idle / end (default)
                                                       v
                                                    PASSIVE
```

Each room has at most one active goal. Goal status is one of `new`,
`in_progress`, `needs_clarification`, `satisfied`, or `abandoned`.

Closure requires deterministic evidence: every required action succeeded and
verified, or the requested answer was delivered, with no missing slot or open
question. Partial failure keeps the goal open and reports exactly which actions
failed. Explicit "never mind" abandons immediately. Inactivity closes a
clarification/follow-up window but is not used to claim that an unfinished goal
succeeded.

## Iridium process topology

The deployed resident topology has three model services:

1. `nova-voice-llm.service`: pinned `llama-server` with Qwen3.5-4B
2. `nova-voice.service`: Python orchestration plus resident Nemotron STT
3. `nova-voice-tts.service`: vLLM-Omni Qwen3-TTS, implemented by separate
   stage-0 talker and stage-1 codec CUDA workers

The lightweight DeepFilterNet sidecar is CPU-only. Service health and the
deployment verifier require all resident services and ports; a model-load
failure never triggers a silent downgrade or live model swap.

## Latency strategy

- keep all three models resident and warm
- use the chunk operating point selected by the local STT bake-off rather than
  hard-coding 320 ms across different architectures
- use stable interim STT only to prefetch read-only room context, likely tools,
  and an LLM-state revision; discard it when the final transcript diverges
- send the LLM compact state and three tools, with thinking disabled
- generate short persona-aware success acknowledgements only after verified
  results; use the same resident LLM renderer for confirmations and dialogue
- benchmark equivalent REST and MCP operations from Iridium, then pin the
  measured hot path behind the semantic provider
- stream finalized sentence/clause units from vLLM-Omni and check cancellation
  at every chunk and semantic-unit boundary
- cancel queued TTS immediately on barge-in while retaining the verified tool result
- preserve resident headroom and measure cross-service GPU contention in the
  live latency/endurance gate

## Satellite lifecycle

Native satellites are headless processes. They capture and play audio, apply
local echo/noise processing, stream framed audio, and expose a local
health/status command. There is no tray/menu quit control in normal operation.
Installation and intentional disablement remain administrator actions.

- **Indium:** signed native macOS helper installed as a per-user LaunchAgent,
  with `RunAtLoad`/`KeepAlive`, bounded restart backoff, and one-time Local
  Network plus microphone consent granted to the signed bundle. The legacy
  LaunchAgent is explicitly associated with that bundle so macOS 15 records
  the Local Network decision against the right app. It is not a LaunchDaemon
  because capture needs the logged-in audio/privacy session.
- **Nocturnium** (address: see `PRIVATEREF.md#1.2`)**:** native Linux client managed by a systemd user
  service under the kiosk user when consuming PipeWire/WirePlumber, with linger,
  audio-session dependencies, automatic restart, watchdog/health reporting, and
  no routine quit UI. A restricted system service is used only for direct ALSA.
- **Iridium:** the same Linux client interface can attach a local mic/speaker,
  but the capture process remains separate from inference workers.

The daemons remain alive and capture continuously regardless of browser or
dashboard state. `always` is the v1 capture policy for Indium and Nocturnium.
An optional read-only OS probe may report dashboard foreground state as an
interpretation feature, but it cannot pause capture, inject JavaScript, call a
page hook, or affect process lifetime.

Capture is continuous; transport is not. Each native daemon runs a low-cost
local activity gate, calibrates the steady room floor for one second, and
retains 400 ms before its trigger plus 800 ms after
activity ends. Silence therefore stays on the edge while the central Silero VAD
still receives complete candidate utterances and remains authoritative. The
dashboard's Satellite noise gate switch applies `satelliteNoiseGateEnabled`
live over existing sockets and can restore continuous transport for A/B tests.

The LAN transport is mutually authenticated TLS WebSocket with locally
provisioned device certificates and a versioned binary envelope. Input uses
16 kHz mono PCM16 in 20 ms frames while the local gate is open; a bypassed
satellite consumes about 32 KB/s before framing overhead. Iridium aggregates frames for central VAD,
wake scoring, and the selected ASR chunk size. Output is framed PCM at the TTS
sample rate and is delivered only to the elected source satellite's own
connection (the response lock, `satellites/playback.py`); every other
satellite receives nothing, including ones sharing the same `roomId`. The
elected source satellite remains the timing anchor: a capable source acknowledges
the first rendered buffer and final completed buffer, and Iridium uses those
edges to fan speaking start/end state through the dashboard. Legacy sources fall
back to first delivered PCM plus a bounded output offset.

Response-lock playback still needs each local AEC graph to recognise the render
program a neighbouring speaker produces, since co-located satellites share
physical air even though only one of them plays. PipeWire routes Nocturnium
playback through `nova_voice_aec_sink` and capture through `nova_voice_aec`;
Indium uses acoustic playback tagging because its input-only CoreAudio path
deliberately does not hold the Mac's output device open. The server also keeps
one post-DSP playback reference per arbitration scope (household by default),
so every microphone in scope can reject the source satellite's echo even when
native cancellation or playback tagging misses a tail. A `playback_cancel`
control frame closes the source satellite's output on barge-in.

Browser microphones are in scope and supported through the dashboard browser
satellite. The client is explicitly foreground/permission-bound, defaults off
per device, and can use push-to-talk or page-bound always-on capture. Caddy
provides the trusted same-origin HTTPS/WSS edge; the dashboard bridge relays the
satellite protocol over mTLS to Iridium. Closing/reloading the page interrupts
this browser client but never the native services. The separate Voice Lab page
remains development-only and shadow-mode-only.

## Failure isolation

- If Iridium is down, satellites remain supervised, report unavailable through
  their own health interface, and reconnect with bounded backoff. Touch controls continue.
- If STT is unhealthy, no inferred action is sent.
- If LLM output fails schema validation, ask for a retry only in an active
  conversation; passive speech is ignored.
- After one Nova mutation, the bounded Ralph verification loop may repeat only
  authoritative state reads; it stops on success, its refresh cap, or its
  wall-clock deadline and never resends the action.
- If Nova action verification still fails, report failure and keep the goal open.
- If TTS fails after a successful action, record a redacted failure metric; do
  not repeat the action.
- Queue limits drop old passive audio before active-conversation audio.

## Privacy and data lifecycle

Audio, model, dashboard, and home-control traffic stays on Iridium or the
household LAN. Models are downloaded during explicit installation and pinned
locally. The orchestrator alone may use configured public web-search providers
for bounded knowledge queries; model and sidecar units otherwise retain their
local-only network policy.

In development:

- raw audio: in-memory ring buffer only, unless visible debug capture is enabled
- interim transcript: memory, optionally same TTL store for diagnostics
- final transcript and derived session text: dedicated SQLite, <=24 hours
- active context: memory plus TTL store for crash recovery, <=24 hours
- persona/preferences: durable config only when deliberately edited
- service logs: no utterance, prompt, response, or secret text

Each record receives `expires_at = transcribed_at + 24 hours`. Deletion runs at
startup and wakes at the earliest expiry, with a periodic safety sweep rather
than a 15-minute policy that can exceed the limit. The database uses no FTS or
virtual tables. `PRAGMA secure_delete=ON`, successful WAL TRUNCATE checkpoints,
no backups/snapshots, redacted logs, and an automated DB/WAL scan make the rule
testable across crash/restart.

## Primary implementation references

- Pipecat processors: https://docs.pipecat.ai/pipecat/fundamentals/custom-frame-processor
- WebSocket over TLS: https://www.rfc-editor.org/info/rfc6455/
- Silero VAD: https://github.com/snakers4/silero-vad
- PipeWire echo cancellation: https://docs.pipewire.org/page_module_echo_cancel.html
- WirePlumber user-service lifecycle: https://pipewire.pages.freedesktop.org/wireplumber/daemon/running.html
- Apple microphone authorization: https://developer.apple.com/documentation/bundleresources/requesting-authorization-for-media-capture-on-macos
- Apple Local Network privacy: https://developer.apple.com/documentation/technotes/tn3179-understanding-local-network-privacy
- Apple ServiceManagement/SMAppService: https://developer.apple.com/documentation/servicemanagement/smappservice
- Apple AVAudioEngine: https://developer.apple.com/documentation/avfaudio/audio-engine
- Browser microphone secure-context requirement: https://w3c.github.io/mediacapture-main/getusermedia.html
- SQLite secure deletion: https://sqlite.org/pragma.html#pragma_secure_delete
- SQLite WAL and TRUNCATE checkpointing: https://sqlite.org/wal.html
