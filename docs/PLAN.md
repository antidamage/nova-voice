# Nova Voice implementation plan

## Outcome

Build a low-latency, completely local voice layer that treats the Nova
dashboard API/MCP as its first tool surface. Iridium performs STT, intent and
emotion interpretation, LLM tool planning, and TTS. Nocturnium, Indium,
Iridium, and later household clients run independent native microphone/speaker
satellites. No satellite is implemented in, hosted by, or stopped with a
dashboard page.

The system supports both:

1. **Affirmative conversations** opened by `beemo`, an accepted direct
   command, or a continuing turn. These have natural multi-turn dialogue and permissive
   pronoun resolution.
2. **Passive room understanding** in which Indium and Nocturnium continuously
   stream audio to Iridium, central VAD delimits every speech segment, and every
   final transcript is classified. Only a high-confidence directive or explicit
   desired state can act.

Examples that must remain distinct:

| Speech | Classification | Action |
| --- | --- | --- |
| "Turn the air con on." | directive | turn it on |
| "Nova, turn the air con on." | addressed directive | turn it on and open a session |
| "Could you turn the air con on?" | polite directive | turn it on |
| "I want the air con to be on." | desired state | turn it on |
| "I gotta turn the air con on." | self-intention | do not act |
| "The air con is off." | observation | do not act while passive |
| "Turn it off." | directive with pronoun | act only when one target is salient |

## Firm architectural choices

- Prototype a pinned Pipecat modular frame pipeline behind owned interfaces.
  Keep it only if the custom local adapters pass latency and maintenance gates.
- Use native, supervised satellite processes and an authenticated local TLS
  WebSocket framed-audio protocol. Keep transport behind an interface.
- Keep exactly one deployed LLM, one STT model, and one TTS model resident.
  Wake-word detection and deterministic DSP/VAD are small pipeline utilities,
  not alternative language or speech models.
- Give the LLM a small, dynamically registered set of semantic tools. Never ask
  a 4B model to invent provider IDs, service names, or raw API bodies.
- Implement Nova as the first `CapabilityProvider`. Benchmark equivalent REST
  and MCP calls from Iridium, use the measured winner for each hot operation,
  and retain MCP for discovery, health, configuration, and administration.
- Enforce a hard repository boundary: no dashboard source imports, shared UI,
  new dashboard routes, or dashboard deployment changes. Only documented API/MCP
  operations cross into a Nova Voice-owned versioned compatibility contract.
- Require provider execution-locality metadata. Every future local-AI task runs
  on Iridium; satellites remain capture/playback clients and never execute tools.
- Treat Home Assistant as the device backend, not the conversation engine. The
  existing HA/Wyoming pipeline can remain available during migration but does
  not own Nova Voice sessions.
- Persist no raw audio by default. Persist development transcript records for a
  maximum of 24 hours and keep normal service logs free of transcript text.

## Phase 0 - Iridium and household preflight

Bring Iridium online without changing the existing Nova/Home Assistant voice
stack. Record the following in a local, uncommitted inventory file:

- Ubuntu version, CPU, RAM, storage, NVIDIA driver, CUDA runtime, and exact
  `nvidia-smi` output
- available mic/speaker devices and PipeWire/ALSA names
- Iridium, Nova, Nocturnium, and Indium LAN names/IPs
- current Beemo model file (if provisioned) and training/version provenance
- current `linux-voice-assistant`, Wyoming, and HA Assist configuration
- Indium CoreAudio input/output IDs and macOS microphone authorization state
- Nocturnium (address: see `PRIVATEREF.md#1.2`) PipeWire/ALSA input/output IDs and kiosk process name

Deliverables:

- non-mutating inventory command/script
- audio loopback and network RTT report
- a rollback note proving the existing HA voice path still works

Gate: do not install models until Iridium can run CUDA inference and all target
satellites can resolve its LAN name.

## Phase 1 - Model and runtime bake-off

Build a repeatable benchmark harness before the application. Pin model revisions
and runtime commits. Test every STT/TTS candidate as part of a complete
LLM+STT+TTS resident set, not as an isolated model whose memory has the GPU alone.

Candidates:

- official Qwen3.5-4B weights converted locally to Q4_K_M through a pinned
  llama.cpp commit, 8K context, one slot, thinking disabled, JSON-Schema/GBNF output
- STT: `nvidia/nemotron-speech-streaming-en-0.6b` at 160/560 ms,
  `nvidia/nemotron-3.5-asr-streaming-0.6b` at `en-GB`/320 ms, and
  `nvidia/parakeet-tdt-0.6b-v3` as the accuracy challenger
- TTS: Qwen3-TTS 12Hz 1.7B and 0.6B CustomVoice in supported precision/SDPA, plus a
  blind-listening trial of current/original Chatterbox when practical

The RTX 2080 Ti is Turing: it cannot use BF16 and the main FlashAttention-2 CUDA
implementation does not support it. The measured Iridium winner is the 0.6B
model in FP32 with PyTorch SDPA: FP16 was numerically unstable. Use a separately
validated Turing-compatible kernel only if it beats that stable baseline; never
silently fall back to CPU for a hot path.

Benchmark with the primary user's real speech in their natural accent (see
`PRIVATEREF.md#3.3`), actual room noise,
Nova entity names, all required emotions, and both short and long responses.
Compare the pre-deployment fallbacks in [MODELS.md](MODELS.md), then lock one
model per category. No fallback model is loaded in the deployed process.

The deployment still contains exactly one selected LLM/STT/TTS set. Selection
occurs before deployment; startup never silently picks a different model.

Gate:

- all three selected models remain loaded for 24 hours without OOM, reload, or
  CPU fallback on a hot path
- measured peak retains at least 1 GiB or 10% of physical VRAM, whichever is larger
- common command end-of-speech to tool dispatch p50 <= 1.2 s and p95 <= 2.0 s
- first response audio p50 <= 1.8 s and p95 <= 3.0 s
- TTS stays faster than real time for typical acknowledgements
- the local speech and tool-call quality thresholds in TESTING.md pass

If no set passes, reduce LLM context/KV precision first, validate actual ASR
dtype and bounded concurrency, then choose the next pre-deployment STT/TTS
candidate. Do not introduce runtime swapping.

## Phase 2 - Core service on Iridium

Create an Ubuntu service composed of small replaceable adapters:

```text
edge activity gate -> audio conditioning -> central VAD/wake -> streaming STT
          -> prosody features -> interpretation/session policy
          -> Nova tool adapter -> response renderer -> TTS -> transport
```

Recommended package boundaries:

```text
nova_voice/
  app/              startup, dependency injection, health
  audio/            framing, VAD, wake, prosody, echo guards
  inference/        STT, LLM and TTS interfaces/adapters
  interpretation/   speech acts, emotion, confidence, pronouns
  sessions/         active goals, turns, closure and room context
  capabilities/     registry, provider interface, schemas, policy and verification
  providers/nova/   Nova skills, alias index, external REST/MCP adapters
  satellites/       connection registry, source election, playback
  personas/         validated persona and tone configuration
  persistence/      transcript TTL repository and janitor
  observability/    redacted metrics, traces and health
```

Use two CUDA-owning processes: a pinned `llama-server` for Qwen3.5 and one Python
voice process holding the selected STT and TTS in a shared PyTorch CUDA context
plus the orchestrator. Both expose health. The orchestrator owns deadlines,
cancellation, backpressure, and bounded queues; streaming ASR has GPU priority.
Model-load failure fails health instead of causing an automatic downgrade.
The core packages must not reference Nova or dashboard types. Provider manifests
register tool schemas, risk classes, verification behavior, and compact skills
through dependency injection. Manifests also declare `iridium_local` or an
explicit household-LAN service; local-AI providers are rejected anywhere but
Iridium. One noisy satellite must not block another. A
single GPU scheduler serializes
LLM/TTS bursts when required while ASR streams retain priority.

Gate: a local WAV can traverse the complete pipeline, execute a mocked tool,
and return audio with component timing recorded and no transcript in logs.

## Phase 3 - Capability layer, Nova provider, and skills

Implement the generic provider contracts and the Nova semantic tool contracts
in [CONTRACTS.md](CONTRACTS.md). The registry loads allowlisted providers from
configuration; a provider owns its aliases, schemas, execution, verification,
risk metadata, and compact skill text.

Candidate translations behind the semantic adapter:

- `nova.query` -> cached `GET /api/state`
- `nova.control` zone operation -> `POST /api/zone`
- `nova.control` entity operation -> `POST /api/entity`
- timers -> `/api/aircon/timer` or `/api/panel-heater/timer`
- computer wake/sleep -> `/api/desktop/wake` or `/api/desktop/sleep`
- `nova.task` -> `/api/tasks` and task-specific routes

Refresh the normalized alias index from `/api/state` and public dashboard event
interfaces where available. The external adapter, not the LLM, resolves friendly
names, rooms, entities, domains, and HA services. Ambiguous aliases return
candidates and cause a clarification turn.

Define `nova-provider-v1` as Nova Voice's compatibility contract. Record
`/api/version` build metadata but do not treat it as a semantic REST version.
Benchmark REST and MCP equivalents on the real LAN, pin the selected path per
semantic operation, and run startup/CI conformance against owned fixtures. The
LLM and skills never know which transport was selected.

Use `/api/mcp` for `nova.dashboard.health`, `nova.ha.discover`, and Nova-specific
administrative capabilities. Keep dashboard/MCP bearer tokens only in the
Iridium provider secret store; never place them in a satellite or LLM prompt.

Build against checked-in Nova Voice contract fixtures and a fake provider. A
read-only conformance suite may exercise a deployed dashboard endpoint, but the
voice build must not compile, copy, or generate code from the dashboard repo.
Adding a future provider must require no edits to audio, STT, interpretation,
sessions, persona, or satellite packages.

Load the compact files under `skills/` into the system prompt. The initial three
skills together are deliberately small enough to remain resident.

Gate: every mutating tool is schema validated, allowlisted, state-verified, and
idempotent where possible. A spoken explicit command counts as user approval
for reversible lights/climate actions. Locks, security, purchases, messaging,
and system-restart tools are not exposed in v1.

## Phase 4 - Interpretation, emotion, persona, and sessions

For every final utterance, derive deterministic acoustic features before the
LLM call: energy, peak energy, median pitch relative to a rolling room baseline,
pitch range, speaking rate, pause ratio, and duration. Do not label emotion from
pitch alone.

Make one short, non-thinking structured LLM pass return:

- probable input emotion, confidence, and intensity
- speech act and addressing probability
- execute/ignore/clarify decision
- resolved goal and a schema-bounded ordered `actions[]` of zero to four semantic calls
- conversation goal state
- response style and whether complex post-tool rendering is required; never an
  unverified success claim

Each action declares order and dependencies. Deterministic policy can remove or
downgrade actions, schema-validates all arguments, and executes independent
low-risk actions in parallel only when provider metadata allows it. Common
verified results use deterministic persona-aware acknowledgements. Partial
failure, complex results, or open-ended dialogue may use a second call to the
same resident LLM after tool results exist.

The response tone is compiled deterministically from the emotion label and
intensity through the selected TTS adapter. Qwen candidates receive an allowlisted
`instruct`; another winner must expose an equally tested bounded mapping. The LLM
cannot send arbitrary TTS control text. Initial labels: `neutral`, `calm`,
`grumpy`, `angry`, `excited`, `bored`, `sad`, and `anxious`.

Persona is configuration. Its style can be cheerful, grumpy, hostile-sounding,
or bored, but these invariants always win:

- understand and execute the user's goal to the best of available local tools
- complaints are at most one short sentence and never replace action
- never fabricate success; report the verified tool result
- never let persona change the requested target/value
- never use passive room context as permission for a high-impact action

Session closure uses both model output and deterministic evidence. A goal is
complete when required slots are filled, every required action is verified, and
no question/pending step remains. Partial failure is reported and keeps the goal
open without repeating successful actions. Then speak the result and close. Keep
a short follow-up window only for clarification, a question, or an explicitly
continuing topic.

Gate: all contrastive speech-act cases and conversation closure scenarios in
TESTING.md pass without special-casing their exact sentences.

## Phase 5 - Native supervised satellites

Build one small, headless satellite protocol/client core with platform audio and
supervisor packaging. It sends device/room ID, protocol version, capture format,
RMS/SNR, playback state, and optional dashboard-foreground context over mutually
authenticated TLS WebSocket with a provisioned device certificate. Continuously
stream 16 kHz mono PCM16 in 20 ms frames; do not edge-VAD-drop speech. Iridium
aggregates frames for central VAD, wake scoring, and STT. Use the TTS native
sample rate for return audio. Add Opus only if measured LAN conditions justify
its complexity. Browser microphones are outside v1.

The client has no normal quit UI. It exposes a read-only health command/socket
and accepts intentional enable/disable only through administrator-owned service
management. Browser close, kiosk refresh, focus changes, and dashboard deploys
cannot kill or uninstall it.

### Nocturnium (address: see `PRIVATEREF.md#1.2`)

- install a native Linux client under the kiosk user for PipeWire/WirePlumber
- supervise PipeWire capture as a systemd user service with linger, audio-session
  dependencies, restart/backoff, watchdog health, and journal fields without speech
  text; use a restricted system service only for direct ALSA capture
- capture/play through PipeWire where available, otherwise a pinned ALSA device
- require PipeWire/native AEC, noise suppression/AGC, and playback-frame tagging
  before live ambient execution
- use `always` capture; an optional read-only foreground probe supplies context
  only and never pauses capture or touches dashboard source/page JavaScript

### Indium

- ship a signed native Swift helper using AVAudioEngine/CoreAudio and the shared
  wire protocol, including voice processing/AEC and playback-reference tagging
- install as a per-user LaunchAgent with `RunAtLoad`, `KeepAlive`, restart
  throttling, and a stable application identity for one-time microphone consent
- do not use a system LaunchDaemon for capture: it lacks the intended logged-in
  user audio/privacy context
- use `always` capture; an optional read-only foreground probe supplies context
  only and never ties capture/process lifetime to the dashboard application/browser
- provide no menu-bar quit item or user-facing stop control
- hold an explicit idle-sleep activity assertion while configured as an always-on
  satellite; allow display sleep

### Iridium and later devices

Package the Linux client independently from the inference service for Iridium's
local audio devices. Later satellite ports implement the same audio/supervisor
interfaces and never require Nova-dashboard changes.

### Multiple microphones

When several satellites hear one utterance, group candidates by time and an
audio/transcript fingerprint. Select the stream with the best SNR/wake score and
discard duplicates. Route speech back to the elected source device unless the
user names another room/device.

Gate: killing/reloading the dashboard or closing its browser does not interrupt
either satellite; killing either satellite process causes supervised recovery;
two satellites in one room cause one command and one response; Nova's own TTS
never re-triggers a command.

## Phase 6 - Retention, security, and observability

- Bind model services to localhost; expose only the authenticated satellite
  framed-audio endpoint and Nova Voice health endpoint on the LAN.
- Use a trusted household CA and mutually authenticated TLS with one provisioned
  certificate per satellite. Do not add a second bearer-token mechanism in v1;
  keep private keys in OS-protected storage and out of git.
- Disable model/runtime telemetry and block runtime internet egress after model
  installation. Health checks and updates remain explicit maintenance actions.
- Store transcript text, interim/final markers, emotion result, and derived
  session text in a dedicated development SQLite database with `expires_at`.
- Set `expires_at = transcribed_at + 24 hours`, enable SQLite secure deletion,
  use no FTS/virtual tables, run the janitor at startup and at the earliest expiry
  with a safety sweep, successfully TRUNCATE-checkpoint WAL after deletion, and
  exclude the database/WAL from backups and snapshots.
- Never put transcript text, prompts, or responses in normal logs. Satellite and
  server metrics use
  utterance IDs, label enums, durations, confidence, and error codes.
- Raw audio stays in bounded memory buffers. A separately enabled debug capture
  mode must use the same <=24-hour TTL and be visible in Nova Voice's own health
  CLI/status endpoint; it must not require a dashboard UI change.

Gate: an automated clock-advance test proves no transcript or derived session
text survives past 24 hours, including after crash/restart.

## Phase 7 - Staged rollout

1. Shadow mode: transcribe/classify and show decisions for at least seven days;
   never mutate Nova.
2. Wake-only mode: execute only inside affirmatively opened sessions.
3. Passive low-risk mode: allow high-confidence light and climate commands.
4. Multi-satellite mode: Nocturnium, then Indium, then other native clients.
5. Disable the old HA microphone/wake path only for a satellite that has passed
   live gates; keep a documented rollback rather than removing HA Assist early.
6. Broaden tools only from measured false-action and ambiguity data.

Every phase must have a one-command stop/disable path. Iridium failure must not
affect touch controls, Home Assistant automations, or the dashboard itself.

## Definition of done for v1

- Iridium runs one resident LLM/STT/TTS set locally with no cloud calls.
- `beemo` begins a turn, but clear low-risk commands work without it.
- all speech segments are transcribed, emotion-evaluated, and speech-act classified
- the two air-con examples at the top behave differently every time
- reversible dashboard controls and tasks work through semantic tools and verify state
- multi-action requests execute a bounded ordered plan and report partial failure
  without repeating already verified actions
- response voice is natural, consistent, and measurably changes for configured tones
- satisfied goals close the session; clarification and unfinished goals keep it open
- Nocturnium and Indium run as supervised native background satellites, survive
  dashboard/browser restarts, and cannot be casually stopped through the UI
- `nova-voice` builds and deploys without importing or modifying dashboard code
- a fake second provider can register a skill and tool without changing core packages
- duplicate microphones and speaker echo do not duplicate/re-trigger commands
- no development transcript or derived conversation text is older than 24 hours
- killing Nova Voice leaves the existing dashboard and HA stack unaffected
