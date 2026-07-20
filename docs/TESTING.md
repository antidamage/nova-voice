# Test and acceptance plan

## Test layers

1. Pure policy tests: speech acts, thresholds, pronouns, goal closure, tone map.
2. Adapter contract tests: STT/LLM/TTS, generic provider, and Nova API fakes.
3. Read-only dashboard integration tests against Nova Voice-owned API/MCP fixtures.
4. Recorded audio replay with deterministic expected decisions.
5. Live Iridium/satellite latency, echo, and endurance tests.
6. Shadow-mode household observation before any passive action is enabled.
7. Repository-boundary and native-service recovery tests.

For manual microphone, interpretation, emotion, and response-voice inspection,
use the mTLS-protected push-to-record page in [DIAGNOSTICS.md](DIAGNOSTICS.md).
It is available only when explicitly enabled in development shadow mode and is
not a continuous/browser satellite.

The deployed authenticated synthetic-speech check on 2026-07-15 completed the
real cache-aware STT, Qwen3.5 interpretation, response renderer, emotion/tone
instruction, and Qwen3-TTS path. The final full-text-quality run returned a
valid 130,604-byte WAV and measured 2.78 s STT, 2.02 s interpretation, and 8.57
s TTS. These are evidence, not acceptance of the latency targets below. The
current upstream Python API buffers the whole waveform, so real microphone
trials and a validated streaming-runtime optimization remain required.

## Independence and extensibility gates

- `nova-voice` contains no imports, workspace links, generated types, assets, or
  build steps from `nova-ha-dashboard`
- dashboard checkout absent: Nova Voice unit/contract builds still pass
- dashboard browser closes/reloads/updates: both satellite streams continue
- dashboard service is unavailable: Nova provider reports unhealthy while audio,
  conversation, and a fake second provider remain operational
- register a fixture provider with one query and one reversible action without
  editing core audio, inference, interpretation, session, or satellite packages
- provider removal deletes no core state and leaves other providers healthy
- run the same domain/contract tests with the Pipecat prototype enabled and with
  a minimal owned test pipeline, proving framework replacement does not alter policy

## Mandatory contrastive language suite

Each device/action must be tested with variations of:

- "Turn the air con on" -> execute
- "Nova, turn the air con on" -> execute and active session
- "Could you please turn the air con on?" -> execute
- "I want the air con to be on" -> execute
- "I gotta turn the air con on" -> ignore
- "I need to go and turn the air con on" -> ignore
- "<household member> said, 'turn the air con on'" -> ignore (use the real
  household name from `PRIVATEREF.md#3.3` when running the suite)
- "The air con is off" -> ignore in passive mode
- "Why is the air con off?" -> answer/query, not turn on
- "Don't turn the air con on" -> do not execute
- "I don't want the air con to be on" -> turn off only when target is unambiguous

Add tense, contractions, politeness, negation, false starts, background speakers,
and STT errors. Do not solve the suite with exact-string rules.

## Pronoun and context suite

- active goal establishes one target; "turn it off" resolves it
- active goal has two targets; "turn it off" asks which
- passive room has one device but no discourse target; "turn it off" asks/ignores
- "make it warmer" resolves current room climate only in an active session
- a closed goal cannot leak its pronoun target into a later unrelated utterance
- context from another room/satellite cannot resolve the pronoun

## Goal closure suite

- one successful light action -> reply, `satisfied`, close
- stale immediate state -> bounded refresh loop observes success, no duplicate action
- failed verification after either Ralph bound -> `in_progress`, report failure,
  no duplicate action
- missing temperature -> `needs_clarification`, keep session open
- answer delivered with no follow-up question -> close
- assistant asks a question -> remain open
- explicit "never mind" -> `abandoned`, close
- timeout during clarification -> close without claiming success
- barge-in during TTS -> cancel audio, retain verified state, process new turn

## Multi-action suite

- two independent lights -> bounded two-action plan, safe parallel execution,
  one verified response
- light then state-dependent brightness -> ordered plan with dependency
- one of two actions fails -> report partial failure, keep goal open, do not
  repeat the successful action
- more than four requested actions -> clarify or safely split; never overflow schema
- duplicated action produced by STT repetition -> normalize to one idempotent action
- mixed supported/unsupported actions -> execute nothing until the user understands
  the unsupported portion unless policy explicitly marks safe partial execution

## Emotion and persona suite

For each label (`calm`, `grumpy`, `angry`, `excited`, `bored`, `sad`, `anxious`,
`neutral`):

- at least 20 real clips, balanced across text whose sentiment agrees/disagrees
  with acoustic delivery
- compare predicted label/confidence with the primary user's annotation
- verify the compiled TTS instruction is allowlisted and intensity bounded
- perform blind A/B listening for naturalness, voice consistency, and perceptible tone
- verify grumpy/negative personas still execute first and complain no more than once
- compare Qwen3-TTS 1.7B and 0.6B plus the selected Chatterbox challengers blind;
  do not reveal model identity until scoring is complete

Initial gates: >=80% broad-label agreement on the local corpus and >=4/5 median
subjective naturalness. Tune mappings and baselines before adding a separate
emotion model.

## Smart-home action safety

- 100% valid JSON/schema on the command corpus
- 100% correct domain/service mapping by the deterministic adapter
- no fabricated entity ID reaches the dashboard
- every mutating result is checked against `/api/state` or an authoritative result
- retries do not toggle twice or create duplicate tasks
- 0 actions on the curated self-intention/quoted/media negative suite
- 0 high-impact operations exposed in the v1 tool registry

Passive rollout requires at least seven days of shadow capture with no
would-have-executed false positive on household media/background speech.

## STT gates

- command word accuracy >=98% for target, action, number, and negation slots
- overall local far-field WER target <=10% at the selected chunk size
- no loss of negation in the mandatory suite
- stable interim text never dispatches an action before the final utterance
- two simultaneous satellites retain independent streaming caches
- locally gated PCM from each enabled primary satellite reaches central VAD;
  pre-roll/tail handling cannot discard a labelled speech segment
- the dashboard noise-gate bypass switches connected satellites back to
  continuous 20 ms PCM without a process restart

## Latency and resource gates

Measure every span with monotonic timestamps:

- audio ingress to interim text
- final speech frame to final transcript
- final transcript to interpretation
- tool dispatch and verification
- response text to first/last TTS audio
- end-to-end command and social reply

Targets:

| Metric | p50 | p95 |
| --- | ---: | ---: |
| final speech -> tool dispatch | <=1.2 s | <=2.0 s |
| final speech -> first response audio | <=1.8 s | <=3.0 s |
| selected Nova REST/MCP hot mutation | <=100 ms | <=300 ms |
| satellite reconnect on healthy LAN | <=2 s | <=5 s |

Resource endurance: 24 hours, all models resident, at least two ASR streams,
periodic commands and TTS, no OOM/reload/CPU hot-path fallback, measured peak
leaving at least 1 GiB or 10% physical VRAM free (whichever is larger), bounded
queues, and no ASR starvation during TTS.

## Wake, duplicate, and echo gates

- tune Beemo on real rooms; target false accepts <=0.2/hour and false reject <=5%
- one utterance heard by two satellites causes exactly one interpretation/tool call
- the best-SNR satellite wins consistently
- response PCM reaches only the elected source satellite (response lock),
  including when another satellite shares its `roomId`, and every microphone
  in the arbitration scope shares the same playback echo reference
- assistant TTS, dashboard control sounds, TV, and music never trigger an action
- barge-in works while the assistant is speaking

## Native satellite supervision gates

- On Nocturnium, run `ops/configure-pipewire-aec.sh` after connecting the
  physical microphone/speaker. It resolves the host's default PipeWire nodes,
  writes the AEC graph, and with `--restart` verifies `nova_voice_aec` nodes
  exist and selects the virtual source/sink as the PipeWire defaults before the
  satellite is started. The live graph must show the PortAudio stream linked to
  those virtual nodes; `echo_cancellation=true` must never silently fall back to
  a physical/dummy/default device.
- On Indium, run `ops/provision-macos-keychains.sh`, install the signed
  LaunchAgent, allow **Nova Voice Satellite** in **System Settings > Privacy &
  Security > Local Network**, and verify its health file reports
  `connected: true` before granting/testing microphone capture. The isolated
  household CA is pinned by the client rather than approved in the user's
  global trust settings, so `security find-identity -p ssl-client` can be a
  false negative. The installer materializes the current user's
  bundle/keychain paths in the plist and registers the signed app with Launch
  Services; no username-specific path belongs in the checked-in template.
- Nocturnium (address: see `PRIVATEREF.md#1.2`) runs as the kiosk user's systemd user service with
  linger when using PipeWire (or a restricted system service for ALSA), and
  recovers from a forced process exit without a dashboard/browser action
- Indium starts at user login under launchd and recovers from a forced process
  exit while retaining the signed app's microphone authorization
- normal users have no dashboard, tray, or menu control that stops the process
- capture remains `always` through dashboard foreground/background, browser close,
  kiosk navigation, and dashboard deploy; an optional foreground probe changes
  metadata only, not audio flow
- loss of Iridium causes bounded reconnect backoff rather than a crash loop
- logs and crash reports contain no raw audio or transcript text

## Retention gates

Use a fake clock plus crash/restart integration test:

- insert final/interim transcripts and derived session text around the 24-hour boundary
- run startup, earliest-expiry, and safety-sweep janitors
- assert no row with `transcribed_at <= now - 24h` remains
- assert the schema has no FTS/virtual transcript tables
- assert the successfully truncated WAL and backup/snapshot paths contain no
  retained transcript copies
- assert raw audio is absent when debug capture is disabled
- enable debug capture, advance 24 hours, and assert audio is deleted too
- scan service logs and traces to prove utterance/prompt/response text is redacted

## Rollout evidence

Each rollout stage produces a short report with model revisions, runtime versions,
corpus hash, metrics, observed false actions, known limitations, and rollback
command. Passive execution is never enabled solely because unit tests pass.
