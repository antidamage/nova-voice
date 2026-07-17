# Implementation status

The checked-in package now provides the standalone text and framed-audio
runtime described by `docs/PLAN.md`. The core is dashboard-independent and
uses only the provider's documented REST/MCP client.

The current source also implements wake-opened, room-local multi-turn context
with a 20-second idle timeout, no-wake follow-ups, direct spoken cancellation,
persona-aware verified-command confirmations, adaptive broad-light shortcuts,
source-satellite-only response audio, global dashboard speaking animation, and
the two-sided configurable-font voice monitor transcript. These paths have only
mocked/unit/static coverage here; no audible validation was performed.

Implemented in this increment:

- a self-contained, responsive development diagnostics page at `/diagnostics`
  for explicit browser microphone capture, real STT/LLM/emotion/policy/TTS
  inspection, timing display, and returned-audio playback;
- an mTLS-protected, shadow-mode-only PCM diagnostics endpoint that reuses the
  native satellite audio runtime, persists no raw audio, and retains resulting
  development transcripts for the same 24-hour TTL;
- a shared bounded `process_pcm` path so diagnostics and elected native
  satellite turns cannot drift into separate inference implementations;
- degraded startup when Nova is temporarily unavailable, with redacted health
  and later refresh rather than taking down the interpreter or janitor;
- schema/policy metadata for capability tools and a blocked-tool enforcement
  point in the executor;
- confirmation-required and policy-less capability calls fail closed before
  any provider mutation;
- provider registration rejects any semantic tool that lacks a policy, or any
  policy that does not belong to an advertised tool;
- positional as well as structured conversation logging is redacted, and the
  service has end-to-end outage, self-intention, and shadow-mode regressions;
- API results and audio-turn logs now carry only monotonic component timings
  (no transcript or prompt text) for the model-latency bake-off;
- dependency-aware action waves with provider-declared safe parallelism,
  provider-failure isolation, and atomic rejection of mixed invalid plans;
- duplicate idempotent-plan coalescing, an idempotent explicit-state Nova
  control policy, removal of non-idempotent `toggle` from the LLM schema, and
  clearer MCP error diagnostics;
- deterministic prosody extraction (energy, pitch, rate proxy, pauses) and
  recurrent Silero state reset between utterances;
- deterministic negated-command suppression and protection against closing a
  session from an unexecuted/shadowed plan;
- cache-aware NeMo 2.x streaming with the native NVIDIA buffer API,
  independent per-satellite caches, configurable chunk size, final padding,
  cleanup, and safe fallback;
- explicit abandonment cues (`never mind`, `cancel that`, etc.) that close a
  room goal without executing actions;
- Turing-aware Qwen TTS dtype selection (`auto` chooses official BF16 when
  supported and stable FP32 otherwise), explicit CUDA-by-default device
  selection, a writable hardened-service Triton cache, and honest CUDA/STT/TTS
  health reporting;
- one shared Python CUDA execution gate for STT/TTS bursts across satellites,
  while keeping the selected model weights resident;
- corrected NeMo streaming PCM frame sizing and mixed-precision encoder/RNNT
  boundaries, verified against the deployed cache-aware model on the 2080 Ti;
- finalized VAD/diagnostic utterances use NeMo's full-buffer path instead of
  replaying completed audio through 160 ms streaming chunks; on Iridium the hot
  benchmark improved from 1.9-2.5 seconds (and an empty streaming result) to
  66 ms with a valid transcript;
- the local interpreter now has an explicit decision table plus a non-mutating
  consistency repair so addressed questions and social/non-tool requests cannot
  be incorrectly mapped to `ignore` or empty `execute` plans;
- the TTS speaker parameter is now the `Serena` female preset, and a small
  bounded in-memory PCM cache avoids regenerating identical responses without
  changing the TTS checkpoint, precision, or sampling quality;
- stricter native satellite identity/supervisor validation, hello acknowledgement
  and receive timeouts, corrected systemd restart-limit placement, native
  `sd_notify` readiness/watchdog support, and periodic health-file refresh;
- Indium CoreAudio playback-start/playback-finished acknowledgements that drive
  the dashboard speaking orb from actual rendering rather than TTS start time;
- a response lock restricting playback to the single elected source satellite
  (`RoomPlaybackRouter.open_stream` targets only that satellite's own
  connection), after a room-scoped fan-out variant briefly caused every
  satellite sharing a room to play at once; one shared post-DSP
  acoustic/transcript echo reference still covers every mic in the
  arbitration scope so a non-playing, co-located satellite rejects the
  source's echo;
- host-specific PipeWire AEC configuration, refusal to capture from a dummy
  device when AEC is required, selection of the virtual AEC source/sink as the
  PipeWire defaults, and portable per-user satellite installation;
- Nova semantic timer/desktop operations behind the provider boundary;
- read-only Iridium inventory and REST/MCP latency probes;
- macOS keychain provisioning that repairs an existing keychain missing its
  private-key identity instead of silently leaving mTLS unusable.
- rollback/stop procedures that leave the existing HA/Wyoming path untouched;
- a CA-preserving, device-scoped satellite identity issuer and an explicit
  native-satellite rollout guide, now including a dedicated browser-diagnostics
  client identity;
- Iridium installation that preserves the separately pinned inference layer
  during base dependency sync (`uv sync --inexact`).

The current Iridium deployment is healthy in development shadow mode: its
authenticated health check reports the Nova provider, Qwen LLM, native
cache-aware NeMo STT, and FP32 Qwen TTS resident. After a complete synthesis
turn the processes use 9,000 MiB of 11,264 MiB, leaving 2,264 MiB headroom. An
authenticated
synthetic-speech turn completed the real STT -> LLM -> tone -> TTS path and
returned a valid 130,604-byte WAV. The diagnostics page is deployed and enabled
at `https://<voice-server-host>:8766/diagnostics` (see `PRIVATEREF.md#1.1`);
browser access still requires the
dedicated household-CA client identity documented in `DIAGNOSTICS.md`.

The selected upstream `qwen-tts` package currently returns full buffered audio;
its `non_streaming_mode` option controls simulated text input rather than audio
packet streaming. A full-text FP32 conversational response measured 8.57 seconds
for TTS. This is stable and quality-preserving but does not meet the interactive
latency target; a future pinned, locally validated streaming runtime for the
same single model remains a performance acceptance gate, not a hidden fallback.

The signed Indium helper is installed as a `RunAtLoad` + `KeepAlive` LaunchAgent,
connected over mTLS, and reporting successful completed playback. Nocturnium is
also connected under its kiosk-user systemd service. Its physical PipeWire AEC
graph is provisioned; deployment verification must additionally prove that the
running PortAudio input/output links terminate on the virtual AEC nodes rather
than merely checking that those nodes exist. Remaining physical/acceptance gates
include the 24-hour resident GPU/OOM test, the primary user's local
speech-quality/latency corpus (see `PRIVATEREF.md#3.3`), and shadow-mode
false-action data. No fallback model is selected at runtime.

Latest read-only Nova check: `/api/state` responded successfully from the LAN
with 26 entities, 8 zones, and 99 normalized aliases; the latest five samples
measured approximately 47 ms median and 49 ms p95. MCP discovery is reachable, but
authenticated MCP POST calls currently return 503 until the deployment's
`NOVA_VOICE_NOVA_MCP_TOKEN` is provisioned.
