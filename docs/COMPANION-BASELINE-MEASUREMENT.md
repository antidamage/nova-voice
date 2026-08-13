# Baseline measurement for the companion offload (NPT-002 / NPT-909)

The premise of this whole project is falsifiable: freeing llama.cpp's single
slot should measurably improve interactive latency. If it does not, the
background routes are still worth having but the hot-path routes are not, and
the route table should stay local. That decision needs numbers from *before*
any route is enabled.

This document specifies what to capture. The run is deliberately not automated
into CI: it puts real load on the household's live voice stack, so it needs a
quiet window and the owner's go-ahead.

## Separate the two quantities that get conflated

End-to-end voice latency and llama.cpp queue time are different things, and
only the second is what a companion can improve.

- **Queue wait** — time between issuing a request to `llama-server` and the
  first token coming back, when another request already holds the slot.
- **Service time** — generation time once the slot is held.
- **End-to-end** — utterance end to first audible response, which also includes
  STT, capability execution, TTS first chunk, network and the satellite's
  jitter buffer.

A companion removes *queue wait* for the workload it takes. Reporting only
end-to-end will hide the effect behind TTS and network variance.

## Per-pass figures

For each of the five real passes (`interpret`, `render_response`,
`extract_self_profile_update`, `confirm_objective`, `classify_icon`), capture
p50 and p95 of:

- queue wait (ms)
- service time (ms)
- prompt tokens in, completion tokens out

Under two conditions:

1. **Idle** — one turn at a time, nothing else touching the GPU.
2. **Contended** — a turn whose plan triggers the verify loop while a
   background pass (`extract_self_profile_update`) and a dashboard
   `classify_icon` are in flight. This is the case the companion is meant to
   fix, and it is the one that is currently invisible.

## Host figures

Sampled once per second for the duration of each run, on the voice host:

- GPU utilisation and VRAM in use (`nvidia-smi --query-gpu=utilization.gpu,memory.used`)
- `llama-server` queue depth, if exposed by its metrics endpoint; otherwise
  infer from concurrent in-flight request count at the client

VRAM matters independently: the stack sits at roughly 9.0/11.3 GB, and the
visualiser shares the card. A baseline that does not record VRAM cannot tell a
latency regression from memory pressure later.

## Interactive figures

- time-to-first-audible, which `api.py` already computes and logs as
  `voice turn time_to_first_audible ... latency_ms=` and records via
  `selected_audio.record_first_audible_ms()`. This is real playback start
  confirmed by the satellite, not an estimate — use it rather than
  re-deriving one.

## Method

1. Pick a quiet window; confirm no deploy or visualiser work is running.
2. Record the git revision and the model ids actually loaded, not the
   configured ones.
3. Run each condition for at least 30 turns — p95 on fewer is noise.
4. Write the results to a dated artifact under `docs/evidence/`, containing no
   transcript text. Counts, timings and token totals only; a latency table is
   not a place for household speech.

## Comparison at NPT-909

Re-run identically with routes enabled and compare: queue occupancy, TTFT,
fallback frequency, companion acceptance rate, and duplicate/error counts. The
route table's final defaults follow from that comparison, not from the
roadmap's suggested values.
