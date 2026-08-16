# Tool-free companion interpretation benchmark — 2026-08-17

## Question

Was the earlier 34.59-second phone interpretation representative, and why did
earlier comparisons appear to swing between a phone win and a tenfold loss?

## Method correction

The earlier sample came through the complete voice-turn harness. It mixed STT,
provider context, live household state, the semantic tool catalogue, memory,
profile extraction, policy, response rendering and route-comparison bookkeeping
with interpretation itself. It was a valid full-stack turn, but not a controlled
interpretation benchmark.

The new hidden `/v1/test/interpretation` diagnostic calls the real routed
planner directly and supplies:

- text input, with no audio or TTS;
- an empty tool catalogue;
- empty household state and memory;
- no active goal or conversation history;
- dry-run semantics, with no policy, provider call or execution;
- one excluded warm-up, followed by interleaved repeated scenarios;
- separate device and server arm timings from comparison mode.

Each tool-free prompt was approximately 10.6 KB and 2,393 estimated context
tokens. The size is mostly Nova's fixed interpretation and safety instructions,
which deliberately remain present on both stacks.

## Results

| Condition | Phone completed | Phone median | Phone p95 | Iridium median | Iridium p95 | Paired median ratio |
|---|---:|---:|---:|---:|---:|---:|
| Burst, 1-second gaps | 19/20 | 16.696 s | 18.451 s | 2.031 s | 2.153 s | 8.06× |
| Paced, 20-second gaps | 7/8 | 16.904 s | 18.696 s | 2.029 s | 2.230 s | 8.12× |

The first burst-run Iridium warm-up was 7.463 seconds, versus a 2.031-second
measured median. The phone warm-ups were 14.428 and 14.183 seconds and were not
slower than the measured distribution.

All 28 measured interpretations planned zero actions. The four interleaved
input shapes were a short social greeting, short conversational question,
observation and self-intention. Scenario order rotated each repeat so one shape
was not systematically cold or systematically hottest.

Two phone arms were not converted into 45-second timeout values or omitted from
the completion count. Server logs show both were rejected after the companion
reported that device state had degraded during generation. Before/after status
snapshots still showed the full/charging tier, proving the transient safety
check catches a condition that coarse polling can miss.

## Interpretation

The 34.59-second number should not be used as the typical tool-free
interpretation latency. It measured a materially larger full-context prompt.
It is plausible for that prompt to take roughly twice the 14–18-second
tool-free range because prefill scales with the context sent to the phone.

It was not, however, evidence that the phone and Iridium were roughly tied.
Once the workload is held constant, both burst and paced measurements put this
Qwen3.5-9B MLX build at about eight times Iridium's warm latency. Cooling gaps
did not reverse the result. The phone's variation is bounded but real, and a
sustained run can cross its device-state safety threshold.

One further fairness bug was found after these runs: Iridium samples
interpretation at temperature 0, while the installed phone build used 0.2.
That changes JSON content and completion length between repetitions. Source now
matches the phone to temperature 0 and logs structural time-to-first-chunk,
total generation, chunk count and character count. The paid build compiled and
signed successfully, but Neptunium locked and became unavailable before it
could be installed, so these two evidence files remain explicitly labelled as
the pre-fix build.

## Retained evidence

- `companion-interpret-tool-free-20260816T223337Z.json` — 20 measured burst samples.
- `companion-interpret-tool-free-20260816T223940Z.json` — 8 measured paced samples.

Neither artifact contains transcript or model-answer text. They retain prompt
structure, timings, result shape, sequence, device tier and every missing arm.
