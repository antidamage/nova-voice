# The contention baseline (NPT-002)

What a voice turn costs on Iridium today, idle and contended, so the offload
has something to be judged against rather than an impression.

Two runs, two days apart, on the live host with the house quiet. Both are dry:
household requests are built and reported, none is sent, nothing in the house
changes.

- `companion-baseline-20260814T071215Z.json` — 25 turns per condition
- `companion-baseline-20260816T003847Z.json` — 16 idle / 18 contended turns

The second run exists because the first was captured and never interpreted. Two
independent runs agreeing is also worth more than one: these numbers are
reproduced, not sampled.

## The numbers (p50, milliseconds)

| Phase | idle 14 Aug | idle 16 Aug | contended 14 Aug | contended 16 Aug |
|---|---|---|---|---|
| `service` | 7,018 | 6,892 | 13,727 | 16,423 |
| `audioTotal` | 9,854 | 9,164 | 16,846 | 18,408 |
| `interpretation` | 5,792 | 5,648 | 6,125 | 5,930 |

From the 16 Aug run, the rest of the idle picture:

| Phase | p50 | p95 | max |
|---|---|---|---|
| `interpretation` | 5,648 | 6,858 | 7,661 |
| `response` | 1,911 | 4,044 | 12,206 |
| `stt` | 1,100 | 1,234 | 1,341 |
| `tts` | 957 | 2,772 | 5,681 |
| `ttsFirstChunk` | 943 | 2,738 | 5,652 |
| `denoise` | 125 | 168 | 169 |
| `providerContext` | 27 | 99 | 104 |
| `speaker` | 16 | 104 | 204 |
| `execution` | 0 | 0.4 | 1.1 |

## What it actually says

**Contention roughly doubles a turn, and it is not because the model gets
slower.** `interpretation` moves by about 3% between conditions — 5,648 → 5,930
on 16 Aug, 5,792 → 6,125 on 14 Aug. `service` moves by 138% and 96%. The extra
seven to nine seconds are **queue wait**: a turn arriving while the single
`--parallel 1` slot is busy sits there until it is free, and the compute itself
is unchanged when it finally runs.

That is the premise of the whole companion project, and it is now measured
rather than assumed. Moving a background pass off the slot does not make
interpretation faster — it removes the wait in front of it. Which means:

- **The win is bounded by how often turns actually collide.** On a quiet
  household that is rarely; during a conversation, where `interpret`,
  `render_response` and `extract_self_profile_update` all fire per turn, it is
  most of the time. The three passes that ship `companion_preferred` are
  precisely the ones stacking up behind the hot path.
- **Measuring an idle system would have shown almost nothing.** An ordinary
  latency check never reproduces this, which is why the harness has a contended
  arm at all.

**Interpretation is the dominant cost either way.** 5.6 s of a 6.9 s idle turn.
Nothing about offloading a background pass changes that, and a companion that
took `interpret` itself would have to beat 5.6 s to help — the on-device model
measured at 1.9–3.1 s for `render_response`, so the ceiling is plausible, but
the quality is what stopped it (see `companion-offload-live-20260815.md`).

**`response` has a long tail.** p50 1.9 s, max 12.2 s. Worth its own look; not
a contention problem, since it is present in the idle arm.

## Comparing against this later

Re-run the same harness and compare `service` p50 in the **contended** arm. That
is the number the offload is supposed to move. `interpretation` p50 is the
control: if it changes, something other than routing changed too, and the
comparison is not clean.

```
/opt/nova-voice/venv/bin/python ops/companion_baseline.py \
    --host https://127.0.0.1:8766 \
    --cert <client.crt> --key <client.key> \
    --repeats 3 --out docs/evidence
```

Run it with the house quiet. It is dry and changes nothing, but it occupies the
GPU for several minutes and will make any real turn slow while it runs.
