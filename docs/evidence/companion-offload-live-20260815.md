# The first LLM work to actually leave Iridium — 2026-08-15

Measured against the deployed stack on Iridium, not a test harness.

## What was proven

A connected companion is offered real work, its answer is used, and Iridium's
single llama.cpp slot is never touched.

```
nova_voice.companion.session  companion session registered
    id=6ac601402f3841cba09f7f4c67098fd0 identity=companion-probe
    locality=home_lan roles=companion
nova_voice.api  classify_icon resolved source=companion reason=companion completed elapsed_ms=2.0
nova_voice.api  classify_icon resolved source=companion reason=companion completed elapsed_ms=2.5
```

`source=companion` is the load-bearing part. The peer answers with the *first*
icon in whatever vocabulary it is sent, so its answers are distinguishable from
the local model's by inspection:

| request | local model answers | observed |
|---|---|---|
| "Take estrogen" | `pill` | `pill` |
| "Wash hair" | `shower` | **`pill`** |

"Wash hair" returning `pill` is only explicable as the peer's answer.

## Latency

| path | wall time |
|---|---|
| local, cold | 7229 ms |
| local, warm | 524–650 ms |
| companion | 41 ms end-to-end; 2.0–2.5 ms inside the router |

The 41 ms includes a fresh mTLS handshake from `curl`. Do not read the ~250x as
the expected win for a *phone*: this peer answers from a lookup table, so the
number measures the protocol, not on-device inference. What it does establish is
that the offload machinery itself costs single-digit milliseconds, so whatever a
phone spends is nearly all model time.

## Why the local numbers vary so much

The 7229 ms first call and the ~550 ms ones after it are the same code path.
The first request pays for a cold model; the difference is not evidence about
routing and should not be quoted as a baseline. Only the warm figure is
comparable.

## What this does and does not establish

**Does:** the deployed server-side path — eligibility, offer, accept, result
validation, and using the result instead of the local model — is sound. When an
offload does not happen, the investigation belongs on the device.

**Does not:** anything about Apple's on-device model. No `SystemLanguageModel`
inference has been measured yet; the phone was locked, which suspends the app
and drops its session.

## Reproducing it

`ops/companion_live_peer.py`, run on Iridium. It advertises `classify_icon`
only by default, and refuses to advertise `render_response` or `interpret`
without an explicit flag — those answers are **spoken aloud in the house**, and
Iridium offers only what a peer advertises, so the advertised list is the gate.

```
ops/issue-satellite-identity.sh companion-probe /tmp/probe-identity
/opt/nova-voice/venv/bin/python ops/companion_live_peer.py \
    --cert /tmp/probe-identity/client.crt --key /tmp/probe-identity/client.key \
    --announced-id companion-probe --jobs 2
```

## On the phone itself

Neptunium (iPhone 17 Pro Max, iOS 26.6), Apple's `SystemLanguageModel`.

### `classify_icon` — a clear win

| | |
|---|---|
| first call | 1261 ms (cold model) |
| warm | 219, 227, 231, 234 ms |
| Iridium, warm | 524–650 ms |

Roughly **2.4x faster than Iridium**, and correct on every sample: *wash hair →
shower*, *pay power bill → bell*, *put washing on → washing-machine*, *take
vitamins → pill*, *take estrogen → pill*. Counters: 5 offered, 5 accepted, 5
completed, 0 failed, 0 fell back.

### `render_response` — latency parity, quality regression

Timing is fine. Of 18 offers, 16 completed on the phone at 1.9–3.1 s, against
1.9–2.7 s locally. Two missed the 4 s deadline, and a miss is expensive: it
costs the deadline *plus* the local render, 7.2 s in one observed turn.

The output is the problem. Same prompts, attributed per-sample from
`render_response resolved source=…` rather than assumed:

| question | Iridium | phone |
|---|---|---|
| favourite colour | "Pink? No way. It's all grey metal and neon rain down here anyway." | "I will say something." |
| how are you feeling | "Feelin' peachy as hell, you know I do, but honestly?" | "I will say something. I can't say that I have a favorite colour, because I am an AI and do not have emotions." |
| a fun fact | "So, that neon flickering we see everywhere came out of film noir back in the forties…" | "I will say something. I'm just an AI, so I don't have feelings or preferences…" |

Three faults, in descending order of how much they matter:

1. **The persona is gone.** Nova's configured personality is absent and
   replaced by stock assistant register — "I'm just an AI, I don't have
   feelings" — which the instructions explicitly forbid. `render_response` *is*
   the assistant's voice, so this is not a rough edge, it is the feature.
2. **It answers the previous question.** Two of three replies answered
   "favourite colour" when asked something else.
3. **"I will say something."** prefixed every reply. That was ours: the shared
   reply prompt ended with *"Return only the response JSON schema"*, which is
   true of llama.cpp (which is additionally schema-constrained at the sampler)
   and nonsense to a device generating plain text. The directive now belongs to
   the local backend only, and `RenderRequest.system` stays transport-neutral.

Fault 3 is fixed; 1 and 2 are not, and are not obviously ours to fix.

### Consequence

`interpret` and `render_response` now **ship routed local**, in code, with the
reasoning in `_default_routes`. The capability stays built and switchable at
runtime through `POST /v1/companion/routing`. It is off until the output is
good, not until the plumbing works.

The non-hot-path passes keep their `companion_preferred` default: those
measured well.

## The lock screen, observed

Mid-measurement the session read `connected: true` with `tier: off,
telemetry is 296s stale`, and every route became ineligible. The socket
survived; the app had been suspended behind the lock screen and stopped
sending. The staleness guard is what stopped Iridium offering work into a hole
— it behaved exactly as designed — but it is the concrete form of the open
question: **a companion that only works while the app is foregrounded cannot
carry hot-path work**, because Iridium would pay the round trip and fall back
on most turns.

## A methodology correction

An earlier attempt at this comparison controlled by *disconnecting the phone*
and assuming everything afterwards was local. It was not: the app comes back on
its own, silently, and several "local" samples were companion-rendered. The
numbers above are attributed per-sample from the server's own
`source=` log line instead. `POST /v1/companion/routing` exists partly so this
never has to be done by inference again.

## Incidental finding

With no companion connected, `/v1/companion/status` reported
`render_response: {fellBack: 27}` — the reply pass had already been routed and
had fallen back to the local model 27 times across live voice turns, exactly as
intended. Routing is invisible until a device is actually there.

---

## Correction, 2026-08-16: it was never the lock screen

The section above attributes `tier: off, telemetry is 296s stale` to the app
being suspended behind the lock screen, and reads the staleness guard as having
worked as designed. The guard did work as designed. The premise was wrong.

**The app never sent telemetry after the handshake at all.** It sent one
reading during `handshake()` and then went silent for the life of the session,
foregrounded or not. Nothing on either side drove liveness: the client had no
periodic send, and the server never sends the `ping` the protocol defines.

Measured on the live system at 09:31 with the phone plugged in, awake, and on
the home LAN:

```
connected: true    locality: home_lan    workloads: all five
tier: "off"        tierReason: "telemetry is 995s stale"
telemetryAgeSeconds:     994.67
lastHeartbeatAgeSeconds: 994.67      <- identical, to the millisecond
```

The two ages being *the same number* is the tell: not "the phone went quiet
when it slept", but "nothing has arrived since the one frame at connect". Every
companion route read `tier off is below reduced` on a charging phone.

So the real behaviour was worse than the note claimed, and in a more boring
way: **the companion went ineligible three minutes after every connect and
stayed that way**, which is why offload appeared to work only in the minutes
right after a reconnect. The earlier `source=companion` results were real; they
were taken inside that window.

### The fix (NPT-205)

`CompanionSession.maintainTelemetry(everySeconds:)` sends on a 45s cycle — a
fraction of the 180s staleness window, so two lost frames still do not age the
device out — and the coordinator additionally sends on every state change that
can move the tier: charge state, battery level, Low Power Mode, thermal state,
foreground/background.

Verified live after installing the fixed build, sampling every 25s:

```
tier=full age=15.8   tier=full age=41.7   tier=full age=19.7   tier=full age=45.3
tier=full age=24.5   tier=full age=3.1    tier=full age=28.7   tier=full age=7.4
```

A sawtooth capped at 45.3s, never approaching 180s. `tierReason: charging`.
All three background routes returned to `eligible`.

Offload confirmed working again end to end, four for four, each answer correct:

| Request | Answer | source | elapsed |
|---|---|---|---|
| Wash hair | `shower` | companion | 940 ms (cold) |
| Take vitamins | `pill` | companion | 628 ms |
| Vacuum lounge | `broom` | companion | 405 ms |
| Buy milk | `cart` | companion | 342 ms |

### What this does and does not settle

It does **not** answer the hot-path question the original section raised. A
suspended app still cannot answer, and the loop stops when iOS suspends it —
correctly, since an app that cannot run cannot do the work. Keeping the device
eligible while suspended needs a background mode, which is M6's audio-session
work rather than a timer.

What it settles is that the previous evidence for that question was not
evidence: the device was ineligible for reasons that had nothing to do with the
lock screen, so nothing measured before 2026-08-16 says anything about how a
locked phone behaves. That measurement has not been taken yet.

## The locked screen, actually measured (2026-08-16)

The correction above says nothing measured before 2026-08-16 describes a locked
phone. It has now been measured, and the answer is different from — and better
than — what both earlier notes assumed.

At 10:09 the phone locked on its own mid-session. At **10:10:37** the server
logged:

```
nova_voice.companion.session companion session released id=a5b6b24ead1149c3b1b59059fe97bb8a
```

and `/v1/companion/status` went to `connected: false` with the session fields
gone entirely. Sampled every 40s for four minutes afterwards: still
disconnected, no reconnection while locked.

**When iOS suspends the app, the socket closes and the session is released
cleanly.** It does not linger as a connected-but-silent session waiting to go
stale. So:

* the staleness guard is not what protects the hot path here — the disconnect
  is, and it is immediate rather than three minutes late;
* there is no window in which Iridium believes a suspended phone is healthy and
  offers work into a hole; and
* the 995s-stale session in the correction above could only ever have been the
  *foreground* liveness bug, which is consistent with what the fix addressed.

This tightens the hot-path question rather than answering it. A companion is
eligible while the app is resident and ineligible the moment it is not, with no
ambiguous middle. Routing `interpret` or `render_response` to the phone would
therefore not risk a stalled turn from a stale session — it would simply fall
back on every turn where the phone was asleep, which is most of them. Making
the phone resident enough to carry hot-path work is M6's audio-session
territory, not something routing configuration can reach.
