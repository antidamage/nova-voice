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

## Incidental finding

With no companion connected, `/v1/companion/status` reported
`render_response: {fellBack: 27}` — the reply pass had already been routed and
had fallen back to the local model 27 times across live voice turns, exactly as
intended. Routing is invisible until a device is actually there.
