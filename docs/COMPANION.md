# The companion topology

Optional. A household device — in practice a phone — that holds an
authenticated session with Nova and answers reasoning workloads on its own
model, so those workloads stop occupying the voice host's single llama.cpp
slot.

This document is for deciding whether to enable it. `COMPANION-RUNBOOK.md` is
for operating it once you have. All identifiers here are placeholders.

## Why it exists

The voice host runs llama.cpp with one parallel request slot. Five passes
compete for it: `interpret`, `render_response`, `extract_self_profile_update`,
`confirm_objective` and `classify_icon`. Two of those are on the spoken hot
path, and the other three are not — but they queue behind it anyway. Moving the
background three to a device that is idle most of the time is the whole idea.

## What it is not

- **Not a second Nova.** The companion has no capability registry, no policy
  engine, no plan store and no credentials. It answers questions.
- **Not a satellite.** Audio capture and playback are a separate role with a
  separate switch. A device may hold both; they do not share a lifecycle.
- **Not required.** Everything it does, the voice host can still do. Disabling
  it restores the previous behaviour exactly.

## The guarantees

These are the properties the design is built around, in the order they matter.

**No eager hedge.** Once the device accepts a workload, the local model does
not also start. Running both would keep the slot occupied and there would be no
point to any of this. The local path starts only on a terminal outcome —
rejection, acceptance timeout, disconnect, invalid result, cancellation, or the
deadline — and then it runs exactly once.

**The device proposes; the host executes.** A returned plan is validated
against the same schemas and the same capability catalogue as a locally
produced one, and every action then runs through ToolPolicy and
HouseholdAuthority on the host. A plan naming a tool that was not offered is
discarded whole rather than having the bad action removed — a plan with a
clause deleted is not the plan the model made, and "turn the lights off and
lock the door" minus one clause still reads as success.

**Failure is ordinary.** A device declining on battery grounds is the system
working. It falls back, it does not count against the failure budget, and it
does not affect health.

**Nothing waits forever.** Every wait — acceptance, result, tool callback,
approval — is bounded and registered, so a device that vanishes mid-job
terminates every caller rather than leaving a future nobody will resolve.

## Locality

Reachability is not authorization. The session's locality is derived from the
**actual peer address** against configured home subnets: `home_lan`, `tailnet`
or `other`. A device connected through a tailnet address is `tailnet` even if
its radio is Wi-Fi, and a device that claims to be at home cannot unlock a
home-LAN-only route by claiming it.

General reasoning replacement requires `home_lan`. Personal-context tools may
be permitted over an authenticated tailnet path when away — a narrower
permission than replacing the host's reasoning.

## The tier model

Telemetry — battery, charging, Low Power Mode, thermal state, app state, model
availability, telemetry age — maps onto `full` / `reduced` / `advisory` / `off`,
with three guards:

- **Hysteresis**, so a reading hovering on a threshold cannot oscillate.
- **Dwell**, so an *improvement* must hold before it is believed. A
  *degradation* applies immediately: a hot or nearly-flat device should stop
  being offered work at once, not a minute later.
- **Staleness**, so telemetry that stopped arriving is treated as no evidence
  of health rather than as continued health.

The device also re-checks itself immediately before accepting, because an offer
can arrive in the gap between a reading and reality changing. Refusing costs
one round trip; accepting and then struggling costs the whole deadline.

## The data boundary

Scoped personal context **may** leave the device in v1. That is a deliberate
decision and the reason calendar and reminder integration is possible at all.
What bounds it:

- Every read tool requires a window, an item cap, or both. No tool can ask for
  "my calendar".
- Payloads carry a sensitivity class. Logs record counts, kinds, latency and
  correlation ids — never titles, notes, coordinates or Health values.
- Retention follows the class: a finished job keeps its structural history for
  a month and loses its content references on a much shorter clock — a day for
  ordinary, an hour for personal, five minutes for Health.
- Health writes are an allowlist of specific sample types with units and
  plausible ranges. There is no generic sample writer.

A stricter mode, where raw records never leave the device and it returns only
derived conclusions, is a plausible future improvement. It is not a
prerequisite and not an implicit promise.

## Approvals

Reads execute under ordinary policy. Every create, update, delete, completion
and Health write becomes a durable approval instead:

1. The proposal is stored with its **exact** target and arguments — approving
   a sentence must execute the action described at the moment it was described,
   not whatever that sentence would resolve to later.
2. The voice turn ends with an acknowledgement. It does not block: a human may
   answer in four seconds or four hours.
3. The device shows a prompt and returns a signed decision. The signature
   covers the approval id, the decision and a per-proposal nonce, so one
   captured "yes" cannot be replayed against a later proposal and a "yes"
   cannot be presented as a "no".
4. On approval, the mutation runs **once**, afterwards, under a stored
   idempotency key.

A duplicate tap, a reconnect, or a restart between the decision and the write
all resolve to one execution. Decisions are first-writer-wins: a later answer
cannot flip an earlier one, because that would make the outcome depend on
network timing.

## Kill switches

| Switch | Effect |
|---|---|
| Per-workload route → `local` | That pass stops going to the device |
| `force_local` | No reasoning goes to the device; the session and personal tools remain |
| `companion_enabled = false` | The whole feature; previous behaviour restored exactly |
| `satellite_enabled = false` | Audio capture only; reasoning and personal tools unaffected |
| `personal_provider_enabled = false` | Personal manifests withdrawn; any existing fallback provider reappears |
| Certificate revocation | The device cannot register at all |

None of these require reinstalling the app, and none can lock the household out
of local control.

## Should you enable it?

Enable the background passes if you have a device that is usually at home,
usually charged, and running a current build. The measured benefit is real and
the failure mode is a fallback nobody notices.

Be more careful with the hot-path passes. Latency parity is not the bar —
`render_response` *is* the assistant's voice, and a faster reply that has lost
the persona is a regression no latency figure pays for. Measure the output, not
just the timing, and keep the route switchable at runtime so it can go back
without a code change.
