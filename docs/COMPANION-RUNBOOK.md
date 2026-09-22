# Companion device runbook

Operating the optional companion role: a household phone that answers Nova's
reasoning workloads on its own model instead of occupying Iridium's single
llama.cpp slot.

Every value in this document is a placeholder. Real hostnames, addresses,
tailnet names, device names and certificate material stay out of git; they live
in the deployment's own configuration files and in the operator's head.

## What it is, in one paragraph

Nova offers a workload to the device. The device accepts or declines. If it
accepts, Nova does **not** also start the local model — that would keep the
slot occupied and defeat the point — and waits for the answer, falling back to
the local path only on a terminal failure. Everything the device returns is
validated by Iridium against the same schemas and the same capability
catalogue as a locally produced plan, and every action still executes on
Iridium under ToolPolicy and HouseholdAuthority. The device proposes; it never
performs.

## Is it working?

```bash
curl -sk --cert <client.crt> --key <client.key> \
  https://<voice-host>:<port>/v1/companion/status
```

The fields that answer the question, in the order worth reading them:

| Field | What it tells you |
|---|---|
| `connected` | Is a device holding an authenticated session at all |
| `locality` | `home_lan`, `tailnet` or `other` — derived from the peer address, never from what the device claims |
| `tier` + `tierReason` | Whether the device is well enough to be offered work, and why not |
| `telemetryAgeSeconds` | How fresh that judgement is |
| `routes[*].eligibility` | Per workload, the exact gate that is stopping it |
| `counters` | Offered / accepted / rejected / completed / failed / fellBack per workload |

### The one diagnostic worth memorising

If `telemetryAgeSeconds` and `lastHeartbeatAgeSeconds` are **the same number**,
nothing has arrived since the session was established. That is a client
liveness fault, not a sleeping device.

A device whose app has been suspended does not go quiet — it **disconnects**,
and `connected` goes false within about a minute. So `connected: true` with
frozen telemetry can only ever be a bug in the client. This is worth knowing
because the symptom looks exactly like a phone in a pocket, and diagnosing it
as one costs a day.

## Turning it off

In increasing order of severity. Each is reversible and none requires
reinstalling the app.

| Goal | Action |
|---|---|
| Stop one workload going to the device | Set that route to `local` |
| Stop **all** reasoning going to the device, keep the session | `force_local = true` |
| Stop the companion entirely | `companion_enabled = false` |
| Stop the device permanently | Revoke its certificate |

`force_local` takes effect immediately and does not disconnect the device or
require the owner to touch the phone, which is the point: it has to be usable
during an incident.

Setting `companion_enabled = false` restores the previous behaviour exactly —
this is the rollback invariant, and there is a test asserting it.

## Route modes

| Mode | Meaning |
|---|---|
| `local` | Always Iridium. The device is never offered it. |
| `companion_preferred` | Offer to the device; run locally if it declines or fails |
| `companion_only` | Device or nothing — returns an explicit unavailable rather than silently doing something different |
| `companion_fallback` | Local first, device only if local fails. Exists for completeness; not a v1 mode |
| `disabled` | Do not run this optional workload anywhere |

The dashboard exposes three of these per pass, in the operator's vocabulary:
**local**, **companion**, **both**, mapping to `local`, `companion_only` and
`companion_preferred`.

### Which routes should be where

Background passes (`classify_icon`, `confirm_objective`,
`extract_self_profile_update`) default to `companion_preferred`. They are
measured faster on-device, and a wrong answer costs a wrong icon rather than a
wrong action in the house.

**What moving them actually buys.** Measured on the live host and reproduced
across two runs: contention roughly doubles a turn — `service` p50 6.9 s idle
against 16.4 s contended — while `interpretation` itself moves about 3%. The
cost is **queue wait**, not slower compute. So offloading a background pass
does not speed anything up; it removes the wait in front of the turn behind it.
That also means the benefit only appears when turns actually collide, which is
during a conversation rather than on a quiet afternoon — and it is why an
idle-only latency check will show you nothing.
See `docs/evidence/companion-baseline-20260816.md`.

Hot-path passes (`interpret`, `render_response`) default to **`local`**, and
that is a measured decision rather than caution. The capability is built and
switchable at runtime; it is off because the on-device output was not good
enough to hold a spoken turn, not because the plumbing is unproven. See
`docs/evidence/companion-offload-live-20260815.md` before changing them.

## Setting up a new device

1. **Issue an identity.** `ops/issue-satellite-identity.sh` mints a household
   client certificate and the iOS-compatible `p12`. The identifier must be a
   safe identifier; the issuance path validates it and refuses anything else.
2. **Configure the endpoints.** The app reads a bundled `companion-config.json`
   holding the LAN endpoint, an optional tailnet endpoint, the announced
   identity and the keychain label. It is untracked and written at build time
   by the machine that has those values.
   *Build settings do not work for this.* Xcode only maps `INFOPLIST_KEY_*` for
   Info.plist keys it already knows and silently drops custom ones, so the
   build succeeds and the app starts unconfigured.
3. **Bundle the household CA** as a DER file. The app anchors it explicitly and
   turns system roots off, so a certificate from any public CA is rejected.
4. **Install and launch.** The app imports the identity into the keychain on
   first launch with `AfterFirstUnlock` accessibility, which is what lets it
   reconnect after a reboot without someone physically unlocking the device.
5. **Confirm** `connected: true` and `locality: home_lan` in the status
   endpoint before enabling any route.

### Locality is derived, never claimed

The device opens the LAN endpoint when it is on home Wi-Fi and the tailnet
endpoint otherwise. Iridium classifies the session from the **actual peer
address** against configured home subnets. A device connected over the tailnet
is `tailnet` even if its radio happens to be Wi-Fi, and a device asserting it
is at home cannot unlock a home-LAN-only route by saying so. Any SSID the
device reports is diagnostic and is never a security gate.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `connected: false` | App suspended, killed, or off the network | Open the app. There is no remote wake — see below |
| `connected: true`, telemetry age == heartbeat age | Client liveness fault | Client bug; the device is not sending |
| `tier: off (…)` | The parenthesised reason names it: flat, hot, no model, or stale | Fix what it names |
| `route is local` | The route is deliberately local | Change it in the dashboard if intended |
| `companion does not advertise <workload>` | Older build, or the engine cannot serve it | Reinstall a current build |
| `connected via tailnet, route needs home_lan` | Away from home | Working as intended |
| `workload is paused by the failure circuit breaker` | Repeated failures tripped it | It releases itself; investigate the failures |
| Every route eligible, nothing offered | The feature or `force_local` switch | Check `enabled` and `forceLocal` |
| `presence: unknown` while the phone is clearly here | Normal. A tailnet session, or no session, is not evidence of anything | Nothing. `unknown` is a real answer |
| A job the device says it holds gets cancelled on reconnect | The server no longer owns it — it was reclaimed or reassigned | Nothing. This is what stops it running twice |

### A deploy disconnects the companion

Deploying restarts `nova-voice`, which drops the companion session. The app
reconnects on its own **only if it is running** — and if the phone is locked at
that moment, it is suspended and cannot. Observed on the 2026-08-16 deploy: the
device stayed `connected: false` afterwards until the app was opened.

So after a deploy, expect to open the app once. There is nothing to fix; it is
the same suspension behaviour as everywhere else, and `presence` correctly
reports `unknown` rather than `away` in the meantime.

### There is no remote wake

The deployment uses the free Apple developer path, so push notifications are
unavailable. Nothing can wake the app remotely. Shortcuts, background refresh
and a geofence are best-effort hints, not guarantees. When the device is not
connected, the honest instruction is "open the app" — do not build a dashboard
action that implies otherwise, and never generate an SSH command for a phone.

### Profiles expire in seven days

Also a consequence of the free path. A device build stops launching about a
week after install and must be reinstalled. Plan any soak test around that.

## Certificate rotation and revocation

Rotating: issue a new identity, rebuild with it, install. The old session is
superseded on the next connect — a newer authenticated session always
supersedes an older one, because a device that has just reconnected is far more
likely to be the live one than a socket that has not yet noticed it is dead.

Revoking: remove the identity from the household CA's trust and restart the
voice service. A revoked device cannot register, and a device that cannot
register cannot influence anything: it holds no credentials for the dashboard,
cannot name a tool it was not offered, and cannot approve its own mutations.

Losing the device is the same operation. Nothing about revocation can lock Nova
out of local control.

## Data that leaves the phone

Scoped calendar, reminder, location and Health results may transit to Iridium
when a request needs them — that is a deliberate v1 decision, not an accident.
What that permits and what it does not:

- Bounded queries only. Every read tool requires a window, an item cap, or
  both; there is no tool that can ask for "my calendar".
- Structural logs only. Counts, kinds, latency and correlation ids are logged;
  titles, notes, coordinates and Health values are not.
- Retention by class. A finished job keeps its workload, attempts, outcome and
  trace id; its content references expire on a per-class clock — a day for
  ordinary, an hour for personal, five minutes for Health.
- Every mutation goes through a durable approval carrying the exact proposed
  change, and executes once afterwards under a stored idempotency key.

## Things that look broken and are not

- **A device declining work.** Rejection is ordinary flow. It does not count
  against the failure budget, because an evening of sensible refusals should
  not disable a healthy device.
- **`fellBack` climbing with `offered` at zero.** The route is local, or the
  device is ineligible. The pass ran locally, as designed.
- **A job cancelled with no result.** A superseded turn. The device is told to
  stop, and a late answer from it is recorded and ignored rather than applied.
