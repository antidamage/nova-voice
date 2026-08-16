# Companion: built, wired, and running

Three different things, and the difference matters. A module can be complete
and tested and still have nothing calling it, which is a fine state to be in
deliberately and a bad one to discover by accident.

Audited 2026-08-16 by checking, for each companion module, whether anything
outside its own tests imports it.

## Running in production

Live on the voice host and exercised by real turns.

| Piece | Evidence |
|---|---|
| `companion/auth.py`, `session.py`, `dispatch.py` | A device authenticates, registers, and supersedes on reconnect |
| `companion/router.py` | `classify_icon` resolves `source=companion` on live requests |
| `companion/routed.py` | Wraps the live interpreter; hot-path routes ship `local` |
| `companion/tiers.py`, `locality.py` | Eligibility and `home_lan` classification, both observed live |
| `companion/compaction.py`, `workloads.py` | Every offered payload goes through them |
| `companion/retention.py` | Swept every five minutes from the API lifespan |
| `companion/presence.py` | Published on `/v1/companion/status`, including when nothing is connected |
| iOS app: transport, telemetry, self-check, diagnostics ring | Installed; telemetry verified at a 45s cadence |

## Built and tested, with no caller yet

Complete and covered, imported by nothing outside their own tests. **This is
the honest state of M4 and most of M5's server side.** Each is waiting on a
specific piece of wiring, named below — none is waiting on more of itself.

| Piece | Waiting on |
|---|---|
| `companion/jobs.py` | Used by `ledger` and `retention`, which are themselves uncalled |
| `companion/ledger.py` | Partly wired: `reconcile` runs on every device registration. The rest waits on a durable-job producer — NPT-408's research/briefing synthesis |
| `companion/approvals.py` | A mutation path that reaches it: `providers/companion` being registered |
| `companion/callbacks.py` | The same durable-job producer |
| ~~`companion/retention.py`~~ | **Wired** — runs on a 5-minute loop from the API lifespan |
| ~~`companion/presence.py`~~ | **Wired** — published on `/v1/companion/status` |
| `providers/companion/` | Deliberately unregistered until the device can answer the calls |

The last one is a choice rather than an omission: registering tools nothing can
answer would put them in planner catalogues, and the model would plan actions
that fail.

## Built, not yet exercised on the device

Compiles and installs; the code path has not run against real data.

- EventKit calendar and reminder reads and mutations
- `HomeLocationRuntime`
- App Intents / Shortcuts entry points
- The bounded diagnostics ring, in anger

## Not built

- HealthKit itself. The validation and allowlist are done; the **entitlement**
  is not added, because a signing failure would leave the app un-installable.
- The satellite (audio) role — all of M6.
- Local synthesis stages in `ResearchManager` / `BriefingManager`, which
  NPT-408 needs before a companion route can fall back to anything.
- NPT-002's contended latency baseline, which needs load against the live
  household stack.

## Cross-cutting test matrix

Against the roadmap's table. "Covered" means a test exercises it against the
real component, not a mock of it.

| Dimension | State |
|---|---|
| Route modes | Covered — every mode, plus the breaker and both operator switches |
| Locality | Covered for `home_lan`/`tailnet`/spoofed. **Not** for a path transition on a real device |
| Identity | Covered: wrong SAN, wrong CA, expired, replayed nonce, superseding session |
| Runtime | Covered for unavailable model, invalid typed result, unsupported workload. **Not** for iOS 27's guarded path |
| Job durability | Covered: duplicate frame, restart, lease expiry, cancel, late result, fallback |
| Tool callback | Covered: unknown tool, unoffered tool, limits, timeout, at-most-once |
| Approval | Covered: approve, deny, expire, duplicate tap, forgery, replay, mid-execution restart |
| Personal data | Covered for bounding, permission states, redaction and retention. **Not** for DST/recurrence, which needs synthetic calendars on a device |
| Audio | Not covered — the role does not exist |
| Resource | Covered for battery, charging, LPM, thermal, stale telemetry. **Not** for memory pressure or deep-tier unload |

The gaps cluster in one place: **everything that needs a physical device doing
something in real time.** That is not an accident of effort — it is the part
that cannot be automated from here, and it is what the next session with the
phone in hand should spend its time on.
