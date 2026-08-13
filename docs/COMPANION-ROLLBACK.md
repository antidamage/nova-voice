# Companion rollout and rollback switches

Every risky capability in this topology names **one** switch that turns it off,
and none of them requires reinstalling or even touching the phone. That is the
point: if the companion starts behaving badly at 2am, the fix is a setting on
Iridium, not a trip to find a device.

All switches are `NOVA_VOICE_COMPANION_*` settings; see
`config/companion.env.example`. Defaults ship the whole topology disabled.

## The switch for each capability

| Capability | Switch | Effect |
|---|---|---|
| Everything | `COMPANION_ENABLED=false` | Restores the previous Iridium-only behaviour exactly. The socket refuses to open, no route is consulted. |
| Companion replacing any Iridium reasoning | `COMPANION_FORCE_LOCAL=true` | Every workload resolves locally. Personal tools stay available — this is how to isolate a reasoning-quality problem without losing Calendar/Reminders. |
| One specific workload | per-workload route → `local` or `disabled` | Route overrides are per row. `local` keeps the pass running on Iridium; `disabled` stops an optional pass entirely. |
| Phone microphone | `COMPANION_SATELLITE_ENABLED=false` | Stops capture. Does not disable personal tools or cancel accepted non-audio jobs. |
| Calendar/Reminders/location/Health | `COMPANION_PERSONAL_PROVIDER_ENABLED=false` | Removes the companion's personal manifests and reveals the existing iCloud fallback where configured. |
| A specific device | `COMPANION_ALLOWED_IDENTITIES` | Naming identities restricts which may connect at all. |
| Legacy unbound satellite identities | `COMPANION_ALLOW_UNBOUND_SATELLITES=false` | Ends the migration window once every native client proves its identity (NPT-108/NPT-910). |
| A phone that fails repeatedly | automatic | The failure budget pauses offers per workload without any operator action, then probes again after the breaker dwell. |

## Why the two role switches are separate

Muting a microphone and disabling an agent are different intentions, and
conflating them produces exactly the wrong behaviour in both directions: you
cannot silence a phone in a meeting without also losing your calendar tools,
and you cannot stop delegating reasoning without going deaf in that room.

So the roles have independent switches, independent state machines and
independent retry budgets. Losing one must not disturb the other.

## Rollback invariants

These hold at every point after server code lands, and each has a test:

1. `companion_enabled=false` restores existing behaviour — no offer is made and
   no route is consulted (`test_companion_config.py`).
2. `force_local=true` prevents phone reasoning while leaving personal tools
   separately controllable.
3. `satellite_enabled=false` stops capture without disabling personal tools or
   accepted non-audio jobs.
4. `personal_provider_enabled=false` removes the phone's personal manifests and
   reveals the iCloud fallback.
5. Deleting or denying the phone's credentials makes the session unavailable.
   It cannot lock Nova out of local control — a companion is never in the path
   of a locally planned action.
6. A phone disconnect never means `away`, never authorizes an action, and never
   cancels an already-approved local execution that has acquired its lease.

## Staged enablement

Enable in this order, soaking each stage before the next. Each stage is a
configuration change, reversible by the switch in its row above.

1. Background classifiers — `classify_icon`, `extract_self_profile_update`.
   Lowest risk: out of turn, and `None` is already a valid outcome.
2. `confirm_objective`. Still inside the verify loop's own budget.
3. `render_response`. First pass whose lateness a listener would notice.
4. `interpret`. Only after the measurement in NPT-310 shows the offload
   actually improves interactive latency.
5. Personal reads, then personal writes behind approval.
6. Continuous microphone.

If a stage's fallback rate or latency is not acceptable, its row stays off. The
route table's final defaults come from measurement (NPT-909), not from the
roadmap's suggested values.
