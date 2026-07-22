# Durable goal and plan engine

Nova's foreground `ActiveGoal` remains the low-latency state for one spoken
turn. Work that must wait, survive a restart, or span more than one turn uses
the separate durable engine under `nova_voice.durable`.

## Records and storage

The engine defines frozen, strict, version-1 records for conversations, events,
goals, plans, plan steps, executions, delegation grants, proactive
interventions, memory references, and audit entries. Unknown fields and naive
timestamps fail validation.

`DurableAgentStore` uses its own SQLite database. Unless
`NOVA_VOICE_DURABLE_DATABASE_PATH` is set, it is placed beside the transcript
database as `durable-agent.sqlite3`. Startup applies numbered migrations,
prunes explicitly expired records, and reports the schema version and record
count through `/health`. Lifecycle writes use optimistic revisions and append
an immutable audit entry in the same transaction.

Backups use SQLite's online backup API and are accepted only after an integrity
and schema-version check. Restore verifies a copied candidate before copying it
into the live database and verifies the result again. Audit entries do not
expire with operational records.

## Execution contract

Each executable step receives one stable idempotency key for its complete
lifetime. A provider bridge must deduplicate that key before applying a side
effect. The engine persists a lease before invocation and completion after the
result. If a process dies after the provider acts but before completion is
stored, a later worker reclaims the expired lease and repeats the same key; it
must observe the original provider result rather than applying the mutation
again.

The runner supports tool, question, approval, wait, timer, event,
verification, retry, and compensation steps. Questions, approvals, and events
pause durably until `resolve_step` records the external decision. Timers and
waits compare timezone-aware stored deadlines. Successful steps are never
replayed because a sibling failed.

Providers declare resource-lock templates in `ToolPolicy`. Independent steps
run together only when every step is `parallel_safe` and the resolved resource
sets are disjoint. A missing declaration conservatively locks the complete
provider. Failed steps may consume their bounded retry budget or activate only
their named compensation step; already successful work remains satisfied.

## Recovery invariants

- migrations and initialization are idempotent;
- plan bundles are inserted atomically;
- optimistic revisions reject concurrent stale writers;
- a live lease cannot be stolen, while an expired lease can be reclaimed;
- one step has one execution record and one stable idempotency key across all
  attempts;
- plans and goals derive terminal or paused state only from persisted steps;
- backup/restore and a new process instance preserve the complete lifecycle.

The foreground voice path does not automatically convert every one-turn action
into a durable plan. Later event, authority, memory, and proactive milestones
will create and administer these records where persistence is required.
