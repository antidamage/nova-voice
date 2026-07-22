# Household authority and administration

Nova classifies every voice turn as `owner`, `recognized_household`, or
`guest`. A recognized speaker defaults to the household class until an owner
assigns an explicit role; an unknown, provisional, or ambiguous speaker is a
guest. Guests can use addressed read-only web knowledge but cannot inspect or
change household state. Recognized household members can read/control the home
and manage tasks. Owners have every capability.

Standing grants can add one capability for one recognized person. Grants may
restrict targets, recipients, locations, local-time windows, weekdays, expiry,
maximum uses, and monetary amount/currency. Every use is durably counted.
Revocation is immediate because the live policy cache is updated in the same
operation as the transactional durable record. Create, update, use, revoke,
role assignment, plan cancellation, and goal cancellation all append immutable
audit entries.

The production Voice endpoints under `/v1/agent/` are protected by the Voice
server's household-CA mutual TLS boundary. The Dashboard uses its provisioned
client identity to expose owner controls in Config > Voice agent authority.
The screen assigns roles, creates/revokes grants, cancels durable goals, and
shows recent audit replay. Direct APIs additionally expose filtered cursor
replay at `/v1/agent/audit` and plan cancellation.

Policy is fail-closed: an absent role, expired/exhausted grant, scope mismatch,
missing amount for a budgeted grant, or unavailable durable authority prevents
the action. Conversational replies are unaffected, and shadow mode still never
executes an otherwise-authorized action.
