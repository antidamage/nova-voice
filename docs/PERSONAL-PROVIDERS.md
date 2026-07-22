# Personal information providers

## iCloud calendar and reminders

Nova's optional `icloud` provider runs locally on Iridium and talks directly to
the owner's explicit iCloud CalDAV calendar and reminders collections. It uses
an Apple app-specific password; the credential and collection URLs stay in the
Voice service environment and are never sent to the dashboard, prompt, trace,
or spoken response.

Configure all four values or leave all four unset. A partial configuration does
not register the provider and therefore does not expose unusable tools:

```text
NOVA_VOICE_ICLOUD_USERNAME=
NOVA_VOICE_ICLOUD_APP_PASSWORD=
NOVA_VOICE_ICLOUD_CALENDAR_URL=
NOVA_VOICE_ICLOUD_REMINDERS_URL=
```

The provider supports bounded calendar and reminder reads, create/update,
calendar/reminder cancellation, and reminder completion. Date-times must be
timezone-aware or paired with an IANA timezone. RFC 5545 recurrence rules are
preserved. Stable UIDs make creation retry-safe, collection mutations are
serialized through provider resource locks, and every write is read back (or
confirmed absent for cancellation) before success is reported.

Household authority remains deterministic. The owner role may use the provider;
other recognized people need grants matching capabilities such as
`icloud.calendar.list` or `icloud.reminders.create`. Unknown/guest voices cannot
use it. Each attempt remains present in the immutable turn trace, and standing
grant use is recorded in the durable audit store.

Provider health appears under `capabilityProviders.icloud` when configured.
An iCloud outage degrades these tools without taking down ordinary local voice
or smart-home control.

## Private notes, lists, and contacts

The always-local `personal` provider stores these records in a separate SQLite
database on Iridium. Name-based writes execute only when one record matches;
ambiguous results return stable record IDs and do not mutate anything. List-item
completion/removal uses the same rule. Contacts retain structured phone, email,
and relationship fields rather than flattening identity into prompt text.

Every successful mutation returns a one-time `undoToken`. Undo is revision
checked: it restores the prior snapshot only if the record has not changed
since the token was issued, and refuses to overwrite newer work. Stable IDs
derived from action IDs make retries idempotent. The database location is set by
`NOVA_VOICE_PERSONAL_DATA_PATH` and provider health is exposed under
`capabilityProviders.personal`.
