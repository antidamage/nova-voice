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

## Weather, media, recipes, documents, and household knowledge

Live weather and media remain replaceable read contracts on the existing Nova
dashboard provider: `nova.query` accepts `weather` and `media`, returns the
dashboard generation timestamp, and includes a `nova://` citation. Media writes
continue through the verified `nova.control` contract, so this milestone does
not duplicate Home Assistant state or bypass its allowlist.

Recipes, text documents, and household knowledge use the Iridium-local
`library` provider. Every search result contains a stable citation, content
SHA-256 revision, bounded excerpt, audience, and update time. Shared search
never returns owner-private records; private search is a separate capability.
Recognized household members receive only `library.search_shared` by default,
while private reads and every write require owner authority or an explicit
grant. Writes are revisioned, resource-locked, retry-safe, and return the same
conflict-safe undo tokens as notes, lists, and contacts.

## Draft-first communications

Email, messages, and invitations use a separate `communications` provider and
database. Voice may resolve one contact and create or preview a draft, but the
immediate voice executor cannot send: `communications.send` is a confirmation
policy and is therefore blocked on that path. The authenticated Voice API must
preview the exact draft revision, return a random one-time approval token to the
Dashboard, and receive that token back in a separate send request.

Only a contact with exactly one address for the selected channel is accepted.
The approval token is stored as a hash, bound to the draft revision, consumed
before the delivery call, and cannot be replayed. Success requires a delivery
receipt from the owner-operated bridge. Failed delivery is retained as failure,
not reported as sent. Pending drafts and delivered invitations/messages can be
cancelled when the bridge verifies cancellation. Draft, preview, authorization,
delivery, failure, and cancellation transitions are append-only audited.

With no bridge configured, drafting remains available and health reports
`configured: false`; external delivery fails closed.

## Governed transactions and bookings

Travel, shopping, bookings, finance, and purchases use `transactions` proposals.
Voice can create or preview a canonical proposal containing category,
counterparty, amount, three-letter currency, summary, and structured details;
the immediate executor cannot commit it. Commitment requires either a
revision-bound one-time owner approval token returned by the authenticated API,
or an active standing budget matching category, currency, optional
counterparty, and sufficient remaining amount.

Budget value is reserved before the external call and restored if the bridge
fails. A proposal becomes committed only when the owner-operated bridge returns
a receipt. Committed work can be cancelled only when the bridge verifies the
compensating cancellation. Proposal, preview, authorization, receipt, failure,
budget, and cancellation state is durable and audited without placing payment
credentials in prompts or traces. With no bridge configured, proposals remain
useful but all real transactions fail closed.

## Durable commitments

The `commitments` provider stores reminders, recurring obligations, deadlines,
and wait-until conditions in the durable agent database. A worker polls active
time commitments, creates exactly one durable proactive intervention per
occurrence, and resumes safely after restart. Household events can satisfy
condition-based commitments without polling an external service.

Daily, weekly, and monthly recurrence advances from the scheduled occurrence,
not from process wake time. An overdue item past its deadline records a miss;
recurring items advance to the first future occurrence while preserving a miss
count. One-time items remain visibly missed. Authenticated devices can list,
snooze, complete, or cancel the same record, and the continuation device is
retained for audit and handoff. All state changes use optimistic revisions, so
two devices cannot silently overwrite each other.

## Asynchronous cited research

The `research` provider turns longer knowledge requests into durable background
jobs and immediately releases the speech path. Jobs survive service restarts,
use a bounded worker pool, can be cancelled while queued or running, and reuse
the configured `web` provider rather than introducing a second search stack.

Completed records retain the backend response, source URLs/domains, a concise
three-sentence spoken summary, and an explicit low/medium/high uncertainty
assessment. A deterministic proactive intervention announces completion once.
The authenticated Voice API and Dashboard administration page expose the full
evidence and citations, while voice receives only the concise summary.

## Briefings and event subscriptions

The `briefings` provider schedules one timezone-aware morning or evening
briefing per local date. Each generated briefing combines commitments with
optional iCloud calendar events, detects overlapping intervals, creates
near-term preparation prompts, and stores the complete agenda for Dashboard
review. Deterministic IDs prevent duplicate briefings and announcements after a
restart.

“Tell me when” requests become durable event subscriptions over Nova’s existing
cursor-backed household event feed. Matching uses an explicit event kind and
payload fields, creates one deterministic intervention per source event, and
supports one-shot or recurring subscriptions. Subscriptions remain visible and
cancellable after triggering.

## Durable conversation continuity

An acoustic conversation keeps bounded recent turn text only while its follow-up
window is open. In parallel, `ConversationContinuityManager` stores a separate
durable structural record: recognized participants, compact model-produced topic
and goal summaries, topic stack, open user questions, unresolved-reference
markers, and linked durable goal IDs. It never copies assistant replies or the
full rolling transcript into durable storage.

These records survive the microphone window and service restarts. Authenticated
APIs can list them and explicitly resolve an open question; every change uses
the durable store’s revision and audit semantics.

Relationship continuity is stored per recognized person with provenance back to
the source conversation records. Narrative continuity is assembled only from
structured topic summaries. Preference and speaking-style learning accepts
explicit phrases such as “I prefer…” or “keep it brief”; it does not infer a
preference from one-off behavior. Before an addressed turn, topic-word overlap
selects relevant callbacks and open questions with source IDs for the model’s
context. Unknown speakers receive none of this context.

Long-form discussion controls are scoped to the durable conversation rather
than global personality. Explicit phrases select brief/normal/deep depth,
deliberate clause pacing, reflective listening, candid/supportive disagreement,
humour, and storytelling. Deep mode allows up to five substantive sentences;
normal mode retains the existing sampled three-sentence option and brief mode is
one short sentence. The renderer receives these controls even when no household
state is relevant.

## Multi-party dialogue

Every live conversation message already carries the recognized speaker’s name
and pronouns. `detect_dialogue_routing` now separately classifies explicit
assistant, known-person, household, and unspecified addressees. A turn prefixed
with another participant’s name is not allocated to the assistant. Explicit
“ask person…” and “tell the household…” phrases compile to the local `dialogue`
provider rather than relying on model inference.

Relay messages preserve sender, person/household audience, ask/tell act, source
conversation, content, status, and per-person delivery acknowledgements. A
recognized person can retrieve only messages for their ID/name plus household
messages they have not acknowledged. Unknown speakers receive no pending relay
content.
