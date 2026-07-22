# Resumable household events

Nova Voice consumes the authenticated Dashboard feed at
`GET /api/agent/events`. `NovaDashboardClient` sends the configured
`NOVA_VOICE_NOVA_MCP_TOKEN`; Dashboard fails closed without the matching
service token.

`HouseholdEventConsumer` validates the complete version-1 response before
accepting a batch. Events cover Home Assistant state, occupancy, device health,
weather, energy, calendar, reminder, and agent-task categories. Each accepted
event becomes a versioned `EventRecord` in the durable agent database with its
Dashboard cursor, payload revision, source, occurrence time, and retention
deadline.

The durable singleton cursor checkpoint advances after each event. If Voice
crashes after inserting an event but before advancing the checkpoint, the same
Dashboard cursor is fetched again; the stable event ID suppresses the duplicate
before the checkpoint advances. A restart loads the checkpoint before polling.
Out-of-order or unexplained cursor gaps fail the batch. An explicit Dashboard
retention reset creates a durable content-free gap record before consumption
continues from the first retained cursor.

Polling runs as a supervised task inside the Voice service. `/health` reports
only structural state: running, cursor, last successful poll time, and error
class. The poller uses capped retry backoff and stops cleanly before the shared
Dashboard client closes.

Settings:

- `NOVA_VOICE_HOUSEHOLD_EVENT_POLL_SECONDS` (default 1 second)
- `NOVA_VOICE_HOUSEHOLD_EVENT_BATCH_SIZE` (default 200)
- `NOVA_VOICE_HOUSEHOLD_EVENT_RETENTION_DAYS` (default 30 days)
