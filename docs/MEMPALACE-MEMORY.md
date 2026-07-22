# Selective conversational memory

Nova Voice uses a dedicated MemPalace 3.x installation on Iridium for durable,
semantic memory. The Voice process remains the policy owner: MemPalace only
persists and retrieves records through a bearer-authenticated loopback service.
It never receives microphone audio and it is not exposed directly to the LAN.

`nova-voice-mempalace.service` listens only on `127.0.0.1:8094`. Deployment
creates a random `NOVA_VOICE_MEMPALACE_TOKEN` in the root-owned Voice environment
file if one does not already exist. The dashboard reaches memory controls through
its existing mTLS connection to the Voice API; it never receives that token.

The supported record types are `profile`, `preference`, `episodic`,
`commitment`, `relationship`, `procedural`, and `household_fact`. Every record
includes owner/audience, source turn and provenance, confidence, sensitivity,
creation/review/access times, expiry, supersession/deletion state, and pinning.

Admission is deliberately conservative. Explicit, low-risk preferences and
commitments from a recognized speaker can be saved. Routine device controls and
transient device state are rejected. Sensitive identity, health, finance,
security, address/phone, or third-party material is not auto-filed and requires
owner review. Unknown or provisional speakers neither retrieve nor create
personal memory.

The service is optional at turn time. A timeout, failed health check, or stopped
MemPalace process returns no memory context and ordinary voice control continues.
`/health` exposes that degraded state under `memory`.

Dashboard controls in **Voice agent authority** list saved memories and support
pinning, correction, expiry, forgetting, manual duplicate consolidation, and a
filesystem backup that immediately opens the copied MemPalace index to verify it
is restorable. The Voice API also exposes JSON export at
`GET /v1/agent/memories/export`.

Recognized speakers can say “what do you remember?”, “pin …”, “forget …”,
“correct memory … to …”, or “expire memory … in N days”. A mutation only runs
when the lookup is unambiguous; otherwise it remains available for review on
the dashboard. Sensitive records remain unavailable to retrieval until the
owner confirms them there.

The current consolidation pass merges exact duplicate active memories, preserves
the newest/pinned record, and marks the duplicate superseded. It deliberately
does not invent conflicts, summaries, or procedures; richer reviewable
consolidation remains a later enhancement.
