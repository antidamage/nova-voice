# Stable contracts

These interfaces isolate the LLM, model runtimes, capability providers, external
Nova dashboard protocol, and satellite clients. Implementations may change
without rewriting behavior policy or importing another repository.

## Capability provider boundary

```typescript
type CapabilityManifest = {
  id: string;
  version: string;
  contractVersion: string;
  executionClass: "iridium_local" | "household_lan_service";
  tools: SemanticToolSchema[];
  skillFiles: string[];
};

interface CapabilityProvider {
  manifest(): CapabilityManifest;
  query(call: CapabilityToolCall, context: ExecutionContext): Promise<ToolResult>;
  execute(call: CapabilityToolCall, context: ExecutionContext): Promise<ToolResult>;
  verify(call: CapabilityToolCall, result: ToolResult): Promise<ToolResult>;
  health(): Promise<ProviderHealth>;
}

type CapabilityToolCall = {
  provider: string;
  tool: string;
  arguments: Record<string, unknown>;
};

type PlannedAction = {
  id: string;
  order: number;
  dependsOn: string[];
  call: CapabilityToolCall;
};
```

The core knows these interfaces only. Provider loading is allowlisted and local;
manifests are validated before any schema enters an LLM request. Provider tools
carry deterministic risk/confirmation/idempotency metadata that a skill cannot
override. The Nova provider owns all dashboard-specific URLs, aliases, auth,
mapping, and result verification. Any provider that performs a local-AI task
must declare `iridium_local`; routing policy rejects attempts to run it on a
satellite or another household device.

The initial Nova manifest uses a Nova Voice-owned compatibility identifier such
as `nova-provider-v1`. Dashboard `/api/version` build metadata is recorded for
diagnostics, but is not treated as a semantic REST API version. Startup and CI
conformance tests decide whether the deployed dashboard is compatible.

## Utterance envelope

The envelope optionally carries `speaker`: a per-turn status (`unknown`, `provisional`,
`pending`, or `recognized`), template/person identifiers, display name, stated pronouns, and
cosine-match confidence. Identity belongs only to that acoustic turn and must not be inherited
from the household conversation window.

```typescript
type Utterance = {
  id: string;
  satelliteId: string;
  roomId: string;
  startedAt: string;
  endedAt: string;
  transcript: string;
  transcriptConfidence: number;
  wakeDetected: boolean;
  wakeScore?: number;
  conversationActive: boolean;
  acoustic: {
    durationMs: number;
    rmsDb: number;
    peakDb: number;
    snrDb?: number;
    pitchMedianRelative?: number;
    pitchRange?: number;
    syllablesPerSecond?: number;
    pauseRatio?: number;
  };
};
```

Raw audio never enters an LLM request.

## Interpretation result

`selfProfileUpdate` is optional and contains `name`, `pronouns`, and a verbatim `evidence`
span. It is valid only for an explicit first-person disclosure in the current addressed turn;
quoted, third-party, media, inferred, and prior-turn identity information must produce `null`.

Speaker-profile management is exposed over authenticated voice-server routes:

- `GET /v1/speaker-profiles`
- `PATCH` / `DELETE /v1/speaker-profiles/{person_id}`
- `PATCH` / `DELETE /v1/speaker-templates/{template_id}`

Responses expose profile/template metadata but never embedding vectors.

The LLM must produce schema-constrained data. Invalid or additional fields fail
closed.

```typescript
type Interpretation = {
  emotion: {
    label: "neutral" | "calm" | "grumpy" | "angry" |
           "excited" | "bored" | "sad" | "anxious";
    confidence: number;
    intensity: number;
    evidence: ("lexical" | "energy" | "pitch" | "rate" | "context")[];
  };
  speechAct: "directive" | "desired_state" | "self_intention" |
             "observation" | "question" | "third_party" |
             "quoted_or_media" | "social" | "unclear";
  addressedProbability: number;
  decision: "execute" | "reply" | "clarify" | "ignore";
  confidence: number;
  activeGoal: {
    summary: string;
    status: "new" | "in_progress" | "needs_clarification" |
            "satisfied" | "abandoned";
    pending: string[];
  };
  actions: PlannedAction[];
  responsePlan: {
    acknowledgementStyle: string;
    preActionSpeech?: string;
    requiresPostToolRendering: boolean;
  };
};
```

Policy may downgrade `execute` to `clarify`/`ignore`; the model cannot upgrade a
blocked action. `actions` is schema-bounded to at most four entries in v1.
Dependencies must refer to earlier entries. The trusted planner/policy may run
independent reversible actions in parallel, but defaults to ordered execution.
No response field may claim an action succeeded before its verifier returns.

## Initial LLM-facing Nova provider tools

Expose these small semantic tools when the Nova provider is enabled.
Other providers may register separate namespaced tools later; the runtime sends
only tools relevant to the active goal rather than every installed schema.

### `nova.query`

```json
{
  "scope": "home|room|target|tasks|health",
  "query": "optional natural target or room"
}
```

Returns normalized target names, current state, units, room, and ambiguity
candidates. It never returns secrets or the full raw HA snapshot.

### `nova.control`

```json
{
  "target": "lounge air con",
  "action": "turn_on|turn_off|set_level|set_temperature|set_color|set_timer|wake|sleep",
  "value": 19,
  "unit": "celsius",
  "durationMinutes": 60,
  "room": "lounge"
}
```

Only `target` and `action` are generally required. The adapter resolves the
target against its alias index, applies domain-specific schemas and configured
bounds, calls the Nova REST endpoint, then verifies the resulting state.

### `nova.lighting_shortcut`

```json
{
  "scope": "indoors|all|outside",
  "action": "on|off"
}
```

A deterministic phrase mapper uses this tool for broad commands such as
"turn all of the lights on" and "turn everything off". `indoors` targets the
dashboard's Home/Everything lighting layer; its `on` endpoint applies the
configured adaptive daylight/sunset preset. Outside lights remain excluded
unless the user explicitly says outside or asks for all lights including outside.

### `nova.task`

```json
{
  "operation": "list|add|update|complete|dismiss|remove",
  "id": "optional existing id",
  "name": "optional task text",
  "start": "optional ISO timestamp",
  "end": "optional ISO timestamp"
}
```

Relative dates are resolved by deterministic date code using the household
timezone. Destructive/ambiguous edits require clarification.

## Tool result

```typescript
type ToolResult = {
  ok: boolean;
  code: "ok" | "ambiguous" | "not_found" | "blocked" |
        "invalid" | "timeout" | "unverified" | "backend_error";
  target?: string;
  requested?: Record<string, unknown>;
  observed?: Record<string, unknown>;
  candidates?: string[];
  message: string;
};
```

The response renderer says an action succeeded only when `ok=true` and the
observed state is consistent. After a single mutation, the configurable Ralph
loop retries only `GET /api/state` verification reads. It is bounded by an
iteration cap and wall-clock deadline and never resends the mutation.

Verified dashboard-control results are passed to the persona-aware response
renderer after verification, with a short deterministic confirmation available
as a failure fallback. Partial failure and open-ended dialogue use the same
resident LLM; this does not introduce another model.

## External dashboard protocol mapping

| Semantic operation | Existing fast path |
| --- | --- |
| state/alias refresh | `GET /api/state`, `/api/events` |
| zone action | `POST /api/zone` |
| entity action | `POST /api/entity` |
| adaptive indoor lights on/off | `GET /api/lights/on`, `/api/lights/off` |
| all indoor and outside lights | `GET /api/all-lights/on`, `/api/all-lights/off` |
| outside lights only | `GET /api/outside-light/on`, `/api/outside-light/off` |
| air-con timer | `POST /api/aircon/timer` |
| panel heater timer | `POST /api/panel-heater/timer` |
| computer wake/sleep | `POST /api/desktop/wake`, `/api/desktop/sleep` |
| task operations | `/api/tasks` and `/api/tasks/:id/*` |
| health/discovery/config | `POST /api/mcp` |

This table documents protocol knowledge, not source-code reuse. Nova Voice owns
its typed client, schemas, fixtures, and compatibility tests. No dashboard
module, generated type, UI component, or internal route implementation is a
Nova Voice dependency.

Deployment benchmarks equivalent REST and MCP operations from Iridium. The
adapter selects the measured hot path per semantic operation and keeps that
choice out of the LLM schema. REST is not assumed faster merely because it is
direct; MCP remains the default surface for discovery, health, and administration.

MCP's mechanical `confirm: true` is set by the trusted adapter only after Nova
Voice policy has established that the spoken request authorizes the reversible
action. The LLM never sets it directly.

## Satellite handshake

```typescript
type SatelliteHello = {
  protocolVersion: 1;
  satelliteId: string;
  displayName: string;
  roomId: string;
  client: "linux-native" | "macos-native" | "browser";
  supervisor: "systemd" | "launchd" | "none";
  capturePolicy: "always" | "push-to-talk";
  dashboardForeground?: boolean;
  capabilities: {
    microphone: boolean;
    speaker: boolean;
    echoCancellation: boolean;
    noiseSuppression: boolean;
    automaticGainControl: boolean;
    playbackEvents?: boolean;
    localVad?: boolean;
  };
};
```

Native connection requires a locally provisioned per-device certificate and
mutually authenticated TLS. A browser uses same-origin WSS to the dashboard
bridge; the bridge holds the mTLS client identity for its upstream connection to
Iridium. After the
JSON hello, audio uses a versioned binary envelope containing sequence, monotonic
timestamp, stream direction, format, playback marker, and PCM payload. Native
satellites capture continuously but normally transmit only locally detected
activity. It calibrates its local noise floor for one second, then keeps 400 ms
pre-roll and an 800 ms silence tail, so it
does not replace Iridium's authoritative VAD or clip labelled speech. The hello
acknowledgement includes `localVadEnabled`; Iridium can later send
`{"type":"local_vad","enabled":false}` to bypass the gate live for diagnostics.
Response PCM is returned only to the elected source satellite's own connection.
`playback`, `playback_done`, and `playback_cancel` control messages start,
finish, or immediately close that source's output stream; room membership is
not a playback fan-out list.
Clients advertising `playbackEvents` return `playback_started` with the matching
`playbackId` when their audio renderer starts the first scheduled buffer, then
`playback_finished` after the last buffer is audibly complete. Missing capability
means Iridium retains the protocol-v1 first-PCM timing fallback.
Small control frames carry capture state, RMS/SNR, optional wake telemetry,
heartbeat, foreground context, and server backpressure. A push-to-talk browser
sends `begin_turn` to arm the next segment as wake-initiated. Browser page
visibility and dashboard UI state are client concerns, not Iridium protocol
authority.

After base VAD silence, Iridium may hold the turn for a bounded semantic
endpoint wait. During an addressed extended pause it may send one short
nonverbal playback stream through the same protocol. That listening earcon is
not a response, action acknowledgement, or completion event and is recorded as
an echo reference. Stable interim text can initiate only read-only context/tool
prefetch; final text must extend the stable prefix before prefetched state is
reused.

All satellites in one acoustic room must advertise the same stable `roomId`.
Iridium stores post-DSP playback references and recent response-text windows by
the configured arbitration scope: household by default, or `roomId` in
room-scoped installations. Every microphone in that scope consults the shared
reference before interpretation. This is distinct from source election, which
still selects only one microphone segment for each utterance.

Speaking animation is not satellite audio routing. Nova Voice turns confirmed
playback edges (or the legacy first-PCM fallback) into one best-effort
speaking-state event for the dashboard service, whose shared event stream fans
it out to every connected dashboard client.

## Conversation and monitor contract

Conversation context is in-memory and keyed by the configured arbitration scope
(`household` by default, or room when configured). A wake-word turn opens it;
each accepted turn refreshes a dashboard-tunable idle timeout whose default is
60 seconds. User and assistant text is appended in role order, and the complete
context is cleared on timeout, explicit abandonment, or a direct speech
interruption. System/persona and initial time/weather context are snapshotted
once per conversation.

The read-only voice monitor records both sides of the exchange as `User:` and
`<Agent name>:` transcript lines. Its transcript-font selector is local UI state
and does not alter agent behavior or retained transcript data.

Iridium also sends each accepted user turn and spoken assistant response to
Nova's `POST /api/voice/transcript` endpoint. The payload contains `role`
(`user` or `assistant`), `text`, `at`, `agentName`, `wakeWords`, `satelliteId`, and `roomId`.
Nova keeps a bounded process-local snapshot and fans new entries out as
`voice-transcript` events on the shared dashboard SSE stream; Iridium remains
the only durable transcript store and retains that data for at most 24 hours.

## Foreground turn trace and cancellation contract

Every handled foreground turn returns an immutable `TurnTrace` containing:

- `traceId`, `utteranceId`, and non-reversible `inputRevision`/
  `contextRevision` hashes
- exactly ordered `capture`, `endpoint`, `contextualize`, `interpret`,
  `authorize`, `execute_query`, `verify`, `render`, `speak`, and `commit` stage
  records with status and elapsed milliseconds
- the policy decision, tool journal, verification evidence revisions, response
  revisions, cancellation decisions, total timing, and terminal status

The trace contains revisions rather than raw transcript, prompt, or response
text and is included in the redacted completion monitor event. Tool manifests
declare `cancellation` as `before_side_effects` (the default), `anytime`, or
`never`. A response playback cancel never implies task cancellation. Queued
actions may be stopped before invocation; only an `anytime` provider may be
cancelled during its call. Once a mutation may have started, its call and
verification finish and the cancellation applies only to later queued actions.
Speech heard during response playback is deterministically classified as true
barge-in, backchannel, cross-talk, or false interruption. Only true barge-in
sets the playback cancellation event; the other classes preserve the active
stream.

## Persona contract

Persona configuration may select:

- name/backstory and concise speaking rules
- compliant mood/attitude, including negative dispositions
- one resident TTS voice and base voice instruction
- complaint budget (maximum one sentence)
- emotion mirroring strength

It may not select tools, permission levels, transcript retention, or execution
thresholds. See `config/persona.example.yaml`.
