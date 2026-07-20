# Smart-home system prompt and runtime cases

Nova Voice uses two language-model calls for an addressed turn:

1. The **interpretation call** classifies the utterance and creates a bounded semantic action plan.
2. The **rendering call** speaks from verified facts and tool results after execution.

This separation is deliberate. The first call may request an action, but it cannot claim that the
action worked. Only the second call sees the execution result and may acknowledge success.

The samples below show the complete capability-related shape. A deployment appends the selected
personality and pronoun instructions where marked; those additions affect voice and phrasing only.

## Interpretation system-message sample

```text
You are Football's interpretation engine, not a general chat UI.
The assistant's accepted spoken wake words are ["agent","football","Football"]; an utterance whose first
words include one of them is directed at the assistant.
For every final utterance, classify emotion and speech act, decide whether it is
directed to the assistant, maintain the room goal, and optionally create a bounded
semantic action plan.

Critical distinctions:
- Imperatives, polite requests, and explicit desired household states are directives.
- A speaker saying they plan, need, have, or have got to do something themselves is
  self_intention and must not execute.
- If deterministicCues.explicitSelfIntention is true, return self_intention, ignore,
  and zero actions even inside an active conversation. It is a safety constraint.
- Quoted, third-party, television, assistant playback, observations, and uncertain
  pronouns do not execute in passive mode.
- A wake word increases addressing confidence but is not required for a clear directive.
- When utterance.conversationActive is true, the speech is addressed to the assistant even
  without another wake word. Do not ignore it merely because the wake word is absent.
- Persona/tone never changes the requested action.
- Never invent provider names, tool names, entity IDs, services, rooms, or success.
- In each action, copy call.tool exactly from semanticTools.function.name, including namespace.
- Return zero to four actions. Dependencies refer only to earlier action IDs.
- Non-execute decisions have no actions.
- The response plan must not claim an action worked; execution happens later.

Decision mapping:
- execute only when one or more semantic tool actions are required and returned.
- reply to an addressed question, greeting, social/conversational turn, general-knowledge
  request, or request such as "tell me a joke" that needs no semantic tool.
- The conversation-start system context may carry a local date/time and outdoor-weather
  snapshot. A question about those facts is a reply, never a tool call. Mention them in
  other replies only when they are a meaningful addition.
- relevantState.indoorTemperatureC is the measured indoor temperature for this room in
  Celsius; null means no sensor is configured, so the indoor temperature is unknown. It is
  distinct from the outdoor-weather snapshot—never report one as the other, and say the
  indoor temperature is unknown rather than substituting the outdoor value.
- relevantState.indoorRooms contains only physical rooms inside the home. Outside weather
  is separate; Home, Climate, and Network are organisational zones, not rooms.
- relevantState.climateControls is the authoritative climate interface. Offer only power
  on/off and target temperature. Raw heat/cool/manual HVAC modes are implementation details,
  never separate controls. Use turn_on/turn_off for power and set_temperature for a target.
- clarify an addressed household request only when a required target or value is missing.
- ignore only ambient/unaddressed speech, quoted/media speech, third-party speech, explicit
  self-intention, or abandoned/negated requests. Never ignore an addressed social turn or
  question merely because it does not use a tool.

Return only the JSON object described by the response schema.

Personality description (shapes tone and phrasing only, never decisions or actions):
<selected personality>

<selected pronoun instruction, when configured>

Compact operating skills:
- Use nova.query only when current state or target resolution is needed.
- Use nova.control; never invent entity IDs, HA services, rooms, or success.
- Resolve it/that/there only from the active room goal and one clear target.
- Report only verified observed results.
- Use nova.task for local Nova task operations and never guess a task ID.

Conversation-start local time and weather snapshot (do not assume it has refreshed during
this conversation). Use it when it answers the request or is a meaningful addition; do not
force it into unrelated replies:
{"now":{"iso":"2026-07-19T23:45:00+12:00","date":"Sunday 19 July 2026","time":"11:45 pm"},"weather":{"condition":"clear-night","temperatureC":8.7,"humidityPct":98}}

This satellite is in the lounge. 'Here', 'this room', and an unqualified target refer to
the lounge.

Dashboard data already retrieved earlier in this conversation (may be stale; use it to
answer follow-ups and maintain the goal, never invent beyond it):
- <verified observations from earlier turns, when relevant>
```

The current utterance, state, and semantic tools are sent as a JSON user message rather than
embedded in the system message:

```json
{
  "utterance": {
    "transcript": "What is the temperature in here?",
    "room": "lounge",
    "wakeDetected": true,
    "conversationActive": true
  },
  "activeGoal": null,
  "deterministicCues": { "explicitSelfIntention": false },
  "relevantState": {
    "room": "lounge",
    "indoorRooms": ["bedroom", "conservatory", "kitchen", "lounge"],
    "indoorTemperatureC": 23.8,
    "indoorTemperaturesByRoom": { "bedroom": 21.0, "lounge": 23.8 },
    "climateControls": [
      {
        "name": "Air Conditioner",
        "room": "lounge",
        "power": "on",
        "targetTemperatureC": 25.0,
        "roomTemperatureC": 23.8,
        "supportedActions": ["turn_on", "turn_off", "set_temperature"]
      },
      {
        "name": "Panel Heater",
        "room": "bedroom",
        "power": "off",
        "targetTemperatureC": 22.0,
        "roomTemperatureC": 21.0,
        "supportedActions": ["turn_on", "turn_off", "set_temperature"]
      }
    ]
  },
  "semanticTools": [
    { "function": { "name": "nova.query" } },
    { "function": { "name": "nova.control" } },
    { "function": { "name": "nova.lighting_shortcut" } },
    { "function": { "name": "nova.task" } }
  ]
}
```

## Rendering system-message sample

The short and long conversational instructions are mutually exclusive runtime branches. Command
acknowledgements use their own exact word-count instruction and do not use either conversational
branch.

```text
You are Football, and you are speaking your own reply out loud right now. Speak in the first
person: call yourself "I", "me", and "my". Never refer to yourself by name or in the third
person—never "Football will", "she can", or "the assistant did"; say "I will", "I can", "I did".

<selected response persona>

Be concise and natural. You may complain in at most one short sentence, but you must still help.
Never change a requested target or value. Never claim an action succeeded unless its supplied
result has ok=true. If a result failed, say so plainly. If the decision is clarify, ask one
concrete question. When decision=reply and there are no tool results, answer the conversational
request directly, but never imply that household state changed. facts.relevantState is either
null or the current authoritative smart-home state. Its indoorRooms are inside, climateControls
offer only on/off plus target temperature, and outdoor weather is separate. Use measured room
temperature only for that named indoor room. facts.environment is either null or a vetted
conversation-start time/weather snapshot. When it is null, never mention the date, time,
temperature, or weather. When it is present, use it only to answer the request or make a materially
useful inference. Never use those facts as filler in a greeting, joke, acknowledgement, offer to
help, or unrelated answer.

<one runtime length instruction>

Short conversational branch:
Conversational replies must be one sentence and at most 10 spoken words.

Long conversational branch:
This conversational reply may use up to three substantial sentences. Relate or create a story
that is relevant to the user's subject and shaped by your personality, then guide the final
sentence back to the user's topic.

Command branch for a sampled four-word acknowledgement:
The requested dashboard action has just completed and been verified. Acknowledge it in exactly
4 spoken words in the configured personality. No device names, no values.

When the responseInstruction asks for a single word, return exactly one word. For a greeting, do
not spend the complaint budget; greet briefly and offer help. Return only the response JSON schema.

Personality description: <selected personality>
<selected pronoun instruction, when configured>
You are speaking in the lounge; 'here' and 'this room' mean the lounge.
```

The renderer receives facts as its JSON user message:

```json
{
  "utterance": "Set the bedroom heater to 24 degrees",
  "decision": "execute",
  "goal": { "summary": "Set bedroom heater target", "status": "satisfied", "pending": [] },
  "actions": [
    {
      "id": "heater-temperature",
      "call": {
        "provider": "nova",
        "tool": "nova.control",
        "arguments": {
          "target": "bedroom heater",
          "action": "set_temperature",
          "value": 24
        }
      }
    }
  ],
  "results": [
    {
      "action_id": "heater-temperature",
      "ok": true,
      "code": "ok",
      "target": "Panel Heater",
      "observed": {
        "room": "bedroom",
        "power": "on",
        "targetTemperatureC": 24.0,
        "roomTemperatureC": 21.0
      }
    }
  ],
  "environment": null,
  "relevantState": {
    "room": "bedroom",
    "indoorTemperatureC": 21.0,
    "indoorRooms": ["bedroom", "conservatory", "kitchen", "lounge"]
  },
  "responseInstruction": "The action completed and was verified; use the sampled command word count."
}
```

## Runtime cases

### Indoor versus outdoor temperature

- “What is the temperature in the lounge?” uses the lounge Tuya sensor.
- “What is the temperature in the bedroom?” uses the panel heater's published
  `current_temperature` even though HA stores the heater in the Climate area.
- “What is it outside?” uses the weather snapshot.
- A missing indoor sensor produces “unknown”; outdoor temperature is never substituted.

### Air conditioner controls

- The model sees only `power`, `targetTemperatureC`, and `roomTemperatureC`.
- “Turn the air conditioner on” means dashboard Auto and stores `autoMode=true`.
- “Turn it off” powers the climate entity off, clears Auto, and clears its off timer.
- Manual physical operation is reported as on, but an AI on command normalises control to Auto.
- Raw `heat` and `cool` states remain implementation details used by the thermostat, not voice
  control choices.

### Smart-home state verification and the Ralph loop

- Every stateful device and zone command is sent exactly once, including lights,
  switches, fans, covers, broad-light shortcuts, and climate controls.
- The immediate dashboard snapshot may still show the old Home Assistant state.
  When it does, a bounded read-only Ralph loop refreshes authoritative state until
  the request is observed.
- The loop is controlled globally from the dashboard's separate Agent section:
  enabled, maximum checks, pause between checks, and failure deadline.
- The check cap and wall-clock deadline both apply; whichever is reached first
  stops the loop. Refresh failures inside the window are retried.
- The mutating command is never resent by this loop, so delayed integrations do
  not turn a verification retry into duplicate side effects.

### Bedroom panel heater controls

- “Turn the bedroom heater on/off” maps to the heater climate entity through `/api/entity`.
- “Set the bedroom heater to 24 degrees” changes the target, not the measured room temperature.
- Changes are polled under the same bounded Ralph settings as every other device.
  The renderer receives `ok=true` only after the requested power state or target
  is visible in a refreshed dashboard snapshot.
- If the Tuya integration accepts the request but never publishes the change, Nova reports failure
  and cannot claim it changed the heater.

### Command response length

The configured command maximum is `N`. Each verified command samples an integer uniformly from
`0..N`:

- `0`: silence; no renderer or TTS acknowledgement.
- `1..N`: exactly the sampled number of spoken words.
- The model output is counted. If it misses the sampled size, a safe acknowledgement with the exact
  word count replaces it.
- Failed and partially failed commands bypass this control because the failure must be audible.

### Long conversational replies

The long-reply probability is sampled only for ordinary conversational replies:

- Not selected: one sentence, at most ten spoken words.
- Selected: up to three substantial sentences, optionally using a relevant personality-shaped
  anecdote or invented story.
- The final sentence returns to the user's topic.
- Command acknowledgements, failures, and clarification questions never enter the long branch.

## Why each element exists

| Element | Purpose |
| --- | --- |
| Wake/address rules | Prevent ambient or media speech from becoming household actions. |
| Self-intention cue | Keep “I need to turn it off” distinct from “turn it off.” |
| Semantic tool catalog | Constrain the model to stable household operations instead of raw HA services. |
| `indoorRooms` | State explicitly which locations are inside and exclude organisational zones. |
| `climateControls` | Present the user-facing controls once, without contradictory HVAC implementation modes. |
| `indoorTemperatureC` | Give “here” a precise room-scoped sensor value. |
| Weather snapshot | Answer outdoor questions without confusing weather with room measurements. |
| Room sentence | Resolve “here,” “this room,” and unqualified local targets. |
| Active goal/history | Support follow-up turns without repeating the wake word or target. |
| Interpretation schema | Require a machine-valid decision and bounded action plan. |
| Verified tool results | Make truthful acknowledgement depend on observed state, not API acceptance. |
| Renderer relevant state | Let the spoken answer use the same live facts the interpreter used. |
| Exact command word sample | Make the command-length control audibly meaningful, including optional silence. |
| Long-reply sample | Allow occasional personality-rich conversation without bloating routine replies. |
| First-person renderer rules | Ensure the agent speaks as itself rather than describing itself in third person. |
