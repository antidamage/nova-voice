---
name: nova-home-control
description: Query or control Nova household devices when speech requests a home state, device action, timer, or current-state answer; use only Nova's semantic local tools.
---

# Nova home control

- Use `nova.query` only when current state or target resolution is needed.
- Use `nova.control`; never invent entity IDs, HA services, rooms, or success.
- Imperatives, polite requests, and explicit desired states are commands.
- A speaker's plan to do something themselves is not a command.
- Resolve `it/that/there` only from the active room goal and one clear target.
- Clarify ambiguous target, value, negation, or unusual duration.
- Put multiple requested changes into the bounded ordered action plan; do not
  drop later actions or merge targets with different values.
- Report only verified observed results. On partial failure, name the failed
  action, retain successful results, and keep the unfinished goal open.
- Treat unavailable tools as unavailable capability; never construct raw API/MCP calls.
- Persona may complain briefly but must still attempt the valid action.
