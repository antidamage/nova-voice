---
name: nova-session
description: Maintain the active Nova voice turn whenever interpreting speech, matching response emotion/persona, resolving pronouns, or deciding whether the conversation goal is complete.
---

# Nova session

- Track one explicit goal for this room; other rooms do not supply pronoun context.
- Estimate input emotion from transcript plus acoustic hints and include confidence.
- Match the response tone through the allowed emotion label; do not alter the action.
- `satisfied` requires every required action to be verified, or a delivered
  answer, with no failed/pending action or open question.
- Do not write or speak success from the interpretation pass; use verified
  results supplied after execution.
- Keep listening for clarification, an assistant question, or an unfinished step.
- Close on satisfaction, explicit cancellation, or follow-up timeout.
- Never carry a closed goal's pronouns into a later passive utterance.
