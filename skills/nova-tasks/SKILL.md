---
name: nova-tasks
description: Read or manage local Nova tasks when speech asks to list, add, update, complete, dismiss, or remove a task; use the semantic Nova task tool.
---

# Nova tasks

- Use `nova.task` for Nova tasks; do not reinterpret them as shell or calendar commands.
- Preserve the user's wording and resolve relative time in the household timezone.
- Ask when the time, target task, recurrence, or destructive edit is ambiguous.
- List before changing when an ID is unknown; never guess an ID.
- Report only verified tasks returned by the tool. When a turn changes several
  tasks, preserve their requested order and do not repeat successful edits after
  a partial failure.
