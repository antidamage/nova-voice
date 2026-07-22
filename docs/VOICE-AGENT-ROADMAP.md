# Nova Fully Featured Voice-Agent Roadmap

  ## Summary

  Nova already has a solid real-time foundation: satellite arbitration, final-buffer STT with a streaming-capable adapter,
  wake/follow-up conversations, speaker
  recognition, structured intent/action planning, provider tools, policy gates, smart-home verification, web lookup, interruption, TTS
  routing, short-lived context, diagnostics, a restart-safe durable goal/plan
  engine, an authenticated resumable household event feed, enforced household
  authority/delegation administration, and 464 passing tests.

  The main gaps are architectural rather than prompt-level:

  - ConversationTracker and SessionManager remain ephemeral turn helpers, while
    the separate durable goal/plan engine now supplies versioned records,
    transactional recovery, leases, waits, approvals, events, retries, and
    compensation. Foreground turns do not yet create durable plans
    automatically.

  - Only Nova home control/tasks and web search are available.
  - The durable engine now consumes normalized Dashboard household events, but
    there is no durable personal memory, commitment tracking, or proactive
    decision engine. Household identity classes, scoped delegation, revocation,
    administration, and audit replay are now implemented.

  - Turn-taking now layers bounded audio-native semantic endpointing, adaptive
    interruption recovery, and stable read-only prefetch on the authoritative
    final transcript; live household tuning and acceptance evidence remain.

  - Deterministic recorded-audio replay and fake-clock household simulation are
    available; the full household corpus, endurance, false-activation shadowing,
    and long-running-task evaluation remain incomplete.

  - Runtime documentation now matches the deployed 60-second default, supported browser microphone paths, Iridium/Nova topology, and
    final-buffer STT/incremental-PCM TTS limits. Live latency, replay, physical-audio, residency, and endurance acceptance evidence is
    still incomplete.

  Target design: retain the cascade architecture and its deterministic safety core, while separating immediate voice interaction from
  durable goals, plans, events, permissions, memory, and background work.

  ## Runtime and Loop Architecture

  1. Audio micro-loop
      - Keep satellite activity gating, central VAD, election, echo suppression, speaker recognition, the streaming-capable STT adapter,
        and response
        locking.

      - Add an audio-native semantic end-of-turn detector after VAD silence, with configurable wait/continue thresholds and a fixed
        maximum pause.

      - Add adaptive interruption classification: distinguish true barge-in, backchannel, cross-talk, and false interruption; resume
        cancelled speech after false positives.

      - Begin safe context retrieval, tool selection, and LLM prefill from stable interim transcripts, but commit interpretation, tools,
        memory, and speech only after the final-turn gate.

      - Stream renderer output into sentence/clause-aware TTS and allow cancellation between semantic units.
      - Before TTS, derive a separate deterministic spoken-text form from the final response. Normalize numbers according to context:
        years such as `2026` become "twenty twenty six", ordinals such as `106th` become "one hundred and sixth", and addresses,
        room numbers, and telephone numbers are spoken digit by digit (`105` becomes "one zero five"). Ordinary counts remain
        cardinals, while decimals, units, currency, dates, versions, URLs, and identifiers use explicit context rules. Keep the original
        response text unchanged in conversation history, APIs, and dashboard transcripts; use the spoken form for TTS, echo matching,
        and playback timing.

      - Use a small deterministic soundboard/backchannel controller for acknowledgements while listening; it must never imply task
        completion.

  2. Foreground turn loop
      - Replace the monolithic service path with explicit stages: capture → endpoint → contextualize → interpret → authorize → execute/
        query → verify → render → speak → commit.

      - Give every turn an immutable trace ID, input revision, context manifest, policy decision, tool journal, verification evidence,
        response revision, and terminal status.

      - Add a bounded knowledge-recovery branch after response drafting and before speech. When an addressed knowledge request produces
        an epistemic failure such as "I don't know", "I'm not sure", or a knowledge-related "I can't do that", automatically perform
        exactly one `web.ask` lookup using the current question and relevant conversation context, then render a grounded answer.

      - Keep the existing web-access setting as the consent gate. Do not invoke this fallback for permission denials, unsupported
        actions, failed device commands, unavailable household sensors, social replies, or unaddressed speech. If web access is disabled
        or the lookup fails, give one honest failure response without recursively retrying. Record the trigger, query, result, and final
        response in the turn trace.

      - Keep deterministic fast paths for common home commands, cancellation, silence, and fixed acknowledgements.
      - Separate “speech cancelled” from “task cancelled”; barge-in stops playback immediately but only cancels an action when its
        provider declares safe cancellation.

  3. Conversation loop
      - Preserve a short room-scoped acoustic conversation window, but introduce a distinct Conversation record with participants, topic
        stack, summaries, unresolved references, open questions, and linked goals.

      - Support corrections, false starts, “actually…”, topic switching, returning to an earlier topic, multi-speaker attribution, and
        explicit private/shared conversation modes.

      - Compact history into structured summaries instead of dropping old messages solely by token count.
      - End only the acoustic window on idle; retain durable summaries and unfinished commitments independently.

  4. Goal and plan loop
      - Replace the single ActiveGoal with durable Goal, Plan, and PlanStep records.
      - Steps may be tool calls, user questions, waits, timers, event conditions, verification checks, approvals, or compensations.
      - Run plan → authorize → act → observe → verify/replan until satisfied, paused, blocked, cancelled, expired, or failed.
      - Persist state before and after every side effect. Use idempotency keys and execution leases so restart/retry cannot repeat
        actions.

      - Support parallel steps only when providers declare compatible resources and concurrency safety.
      - Preserve successful steps during partial failure; retry or compensate only failed work.

  5. Proactive event loop
      - Consume normalized Home Assistant, reminder, calendar, device-health, occupancy, weather, energy, and agent-task events through
        an authenticated dashboard event stream.

      - Convert events into candidate interventions, deduplicate them, assess urgency/confidence/relevance, check delegation and quiet-
        hour policy, then choose act, speak, notify, defer, aggregate, or ignore.

      - Under the selected voice-forward policy, speak routine useful interventions when an eligible occupied room exists; otherwise
        queue them for the next interaction or dashboard notification.

      - Never interrupt active human speech, media, sleep/quiet hours, emergencies already being handled, or a higher-priority Nova
        utterance.

      - Track whether suggestions were accepted, dismissed, redundant, or annoying and use that evidence to tune policies.

  6. Memory and learning loop
      - Require a dedicated MemPalace installation on Iridium, exposed as a private LAN service for Nova Voice. Nova Voice remains on
        its existing host and owns memory policy and lifecycle; Iridium provides the durable memory storage and retrieval backend.

      - Use MemPalace for salient, non-routine memories that emerge from longer voice conversations, especially durable facts about
        users, their preferences, circumstances, ongoing concerns, and needs. Do not store routine smart-home commands or transient
        device state such as a light being turned on.

      - Maintain separate profile, preference, episodic, commitment, relationship, procedural, and household-fact memories.
      - Every memory carries owner/audience, source turn, provenance, confidence, sensitivity, creation/review/access times, expiry,
        supersession links, and deletion state.

      - Auto-save low-risk preferences and commitments; sensitive identity, health, finance, security, or third-party information
        requires explicit confirmation.

      - Run asynchronous consolidation during idle periods: merge duplicates, detect conflicts, summarize episodes, refresh preference
        confidence, and propose—not silently apply—new procedures.

      - Compile each prompt from a fixed context hierarchy: identity/policy, current participants, active conversation, active goals,
        selected memory, live household state, and relevant tools.

      - Provide spoken and dashboard operations for “what do you remember?”, correction, pinning, expiry changes, export, and
        forgetting. This follows current memory-first harness work while keeping user review and provenance stronger than unconstrained
        self-editing. Letta Context Constitution (https://www.letta.com/blog/context-constitution/), Memory Models
        (https://www.letta.com/blog/towards-agents-that-learn/)

  7. Operations and evaluation loop
      - Continuously record redacted structural traces, latency, queue depth, tool outcomes, policy decisions, memory retrieval quality,
        proactive outcomes, and interruption behavior.

      - Replay failures against pinned models/prompts/contracts before deployment.
      - Keep an evaluation registry tying every model, prompt, skill, policy, and provider version to its scenario results.
      - Run nightly offline regression and memory consolidation without permitting autonomous code, prompt, permission, or policy
        changes.

  ## Execution Task List

  This is the authoritative work queue for the roadmap. Work one task at a time unless a dependency explicitly requires coordinated
  Voice and Dashboard changes. A task is complete only when its implementation, focused tests, full regression suite, documentation,
  and required deployment/health verification are complete. Documentation-only tasks do not require deployment. Do not mark a parent
  gate complete because one of its examples works.

  Status: `[x]` deployed/accepted, `[~]` implemented and awaiting deployment/acceptance, `[ ]` pending. Current progress: 34 of 58 tasks complete. Next blocked gate: **T0-15/T0-16**, then **T1-16**.

  ### Milestone decision map

  Milestones represent usable feature outcomes, not dates or arbitrary batches. Every task belongs to exactly one milestone. A milestone
  becomes **Complete** only when every listed task is complete; it is **Ready** when all milestone dependencies are complete; otherwise
  it is **Blocked**. Update milestone state whenever task checkboxes change. Work may proceed within a ready milestone in the task order
  below, but a milestone gate cannot be skipped by completing only its last task.

  Milestone progress: 12 of 26 complete. In progress: **M0-06 Dependable real-time core accepted**.

  | State | Milestone | Completed feature outcome | Required tasks | Milestone dependencies |
  | --- | --- | --- | --- | --- |
  | **Complete** | **M0-01 — Documentation truth** | Operators and agents have one accurate account of deployed voice behavior. | `T0-01` | None |
  | **Complete** | **M0-02 — Traceable foreground turns** | Every foreground turn runs through explicit stages with a complete immutable trace and safe cancellation semantics. | `T0-02`–`T0-04` | None |
  | **Complete** | **M0-03 — Natural turn-taking** | Nova detects semantic turn completion, handles interruptions/backchannels, prefetches safely, and streams cancellable speech. | `T0-05`–`T0-09` | `M0-02` |
  | **Complete** | **M0-04 — Knowledge and speech reliability** | Knowledge gaps recover through one safe web lookup and numbers are spoken contextually. | `T0-10`–`T0-11` | None |
  | **Complete** | **M0-05 — Reproducible household simulation** | Recorded audio and fake-clock household scenarios reproduce failures deterministically. | `T0-12`–`T0-13` | `M0-02` |
  | In progress | **M0-06 — Dependable real-time core accepted** | Live latency, endurance, corpus, residency, and streaming gates prove the Tier 0 core. | `T0-14`–`T0-16` | `M0-01`, `M0-03`, `M0-04`, `M0-05`, `MOPS-01` |
  | **Complete** | **M1-01 — Durable goal and plan engine** | Goals and plans survive restarts, execute exactly once, support waits/approvals, and recover partial failures. | `T1-01`–`T1-05` | `M0-02` |
  | **Complete** | **M1-02 — Resumable household event backbone** | Voice consumes authenticated, normalized, cursor-based household events without duplication. | `T1-06` | `M1-01` |
  | **Complete** | **M1-03 — Household authority and administration** | Identity classes, delegation grants, revocation, APIs, Dashboard controls, and audit replay are enforced end to end. | `T1-07`–`T1-09` | `M1-01` |
  | **Complete** | **M1-04 — Selective durable conversational memory** | MemPalace stores, retrieves, consolidates, reviews, corrects, and forgets salient memories under provenance and sensitivity policy. | `T1-10`–`T1-13` | `M1-01` |
  | **Complete** | **M1-05 — Proactive home autonomy** | Extended home providers, simulated automation authoring, proactive intervention, quiet-hour policy, and feedback operate under grants. | `T1-14`–`T1-15` | `M1-01`, `M1-02`, `M1-03` |
  | Blocked | **M1-06 — Home-autonomy harness accepted** | Tier 1 passes restart, exactly-once, permission, simulation, audit, memory recovery, and high-impact safety gates. | `T1-16` | `M0-06`, `M1-01`–`M1-05` |
  | Blocked | **M2-01 — Personal information providers** | Calendar, reminders, notes, lists, contacts, weather, media, recipes, documents, and household knowledge are safely available. | `T2-01`–`T2-03` | `M1-06` |
  | Blocked | **M2-02 — Safe communications and transactions** | Messages, invitations, travel, shopping, bookings, finance, and purchases use preview, validation, grants, budgets, verification, and undo. | `T2-04`–`T2-05` | `M1-03`, `M2-01` |
  | Blocked | **M2-03 — Commitments, research, and briefings** | Multi-day commitments, cited research, briefings, conflicts, preparation prompts, and event subscriptions survive restarts. | `T2-06`–`T2-08` | `M1-01`, `M1-02`, `M2-01` |
  | Blocked | **M2-04 — Personal digital assistant accepted** | Tier 2 passes multi-day, timezone, recipient, amount, audit, cancellation, and undo gates. | `T2-09` | `M2-01`–`M2-03` |
  | Blocked | **M3-01 — Durable conversational continuity** | Topics, relationships, narrative summaries, open threads, preferences, and long-form discussion persist appropriately. | `T3-01`–`T3-03` | `M0-03`, `M1-01`, `M1-04` |
  | Blocked | **M3-02 — Private multi-party dialogue** | Speaker attribution, addressee detection, turn allocation, and private/shared memory boundaries work for household conversations. | `T3-04`–`T3-05` | `M1-03`, `M1-04`, `M3-01` |
  | Blocked | **M3-03 — Adaptive multilingual speech** | Code-switching, pronunciation, night/whisper delivery, accessibility pacing, and per-user speech preferences are supported. | `T3-06` | `M0-03`, `M3-01` |
  | Blocked | **M3-04 — Conversational continuity accepted** | Tier 3 passes longitudinal memory, correction, privacy, multi-party, naturalness, and authority-separation gates. | `T3-07` | `M3-01`–`M3-03` |
  | Blocked | **M4-01 — Governed multimodal inputs** | Replaceable visual/document providers accept explicit user shares and permitted camera snapshots with provenance. | `T4-01`–`T4-02` | `M2-04` |
  | Blocked | **M4-02 — Household digital twin** | Nova simulates household behavior, explains causes, rehearses automation, and evaluates energy scenarios. | `T4-03` | `M1-05` |
  | Blocked | **M4-03 — Visual assistance and continuity** | Visual help, object/location context, maintenance walkthroughs, and cross-device continuity obey memory policy. | `T4-04` | `M1-04`, `M4-01`, `M4-02` |
  | Blocked | **M4-04 — Frontier capabilities accepted** | Offline optimizers pass their safety/evaluation gate while the production speech path remains cascade-based. | `T4-05` | `M4-01`–`M4-03`, `MOPS-01` |
  | **Complete** | **MOPS-01 — Reproducible observability and evaluation** | Redacted telemetry, pinned replay, the evaluation registry, and deterministic/model graders support release decisions. | `OPS-01`–`OPS-03` | `M0-02`, `M0-05` |
  | Blocked | **MOPS-02 — Continuous validation and staged rollout** | Nightly regression/consolidation and fixture-to-autonomy promotion run with revocation and rollback. | `OPS-04`–`OPS-05` | `M1-04`, `MOPS-01` |

  ### Tier 0 task queue — dependable real-time core

  - [x] **T0-01 — Reconcile runtime documentation.** Align conversation duration, the currently supported in-scope browser microphone/
    satellite path, Iridium/Nova topology, streaming limitations, test counts, and acceptance claims with deployed behavior.
  - [x] **T0-02 — Define structured TurnTrace.** Add the immutable trace schema, trace IDs, input/context revisions, policy decision,
    tool journal, verification evidence, response revisions, timing fields, and terminal status without changing turn behavior.
  - [x] **T0-03 — Extract the foreground turn state machine.** Split capture → endpoint → contextualize → interpret → authorize →
    execute/query → verify → render → speak → commit into explicit tested stages using `TurnTrace` (`T0-02`).
  - [x] **T0-04 — Separate speech and task cancellation.** Make barge-in stop playback immediately while task cancellation follows
    provider-declared safety; cover cancellation before, during, and after side effects (`T0-03`).
  - [x] **T0-05 — Add semantic endpointing.** Run an audio-native end-of-turn decision after VAD silence with wait/continue thresholds,
    a maximum pause, latency metrics, and recorded-audio tests (`T0-03`).
  - [x] **T0-06 — Add adaptive interruption recovery.** Classify true barge-in, backchannel, cross-talk, and false interruption, and
    resume cancelled speech after false positives (`T0-04`, `T0-05`).
  - [x] **T0-07 — Add stable-interim prefetch.** Prefetch safe context, likely tools, and LLM state from stable interim transcripts but
    commit no tools, memory, or speech before the final-turn gate (`T0-03`, `T0-05`).
  - [x] **T0-08 — Finish sentence-streaming TTS.** Stream sentence/clause units, cancel between units, and prove actual first-audio and
    cancellation behavior against the selected production backend (`T0-04`).
  - [x] **T0-09 — Add deterministic listening acknowledgements.** Introduce a bounded soundboard/backchannel controller whose output
    never implies completion and never interferes with endpointing or interruption classification (`T0-05`, `T0-06`).
  - [x] **T0-10 — Add bounded knowledge-failure web recovery.** One addressed, policy-checked lookup and grounded rerender are deployed,
    with operational/local/permission/social/unaddressed exclusions and non-recursive failure handling.
  - [x] **T0-11 — Add context-aware spoken-number normalization.** Deterministic TTS-boundary handling for years, ordinals, counts,
    addresses, rooms, phones, dates, times, currency, versions, and IPs is deployed while canonical response text remains unchanged.
  - [x] **T0-12 — Build recorded-audio replay.** Create a deterministic runner for saved far-field, echo, interruption, disfluency, and
    false-activation samples with pinned expected traces and latency measurements (`T0-02`).
  - [x] **T0-13 — Build the fake-clock household simulator.** Add deterministic providers, delayed state convergence, failures,
    occupancy, concurrent speakers, and repeatable time/event control (`T0-03`).
  - [x] **T0-14 — Build live latency and endurance runners.** Measure endpoint, first audio, queue depth, interruption recovery,
    resource residency, and duplicate-mutation invariants over sustained runs (`T0-08`, `OPS-01`).
  - [ ] **T0-15 — Complete the household acceptance corpus.** Run far-field, echo/barge-in, false-activation shadow, and spoken-number
    cases with recorded evidence (`T0-06`, `T0-11`, `T0-12`).
  - [ ] **T0-16 — Pass the Tier 0 gate.** Complete 24-hour residency and production streaming-TTS validation with no regressions,
    duplicate mutations, or unresolved acceptance failures (`T0-01` through `T0-15`).

  ### Tier 1 task queue — durable home-autonomy harness

  - [x] **T1-01 — Define durable agent records.** Specify versioned Conversation, Event, Goal, Plan, PlanStep, Execution,
    DelegationGrant, ProactiveIntervention, Memory reference, and audit schemas (`T0-02`, `T0-03`).
  - [x] **T1-02 — Implement the durable stores and migrations.** Persist every lifecycle transition transactionally with restart tests,
    retention rules, backup/restore, and schema migration coverage (`T1-01`).
  - [x] **T1-03 — Implement execution leases and idempotency.** Persist before/after side effects and prove crash/retry cannot duplicate
    mutations (`T1-02`).
  - [x] **T1-04 — Implement the durable plan runner.** Support tool, question, approval, wait, timer, event, verification, retry, and
    compensation steps through satisfied/paused/blocked/cancelled/expired/failed states (`T1-03`).
  - [x] **T1-05 — Add safe plan concurrency and partial-failure recovery.** Use provider resource declarations, preserve successful
    work, and retry or compensate only failed steps (`T1-04`).
  - [x] **T1-06 — Add the authenticated resumable dashboard event stream.** Normalize HA, occupancy, device-health, weather, energy,
    reminder, calendar, and agent-task events with cursors and deduplication (`T1-01`; Voice + Dashboard).
  - [x] **T1-07 — Implement household identity policy classes.** Enforce owner, recognized-household, and unknown/guest capabilities in
    deterministic policy tests (`T1-01`).
  - [x] **T1-08 — Implement standing delegation grants.** Support scoped capabilities, targets, budgets, recipients, locations,
    schedules, expiry, notification, and immediate revocation (`T1-07`).
  - [x] **T1-09 — Add goal/execution/grant/audit administration.** Provide authenticated Voice APIs and Dashboard screens for inspect,
    cancel, revoke, and trace replay (`T1-02`, `T1-08`; Voice + Dashboard).
  - [x] **T1-10 — Install and operate MemPalace on Iridium.** Provision a private authenticated service with health checks, backups,
    restore verification, and graceful Nova Voice degradation when memory is unavailable.
  - [x] **T1-11 — Define and integrate the memory contract.** Implement typed profile, preference, episodic, commitment, relationship,
    procedural, and household-fact memories with provenance, audience, sensitivity, confidence, expiry, supersession, and deletion
    (`T1-01`, `T1-10`).
  - [x] **T1-12 — Implement selective memory formation and retrieval.** Save salient non-routine conversation facts and needs, reject
    routine commands/transient device state, confirm sensitive memories, and compile prompts using the fixed context hierarchy
    (`T1-11`).
  - [x] **T1-13 — Implement memory consolidation and user controls.** Add conflict/duplicate consolidation plus spoken and Dashboard
    review, correction, pin, expiry, export, and forget operations (`T1-12`; Voice + Dashboard).
  - [x] **T1-14 — Extend home providers.** Add scenes, automations, timers, schedules, energy, occupancy, device health, maintenance,
    media, and safe diagnostics with contracts and verification (`T1-04`).
  - [x] **T1-15 — Implement automation authoring and proactive intervention.** Deliver draft → simulate → explain → approve → activate
    → monitor → rollback plus deduplicated voice/notification decisions, quiet-hour policy, and feedback tracking (`T1-06`, `T1-08`,
    `T1-14`).
  - [~] **T1-16 — Pass the Tier 1 gate.** The fail-closed structural evidence harness is implemented; final owner acceptance remains
    deferred to the consolidated acceptance pass. Prove restart-safe exactly-once behavior, permission correctness, pre-activation
    simulation, auditable proactive reasons, MemPalace recovery, and zero unapproved high-impact mutations (`T1-01` through `T1-15`).

  ### Tier 2 task queue — personal digital assistant

  - [x] **T2-01 — Add iCloud calendar and reminders.** Implement read/write contracts, recurrence, timezone handling, verification,
    cancellation, and audit under explicit grants (`T1-16`).
  - [ ] **T2-02 — Add notes, lists, and contacts.** Provide identity-safe lookup and mutation with ambiguity handling and undo
    (`T2-01`).
  - [ ] **T2-03 — Add weather, media, recipes, documents, and household knowledge providers.** Keep replaceable contracts, citations,
    privacy boundaries, and read/write classification (`T2-02`).
  - [ ] **T2-04 — Add email, messages, and invitations.** Require draft/preview, recipient validation, explicit send authority, delivery
    verification, cancellation, and audit (`T1-08`, `T2-02`).
  - [ ] **T2-05 — Add travel, shopping, bookings, finance, and purchase providers.** Enforce recipient/amount validation, standing
    budgets, confirmation boundaries, and compensating cancellation (`T2-04`).
  - [ ] **T2-06 — Implement durable commitments.** Support reminders, recurrence, wait-until conditions, deadlines, missed-commitment
    recovery, and cross-device continuation (`T1-04`, `T2-01`).
  - [ ] **T2-07 — Implement asynchronous cited research.** Gather sources, preserve citations and uncertainty, and return concise spoken
    results plus Dashboard detail (`T1-04`, `T2-03`).
  - [ ] **T2-08 — Implement briefings and subscriptions.** Add morning/evening briefings, schedule conflicts, preparation prompts, and
    “tell me when…” event subscriptions (`T1-06`, `T2-06`).
  - [ ] **T2-09 — Pass the Tier 2 gate.** Complete multi-day restart scenarios, timezone tests, recipient/amount verification,
    external-effect audits, and visible cancellation/undo (`T2-01` through `T2-08`).

  ### Tier 3 task queue — conversational depth and continuity

  - [ ] **T3-01 — Add durable conversation/topic records.** Persist participants, topic stack, summaries, unresolved references, open
    questions, and linked goals independently of the acoustic window (`T1-02`, `T1-11`).
  - [ ] **T3-02 — Add relationship continuity.** Implement narrative summaries, relevant callbacks, preference learning, open-thread
    resurfacing, and conversation-specific speaking style without fabricated recollection (`T3-01`).
  - [ ] **T3-03 — Add long-form discussion mode.** Support user-controlled depth, deliberate pauses, clarification, reflective
    listening, disagreement, humour, and storytelling (`T0-05`, `T3-01`).
  - [ ] **T3-04 — Add multi-party dialogue.** Implement speaker-attributed history, addressee detection, turn allocation, and “ask
    Addie”/“tell the household” semantics (`T3-01`).
  - [ ] **T3-05 — Enforce private/shared memory boundaries.** Apply participant/audience policy to every retrieval, write, callback,
    correction, export, and forget operation (`T1-07`, `T1-11`, `T3-04`).
  - [ ] **T3-06 — Add multilingual and user-specific speech support.** Implement code-switching, pronunciation dictionaries,
    whisper/night mode, accessibility pacing, and per-user speech preferences (`T0-08`, `T3-01`).
  - [ ] **T3-07 — Pass the Tier 3 gate.** Complete longitudinal precision, contradiction/correction, privacy-boundary, multi-party,
    naturalness, and action-authority-separation evaluations (`T3-01` through `T3-06`).

  ### Tier 4 task queue — frontier capabilities

  - [ ] **T4-01 — Define replaceable multimodal provider contracts.** Cover Dashboard screens, user-shared images, camera snapshots,
    documents, and device diagrams with provenance and permission metadata (`T2-09`).
  - [ ] **T4-02 — Implement visual/document inputs.** Add explicit user sharing plus doorbell/camera snapshot access under household
    privacy and retention policy (`T4-01`).
  - [ ] **T4-03 — Build the household digital twin.** Support simulation, causal explanations, automation rehearsal, energy
    optimization, and “what would happen if…” queries (`T1-14`, `T1-15`).
  - [ ] **T4-04 — Add visual assistance and object/location continuity.** Provide maintenance walkthroughs, proactive visual help, and
    cross-device context without silently creating sensitive memories (`T1-12`, `T4-02`, `T4-03`).
  - [ ] **T4-05 — Add offline optimizer workers and pass the Tier 4 gate.** Run bounded research/automation/memory/plan evaluators off
    the speech path; compare duplex models only as evaluation targets and retain the cascade runtime (`OPS-03`, `T4-01` through
    `T4-04`).

  ### Cross-cutting operations task queue

  - [x] **OPS-01 — Add redacted structural telemetry.** Record latency, queues, tool/policy outcomes, memory retrieval quality,
    proactive outcomes, and interruption behavior without retaining excluded audio/transcripts (`T0-02`).
  - [x] **OPS-02 — Add pinned failure replay.** Replay failures against exact model, prompt, contract, skill, policy, and provider
    versions before deployment (`T0-12`, `OPS-01`).
  - [x] **OPS-03 — Build the evaluation registry and graders.** Track scenario results and implement deterministic outcome, policy,
    trace, and selective model graders with task/policy/latency/memory/proactivity metrics (`T0-13`, `OPS-01`).
  - [ ] **OPS-04 — Add nightly offline regression and consolidation.** Run pinned evaluation and memory maintenance without autonomous
    code, prompt, permission, or policy changes (`T1-13`, `OPS-02`, `OPS-03`).
  - [ ] **OPS-05 — Implement staged rollout controls.** Automate fixture → replay → shadow → owner canary → household → standing
    autonomy promotion with instant revocation and rollback (`OPS-02`, `OPS-03`).

  ## Implementation Tiers

  ### Tier 0 — Make the existing voice core dependable

  - Reconcile documentation and runtime defaults, especially conversation duration, browser satellites, current deployment topology, and
    streaming limitations.

  - Extract the foreground turn state machine and structured TurnTrace without changing behavior.
  - Implement semantic endpointing, adaptive interruption/false-interruption recovery, stable-interim prefetch, sentence-streaming TTS,
    and explicit cancellation semantics.

  - Make web search the single bounded fallback for addressed knowledge-answer failures. Give the normal planner the first opportunity
    to select `web.ask`, then catch epistemic failure responses before speech and run one traced lookup through the normal read-only
    authorization and provider path. Never turn operational, permission, or household-state failures into web searches.

  - Add a deterministic, context-aware spoken-number normalizer on the finalized response-to-TTS boundary. Do not put a model or
    sub-agent call on this latency-critical path, and do not replace the canonical response text stored or shown to users.

  - Build recorded-audio replay, fake-clock household simulation, deterministic provider fixtures, and live latency/endurance runners.
  - Finish existing acceptance gates: far-field corpus, echo/barge-in trials, shadow false-action period, 24-hour residency, and actual
    streaming-TTS validation.

  - Gate: no regression in existing tests; no duplicate mutations; measured endpoint, interruption, and first-audio improvements on the
    household corpus; knowledge failures either produce one grounded web answer or an honest terminal failure; and the spoken-number
    corpus passes for years, ordinals, counts, addresses, room numbers, and telephone numbers.

  Semantic endpointing and preemptive generation should borrow from current cascade techniques rather than adopting a duplex model.
  Pipecat Smart Turn (https://github.com/pipecat-ai/smart-turn), LiveKit turn tuning
  (https://docs.livekit.io/agents/logic/turns/tuning/), LTS-VoiceAgent (https://arxiv.org/abs/2601.19952)

  ### Tier 1 — Home-autonomy harness

  - Bring Iridium online and provision a private, authenticated, backed-up MemPalace service before enabling durable voice-agent
    memory. Define health checks and degraded behavior so Nova Voice continues without durable recall when Iridium is unavailable.

  - Integrate Nova Voice with MemPalace for memory candidate extraction, policy filtering, storage, retrieval, correction, and
    deletion. The write gate must reject routine command history and transient household/device state while retaining salient details
    from substantive conversations.

  - Add durable event, goal, plan, execution, delegation, audit, and proactive-intervention stores.
  - Add owner/household/unknown policy classes:
      - Owner: administer grants, memories, automations, security, spending, and external providers.
      - Recognized household: ordinary home control, shared tasks/media, and explicitly shared grants.
      - Unknown/guest: conversation, public information, and allowlisted room-local reversible controls only.

  - Add standing DelegationGrant scopes with capability, targets, values/budgets, recipients, locations, schedule, expiry, notification
    rule, and revocation.

  - Extend smart-home providers for scenes, automations, timers, schedules, energy, occupancy, device health, maintenance, media, and
    safe diagnostics.

  - Support natural-language automation authoring through draft → simulate against historical/current state → explain → approve →
    activate → monitor → rollback.

  - Add proactive home behaviors: forgotten devices, unusual energy use, climate optimization, doors/windows versus heating, device
    offline alerts, expiring timers, and automation failure recovery.

  - Gate: crash-safe exactly-once behavior, permission correctness, simulation before automation activation, auditable reasons for every
    proactive action, and zero unapproved high-impact mutations.

  ### Tier 2 — Personal digital assistant

  - Expand providers in this order:
      1. Existing Nova tasks plus iCloud calendar/reminders.
      2. Notes, lists, contacts, weather, media, recipes, documents, and household knowledge.
      3. Email/messages and invitations with draft, preview, recipient validation, send, and delivery verification.
      4. Travel, shopping, bookings, finances, and purchases under explicit standing grants and budgets.

  - Add durable reminders, recurring commitments, “wait until” conditions, follow-up deadlines, briefings, and cross-device
    continuation.

  - Support research tasks that gather sources asynchronously, preserve citations, identify uncertainty, and return a concise spoken
    summary plus dashboard detail.

  - Add morning/evening briefings, schedule-conflict detection, preparation prompts, missed-commitment recovery, and “tell me when…”
    event subscriptions.

  - Gate: multi-day restart-safe scenarios, calendar/time-zone correctness, recipient and amount verification, external-effect audit
    trails, and user-visible cancellation/undo.

  ### Tier 3 — Conversational depth and relationship continuity

  - Add topic/relationship memory, narrative summaries, callbacks to relevant prior discussions, preference learning, open-thread
    resurfacing, and conversation-specific speaking style.

  - Support long-form discussion mode with less restrictive response length, deliberate pauses, clarification, reflective listening,
    disagreement, humour, storytelling, and user-controlled depth.

  - Add multi-party dialogue: speaker-attributed history, addressee detection, shared versus private memory boundaries, turn allocation,
    and “ask Addie”/“tell the household” semantics.

  - Add multilingual/code-switching support, pronunciation dictionaries, whisper/night mode, accessibility pacing, and user-specific
    speech preferences.

  - Keep conversation separate from action authority: warmth, confidence, or remembered preference must never weaken provider policy.
  - Gate: longitudinal memory precision, contradiction/correction tests, privacy-boundary tests, human scoring for relevance and
    naturalness, and no fabricated recollections.

  ### Tier 4 — Frontier home-agent capabilities

  - Add multimodal inputs through replaceable providers: dashboard screens, user-shared images, doorbell/camera snapshots, documents,
    and device diagrams.

  - Build a household digital twin for simulation, causal explanations, automation rehearsal, energy optimization, and “what would
    happen if…” questions.

  - Add proactive visual assistance, object/location memory, maintenance walkthroughs, and cross-device context continuity—capabilities
    aligned with current universal-assistant research. Google DeepMind Project Astra (https://deepmind.google/models/project-astra/)

  - Add offline evaluator/optimizer workers for research, automation design, memory maintenance, and complex plans; never put iterative
    evaluator loops on the latency-critical speech path.

  - Evaluate but do not adopt full-duplex speech models under the selected cascade-only direction. PersonaPlex and similar systems
    remain comparison targets for turn-taking naturalness, not runtime dependencies. NVIDIA PersonaPlex
    (https://research.nvidia.com/labs/adlr/personaplex/)

  ## Public Interfaces and Ownership

  - Nova Voice owns Conversation, Goal, Plan, Execution, memory selection/policy/lifecycle, DelegationGrant, ProactiveIntervention, and
    TurnTrace. The MemPalace service on Iridium owns durable memory persistence and retrieval behind a private authenticated API.
  - The dashboard remains the source of truth for Home Assistant state, dashboard tasks/configuration, normalized household events, and
    administration UI.

  - Add an authenticated resumable dashboard event API carrying eventId, source, type, entity/room/person, observed state, timestamp,
    sensitivity, and deduplication key.

  - Add Voice APIs for listing/inspecting/cancelling goals and executions; reviewing memories; managing delegation grants; viewing
    proactive decisions; and replaying redacted traces.
      - read/write/external effect classification;
      - required permission scope and risk;
    and explicit exclusion of short-lived transcripts/audio.

  ## Test and Acceptance Plan

  - Preserve the 442-test baseline and add model-independent state-machine/property tests for all lifecycle transitions and invariants.
  - Build a virtual household with fake time, occupancy, HA entities, calendars, communications, failures, delayed state convergence,
    and concurrent speakers.

  - Add scenario suites for:
      - smart-home automation, proactive intervention, partial failure, compensation, and restart;
      - reminders, calendar conflicts, communications, research, and multi-day commitments;
      - corrections, interruptions, disfluency, topic shifts, multi-party conversation, and memory conflict;
      - permission escalation, prompt injection through tools/web/documents, unknown speakers, revoked grants, and quiet hours;
      - addressed knowledge failures that trigger exactly one web lookup, plus negative cases for permission, device, sensor, social, and
        unaddressed failures;
      - spoken-number normalization for `2026`, `106th`, ordinary cardinals, addresses such as `105 Main Street`, room numbers, telephone
        numbers, decimals, units, currency, dates, versions, URLs, and identifiers.

  - Score task success, policy compliance, side-effect correctness, pass-at-k, latency, interruption recovery, false activations, memory
    precision/recall, proactive precision/dismissal rate, and human conversation preference.

  - Adapt τ-bench-style multi-turn tool/user scenarios and Full-Duplex-Bench-v3 audio/tool cases to Nova’s household simulator; recent
    voice benchmarks identify disfluency, self-correction, and multi-step tool reasoning as persistent failure points. τ-bench
    (https://arxiv.org/abs/2406.12045), Full-Duplex-Bench-v3 (https://arxiv.org/abs/2604.04847)

  - Require trace-based graders, deterministic outcome graders, policy graders, and selective model judges rather than a single
    aggregate score. Anthropic agent eval guidance (https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)

  - Roll out each tier through fixture simulation → recorded replay → shadow mode → owner-only canary → recognized household → standing
    autonomy, with instant grant revocation and rollback.

  ## Assumptions and Selected Defaults

  - Home autonomy leads the roadmap; personal-assistant breadth and conversational depth follow on the same harness.
  - The authoritative voice architecture remains STT → structured text reasoning/policy/tools → TTS. No end-to-end duplex-model
    migration is planned.

  - Nova is voice-forward: useful proactive information is spoken when an eligible room is occupied, subject to quiet hours,
    interruption policy, deduplication, and user feedback.

  - Browser microphones are in scope and currently supported. Documentation and tests must treat the browser microphone/satellite path
    as a present capability, not a future or unsupported architecture.

  - Web search is the automatic fallback for addressed knowledge-answer failures when web access is enabled. It runs at most once per
    turn and never converts operational, authorization, or household-state failures into external searches.

  - Number pronunciation is a deterministic TTS-boundary transform: use contextual years, ordinals, and cardinals; say address, room,
    and telephone digits individually with "zero"; and retain the unmodified response text for display and conversation history.

  - Durable low-risk memories are formed automatically with provenance and review; sensitive memories require confirmation.
  - Durable conversational memory is backed by MemPalace on Iridium and is selective: retain salient user context and needs from
    substantive voice chats, not routine commands or transient smart-home state.
  - Broad autonomy is enabled through explicit, scoped, expiring standing grants—not unrestricted global permission.
  - Household permissions use owner, recognized-household, and unknown/guest classes.
  - Feasibility and model/hardware selection are deferred; interfaces remain replaceable so later experiments do not rewrite the
    harness.
