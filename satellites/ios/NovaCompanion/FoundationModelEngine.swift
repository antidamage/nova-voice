import Foundation
import NovaCompanionKit

#if canImport(FoundationModels)
    import FoundationModels
#endif

/// Answers offered jobs with Apple's on-device model.
///
/// Lives in the app rather than the shared package because `FoundationModels`
/// is an OS framework: the package must keep building headlessly on the build
/// host, where availability cannot be assumed.
///
/// Availability is a runtime fact, not a compile-time one. The model can be
/// absent because the OS is older, because the device does not support it, or
/// because the user has not enabled it — so this reports "not available" and
/// lets the session reject cleanly, which costs Iridium one round trip. The
/// alternative, accepting and then failing, costs it the whole completion
/// deadline before it falls back.
struct FoundationModelEngine: CompanionSession.Engine {
    enum EngineError: Error {
        case unavailable
        case unsupportedWorkload(CompanionWorkload)
        case malformedResponse
    }

    /// Workloads this engine will answer, in the order they were moved.
    ///
    /// `classify_icon` first on purpose: it is out of the spoken turn, its
    /// answer is a single token from a closed vocabulary, and a wrong answer
    /// costs a wrong glyph rather than a wrong action in the house — the
    /// cheapest possible thing to move first.
    ///
    /// The other two are also off the spoken path, and each earns its place
    /// for a different reason. `extract_self_profile_update` runs *beside*
    /// interpretation on the same turn, so on Iridium the two queue behind one
    /// llama.cpp slot; moving it stops them serialising. `confirm_objective`
    /// fires repeatedly while a multi-device command settles — every call
    /// lands exactly when the house is busiest.
    ///
    /// `render_response` is implemented below but **deliberately not
    /// advertised**. Measured on this device it matched Iridium for speed and
    /// then lost Nova entirely: flat "I'm just an AI, I don't have feelings"
    /// replies where the local model speaks in character, answering the
    /// previous question rather than the current one. That pass *is* the
    /// assistant's voice, so it is not a rough edge to ship and polish later.
    ///
    /// Advertising is the gate that matters. Iridium offers only what a device
    /// says it can do, so leaving it out of this list stops the offers at the
    /// source — before, and independently of, whatever the server's route table
    /// happens to say. See docs/evidence/companion-offload-live-20260815.md.
    ///
    /// `interpret` is absent for the same reason and worse stakes: it plans the
    /// actions.
    static let supported: [CompanionWorkload] = [
        .classifyIcon, .extractSelfProfileUpdate, .confirmObjective,
    ]

    /// Why the model is or is not usable, in words. Availability has several
    /// distinct causes — unsupported device, feature disabled, model still
    /// downloading — and they need different actions from the owner, so
    /// collapsing them to a bool loses the only useful part.
    var availabilityDescription: String {
        #if canImport(FoundationModels)
            if #available(iOS 26.0, *) {
                switch SystemLanguageModel.default.availability {
                case .available:
                    return "available"
                case .unavailable(let reason):
                    return "unavailable: \(reason)"
                @unknown default:
                    return "unavailable: unknown"
                }
            }
            return "unavailable: requires iOS 26"
        #else
            return "unavailable: FoundationModels not compiled in"
        #endif
    }

    var isAvailable: Bool {
        #if canImport(FoundationModels)
            if #available(iOS 26.0, *) {
                if case .available = SystemLanguageModel.default.availability { return true }
            }
        #endif
        return false
    }

    func run(
        workload: CompanionWorkload,
        payload: JSONValue,
        contextTokens: Int
    ) async throws -> JSONValue? {
        guard Self.supported.contains(workload) else {
            throw EngineError.unsupportedWorkload(workload)
        }
        guard isAvailable else { throw EngineError.unavailable }

        switch workload {
        case .classifyIcon:
            return try await classifyIcon(payload: payload)
        case .extractSelfProfileUpdate:
            return try await extractSelfProfileUpdate(payload: payload)
        case .confirmObjective:
            return try await confirmObjective(payload: payload)
        case .renderResponse:
            return try await renderResponse(payload: payload)
        case .interpret:
            return try await interpret(payload: payload)
        default:
            throw EngineError.unsupportedWorkload(workload)
        }
    }

    /// Pick the glyph that best represents a reminder's name.
    ///
    /// The vocabulary is closed and comes with the job, so the answer is
    /// re-checked against it here as well as on the server. A schema is a
    /// strong guarantee, not a substitute for validating what crossed a
    /// network — and a value outside the vocabulary would render as a missing
    /// glyph rather than an error anyone would notice.
    private func classifyIcon(payload: JSONValue) async throws -> JSONValue? {
        let name = payload["name"]?.stringValue ?? ""
        let icons = (payload["icons"]?.arrayValue ?? []).compactMap(\.stringValue)
        guard !name.isEmpty, !icons.isEmpty else { return nil }

        #if canImport(FoundationModels)
            if #available(iOS 26.0, *) {
                let session = LanguageModelSession(
                    instructions: """
                        You choose one icon id for a reminder's name.
                        Answer with exactly one id from the list and nothing else.
                        If none fits, answer with the single word none.
                        """
                )
                let prompt = """
                    Reminder: \(name)
                    Icons: \(icons.joined(separator: ", "))
                    """
                let response = try await session.respond(to: prompt)
                let chosen = response.content
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    .lowercased()
                guard let match = icons.first(where: { $0.lowercased() == chosen }) else {
                    // "none" and anything invented both land here, and both
                    // are answered rather than rejected: no glyph fits is a
                    // real conclusion, and rejecting would send Iridium off to
                    // reach the same one on its own model.
                    return .object(["icon": .null])
                }
                return .object(["icon": .string(match)])
            }
        #endif
        throw EngineError.unavailable
    }

    /// Pull an explicit first-person name or pronoun disclosure out of a turn.
    ///
    /// Almost every turn contains none, so "nothing was disclosed" is answered
    /// explicitly rather than rejected — a rejection would hand the common
    /// case straight back to Iridium and offload only the rare one.
    ///
    /// Inference is not the job. "I'm Adeline" discloses a name; "I'm cold"
    /// and "I'm at Adeline's place" do not, and a model that helpfully fills
    /// the field in would rename the household from a passing remark. That is
    /// why `evidence` is required alongside: a claim with no quotable span is
    /// discarded here rather than trusted.
    private func extractSelfProfileUpdate(payload: JSONValue) async throws -> JSONValue? {
        let transcript = payload["transcript"]?.stringValue ?? ""
        guard !transcript.isEmpty else { return .object(["update": .null]) }

        #if canImport(FoundationModels)
            if #available(iOS 26.0, *) {
                let session = LanguageModelSession(
                    instructions: """
                        You detect whether a speaker states their own name or \
                        their own pronouns.
                        Report only what is explicitly stated about the speaker \
                        themselves. Never infer, never guess from context, and \
                        never report a name belonging to someone else.
                        If nothing is explicitly stated, set disclosed to false.
                        Quote the exact words as evidence.
                        """
                )
                let generated = try await session.respond(
                    to: "Utterance: \(transcript)",
                    generating: GeneratedSelfProfile.self
                ).content

                guard generated.disclosed else { return .object(["update": .null]) }
                let name = Self.tidy(generated.name)
                let pronouns = Self.tidy(generated.pronouns)
                let evidence = Self.tidy(generated.evidence)
                // Iridium's schema requires at least one of the two plus
                // evidence. Filtering here rather than sending a shape it will
                // reject keeps a wasted round trip out of the spoken turn.
                guard name != nil || pronouns != nil, let evidence else {
                    return .object(["update": .null])
                }
                // The evidence must actually be in what was said. A model that
                // paraphrases its own justification is the failure mode this
                // catches, and it is the one that would rename the household.
                guard transcript.localizedCaseInsensitiveContains(evidence) else {
                    return .object(["update": .null])
                }
                return .object([
                    "update": .object([
                        "name": name.map(JSONValue.string) ?? .null,
                        "pronouns": pronouns.map(JSONValue.string) ?? .null,
                        "evidence": .string(String(evidence.prefix(200))),
                    ])
                ])
            }
        #endif
        throw EngineError.unavailable
    }

    /// Judge whether each still-pending device objective is now satisfied.
    ///
    /// Called repeatedly from Iridium's verification loop while a multi-device
    /// command settles, which is what makes it worth moving: those calls land
    /// on Iridium exactly when it is busiest actually driving the devices.
    ///
    /// Only the targets that were sent may appear in the answer, and
    /// `all_confirmed` is computed here rather than taken from the model — it
    /// is derivable, and Iridium rejects the whole verdict if the two disagree.
    private func confirmObjective(payload: JSONValue) async throws -> JSONValue? {
        let transcript = payload["transcript"]?.stringValue ?? ""
        let pending = payload["pending"]?.arrayValue ?? []
        guard !pending.isEmpty else {
            return .object(["items": .array([]), "all_confirmed": .bool(true)])
        }

        let targets = pending.compactMap { $0["target"]?.stringValue }
        guard !targets.isEmpty else { return nil }

        #if canImport(FoundationModels)
            if #available(iOS 26.0, *) {
                let described = pending.map { entry -> String in
                    let target = entry["target"]?.stringValue ?? "?"
                    let objective = entry["objective"]?.stringValue ?? "?"
                    let observed = entry["observed"].map(Self.describe) ?? "unknown"
                    return "- target: \(target)\n  objective: \(objective)\n  observed: \(observed)"
                }.joined(separator: "\n")

                let session = LanguageModelSession(
                    instructions: """
                        You judge whether each device's observed state now \
                        satisfies its stated objective.
                        Judge only from the observed state given. Do not assume \
                        a command succeeded because it was issued.
                        Answer for every target listed, once each, using the \
                        target's exact name. Give a short reason for each.
                        """
                )
                let generated = try await session.respond(
                    to: """
                        Request: \(transcript)
                        Targets:
                        \(described)
                        """,
                    generating: GeneratedVerdict.self
                ).content

                // One verdict per target that was asked about, in the order
                // they were asked. A target the model skipped counts as
                // unconfirmed, which keeps the loop waiting rather than
                // declaring success it never actually judged.
                var items: [JSONValue] = []
                var allConfirmed = true
                for target in targets {
                    let match = generated.items.first {
                        $0.target.compare(target, options: .caseInsensitive) == .orderedSame
                    }
                    let confirmed = match?.confirmed ?? false
                    if !confirmed { allConfirmed = false }
                    let reason = Self.tidy(match?.reason) ?? "no verdict returned for this target"
                    items.append(
                        .object([
                            "target": .string(target),
                            "confirmed": .bool(confirmed),
                            "reason": .string(String(reason.prefix(200))),
                        ])
                    )
                }
                return .object(["items": .array(items), "all_confirmed": .bool(allConfirmed)])
            }
        #endif
        throw EngineError.unavailable
    }

    /// Speak Nova's reply for this turn.
    ///
    /// The instructions arrive with the job and are used verbatim. They are
    /// long, specific, and load-bearing — they decide whether the assistant may
    /// claim a device changed, how many words it gets, whether it may mention
    /// the weather — so nothing here paraphrases or supplements them. A phone
    /// that improvised its own version of the reply contract would produce
    /// replies that sound right and are wrong.
    ///
    /// Whatever comes back is still re-checked on Iridium against the same
    /// word budgets the local model is held to.
    private func renderResponse(payload: JSONValue) async throws -> JSONValue? {
        guard let instructions = payload["instructions"]?.stringValue, !instructions.isEmpty,
            let facts = payload["facts"]
        else { return nil }

        #if canImport(FoundationModels)
            if #available(iOS 26.0, *) {
                let session = LanguageModelSession(instructions: instructions)
                var prompt = ""
                // Prior turns first, in the order they happened, so the reply
                // lands as a continuation rather than a fresh answer.
                //
                // Fenced and labelled rather than concatenated. Iridium's model
                // receives these as separate chat messages and cannot confuse
                // them with the request; flattened into one string they are
                // just more text, and on device the reply came back answering
                // the *previous* turn's question.
                let history = payload["history"]?.arrayValue ?? []
                if !history.isEmpty {
                    prompt += "Earlier in this conversation, for context only:\n"
                    for message in history {
                        guard let role = message["role"]?.stringValue,
                            let content = message["content"]?.stringValue
                        else { continue }
                        prompt += "\(role): \(content)\n"
                    }
                    prompt += "\n"
                }
                prompt += "Reply to this turn, and only this turn:\n"
                prompt += Self.encode(facts)

                var options = GenerationOptions()
                if let maxTokens = payload["maxTokens"]?.intValue {
                    options = GenerationOptions(maximumResponseTokens: maxTokens)
                }
                let text = try await session.respond(to: prompt, options: options).content
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                // Empty is a failure, not an answer: a turn with nothing to say
                // says nothing at all out loud, and that must not be something
                // a phone can cause by accident.
                guard !text.isEmpty else { return nil }
                return .object(["text": .string(text)])
            }
        #endif
        throw EngineError.unavailable
    }

    /// Classify a turn and, when it asks for one, plan the actions.
    ///
    /// The riskiest thing this device does, and the shape reflects that. The
    /// model is asked for a flat set of fields plus at most a few tool calls,
    /// rather than for Nova's full `Interpretation` — that schema carries
    /// cross-field rules a generation schema cannot express, and asking a small
    /// model to satisfy them produces failed generations rather than careful
    /// answers.
    ///
    /// Everything consequential is re-derived or re-checked on Iridium: it
    /// validates the result against the real schema and rejects any action
    /// naming a tool this turn did not offer, discarding the whole plan rather
    /// than dropping a clause. Nothing here is trusted to be safe; it is
    /// trusted only to be a suggestion.
    private func interpret(payload: JSONValue) async throws -> JSONValue? {
        let turn = payload["turn"]
        guard let transcript = turn?["utterance"]?["transcript"]?.stringValue,
            !transcript.isEmpty
        else { return nil }

        #if canImport(FoundationModels)
            if #available(iOS 26.0, *) {
                let instructions = payload["instructions"]?.stringValue ?? ""
                let session = LanguageModelSession(instructions: instructions)

                var prompt = ""
                if let tools = payload["semanticTools"], case .array(let list) = tools,
                    !list.isEmpty
                {
                    prompt += "Tools you may call:\n\(Self.encode(tools))\n\n"
                }
                if let state = payload["relevantState"], state != .object([:]) {
                    prompt += "Household state:\n\(Self.encode(state))\n\n"
                }
                // Whatever Iridium had to drop to fit this device is stated, so
                // the plan can be honest about seeing a partial listing rather
                // than answering as though it saw everything.
                if let note = payload["compactionNote"]?.stringValue {
                    prompt += "\(note)\n\n"
                }
                prompt += "Classify this turn and plan any actions it asks for:\n"
                prompt += Self.encode(turn ?? .null)

                let generated = try await session.respond(
                    to: prompt, generating: GeneratedInterpretation.self
                ).content
                return Self.interpretation(from: generated, transcript: transcript)
            }
        #endif
        throw EngineError.unavailable
    }

    /// Map the flat generated shape onto Nova's wire schema.
    ///
    /// Clamping rather than trusting: probabilities out of range, an unknown
    /// enum value or a malformed tool name would each fail validation on
    /// Iridium and cost the fallback, so they are corrected here where the
    /// correct value is knowable and dropped where it is not.
    @available(iOS 26.0, *)
    private static func interpretation(
        from generated: GeneratedInterpretation,
        transcript: String
    ) -> JSONValue {
        var actions: [JSONValue] = []
        for (index, action) in generated.actions.prefix(6).enumerated() {
            let provider = tidy(action.provider)?.lowercased()
            let name = tidy(action.name)
            guard let provider, let name else { continue }
            actions.append(
                .object([
                    "id": .string("companion-\(index)"),
                    // Sequential, and dependency-free. A model asked to invent
                    // an execution order invents dependencies with it, and a
                    // wrong dependency deadlocks the plan rather than
                    // misordering it. Spoken clauses run in the order spoken.
                    "order": .number(Double(index)),
                    "depends_on": .array([]),
                    "call": .object([
                        "provider": .string(provider),
                        "tool": .string(name),
                        "arguments": decodeArguments(action.argumentsJSON),
                    ]),
                ])
            )
        }

        return .object([
            "emotion": .object([
                "label": .string(oneOf(generated.emotion, Self.EMOTIONS, fallback: "neutral")),
                "confidence": .number(clamp(generated.emotionConfidence)),
                "intensity": .number(clamp(generated.emotionIntensity)),
                "evidence": .array([]),
            ]),
            "speech_act": .string(oneOf(generated.speechAct, Self.SPEECH_ACTS, fallback: "unclear")),
            "addressed_probability": .number(clamp(generated.addressedProbability)),
            "decision": .string(oneOf(generated.decision, Self.DECISIONS, fallback: "ignore")),
            "confidence": .number(clamp(generated.confidence)),
            "active_goal": .object([
                "summary": .string(tidy(generated.goalSummary) ?? transcript),
                "status": .string(oneOf(generated.goalStatus, Self.GOAL_STATUSES, fallback: "new")),
                "pending": .array([]),
            ]),
            "actions": .array(actions),
            "response_plan": .object([
                "acknowledgement_style": .string("concise"),
                "pre_action_speech": .null,
                "requires_post_tool_rendering": .bool(!actions.isEmpty),
            ]),
            "self_profile_update": .null,
        ])
    }

    /// Tool arguments arrive as a JSON string: a generation schema cannot
    /// describe a shape that differs per tool. Unparseable means no arguments
    /// rather than a discarded action, and Iridium validates them anyway.
    private static func decodeArguments(_ text: String) -> JSONValue {
        guard let data = text.data(using: .utf8),
            let value = try? JSONDecoder().decode(JSONValue.self, from: data),
            case .object = value
        else { return .object([:]) }
        return value
    }

    private static func oneOf(_ value: String, _ allowed: Set<String>, fallback: String) -> String {
        let candidate = value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return allowed.contains(candidate) ? candidate : fallback
    }

    private static func clamp(_ value: Double) -> Double {
        min(max(value, 0), 1)
    }

    /// The structured input, as JSON, exactly as Iridium's own model receives it.
    private static func encode(_ value: JSONValue) -> String {
        guard let data = try? JSONEncoder().encode(value),
            let text = String(data: data, encoding: .utf8)
        else { return "{}" }
        return text
    }

    /// Nova's enums, mirrored. A value outside these fails validation on
    /// Iridium and costs the fallback, so an unrecognised answer is corrected
    /// to the safe member here instead — `ignore` for a decision, `unclear`
    /// for a speech act: the ones that do the least if the model was wrong.
    private static let EMOTIONS: Set<String> = [
        "neutral", "calm", "grumpy", "angry", "excited", "bored", "sad", "anxious",
    ]
    private static let SPEECH_ACTS: Set<String> = [
        "directive", "desired_state", "self_intention", "observation", "question",
        "third_party", "quoted_or_media", "social", "unclear",
    ]
    private static let DECISIONS: Set<String> = ["execute", "reply", "clarify", "ignore"]
    private static let GOAL_STATUSES: Set<String> = [
        "new", "in_progress", "needs_clarification", "satisfied", "abandoned",
    ]

    /// Trim, and treat blank or the model's own filler as absent.
    private static func tidy(_ value: String?) -> String? {
        guard let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines),
            !trimmed.isEmpty
        else { return nil }
        // Small models answer optional string fields with a word rather than
        // leaving them out, and "none" as a name is worse than no name.
        let placeholders: Set<String> = ["none", "null", "n/a", "na", "unknown", "unspecified"]
        return placeholders.contains(trimmed.lowercased()) ? nil : trimmed
    }

    /// Render an observed-state blob compactly for the prompt.
    private static func describe(_ value: JSONValue) -> String {
        switch value {
        case .object(let fields):
            return fields
                .sorted { $0.key < $1.key }
                .map { "\($0.key)=\(describe($0.value))" }
                .joined(separator: " ")
        case .array(let items):
            return items.map(describe).joined(separator: ", ")
        case .string(let text):
            return text
        case .number(let number):
            // Whole numbers read better without a decimal tail — "22" not
            // "22.0" — but the conversion is guarded because a value outside
            // Int's range would trap, and a prompt is not worth a crash.
            if number == number.rounded(), let whole = Int(exactly: number.rounded()) {
                return String(whole)
            }
            return String(number)
        case .bool(let flag):
            return flag ? "true" : "false"
        case .null:
            return "unknown"
        }
    }
}

#if canImport(FoundationModels)
    /// The typed shapes the on-device model generates against.
    ///
    /// They are deliberately *not* Iridium's schemas. Iridium's models carry
    /// cross-field rules — a self-profile update must contain a name or
    /// pronouns, `all_confirmed` may not contradict its items — that a
    /// generation schema cannot express, so the model is given a flat shape it
    /// can always satisfy and the rules are applied afterwards, in code. The
    /// alternative is a model that fails generation on the ordinary case.
    @available(iOS 26.0, *)
    @Generable
    struct GeneratedSelfProfile {
        @Guide(description: "True only if the speaker explicitly stated their own name or pronouns")
        var disclosed: Bool
        @Guide(description: "The speaker's own name, exactly as stated, or empty if not stated")
        var name: String
        @Guide(description: "The speaker's own pronouns, exactly as stated, or empty if not stated")
        var pronouns: String
        @Guide(description: "The exact words from the utterance that state it, or empty")
        var evidence: String
    }

    @available(iOS 26.0, *)
    @Generable
    struct GeneratedVerdict {
        @Guide(description: "One entry per target, using the exact target name given")
        var items: [GeneratedVerdictItem]
    }

    /// Flat on purpose. Nova's `Interpretation` nests emotion, goal and plan,
    /// and carries cross-field rules a generation schema cannot express; a
    /// small model asked for that shape fails generation rather than answering
    /// carefully. The nesting is rebuilt in code, where the rules can be.
    ///
    /// Tool arguments are a JSON *string* for the same reason: their shape
    /// differs per tool, so no single schema describes them.
    @available(iOS 26.0, *)
    @Generable
    struct GeneratedInterpretation {
        @Guide(description: "One of: neutral, calm, grumpy, angry, excited, bored, sad, anxious")
        var emotion: String
        @Guide(description: "How sure you are of the emotion, 0 to 1")
        var emotionConfidence: Double
        @Guide(description: "How strong the emotion is, 0 to 1")
        var emotionIntensity: Double
        @Guide(
            description:
                "One of: directive, desired_state, self_intention, observation, question, "
                + "third_party, quoted_or_media, social, unclear"
        )
        var speechAct: String
        @Guide(description: "Probability this was said to the assistant, 0 to 1")
        var addressedProbability: Double
        @Guide(
            description:
                "One of: execute (run the tools), reply (answer in words), clarify (ask a "
                + "question), ignore (not addressed to you)"
        )
        var decision: String
        @Guide(description: "How sure you are of the decision, 0 to 1")
        var confidence: Double
        @Guide(description: "One short sentence naming what the speaker wants")
        var goalSummary: String
        @Guide(
            description: "One of: new, in_progress, needs_clarification, satisfied, abandoned"
        )
        var goalStatus: String
        @Guide(description: "Tools to call, only from the list given, empty if none are needed")
        var actions: [GeneratedAction]
    }

    @available(iOS 26.0, *)
    @Generable
    struct GeneratedAction {
        @Guide(description: "The provider part of the tool name, before the dot")
        var provider: String
        @Guide(description: "The tool name, after the dot")
        var name: String
        @Guide(description: "The tool's arguments as a JSON object string, or {} if none")
        var argumentsJSON: String
    }

    @available(iOS 26.0, *)
    @Generable
    struct GeneratedVerdictItem {
        @Guide(description: "The target's exact name as given")
        var target: String
        @Guide(description: "True only if the observed state satisfies the objective")
        var confirmed: Bool
        @Guide(description: "A short reason, under 200 characters")
        var reason: String
    }
#endif
