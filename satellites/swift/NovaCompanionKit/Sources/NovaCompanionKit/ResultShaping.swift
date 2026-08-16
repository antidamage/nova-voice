import Foundation

/// Turning what a small on-device model says into what Iridium will accept.
///
/// This lives in the kit rather than beside the model for one reason: it is
/// where the mistakes are. An answer that fails Iridium's validation costs the
/// whole attempt and falls back, so every value the device emits has to land
/// inside Nova's own enums and its own field names — and a name that is merely
/// *plausible* fails silently on every plan until someone reads a log.
///
/// That has happened once already: actions were emitted with `name` where the
/// domain wants `tool`, and without the `order` a `PlannedAction` requires.
/// Both are the kind of thing a test catches instantly and a code review does
/// not.
public enum CompanionResultShaping {
    // MARK: - Nova's enums, mirrored

    /// A value outside these fails validation on Iridium and costs the
    /// fallback, so an unrecognised answer is corrected here instead — to the
    /// member that does the least if the model was wrong.
    public static let emotions: Set<String> = [
        "neutral", "calm", "grumpy", "angry", "excited", "bored", "sad", "anxious",
    ]
    public static let speechActs: Set<String> = [
        "directive", "desired_state", "self_intention", "observation", "question",
        "third_party", "quoted_or_media", "social", "unclear",
    ]
    public static let decisions: Set<String> = ["execute", "reply", "clarify", "ignore"]
    public static let goalStatuses: Set<String> = [
        "new", "in_progress", "needs_clarification", "satisfied", "abandoned",
    ]

    /// The safe members. `ignore` does nothing; `unclear` asserts nothing.
    public static let safeDecision = "ignore"
    public static let safeSpeechAct = "unclear"

    // MARK: - Coercion

    public static func oneOf(_ value: String, _ allowed: Set<String>, fallback: String) -> String {
        let candidate = value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return allowed.contains(candidate) ? candidate : fallback
    }

    public static func clamp(_ value: Double) -> Double {
        // NaN compares false against everything, so `min(max(...))` alone
        // would pass it straight through into a field typed 0...1.
        guard value.isFinite else { return 0 }
        return min(max(value, 0), 1)
    }

    /// Trim, and treat blank or the model's own filler as absent.
    ///
    /// Small models answer an optional string field with a word rather than
    /// leaving it out, and "none" stored as somebody's name is worse than no
    /// name at all.
    public static func tidy(_ value: String?) -> String? {
        guard let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines),
            !trimmed.isEmpty
        else { return nil }
        let placeholders: Set<String> = ["none", "null", "n/a", "na", "unknown", "unspecified"]
        return placeholders.contains(trimmed.lowercased()) ? nil : trimmed
    }

    /// Parse a model-authored arguments blob, or give back an empty object.
    ///
    /// Never nil and never a non-object: the domain field is a mapping, and an
    /// action with malformed arguments should be an action with *no* arguments
    /// rather than a plan that fails to parse as a whole.
    public static func decodeArguments(_ text: String) -> JSONValue {
        guard let data = text.data(using: .utf8),
            let value = try? JSONDecoder().decode(JSONValue.self, from: data),
            case .object = value
        else { return .object([:]) }
        return value
    }

    // MARK: - The action shape

    /// One planned action in the shape `PlannedAction` actually requires.
    ///
    /// `tool`, not `name`. `order`, always. `depends_on`, even when empty.
    /// Getting any of these wrong produces a plan that parses on this side and
    /// is rejected on the other.
    public static func action(
        index: Int, provider: String, tool: String, argumentsJSON: String
    ) -> JSONValue {
        .object([
            "id": .string("companion-\(index)"),
            // Sequential and dependency-free. A model asked to invent an
            // execution order invents dependencies with it, and a wrong
            // dependency deadlocks a plan rather than merely misordering it.
            // Spoken clauses run in the order they were spoken.
            "order": .number(Double(index)),
            "depends_on": .array([]),
            "call": .object([
                "provider": .string(provider),
                "tool": .string(tool),
                "arguments": decodeArguments(argumentsJSON),
            ]),
        ])
    }

    /// Render an observed-state blob compactly for a 4k-token prompt.
    public static func describe(_ value: JSONValue) -> String {
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
            return flag ? "yes" : "no"
        case .null:
            return "unknown"
        }
    }
}
