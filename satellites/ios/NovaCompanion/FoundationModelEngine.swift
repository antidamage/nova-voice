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

    /// Workloads this engine will answer. `classify_icon` first on purpose: it
    /// is out of the spoken turn, its answer is a single token from a closed
    /// vocabulary, and a wrong answer costs a wrong glyph rather than a wrong
    /// action in the house — the cheapest possible thing to move first.
    static let supported: [CompanionWorkload] = [.classifyIcon]

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
                    // "none" and anything invented both land here. Returning
                    // nil makes Iridium fall back rather than accept a glyph
                    // the vocabulary never offered.
                    return nil
                }
                return .object(["icon": .string(match)])
            }
        #endif
        throw EngineError.unavailable
    }
}
