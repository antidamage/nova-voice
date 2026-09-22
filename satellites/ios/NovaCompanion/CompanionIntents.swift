import AppIntents
import Foundation

/// Shortcuts entry points for the companion role.
///
/// These exist because of a hard constraint rather than as a nicety: this app
/// is installed through the free developer path, which cannot use push
/// notifications. **Nothing can wake it remotely.** If iOS has suspended it,
/// the only thing that brings it back is the owner doing something on the
/// device, and asking them to find and open an app they otherwise never look
/// at is a poor answer.
///
/// A Shortcut is a better one: it can sit on the Home Screen, in the Action
/// Button, or in an automation that runs on arriving home. None of that is a
/// guarantee — iOS decides — so everything here is described as best-effort,
/// and the fallback is always that Nova keeps working on its own.
struct ReconnectCompanionIntent: AppIntent {
    static var title: LocalizedStringResource = "Reconnect Nova Companion"
    // A single literal, not a concatenation: `IntentDescription` takes a
    // `LocalizedStringResource`, which can be formed from a string *literal*
    // but not from a String assembled at runtime.
    static var description = IntentDescription(
        "Brings the Nova companion back online so it can take reasoning work from Nova. Nova keeps working normally either way."
    )

    /// Opening the app is the point, not a side effect. A Shortcut that ran
    /// entirely in the background would not achieve the one thing that is
    /// actually needed, which is the process being resident.
    static var openAppWhenRun = true

    @MainActor
    func perform() async throws -> some IntentResult & ProvidesDialog {
        let coordinator = CompanionIntentBridge.shared.coordinator
        coordinator?.start()
        // Deliberately hedged. The socket may not be established by the time
        // this returns, and claiming a connection that has not happened would
        // be worse than saying nothing.
        return .result(dialog: "Nova Companion is starting up.")
    }
}

struct CompanionStatusIntent: AppIntent {
    static var title: LocalizedStringResource = "Nova Companion status"
    static var description = IntentDescription(
        "Says whether this device is currently connected to Nova."
    )

    @MainActor
    func perform() async throws -> some IntentResult & ProvidesDialog {
        guard let status = CompanionIntentBridge.shared.coordinator?.status else {
            return .result(dialog: "Nova Companion has not started yet.")
        }
        if status.connected {
            return .result(dialog: "Connected to Nova.")
        }
        // The detail already says *why* — no endpoint, no model, reconnecting —
        // so passing it through beats a generic "not connected".
        return .result(dialog: "Not connected. \(status.detail).")
    }
}

struct CompanionShortcuts: AppShortcutsProvider {
    static var appShortcuts: [AppShortcut] {
        AppShortcut(
            intent: ReconnectCompanionIntent(),
            phrases: [
                "Reconnect \(.applicationName)",
                "Start \(.applicationName)",
            ],
            shortTitle: "Reconnect",
            systemImageName: "arrow.clockwise"
        )
        AppShortcut(
            intent: CompanionStatusIntent(),
            phrases: ["\(.applicationName) status"],
            shortTitle: "Status",
            systemImageName: "waveform"
        )
    }
}

/// Lets an intent reach the live coordinator without owning its lifecycle.
///
/// An intent may run before the UI has been built, so it cannot construct a
/// coordinator of its own — a second one would open a second session and be
/// superseded immediately, which looks like a reconnect loop from the server.
@MainActor
final class CompanionIntentBridge {
    static let shared = CompanionIntentBridge()
    weak var coordinator: CompanionCoordinator?
    private init() {}
}
