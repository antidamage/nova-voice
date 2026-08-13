import SwiftUI

/// The Nova companion app.
///
/// Two independent roles live here eventually — a voice satellite and a
/// reasoning companion — and the coordinator's job is to keep them independent.
/// Turning off the microphone must not tear down reasoning or personal tools,
/// and losing the companion socket must not stop audio playback. Nothing is
/// wired to Nova yet: this first cut proves the app builds against the shared
/// wire package and can render its own status.
@main
struct NovaCompanionApp: App {
    var body: some Scene {
        WindowGroup {
            StatusView()
        }
    }
}
