import NovaCompanionKit
import SwiftUI
import UIKit

/// Why each role is or is not doing anything, in the owner's words.
///
/// The requirement this screen exists for is that the owner can tell why
/// something is inactive *without consulting logs*, so every row carries a
/// reason and "not built yet" is a reason like any other.
struct StatusView: View {
    @StateObject private var companion = CompanionCoordinator()
    @State private var copied = false

    var body: some View {
        NavigationStack {
            List {
                Section("Companion role") {
                    LabeledContent("State") {
                        HStack(spacing: 6) {
                            Circle()
                                .fill(companion.status.connected ? Color.green : Color.secondary)
                                .frame(width: 8, height: 8)
                            Text(companion.status.detail)
                                .foregroundStyle(.secondary)
                        }
                    }
                    // "Connected" is not the whole truth on its own: Nova stops
                    // offering work to a device whose telemetry has gone stale,
                    // so this row is what distinguishes a working companion
                    // from one that is merely still holding a socket open.
                    LabeledContent("Last reported") {
                        if let sent = companion.status.lastTelemetryAt {
                            Text(sent, style: .relative).foregroundStyle(.secondary)
                        } else {
                            Text("Never").foregroundStyle(.secondary)
                        }
                    }
                    LabeledContent("Jobs accepted", value: "\(companion.status.acceptedJobs)")
                    if let error = companion.status.lastError {
                        LabeledContent("Last error") {
                            Text(error).font(.footnote).foregroundStyle(.secondary)
                        }
                    }
                }

                Section("Satellite role") {
                    LabeledContent("State") {
                        Text("Not built yet").foregroundStyle(.secondary)
                    }
                }

                // "Can run", not "runs": Nova decides which of these it
                // actually sends, per pass, from the dashboard.
                Section("Workloads this device can run") {
                    ForEach(FoundationModelEngine.supported, id: \.rawValue) { workload in
                        Text(workload.rawValue).font(.callout.monospaced())
                    }
                }

                Section("Diagnostics") {
                    Button("Copy recent activity") {
                        Task {
                            UIPasteboard.general.string = await companion.exportDiagnostics()
                            copied = true
                        }
                    }
                    if copied {
                        Text("Copied. It records connections, jobs and power changes — no "
                            + "message contents.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }

                Section {
                    Button(companion.status.connected ? "Disconnect" : "Connect") {
                        if companion.status.connected {
                            companion.stop()
                        } else {
                            companion.start()
                        }
                    }
                } footer: {
                    Text(
                        "Nova offers work; this device answers it. Every action it asks for "
                        + "still runs on Nova, under Nova's own rules."
                    )
                }
            }
            .navigationTitle("Nova Companion")
        }
        .onAppear {
            // Published here rather than owned by the intent: an intent that
            // built its own coordinator would open a second session, be
            // superseded at once, and look like a reconnect loop from Nova's
            // side.
            CompanionIntentBridge.shared.coordinator = companion
            companion.start()
        }
    }
}

#Preview {
    StatusView()
}
