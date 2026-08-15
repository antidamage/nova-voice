import NovaCompanionKit
import SwiftUI

/// Why each role is or is not doing anything, in the owner's words.
///
/// The requirement this screen exists for is that the owner can tell why
/// something is inactive *without consulting logs*, so every row carries a
/// reason and "not built yet" is a reason like any other.
struct StatusView: View {
    @StateObject private var companion = CompanionCoordinator()

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

                Section("Workloads this device offers") {
                    ForEach(FoundationModelEngine.supported, id: \.rawValue) { workload in
                        Text(workload.rawValue).font(.callout.monospaced())
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
        .onAppear { companion.start() }
    }
}

#Preview {
    StatusView()
}
