import NovaCompanionKit
import SwiftUI

/// Why a role is inactive, in the owner's words.
///
/// The roadmap's requirement for this screen is that the owner can tell why
/// something is off *without consulting logs* — so every row carries a reason,
/// and "not configured yet" is a reason like any other rather than a blank.
struct StatusRow: Identifiable {
    let id = UUID()
    let title: String
    let detail: String
    let active: Bool
}

struct StatusView: View {
    private let rows: [StatusRow] = [
        StatusRow(
            title: "Companion role",
            detail: "Not connected — transport not built yet",
            active: false
        ),
        StatusRow(
            title: "Satellite role",
            detail: "Not connected — home-LAN audio not built yet",
            active: false
        ),
        StatusRow(
            title: "Protocol",
            detail: "v\(CompanionProtocol.version), \(CompanionWorkload.allCases.count) workloads",
            active: true
        ),
    ]

    var body: some View {
        NavigationStack {
            List {
                Section("Status") {
                    ForEach(rows) { row in
                        HStack(alignment: .firstTextBaseline) {
                            Circle()
                                .fill(row.active ? Color.green : Color.secondary)
                                .frame(width: 8, height: 8)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(row.title)
                                Text(row.detail)
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                }

                Section("Workloads this device can be offered") {
                    ForEach(CompanionWorkload.allCases, id: \.rawValue) { workload in
                        Text(workload.rawValue)
                            .font(.callout.monospaced())
                    }
                }
            }
            .navigationTitle("Nova Companion")
        }
    }
}

#Preview {
    StatusView()
}
