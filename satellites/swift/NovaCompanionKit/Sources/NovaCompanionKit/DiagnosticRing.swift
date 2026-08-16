import Foundation

/// A small, bounded record of what this device has been doing.
///
/// Exists because the failures worth diagnosing here are the ones that leave no
/// trace anywhere else. A phone that quietly stopped reconnecting, or accepted
/// work and then declined everything, looks identical from Iridium's side to a
/// phone that was simply switched off — and the owner is not going to be
/// holding a Mac with a console attached at the moment it happens.
///
/// The hard rule is that this holds **events, never payloads**. Every entry is
/// a category, a short structural detail and a timestamp. No prompt, no
/// calendar title, no reminder note, no coordinate and no Health value ever
/// enters it, which is what makes it safe to export in one action without a
/// review step the owner would skip anyway.
public actor DiagnosticRing {
    public enum Category: String, Codable, Sendable, CaseIterable {
        case auth
        case connection
        case job
        case tool
        case power
        case route
    }

    public struct Entry: Codable, Sendable, Equatable {
        public let at: Date
        public let category: Category
        public let event: String
        /// Structural only: counts, durations, ids, reasons. Never content.
        public let detail: String?
    }

    /// Bounded by construction. A ring that grows is a log file, and a log file
    /// on a phone is something nobody rotates.
    public let capacity: Int
    private var entries: [Entry] = []

    public init(capacity: Int = 200) {
        self.capacity = max(1, capacity)
        entries.reserveCapacity(self.capacity)
    }

    public func record(
        _ category: Category,
        _ event: String,
        detail: String? = nil,
        at moment: Date = Date()
    ) {
        entries.append(
            Entry(
                at: moment,
                category: category,
                event: String(event.prefix(64)),
                // Truncated rather than rejected: a caller that accidentally
                // passes something long should lose the tail, not the event.
                detail: detail.map { String($0.prefix(200)) }
            )
        )
        if entries.count > capacity {
            // Oldest first. The recent past is what explains a fault now; the
            // distant past is what a fresh install would not have had either.
            entries.removeFirst(entries.count - capacity)
        }
    }

    public func snapshot() -> [Entry] { entries }

    public func clear() { entries.removeAll(keepingCapacity: true) }

    /// A plain-text export for the owner to send on, newest last.
    ///
    /// Deliberately not JSON: this is read by a person deciding whether to
    /// bother reporting something, and it has to be legible in a message
    /// without a viewer.
    public func export() -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        let lines = entries.map { entry -> String in
            let stamp = formatter.string(from: entry.at)
            let detail = entry.detail.map { " \($0)" } ?? ""
            return "\(stamp) [\(entry.category.rawValue)] \(entry.event)\(detail)"
        }
        return lines.joined(separator: "\n")
    }
}
