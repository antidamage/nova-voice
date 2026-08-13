import Foundation

/// The socket the transport talks through.
///
/// Abstracted so the transport's decisions — what to queue, what to drop, when
/// to give up — are testable without a network, a server, or a phone. The
/// URLSession implementation is a thin conformance; everything worth getting
/// wrong lives above this line.
public protocol CompanionSocket: Actor {
    func send(_ text: String) async throws
    func receive() async throws -> String
    func close()
}

public enum TransportState: Equatable, Sendable {
    case idle
    case connecting
    /// Authenticated and registered — the only state in which jobs may flow.
    case ready
    case backingOff(seconds: Double)
    case stopped
}

public enum TransportError: Error, Equatable {
    case notReady
    case queueFull
    case duplicateCorrelation(String)
}

/// Outbound frames waiting for the socket, with a hard ceiling.
///
/// Bounded because the alternative is worse than dropping: a phone that loses
/// its connection mid-conversation would otherwise accumulate frames until iOS
/// kills the app for memory — taking the satellite role down with it. Dropping
/// the *oldest* is deliberate. These are job replies and telemetry, where the
/// newest reading is the useful one; a stale telemetry frame delivered late is
/// actively misleading, because the server ages telemetry to decide whether to
/// trust the device at all.
public struct BoundedFrameQueue: Sendable {
    public private(set) var frames: [String] = []
    public private(set) var dropped = 0
    public let capacity: Int

    public init(capacity: Int = 64) {
        precondition(capacity > 0)
        self.capacity = capacity
    }

    public mutating func append(_ frame: String) {
        frames.append(frame)
        while frames.count > capacity {
            frames.removeFirst()
            dropped += 1
        }
    }

    public mutating func drain() -> [String] {
        defer { frames.removeAll(keepingCapacity: true) }
        return frames
    }

    public var isEmpty: Bool { frames.isEmpty }
    public var count: Int { frames.count }
}

/// Matches replies to the requests that are waiting for them.
///
/// The server answers a `tool_call` with a `tool_result` carrying the same
/// `callId`, and answers nothing else in order. Without correlation a slow
/// first call would be resolved by the second call's reply — which is the kind
/// of bug that surfaces as an occasional wrong answer rather than a crash.
/// ``Value`` is `Sendable` because a reply is resolved on whichever task read
/// it off the socket and consumed on whichever task is awaiting it — the
/// compiler is right to insist the two cannot share mutable state.
public struct FrameCorrelator<Value: Sendable>: ~Copyable {
    private var waiting: [String: CheckedContinuation<Value, Error>] = [:]

    public init() {}

    public mutating func register(
        _ id: String,
        continuation: CheckedContinuation<Value, Error>
    ) {
        if let existing = waiting.removeValue(forKey: id) {
            // Reusing a correlation id in flight is a client bug. Fail the
            // older waiter rather than leaking it forever.
            existing.resume(throwing: TransportError.duplicateCorrelation(id))
        }
        waiting[id] = continuation
    }

    public mutating func resolve(_ id: String, with value: Value) -> Bool {
        guard let continuation = waiting.removeValue(forKey: id) else { return false }
        continuation.resume(returning: value)
        return true
    }

    /// Fail everything still waiting. Called on disconnect: a caller must never
    /// be left awaiting a reply that can no longer arrive.
    public mutating func failAll(with error: Error) -> Int {
        let count = waiting.count
        for continuation in waiting.values {
            continuation.resume(throwing: error)
        }
        waiting.removeAll()
        return count
    }

    public var pendingCount: Int { waiting.count }
}

/// Whether the companion should currently be trying to hold a connection.
///
/// Kept separate from the socket so the two roles can be switched
/// independently, which the roadmap requires: turning the microphone off must
/// not tear down reasoning, and vice versa.
public struct RoleGate: Sendable, Equatable {
    public var companionEnabled: Bool
    public var satelliteEnabled: Bool
    /// The satellite role is home-LAN only; the companion role may run over the
    /// tailnet for permitted personal work.
    public var onHomeNetwork: Bool

    public init(
        companionEnabled: Bool = true,
        satelliteEnabled: Bool = true,
        onHomeNetwork: Bool = false
    ) {
        self.companionEnabled = companionEnabled
        self.satelliteEnabled = satelliteEnabled
        self.onHomeNetwork = onHomeNetwork
    }

    public var shouldConnectCompanion: Bool { companionEnabled }
    public var shouldConnectSatellite: Bool { satelliteEnabled && onHomeNetwork }
}
