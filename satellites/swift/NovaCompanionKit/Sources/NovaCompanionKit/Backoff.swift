import Foundation

/// Reconnect delays for a companion that keeps losing its socket.
///
/// Mirrors the Python satellite client (`satellites/client.py`): start at one
/// second, double, cap at thirty. Deliberately the same shape so a household
/// running both does not have two different reconnect personalities to reason
/// about during an outage.
///
/// The addition here is **jitter**, which the phone needs and a fixed satellite
/// does not. A phone reconnects on every network transition — Wi-Fi to
/// cellular, waking from suspend, walking out the door — so without jitter a
/// server restart would bring every retry back in lockstep with whatever else
/// was disconnected at the same moment.
public struct ReconnectBackoff: Sendable {
    public let initialSeconds: Double
    public let maximumSeconds: Double
    public let multiplier: Double
    /// Fraction of the delay that is randomised, 0...1. At 0.2 a nominal 10 s
    /// wait lands somewhere in 8...10 s.
    public let jitter: Double

    private var attempt = 0

    public init(
        initialSeconds: Double = 1.0,
        maximumSeconds: Double = 30.0,
        multiplier: Double = 2.0,
        jitter: Double = 0.2
    ) {
        precondition(initialSeconds > 0, "initial delay must be positive")
        precondition(maximumSeconds >= initialSeconds, "maximum must not be below initial")
        precondition(multiplier >= 1, "multiplier must not shrink the delay")
        precondition((0...1).contains(jitter), "jitter is a fraction of the delay")
        self.initialSeconds = initialSeconds
        self.maximumSeconds = maximumSeconds
        self.multiplier = multiplier
        self.jitter = jitter
    }

    /// The delay this attempt would wait, before jitter. Exposed for tests and
    /// for status reporting — an operator wants "retrying in 30s", not a
    /// number that moved while they read it.
    public func nominalDelay(forAttempt attempt: Int) -> Double {
        guard attempt > 0 else { return 0 }
        let raw = initialSeconds * pow(multiplier, Double(attempt - 1))
        return min(maximumSeconds, raw)
    }

    /// Advance and return the next delay, jitter applied.
    public mutating func next(using generator: inout some RandomNumberGenerator) -> Double {
        attempt += 1
        let nominal = nominalDelay(forAttempt: attempt)
        guard jitter > 0 else { return nominal }
        let lowest = nominal * (1 - jitter)
        return Double.random(in: lowest...nominal, using: &generator)
    }

    public mutating func next() -> Double {
        var generator = SystemRandomNumberGenerator()
        return next(using: &generator)
    }

    /// Called on a successful connection. The next failure starts from the
    /// bottom again — a device that has been happily connected for an hour
    /// should not inherit the delay from an outage last week.
    public mutating func reset() {
        attempt = 0
    }

    public var attemptCount: Int { attempt }
}
