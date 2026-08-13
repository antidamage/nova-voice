import Foundation
import XCTest

@testable import NovaCompanionKit

/// The transport's decisions, tested without a network.
///
/// Everything worth getting wrong here — what gets dropped when the socket is
/// gone, which reply resolves which request, how fast a flapping phone comes
/// back — is decidable from pure logic, so none of it needs a server, a
/// simulator, or the phone that is currently unprovisionable.
final class BackoffTests: XCTestCase {
    /// Deterministic generator so a jittered delay is still assertable.
    private struct FixedGenerator: RandomNumberGenerator {
        var value: UInt64
        mutating func next() -> UInt64 { value }
    }

    func testDelaysDoubleUpToTheCeiling() {
        let backoff = ReconnectBackoff(initialSeconds: 1, maximumSeconds: 30, jitter: 0)
        XCTAssertEqual(backoff.nominalDelay(forAttempt: 1), 1)
        XCTAssertEqual(backoff.nominalDelay(forAttempt: 2), 2)
        XCTAssertEqual(backoff.nominalDelay(forAttempt: 3), 4)
        XCTAssertEqual(backoff.nominalDelay(forAttempt: 6), 30, "should cap, not keep doubling")
        XCTAssertEqual(backoff.nominalDelay(forAttempt: 40), 30)
    }

    func testTheCapMatchesTheSatelliteClient() {
        // Both clients reconnect the same way on purpose; a household running
        // both should not have two reconnect personalities during an outage.
        XCTAssertEqual(ReconnectBackoff().maximumSeconds, 30)
        XCTAssertEqual(ReconnectBackoff().initialSeconds, 1)
    }

    func testAttemptZeroWaitsNotAtAll() {
        XCTAssertEqual(ReconnectBackoff().nominalDelay(forAttempt: 0), 0)
    }

    func testJitterOnlyEverShortensTheWait() {
        var backoff = ReconnectBackoff(initialSeconds: 10, maximumSeconds: 10, jitter: 0.2)
        var generator = SystemRandomNumberGenerator()
        for _ in 0..<200 {
            let delay = backoff.next(using: &generator)
            XCTAssertGreaterThanOrEqual(delay, 8)
            XCTAssertLessThanOrEqual(delay, 10)
        }
    }

    func testJitterIsDisableableForDeterminism() {
        var backoff = ReconnectBackoff(initialSeconds: 5, maximumSeconds: 5, jitter: 0)
        XCTAssertEqual(backoff.next(), 5)
        XCTAssertEqual(backoff.next(), 5)
    }

    func testASuccessfulConnectionResetsTheDelay() {
        var backoff = ReconnectBackoff(initialSeconds: 1, maximumSeconds: 30, jitter: 0)
        for _ in 0..<5 { _ = backoff.next() }
        XCTAssertEqual(backoff.attemptCount, 5)

        backoff.reset()

        XCTAssertEqual(backoff.attemptCount, 0)
        // An hour of healthy connection must not inherit last week's outage.
        XCTAssertEqual(backoff.next(), 1)
    }
}

final class BoundedFrameQueueTests: XCTestCase {
    func testFramesSurviveUntilTheyAreDrained() {
        var queue = BoundedFrameQueue(capacity: 4)
        queue.append("a")
        queue.append("b")
        XCTAssertEqual(queue.count, 2)
        XCTAssertEqual(queue.drain(), ["a", "b"])
        XCTAssertTrue(queue.isEmpty)
    }

    func testTheQueueIsBoundedRatherThanUnbounded() {
        // Unbounded, a phone that lost its socket mid-conversation would grow
        // until iOS killed the app — taking the satellite role with it.
        var queue = BoundedFrameQueue(capacity: 3)
        for index in 0..<10 { queue.append("frame-\(index)") }

        XCTAssertEqual(queue.count, 3)
        XCTAssertEqual(queue.dropped, 7)
    }

    func testTheOldestFramesAreDroppedNotTheNewest() {
        // Telemetry ages: the server uses staleness to decide whether to trust
        // the device, so a late old reading is worse than no reading.
        var queue = BoundedFrameQueue(capacity: 2)
        queue.append("oldest")
        queue.append("middle")
        queue.append("newest")

        XCTAssertEqual(queue.drain(), ["middle", "newest"])
    }

    func testDrainKeepsCapacityForReuse() {
        var queue = BoundedFrameQueue(capacity: 2)
        queue.append("a")
        _ = queue.drain()
        queue.append("b")
        XCTAssertEqual(queue.drain(), ["b"])
    }
}

final class FrameCorrelatorTests: XCTestCase {
    func testAReplyResolvesOnlyItsOwnRequest() async throws {
        // Without correlation, a slow first call is resolved by the second
        // call's reply — a wrong answer rather than a crash.
        let box = Box()

        async let first: String = withCheckedThrowingContinuation { continuation in
            Task { await box.register("call-1", continuation) }
        }
        async let second: String = withCheckedThrowingContinuation { continuation in
            Task { await box.register("call-2", continuation) }
        }

        try await Task.sleep(nanoseconds: 50_000_000)
        await box.resolve("call-2", "second reply")
        await box.resolve("call-1", "first reply")

        let results = try await (first, second)
        XCTAssertEqual(results.0, "first reply")
        XCTAssertEqual(results.1, "second reply")
    }

    func testADisconnectFailsEveryWaiter() async {
        // A caller must never be left awaiting a reply that cannot arrive.
        let box = Box()
        async let pending: String = withCheckedThrowingContinuation { continuation in
            Task { await box.register("call-1", continuation) }
        }
        try? await Task.sleep(nanoseconds: 50_000_000)
        let failed = await box.failAll(TransportError.notReady)
        XCTAssertEqual(failed, 1)

        do {
            _ = try await pending
            XCTFail("a waiter survived a disconnect")
        } catch {
            XCTAssertEqual(error as? TransportError, .notReady)
        }
    }

    func testAnUnknownReplyIsReportedRatherThanIgnored() async {
        let box = Box()
        let matched = await box.resolve("never-registered", "reply")
        XCTAssertFalse(matched)
    }

    /// Wraps the non-copyable correlator so tests can share it across tasks.
    private actor Box {
        private var correlator = FrameCorrelator<String>()

        func register(_ id: String, _ continuation: CheckedContinuation<String, Error>) {
            correlator.register(id, continuation: continuation)
        }

        @discardableResult
        func resolve(_ id: String, _ value: String) -> Bool {
            correlator.resolve(id, with: value)
        }

        func failAll(_ error: Error) -> Int {
            correlator.failAll(with: error)
        }
    }
}

final class RoleGateTests: XCTestCase {
    func testTheMicrophoneIsHomeLanOnly() {
        var gate = RoleGate(companionEnabled: true, satelliteEnabled: true, onHomeNetwork: false)
        XCTAssertFalse(gate.shouldConnectSatellite, "audio must not leave the home LAN")
        XCTAssertTrue(gate.shouldConnectCompanion, "personal tools stay available away")

        gate.onHomeNetwork = true
        XCTAssertTrue(gate.shouldConnectSatellite)
    }

    func testTheRolesSwitchIndependently() {
        // Muting a microphone and disabling an agent are different intentions.
        var gate = RoleGate(companionEnabled: true, satelliteEnabled: true, onHomeNetwork: true)

        gate.satelliteEnabled = false
        XCTAssertFalse(gate.shouldConnectSatellite)
        XCTAssertTrue(gate.shouldConnectCompanion)

        gate = RoleGate(companionEnabled: false, satelliteEnabled: true, onHomeNetwork: true)
        XCTAssertFalse(gate.shouldConnectCompanion)
        XCTAssertTrue(gate.shouldConnectSatellite)
    }
}
