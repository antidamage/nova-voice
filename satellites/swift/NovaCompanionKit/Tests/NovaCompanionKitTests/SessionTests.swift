import Foundation
import XCTest

@testable import NovaCompanionKit

/// The offload itself: a job offered by Iridium, answered by the phone.
///
/// Driven over an in-memory socket against a stub engine, so the behaviour is
/// testable without a device, a model, or Apple's availability rules — which
/// matters because those are exactly the things that cannot be automated.
final class CompanionSessionTests: XCTestCase {
    /// Both ends of a socket in memory. The session sends into `toServer` and
    /// reads from `toPeer`; the test drives the opposite sides.
    private actor Wire: CompanionSocket {
        private var toPeer: [String] = []
        private var toServer: [String] = []
        private var waiters: [CheckedContinuation<String, Error>] = []

        func send(_ text: String) async throws { toServer.append(text) }

        func receive() async throws -> String {
            if !toPeer.isEmpty { return toPeer.removeFirst() }
            return try await withCheckedThrowingContinuation { waiters.append($0) }
        }

        func close() {
            for waiter in waiters { waiter.resume(throwing: TransportError.notReady) }
            waiters.removeAll()
        }

        /// Queue a frame for the session to read.
        func offer(_ text: String) {
            if waiters.isEmpty {
                toPeer.append(text)
            } else {
                waiters.removeFirst().resume(returning: text)
            }
        }

        func sent() -> [String] { toServer }
    }

    private struct StubSigner: ChallengeSigner {
        var signature = Data("signed".utf8)
        func certificateChainPEM() throws -> [String] { ["-----BEGIN CERTIFICATE-----"] }
        func sign(_ material: Data) throws -> Data { signature }
    }

    private struct StubEngine: CompanionSession.Engine {
        let result: JSONValue?
        func run(workload: CompanionWorkload, payload: JSONValue, contextTokens: Int)
            async throws -> JSONValue?
        {
            result
        }
    }

    private struct FailingEngine: CompanionSession.Engine {
        struct Boom: Error {}
        func run(workload: CompanionWorkload, payload: JSONValue, contextTokens: Int)
            async throws -> JSONValue?
        {
            throw Boom()
        }
    }

    private let codec = CompanionCodec()

    private func encode<T: Encodable>(_ value: T, as type: CompanionMessageType) throws -> String {
        String(decoding: try codec.encode(value, as: type), as: UTF8.self)
    }

    private func makeOffer(workload: CompanionWorkload = .classifyIcon) throws -> String {
        let now = Date(timeIntervalSince1970: 1_786_000_000)
        let envelope = JobEnvelope(
            jobId: "job-1",
            attemptId: "attempt-1",
            idempotencyKey: "key-1",
            workload: workload,
            inputRevision: "rev-1",
            schemaVersion: 1,
            resultSchema: "\(workload.rawValue).v1",
            sensitivity: .ordinary,
            locality: .homeLan,
            traceId: "trace-1",
            createdAt: now,
            acceptDeadline: now.addingTimeInterval(2),
            completeDeadline: now.addingTimeInterval(10)
        )
        return try encode(
            JobOffer(envelope: envelope, payload: .object([:]), callbackBudget: 12, contextTokens: 4096),
            as: .jobOffer
        )
    }

    private func session(
        wire: Wire,
        engine: any CompanionSession.Engine,
        workloads: [CompanionWorkload] = [.classifyIcon]
    ) -> CompanionSession {
        CompanionSession(
            socket: wire,
            signer: StubSigner(),
            announcedId: "companion-1",
            workloads: workloads,
            engine: engine,
            telemetry: {
                CompanionTelemetry(
                    battery: 1, charging: true,
                    models: ModelAvailability(hotAvailable: true, hotContextTokens: 4096)
                )
            }
        )
    }

    func testHandshakeProvesIdentityBeforeRegistering() async throws {
        let wire = Wire()
        let session = session(wire: wire, engine: StubEngine(result: .object([:])))

        await wire.offer(
            try encode(
                AuthChallenge(
                    nonce: "nonce-value",
                    supportedVersions: [1],
                    expiresAt: Date(timeIntervalSince1970: 1_786_000_030)
                ),
                as: .authChallenge
            )
        )
        await wire.offer(
            try encode(
                HelloAck(
                    protocolVersion: 1,
                    authenticatedId: "companion-1",
                    locality: .homeLan,
                    sessionId: "session-1",
                    heartbeatSeconds: 20
                ),
                as: .helloAck
            )
        )

        try await session.handshake(appVersion: "test", osVersion: "test")

        let registered = await session.sessionId
        XCTAssertEqual(registered, "session-1")

        // The signed response must precede the hello: a peer that has not
        // proved its identity may not register one.
        let sent = await wire.sent()
        XCTAssertEqual(sent.count, 2)
        XCTAssertTrue(sent[0].contains("auth_response"))
        XCTAssertTrue(sent[1].contains("\"type\":\"hello\""))
    }

    func testAHandshakeThatDoesNotStartWithAChallengeFails() async throws {
        let wire = Wire()
        let session = session(wire: wire, engine: StubEngine(result: .object([:])))
        await wire.offer(try encode(Ping(sentAt: Date()), as: .ping))

        do {
            try await session.handshake(appVersion: "test", osVersion: "test")
            XCTFail("a session registered without proving its identity")
        } catch {
            XCTAssertEqual(
                error as? CompanionSession.SessionError,
                .handshakeFailed("expected an auth challenge")
            )
        }
    }

    func testAnOfferedJobIsAcceptedAndAnswered() async throws {
        // The whole point: work offered by Iridium is answered by the phone.
        let wire = Wire()
        let session = session(
            wire: wire, engine: StubEngine(result: .object(["icon": .string("pill")]))
        )
        await wire.offer(try makeOffer())

        let serving = Task { try await session.serve() }
        try await Task.sleep(nanoseconds: 200_000_000)
        await wire.close()
        _ = await serving.result

        let sent = await wire.sent()
        XCTAssertTrue(sent.contains { $0.contains("job_accept") })
        let result = try XCTUnwrap(sent.first { $0.contains("job_result") })
        XCTAssertTrue(result.contains("pill"))
        let accepted = await session.acceptedJobs
        XCTAssertEqual(accepted, 1)
    }

    func testAnUnsupportedWorkloadIsRejectedRatherThanAccepted() async throws {
        // Rejecting costs Iridium one round trip; accepting and then failing
        // costs it the whole completion deadline before it falls back.
        let wire = Wire()
        let session = session(
            wire: wire,
            engine: StubEngine(result: .object([:])),
            workloads: [.classifyIcon]
        )
        await wire.offer(try makeOffer(workload: .interpret))

        let serving = Task { try await session.serve() }
        try await Task.sleep(nanoseconds: 200_000_000)
        await wire.close()
        _ = await serving.result

        let sent = await wire.sent()
        XCTAssertTrue(sent.contains { $0.contains("job_reject") })
        XCTAssertFalse(sent.contains { $0.contains("job_accept") })
        let rejected = await session.rejectedJobs
        XCTAssertEqual(rejected, 1)
    }

    func testAnEngineFailureIsReportedSoIridiumCanFallBack() async throws {
        let wire = Wire()
        let session = session(wire: wire, engine: FailingEngine())
        await wire.offer(try makeOffer())

        let serving = Task { try await session.serve() }
        try await Task.sleep(nanoseconds: 200_000_000)
        await wire.close()
        _ = await serving.result

        let sent = await wire.sent()
        // Silence here would strand the turn until its deadline elapsed.
        XCTAssertTrue(sent.contains { $0.contains("job_failed") })
    }

    func testAnEngineThatDeclinesReportsFailureRatherThanEmptyResult() async throws {
        let wire = Wire()
        let session = session(wire: wire, engine: StubEngine(result: nil))
        await wire.offer(try makeOffer())

        let serving = Task { try await session.serve() }
        try await Task.sleep(nanoseconds: 200_000_000)
        await wire.close()
        _ = await serving.result

        let sent = await wire.sent()
        XCTAssertTrue(sent.contains { $0.contains("job_failed") })
        XCTAssertFalse(sent.contains { $0.contains("job_result") })
    }

    func testAnUnknownFrameDoesNotKillTheSession() async throws {
        // The server may be newer than this build. Dropping the session over a
        // frame we simply do not use would be a self-inflicted outage.
        let wire = Wire()
        let session = session(
            wire: wire, engine: StubEngine(result: .object(["icon": .string("pill")]))
        )
        await wire.offer(#"{"type":"something_from_the_future"}"#)
        await wire.offer(try makeOffer())

        let serving = Task { try await session.serve() }
        try await Task.sleep(nanoseconds: 200_000_000)
        await wire.close()
        _ = await serving.result

        let sent = await wire.sent()
        XCTAssertTrue(sent.contains { $0.contains("job_result") })
    }

    func testAPingIsAnsweredWithAHeartbeat() async throws {
        let wire = Wire()
        let session = session(wire: wire, engine: StubEngine(result: .object([:])))
        await wire.offer(try encode(Ping(sentAt: Date()), as: .ping))

        let serving = Task { try await session.serve() }
        try await Task.sleep(nanoseconds: 200_000_000)
        await wire.close()
        _ = await serving.result

        let sent = await wire.sent()
        XCTAssertTrue(sent.contains { $0.contains("heartbeat") })
    }
}
