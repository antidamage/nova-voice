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

        private var sendsFail = false

        func send(_ text: String) async throws {
            if sendsFail { throw TransportError.notReady }
            toServer.append(text)
        }

        /// Stand in for a socket that has gone away without being closed.
        func failSends() { sendsFail = true }

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
        func run(
            workload: CompanionWorkload,
            payload: JSONValue,
            contextTokens: Int,
            tools: any CompanionTools
        ) async throws -> JSONValue? {
            result
        }
    }

    private struct FailingEngine: CompanionSession.Engine {
        struct Boom: Error {}
        func run(
            workload: CompanionWorkload,
            payload: JSONValue,
            contextTokens: Int,
            tools: any CompanionTools
        ) async throws -> JSONValue? {
            throw Boom()
        }
    }

    /// Calls one tool, then answers with whatever came back.
    private struct ToolUsingEngine: CompanionSession.Engine {
        let provider: String
        let tool: String
        let outcome: Outcome

        final class Outcome: @unchecked Sendable {
            var error: Error?
            var observedCode: String?
            var finished = false
        }

        func run(
            workload: CompanionWorkload,
            payload: JSONValue,
            contextTokens: Int,
            tools: any CompanionTools
        ) async throws -> JSONValue? {
            do {
                let result = try await tools.call(
                    provider: provider, tool: tool, arguments: .object([:])
                )
                outcome.observedCode = result.code
            } catch {
                outcome.error = error
                throw error
            }
            outcome.finished = true
            return .object(["icon": .string("pill")])
        }
    }

    private let codec = CompanionCodec()

    private func encode<T: Encodable>(_ value: T, as type: CompanionMessageType) throws -> String {
        String(decoding: try codec.encode(value, as: type), as: UTF8.self)
    }

    private func makeOffer(
        workload: CompanionWorkload = .classifyIcon,
        catalogue: [String]? = nil
    ) throws -> String {
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
            JobOffer(
                envelope: envelope,
                payload: .object([:]),
                callbackBudget: 12,
                contextTokens: 4096,
                toolCatalogue: catalogue,
                callbackDeadlineSeconds: nil,
                callbackBudgetSeconds: nil,
                maxConcurrentCallbacks: nil
            ),
            as: .jobOffer
        )
    }

    private func session(
        wire: Wire,
        engine: any CompanionSession.Engine,
        workloads: [CompanionWorkload] = [.classifyIcon],
        telemetry: @escaping @Sendable () -> CompanionTelemetry = {
            CompanionTelemetry(
                battery: 1, charging: true,
                models: ModelAvailability(hotAvailable: true, hotContextTokens: 4096)
            )
        }
    ) -> CompanionSession {
        CompanionSession(
            socket: wire,
            signer: StubSigner(),
            announcedId: "companion-1",
            workloads: workloads,
            engine: engine,
            telemetry: telemetry
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

    // MARK: - Self-check before accepting (NPT-702)

    func testAHotPhoneDeclinesRatherThanAcceptingAndStruggling() async throws {
        // Rejecting costs Iridium one round trip. Accepting and then failing
        // costs it the job's whole completion deadline, because the attempt is
        // held until it times out.
        let wire = Wire()
        let session = session(
            wire: wire,
            engine: StubEngine(result: .object(["icon": .string("pill")])),
            telemetry: {
                CompanionTelemetry(
                    battery: 1,
                    charging: true,
                    thermalState: .critical,
                    models: ModelAvailability(hotAvailable: true, hotContextTokens: 4096)
                )
            }
        )
        await wire.offer(try makeOffer())

        let serving = Task { try await session.serve() }
        try await Task.sleep(nanoseconds: 200_000_000)
        await wire.close()
        _ = await serving.result

        let sent = await wire.sent()
        let reject = try XCTUnwrap(sent.first { $0.contains("job_reject") })
        XCTAssertTrue(reject.contains("\"thermal\""))
        XCTAssertFalse(sent.contains { $0.contains("job_accept") })
    }

    func testAFlatPhoneDeclinesWithARetryHint() async throws {
        // Worth re-offering, but only once something has charged — so the
        // reason carries a hint rather than leaving the server to guess.
        let wire = Wire()
        let session = session(
            wire: wire,
            engine: StubEngine(result: .object([:])),
            telemetry: {
                CompanionTelemetry(
                    battery: 0.05,
                    charging: false,
                    models: ModelAvailability(hotAvailable: true, hotContextTokens: 4096)
                )
            }
        )
        await wire.offer(try makeOffer())

        let serving = Task { try await session.serve() }
        try await Task.sleep(nanoseconds: 200_000_000)
        await wire.close()
        _ = await serving.result

        let sent = await wire.sent()
        let reject = try XCTUnwrap(sent.first { $0.contains("job_reject") })
        XCTAssertTrue(reject.contains("\"battery\""))
        XCTAssertTrue(reject.contains("retryAfterSeconds"))
    }

    func testAHealthyPhoneStillAccepts() async throws {
        // The self-check must not become a reason nothing ever runs.
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
        XCTAssertTrue(sent.contains { $0.contains("job_result") })
    }

    func testTheSelfCheckAgreesWithTheServerAboutTheBatteryFloor() {
        // Two components disagreeing about "too flat to help" would produce a
        // device that is offered work it always refuses.
        let check = CompanionSelfCheck()
        XCTAssertEqual(check.batteryFloor, 0.15)
    }

    func testDegradationIsDetectedButImprovementIsNot() {
        let cool = CompanionTelemetry(battery: 1, charging: true, thermalState: .nominal)
        let hot = CompanionTelemetry(battery: 1, charging: true, thermalState: .serious)

        XCTAssertTrue(hot.degraded(from: cool))
        XCTAssertFalse(cool.degraded(from: hot))
        XCTAssertFalse(cool.degraded(from: cool))
    }

    func testComingOffChargeCountsAsDegradation() {
        let plugged = CompanionTelemetry(battery: 0.4, charging: true)
        let unplugged = CompanionTelemetry(battery: 0.4, charging: false)

        XCTAssertTrue(unplugged.degraded(from: plugged))
    }

    // MARK: - Liveness (NPT-205)

    func testTelemetryKeepsArrivingRatherThanStoppingAfterTheHandshake() async throws {
        // The regression this exists for: the handshake sent one reading and
        // the device then went silent, so Iridium aged it out of the routing
        // table three minutes later. Connected, advertising every workload,
        // and offered nothing — measured live at 995s stale while charging.
        let wire = Wire()
        let session = session(wire: wire, engine: StubEngine(result: .object([:])))

        let pump = Task { await session.maintainTelemetry(everySeconds: 0.05) }
        try await Task.sleep(nanoseconds: 300_000_000)
        pump.cancel()

        let sent = await wire.sent()
        let readings = sent.filter { $0.contains("\"type\":\"telemetry\"") }
        XCTAssertGreaterThanOrEqual(readings.count, 2, "telemetry stopped after the first send")
    }

    func testTheTelemetryLoopStopsWhenTheSocketGoesAway() async throws {
        // Reconnecting is the coordinator's job. Looping here would spin
        // against a dead socket for as long as the app stayed alive.
        let wire = Wire()
        let session = session(wire: wire, engine: StubEngine(result: .object([:])))
        await wire.failSends()

        let pump = Task { await session.maintainTelemetry(everySeconds: 0.05) }
        // If the loop gives up on a dead socket it returns; if it does not,
        // the timeout wins and this fails.
        let exited = await withTaskGroup(of: Bool.self) { group in
            group.addTask {
                await pump.value
                return true
            }
            group.addTask {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                return false
            }
            let first = await group.next() ?? false
            group.cancelAll()
            return first
        }
        pump.cancel()

        let sent = await wire.sent()
        XCTAssertTrue(sent.isEmpty)
        XCTAssertTrue(exited, "the loop kept retrying against a dead socket")
    }

    // MARK: - Tool callbacks and cancellation (NPT-308, NPT-309)

    func testAnEngineCanCallAToolAndWaitForIridiumsAnswer() async throws {
        // The loop that makes reasoning on the phone useful for anything more
        // than a single typed question.
        let wire = Wire()
        let outcome = ToolUsingEngine.Outcome()
        let engine = ToolUsingEngine(provider: "nova", tool: "light_set", outcome: outcome)
        let session = session(wire: wire, engine: engine)
        await wire.offer(try makeOffer(catalogue: ["nova.light_set"]))

        let serving = Task { try await session.serve() }
        try await Task.sleep(nanoseconds: 200_000_000)

        // The job is parked on its callback, so the receive loop must still be
        // reading — this frame proves it is.
        let parked = await wire.sent()
        let call = try XCTUnwrap(parked.first { $0.contains("tool_call") })
        let callId = try XCTUnwrap(
            (try JSONSerialization.jsonObject(with: Data(call.utf8)) as? [String: Any])?["callId"]
                as? String
        )
        await wire.offer(
            try encode(
                ToolResult(
                    jobId: "job-1",
                    attemptId: "attempt-1",
                    callId: callId,
                    ok: true,
                    code: "ok",
                    message: "done",
                    observed: nil,
                    sensitivity: .ordinary
                ),
                as: .toolResult
            )
        )
        try await Task.sleep(nanoseconds: 300_000_000)
        await wire.close()
        _ = await serving.result

        XCTAssertEqual(outcome.observedCode, "ok")
        XCTAssertTrue(outcome.finished)
        let sent = await wire.sent()
        XCTAssertTrue(sent.contains { $0.contains("job_result") })
    }

    func testAToolOutsideTheOfferedCatalogueIsNotEvenSent() async throws {
        // Iridium would refuse it. Sending it anyway spends a round trip and a
        // slice of the callback budget to be told so.
        let wire = Wire()
        let outcome = ToolUsingEngine.Outcome()
        let engine = ToolUsingEngine(provider: "nova", tool: "unlock_door", outcome: outcome)
        let session = session(wire: wire, engine: engine)
        await wire.offer(try makeOffer(catalogue: ["nova.light_set"]))

        let serving = Task { try await session.serve() }
        try await Task.sleep(nanoseconds: 200_000_000)
        await wire.close()
        _ = await serving.result

        XCTAssertEqual(outcome.error as? ToolCallError, .notOffered("nova.unlock_door"))
        let sent = await wire.sent()
        XCTAssertFalse(sent.contains { $0.contains("tool_call") })
    }

    func testCancellingAJobUnblocksAnEngineWaitingOnACallback() async throws {
        // Without this the engine waits for a `tool_result` Iridium has
        // already stopped intending to send, and the job never ends.
        let wire = Wire()
        let outcome = ToolUsingEngine.Outcome()
        let engine = ToolUsingEngine(provider: "nova", tool: "light_set", outcome: outcome)
        let session = session(wire: wire, engine: engine)
        await wire.offer(try makeOffer(catalogue: ["nova.light_set"]))

        let serving = Task { try await session.serve() }
        try await Task.sleep(nanoseconds: 200_000_000)
        let parked = await wire.sent()
        XCTAssertTrue(parked.contains { $0.contains("tool_call") })

        await wire.offer(
            try encode(
                JobCancel(jobId: "job-1", attemptId: "attempt-1", reason: .superseded),
                as: .jobCancel
            )
        )
        try await Task.sleep(nanoseconds: 300_000_000)
        await wire.close()
        _ = await serving.result

        XCTAssertTrue(outcome.error is CancellationError)
        XCTAssertFalse(outcome.finished)
        let cancelled = await session.cancelledJobs
        XCTAssertEqual(cancelled, 1)
        // A cancelled attempt already has its terminal frame on Iridium's side.
        let sent = await wire.sent()
        XCTAssertFalse(sent.contains { $0.contains("job_failed") })
        XCTAssertFalse(sent.contains { $0.contains("job_result") })
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
