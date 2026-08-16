import Foundation

/// Runs one companion session: authenticate, register, then serve jobs.
///
/// This is the piece that makes an offload actually happen. Everything else
/// was scaffolding for it — Iridium could already offer work, and the app
/// could already speak the wire format, but nothing joined the two.
///
/// The reasoning engine is injected so the session can be driven end to end
/// against a stub, without a model, a device, or Apple's availability rules.
public actor CompanionSession {
    /// Whatever actually answers a job. `SystemLanguageModel` in the app; a
    /// deterministic stub in tests.
    public protocol Engine: Sendable {
        /// Return the workload's result payload, or nil to reject the job.
        ///
        /// `tools` is the bounded way back into the house. An engine that only
        /// answers a typed question ignores it; one that needs to look
        /// something up mid-reasoning calls it and waits.
        func run(
            workload: CompanionWorkload,
            payload: JSONValue,
            contextTokens: Int,
            tools: any CompanionTools
        ) async throws -> JSONValue?
    }

    public enum SessionError: Error, Equatable {
        case handshakeFailed(String)
        case rejected(String)
    }

    /// Answers Calendar, Reminders, location and Health calls.
    ///
    /// Separate from `Engine` on purpose: reasoning and personal context are
    /// different permissions, different failure modes and different reasons to
    /// be unavailable. A device with no calendar access should still be able
    /// to answer a reasoning job.
    public protocol PersonalHandler: Sendable {
        func run(_ call: PersonalCall) async throws -> PersonalResult
    }

    /// One job's callback state: the work itself, and every call waiting on it.
    private struct RunningJob {
        /// The durable identity, kept alongside the attempt so a cancel that
        /// names only the job can still find it.
        let jobId: String
        var task: Task<Void, Never>?
        var pending: [String: CheckedContinuation<ToolResult, Error>] = [:]
        var catalogue: [String]
    }

    private let socket: any CompanionSocket
    private let signer: ChallengeSigner
    private let announcedId: String
    private let roles: [CompanionRole]
    private let engine: any Engine
    private let codec = CompanionCodec()
    private let telemetry: @Sendable () -> CompanionTelemetry
    private let workloads: [CompanionWorkload]
    private let selfCheck: CompanionSelfCheck
    private let diagnostics: DiagnosticRing?
    private let personal: (any PersonalHandler)?

    public private(set) var sessionId: String?
    public private(set) var acceptedJobs = 0
    public private(set) var rejectedJobs = 0
    public private(set) var cancelledJobs = 0

    private var running: [String: RunningJob] = [:]
    private var personalTasks: [String: Task<Void, Never>] = [:]

    public init(
        socket: any CompanionSocket,
        signer: ChallengeSigner,
        announcedId: String,
        roles: [CompanionRole] = [.companion],
        workloads: [CompanionWorkload],
        engine: any Engine,
        telemetry: @escaping @Sendable () -> CompanionTelemetry,
        selfCheck: CompanionSelfCheck = CompanionSelfCheck(),
        diagnostics: DiagnosticRing? = nil,
        personal: (any PersonalHandler)? = nil
    ) {
        self.selfCheck = selfCheck
        self.diagnostics = diagnostics
        self.personal = personal
        self.socket = socket
        self.signer = signer
        self.announcedId = announcedId
        self.roles = roles
        self.workloads = workloads
        self.engine = engine
        self.telemetry = telemetry
    }

    /// Complete the handshake. Throws rather than returning a half-open
    /// session: a companion that failed to prove its identity must not go on
    /// to accept jobs.
    public func handshake(appVersion: String, osVersion: String) async throws {
        guard case .authChallenge(let challenge) = try codec.decode(
            Data(try await socket.receive().utf8)
        ) else {
            throw SessionError.handshakeFailed("expected an auth challenge")
        }

        let material = CompanionChallenge.material(
            nonce: challenge.nonce,
            protocolVersion: CompanionProtocol.version,
            announcedId: announcedId,
            roles: roles
        )
        let response = AuthResponse(
            announcedId: announcedId,
            roles: roles,
            certificateChain: try signer.certificateChainPEM(),
            signature: try signer.sign(material).base64EncodedString()
        )
        try await socket.send(
            String(decoding: try codec.encode(response, as: .authResponse), as: UTF8.self)
        )

        let hello = CompanionHello(
            protocolVersion: CompanionProtocol.version,
            schemaVersions: [1],
            displayName: "Nova Companion",
            roles: roles,
            appVersion: appVersion,
            osVersion: osVersion,
            workloads: workloads,
            personalTools: [],
            telemetry: telemetry(),
            // What this process still holds. Omitted entirely when empty, so a
            // server that predates the field is not sent something its strict
            // models will reject — see the note on `CompanionHello`.
            activeJobs: running.isEmpty ? nil : running.values.map(\.jobId)
        )
        try await socket.send(
            String(decoding: try codec.encode(hello, as: .hello), as: UTF8.self)
        )

        guard case .helloAck(let ack) = try codec.decode(
            Data(try await socket.receive().utf8)
        ) else {
            throw SessionError.handshakeFailed("expected a hello acknowledgement")
        }
        sessionId = ack.sessionId
        await diagnostics?.record(.auth, "registered", detail: "session \(ack.sessionId)")
    }

    /// Serve frames until the socket closes. Returns when the peer goes away.
    public func serve() async throws {
        while true {
            let raw: String
            do {
                raw = try await socket.receive()
            } catch {
                return
            }
            let message: CompanionMessage
            do {
                message = try codec.decode(Data(raw.utf8))
            } catch {
                // A frame this build does not understand is not fatal: the
                // server may be newer. Ignore it rather than dropping a
                // session that is otherwise working.
                continue
            }
            switch message {
            case .jobOffer(let offer):
                // Started, not awaited. A job that blocks the receive loop can
                // never see its own `job_cancel` or the `tool_result` it is
                // waiting for — the frames would sit unread behind the work
                // that needs them. This is what makes cancellation and the
                // callback loop possible at all.
                start(offer)
            case .ping:
                try? await send(Heartbeat(sentAt: Date()), as: .heartbeat)
            case .jobCancel(let cancel):
                cancelJob(attemptId: cancel.attemptId, jobId: cancel.jobId)
            case .toolResult(let result):
                deliver(result)
            case .personalCall(let call):
                // Started, not awaited, for the same reason jobs are: a
                // calendar query that blocks the receive loop would also block
                // the cancel and the heartbeat behind it.
                servePersonalCall(call)
            default:
                continue
            }
        }
    }

    // MARK: - Job lifecycle

    private func start(_ offer: JobOffer) {
        let attemptId = offer.envelope.attemptId
        running[attemptId] = RunningJob(
            jobId: offer.envelope.jobId, catalogue: offer.toolCatalogue ?? []
        )
        let task = Task { [weak self] in
            await self?.handle(offer)
            await self?.finish(attemptId: attemptId)
        }
        // The offer may already have finished and cleaned up by the time this
        // returns, so only record the task if the entry is still there.
        if running[attemptId] != nil {
            running[attemptId]?.task = task
        }
    }

    private func finish(attemptId: String) {
        guard let job = running.removeValue(forKey: attemptId) else { return }
        // Anything still waiting is waiting for a turn that is over.
        for (_, continuation) in job.pending {
            continuation.resume(throwing: CancellationError())
        }
    }

    /// Cancel by attempt, or by job when the server does not name an attempt.
    ///
    /// Reconnect reconciliation cancels by *job*: the server has decided this
    /// device should not be working on it, and which attempt the device
    /// happens to hold is beside the point — and after a restart the server no
    /// longer knows. Matching on either is what makes that message land.
    private func cancelJob(attemptId: String, jobId: String? = nil) {
        var key: String? = running[attemptId] != nil ? attemptId : nil
        if key == nil, let jobId {
            key = running.first { $0.value.jobId == jobId }?.key
        }
        guard let key, let job = running[key] else { return }
        let attemptId = key
        cancelledJobs += 1
        Task { await diagnostics?.record(.job, "cancelled", detail: attemptId) }
        job.task?.cancel()
        // Failing the pending calls is what actually unblocks the engine: a
        // cancelled Task does not resume a continuation on its own, so an
        // engine parked on a callback would otherwise wait for a `tool_result`
        // Iridium has already stopped intending to send.
        finish(attemptId: attemptId)
    }

    private func deliver(_ result: ToolResult) {
        guard let continuation = running[result.attemptId]?.pending
            .removeValue(forKey: result.callId)
        else {
            // A result for a call that was cancelled, or already answered.
            return
        }
        continuation.resume(returning: result)
    }

    /// Issue one tool call and wait for Iridium's answer.
    ///
    /// The catalogue is checked here as well as on the server. That is not
    /// defence in depth — Iridium's check is the real one and this cannot
    /// weaken it — it just saves a round trip and a slice of the callback
    /// budget on a call that would certainly be refused.
    fileprivate func performToolCall(
        attemptId: String,
        jobId: String,
        provider: String,
        tool: String,
        arguments: JSONValue
    ) async throws -> ToolResult {
        guard let job = running[attemptId] else { throw CancellationError() }
        guard job.catalogue.contains("\(provider).\(tool)") || job.catalogue.contains(tool) else {
            throw ToolCallError.notOffered("\(provider).\(tool)")
        }
        let callId = UUID().uuidString
        try await send(
            ToolCall(
                jobId: jobId,
                attemptId: attemptId,
                callId: callId,
                provider: provider,
                tool: tool,
                arguments: arguments
            ),
            as: .toolCall
        )
        return try await withCheckedThrowingContinuation { continuation in
            guard running[attemptId] != nil else {
                return continuation.resume(throwing: CancellationError())
            }
            running[attemptId]?.pending[callId] = continuation
        }
    }

    private func handle(_ offer: JobOffer) async {
        let envelope = offer.envelope

        // Decide before accepting. Accepting and then failing costs Iridium the
        // whole completion deadline before it falls back, whereas a rejection
        // costs it one round trip — so an engine that cannot serve this
        // workload must say so now.
        guard workloads.contains(envelope.workload) else {
            await reject(envelope, reason: .unsupportedWorkload, detail: nil, retryAfter: nil)
            return
        }

        // Re-read the device *now* rather than trusting the telemetry the
        // server decided on. An offer can arrive in the gap between a reading
        // and reality changing — the phone reports nominal thermals, goes into
        // a pocket in the sun, and is offered work against a reading that is
        // already wrong.
        let state = telemetry()
        if let refusal = selfCheck.refusal(for: state, workload: envelope.workload) {
            await reject(
                envelope,
                reason: refusal.reason,
                detail: refusal.detail,
                retryAfter: refusal.retryAfter
            )
            return
        }

        acceptedJobs += 1
        await diagnostics?.record(.job, "accepted", detail: envelope.workload.rawValue)
        try? await send(
            JobAccept(jobId: envelope.jobId, attemptId: envelope.attemptId, tier: .hot),
            as: .jobAccept
        )

        do {
            guard
                let result = try await engine.run(
                    workload: envelope.workload,
                    payload: offer.payload,
                    contextTokens: offer.contextTokens,
                    tools: SessionTools(
                        session: self,
                        jobId: envelope.jobId,
                        attemptId: envelope.attemptId,
                        catalogue: offer.toolCatalogue ?? []
                    )
                )
            else {
                try await send(
                    JobFailed(
                        jobId: envelope.jobId,
                        attemptId: envelope.attemptId,
                        reason: .modelError,
                        detail: "engine produced no result"
                    ),
                    as: .jobFailed
                )
                return
            }
            // The device may have got into trouble while this was running. A
            // long job finishing on an overheating phone makes the overheating
            // worse, and the answer is not worth that — Iridium can run it
            // locally at no cost to this device.
            if telemetry().degraded(from: state) {
                try await send(
                    JobFailed(
                        jobId: envelope.jobId,
                        attemptId: envelope.attemptId,
                        reason: .internalError,
                        detail: "device state degraded while the job was running"
                    ),
                    as: .jobFailed
                )
                return
            }
            try await send(
                JobResult(
                    jobId: envelope.jobId,
                    attemptId: envelope.attemptId,
                    result: result,
                    tokensUsed: nil
                ),
                as: .jobResult
            )
        } catch {
            // A cancelled job says nothing. Iridium cancelled it, has already
            // moved on, and a `job_failed` arriving afterwards would be a
            // second terminal frame for an attempt that already has one.
            if error is CancellationError || Task.isCancelled { return }
            try? await send(
                JobFailed(
                    jobId: envelope.jobId,
                    attemptId: envelope.attemptId,
                    reason: .modelError,
                    detail: String(describing: error).prefix(200).description
                ),
                as: .jobFailed
            )
        }
    }

    // MARK: - Personal context

    private func servePersonalCall(_ call: PersonalCall) {
        let task = Task { [weak self] in
            guard let self else { return }
            await self.answerPersonalCall(call)
            await self.finishedPersonalCall(call.callId)
        }
        personalTasks[call.callId] = task
    }

    private func finishedPersonalCall(_ callId: String) {
        personalTasks.removeValue(forKey: callId)
    }

    private func answerPersonalCall(_ call: PersonalCall) async {
        guard let handler = personal else {
            await send(
                PersonalResult(
                    callId: call.callId,
                    ok: false,
                    code: "unavailable",
                    message: "this device has no personal-context runtime",
                    sensitivity: .personal,
                    items: [],
                    truncated: false
                )
            )
            return
        }
        do {
            let answer = try await handler.run(call)
            await send(answer)
        } catch {
            // Reported, never silent: Nova is waiting on this call with a
            // deadline, and a device that says nothing costs the whole
            // deadline before the caller learns anything.
            await send(
                PersonalResult(
                    callId: call.callId,
                    ok: false,
                    code: "backend_error",
                    // The error *type*, not its description: a description can
                    // name a calendar or a list, and this frame is meant to
                    // carry only what went wrong.
                    message: String(describing: type(of: error)),
                    sensitivity: .personal,
                    items: [],
                    truncated: false
                )
            )
        }
    }

    private func send(_ result: PersonalResult) async {
        try? await send(result, as: .personalResult)
    }

    private func reject(
        _ envelope: JobEnvelope,
        reason: RejectReason,
        detail: String?,
        retryAfter: Double?
    ) async {
        rejectedJobs += 1
        await diagnostics?.record(
            .job,
            "rejected",
            // The workload and the reason, which is all anyone needs to tell
            // "it declined on battery" from "it does not do that workload".
            detail: "\(envelope.workload.rawValue) \(reason.rawValue)"
        )
        try? await send(
            JobReject(
                jobId: envelope.jobId,
                attemptId: envelope.attemptId,
                reason: reason,
                detail: detail,
                retryAfterSeconds: retryAfter
            ),
            as: .jobReject
        )
    }

    public func sendTelemetry() async throws {
        try await send(TelemetryMessage(telemetry: telemetry()), as: .telemetry)
    }

    /// Keep the server's picture of this device from going stale.
    ///
    /// Iridium treats telemetry older than its staleness window as *no
    /// evidence of health* rather than as continued health, and stops offering
    /// work. That is the right rule — a device that has gone quiet may be
    /// asleep, wedged, or gone — but nothing was driving the other half of it.
    /// The handshake sent one reading and the device then went silent, so the
    /// companion became ineligible three minutes after every connect and
    /// stayed that way while plugged in and idle in the same room. Measured
    /// live on 2026-08-16: `tier: off, telemetry is 995s stale`, every route
    /// reporting `tier off is below reduced`, on a charging phone.
    ///
    /// The interval must leave room for a missed frame or two inside that
    /// window, so it is a fraction of it rather than just under it.
    ///
    /// This stops when the OS suspends the app, and that is honest rather than
    /// a defect: a suspended app genuinely cannot answer a job, so going
    /// ineligible is the correct outcome. Keeping the device eligible while
    /// suspended would need a background mode, which is M6's audio-session
    /// work, not a timer.
    public func maintainTelemetry(
        everySeconds interval: Double,
        onSent: (@Sendable () -> Void)? = nil
    ) async {
        while !Task.isCancelled {
            try? await Task.sleep(nanoseconds: UInt64(interval * 1_000_000_000))
            if Task.isCancelled { return }
            do {
                try await sendTelemetry()
                onSent?()
            } catch {
                // The socket has gone. Reconnecting is the coordinator's job;
                // looping here would just fail forever against a dead socket.
                return
            }
        }
    }

    private func send<T: Encodable>(_ value: T, as type: CompanionMessageType) async throws {
        try await socket.send(
            String(decoding: try codec.encode(value, as: type), as: UTF8.self)
        )
    }
}

/// The bounded loop back into Nova, handed to an engine for one job.
///
/// Deliberately small. The device may *ask*; every decision about whether the
/// action is allowed, needs approval, or may run at all stays on Iridium, which
/// is also why nothing here carries a credential.
public protocol CompanionTools: Sendable {
    /// Exactly what this job may call, as `provider.tool`. Empty means the job
    /// was offered no tools, which is the ordinary case.
    var catalogue: [String] { get }

    func call(provider: String, tool: String, arguments: JSONValue) async throws -> ToolResult
}

public enum ToolCallError: Error, Equatable {
    /// Named a tool this job was not offered. Iridium would refuse it too.
    case notOffered(String)
}

struct SessionTools: CompanionTools {
    let session: CompanionSession
    let jobId: String
    let attemptId: String
    let catalogue: [String]

    func call(provider: String, tool: String, arguments: JSONValue) async throws -> ToolResult {
        try await session.performToolCall(
            attemptId: attemptId,
            jobId: jobId,
            provider: provider,
            tool: tool,
            arguments: arguments
        )
    }
}

/// Signs the server's challenge with the device's private key.
public protocol ChallengeSigner: Sendable {
    func certificateChainPEM() throws -> [String]
    func sign(_ material: Data) throws -> Data
}
