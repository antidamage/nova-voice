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
        func run(workload: CompanionWorkload, payload: JSONValue, contextTokens: Int) async throws
            -> JSONValue?
    }

    public enum SessionError: Error, Equatable {
        case handshakeFailed(String)
        case rejected(String)
    }

    private let socket: any CompanionSocket
    private let signer: ChallengeSigner
    private let announcedId: String
    private let roles: [CompanionRole]
    private let engine: any Engine
    private let codec = CompanionCodec()
    private let telemetry: @Sendable () -> CompanionTelemetry
    private let workloads: [CompanionWorkload]

    public private(set) var sessionId: String?
    public private(set) var acceptedJobs = 0
    public private(set) var rejectedJobs = 0

    public init(
        socket: any CompanionSocket,
        signer: ChallengeSigner,
        announcedId: String,
        roles: [CompanionRole] = [.companion],
        workloads: [CompanionWorkload],
        engine: any Engine,
        telemetry: @escaping @Sendable () -> CompanionTelemetry
    ) {
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
            telemetry: telemetry()
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
                await handle(offer)
            case .ping:
                try? await send(Heartbeat(sentAt: Date()), as: .heartbeat)
            case .jobCancel:
                // Nothing to unwind yet: jobs are answered inline, so a cancel
                // can only arrive for work already finished.
                continue
            default:
                continue
            }
        }
    }

    private func handle(_ offer: JobOffer) async {
        let envelope = offer.envelope

        // Decide before accepting. Accepting and then failing costs Iridium the
        // whole completion deadline before it falls back, whereas a rejection
        // costs it one round trip — so an engine that cannot serve this
        // workload must say so now.
        guard workloads.contains(envelope.workload) else {
            rejectedJobs += 1
            try? await send(
                JobReject(
                    jobId: envelope.jobId,
                    attemptId: envelope.attemptId,
                    reason: .unsupportedWorkload,
                    detail: nil,
                    retryAfterSeconds: nil
                ),
                as: .jobReject
            )
            return
        }

        acceptedJobs += 1
        try? await send(
            JobAccept(jobId: envelope.jobId, attemptId: envelope.attemptId, tier: .hot),
            as: .jobAccept
        )

        do {
            guard
                let result = try await engine.run(
                    workload: envelope.workload,
                    payload: offer.payload,
                    contextTokens: offer.contextTokens
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

    public func sendTelemetry() async throws {
        try await send(TelemetryMessage(telemetry: telemetry()), as: .telemetry)
    }

    private func send<T: Encodable>(_ value: T, as type: CompanionMessageType) async throws {
        try await socket.send(
            String(decoding: try codec.encode(value, as: type), as: UTF8.self)
        )
    }
}

/// Signs the server's challenge with the device's private key.
public protocol ChallengeSigner: Sendable {
    func certificateChainPEM() throws -> [String]
    func sign(_ material: Data) throws -> Data
}
