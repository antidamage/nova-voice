import Foundation

/// Companion protocol v1, mirroring `src/nova_voice/companion/protocol.py`.
///
/// The two sides are kept in agreement by fixtures rather than by a code
/// generator: the server writes `docs/companion-protocol/*.json` from its own
/// models, and this package's tests decode those exact files. A change on
/// either side that the other cannot parse fails a test instead of failing in
/// a room at two in the morning.
public enum CompanionProtocol {
    public static let version = 1
    /// Frames larger than this are rejected before parsing on the server, so
    /// the client must not build one either.
    public static let maxFrameBytes = 256 * 1024
}

public enum CompanionWorkload: String, Codable, Sendable, CaseIterable {
    case interpret
    case renderResponse = "render_response"
    case confirmObjective = "confirm_objective"
    case extractSelfProfileUpdate = "extract_self_profile_update"
    case classifyIcon = "classify_icon"
}

public enum Sensitivity: String, Codable, Sendable {
    case ordinary, personal, location, health, mutation
}

public enum Locality: String, Codable, Sendable {
    case homeLan = "home_lan"
    case tailnet
    case other
}

public enum CompanionRole: String, Codable, Sendable {
    case companion, satellite
}

public enum RejectReason: String, Codable, Sendable {
    case battery, thermal, memory, busy, disabled
    case modelUnavailable = "model_unavailable"
    case unsupportedWorkload = "unsupported_workload"
    case unsupportedSchema = "unsupported_schema"
    case deadlineTooShort = "deadline_too_short"
}

public enum FailureReason: String, Codable, Sendable {
    case timeout, cancelled, internalError = "internal"
    case modelError = "model_error"
    case invalidResult = "invalid_result"
    case toolLimit = "tool_limit"
}

public enum ThermalState: String, Codable, Sendable {
    case nominal, fair, serious, critical
}

public enum AppState: String, Codable, Sendable {
    case foreground, background, inactive
}

public enum PermissionStatus: String, Codable, Sendable {
    case available, denied, restricted, partial, unknown
}

// MARK: - Envelope

/// Identity and lifetime carried by every job-related frame.
///
/// `attemptId` is what makes a late result detectable: a job may be retried,
/// each offer creates a new attempt, and a frame naming a stale one is ignored.
/// `idempotencyKey` is stable across attempts *and* across a local fallback,
/// so a retry cannot duplicate a downstream write.
public struct JobEnvelope: Codable, Sendable {
    public let jobId: String
    public let attemptId: String
    public let idempotencyKey: String
    public let workload: CompanionWorkload
    public let inputRevision: String
    public let schemaVersion: Int
    public let resultSchema: String
    public let sensitivity: Sensitivity
    public let locality: Locality
    public let traceId: String
    public let createdAt: Date
    public let acceptDeadline: Date
    public let completeDeadline: Date
}

// MARK: - Authentication

public struct AuthChallenge: Codable, Sendable {
    public let nonce: String
    public let supportedVersions: [Int]
    public let expiresAt: Date
}

public struct AuthResponse: Codable, Sendable {
    public let protocolVersion: Int
    public let announcedId: String
    public let roles: [CompanionRole]
    public let certificateChain: [String]
    public let signature: String

    public init(
        protocolVersion: Int = CompanionProtocol.version,
        announcedId: String,
        roles: [CompanionRole],
        certificateChain: [String],
        signature: String
    ) {
        self.protocolVersion = protocolVersion
        self.announcedId = announcedId
        self.roles = roles
        self.certificateChain = certificateChain
        self.signature = signature
    }
}

public struct HelloAck: Codable, Sendable {
    public let protocolVersion: Int
    public let authenticatedId: String
    public let locality: Locality
    public let sessionId: String
    public let heartbeatSeconds: Double
}

// MARK: - Capability and runtime state

public struct ModelAvailability: Codable, Sendable {
    public var hotAvailable: Bool
    public var hotContextTokens: Int
    public var deepAvailable: Bool
    public var deepModelId: String?

    public init(
        hotAvailable: Bool = false,
        hotContextTokens: Int = 4096,
        deepAvailable: Bool = false,
        deepModelId: String? = nil
    ) {
        self.hotAvailable = hotAvailable
        self.hotContextTokens = hotContextTokens
        self.deepAvailable = deepAvailable
        self.deepModelId = deepModelId
    }
}

public struct PermissionState: Codable, Sendable {
    public var calendar: PermissionStatus
    public var reminders: PermissionStatus
    public var location: PermissionStatus
    public var health: PermissionStatus

    public init(
        calendar: PermissionStatus = .unknown,
        reminders: PermissionStatus = .unknown,
        location: PermissionStatus = .unknown,
        health: PermissionStatus = .unknown
    ) {
        self.calendar = calendar
        self.reminders = reminders
        self.location = location
        self.health = health
    }
}

public struct CompanionTelemetry: Codable, Sendable {
    public var battery: Double
    public var charging: Bool
    public var lowPowerMode: Bool
    public var thermalState: ThermalState
    public var appState: AppState
    public var models: ModelAvailability
    public var permissions: PermissionState
    /// Diagnostic only. The server classifies locality from the peer address;
    /// a device claiming to be home cannot unlock a home-LAN route by saying so.
    public var reportedNetwork: String?

    public init(
        battery: Double = 1,
        charging: Bool = false,
        lowPowerMode: Bool = false,
        thermalState: ThermalState = .nominal,
        appState: AppState = .background,
        models: ModelAvailability = ModelAvailability(),
        permissions: PermissionState = PermissionState(),
        reportedNetwork: String? = nil
    ) {
        self.battery = battery
        self.charging = charging
        self.lowPowerMode = lowPowerMode
        self.thermalState = thermalState
        self.appState = appState
        self.models = models
        self.permissions = permissions
        self.reportedNetwork = reportedNetwork
    }
}

public struct CompanionHello: Codable, Sendable {
    public let protocolVersion: Int
    public let schemaVersions: [Int]
    public let displayName: String
    public let roles: [CompanionRole]
    public let appVersion: String
    public let osVersion: String
    public let workloads: [CompanionWorkload]
    public let personalTools: [String]
    public let telemetry: CompanionTelemetry
    /// Durable jobs this device believes it is still working on.
    ///
    /// Empty after a restart — which is the point. The server compares this
    /// with what it thinks it leased, so an empty claim from a device that
    /// crashed mid-job is how that job gets released rather than waiting out
    /// its lease.
    ///
    /// **Optional, and omitted when there is nothing to claim.** The server's
    /// message models are `extra="forbid"`, so a hello carrying a field an
    /// older server does not know is rejected and the socket closes. Sending
    /// it unconditionally broke a live device against a server whose matching
    /// change had not shipped yet — the client updates the moment it is
    /// installed, the server updates on a deploy, and the two are not the same
    /// moment. Omitting the empty case means the field only appears once there
    /// is something to say, by which time the server understands it.
    public let activeJobs: [String]?
}

// MARK: - Job lifecycle

public struct JobOffer: Codable, Sendable {
    public let envelope: JobEnvelope
    public let payload: JSONValue
    public let callbackBudget: Int
    public let contextTokens: Int
    /// Exactly the tools this attempt may call back for, as `provider.tool`.
    ///
    /// The device is told the catalogue rather than left to infer it from the
    /// prompt. Nothing here is a security boundary — Iridium refuses anything
    /// outside the list regardless of what the device believes — but naming a
    /// tool that will certainly be refused wastes a round trip and a slice of
    /// the callback budget, so the runtime should read this rather than guess.
    ///
    /// Optional because these arrived after the first committed fixtures: an
    /// offer from an older server decodes with them absent rather than
    /// failing the whole frame.
    public let toolCatalogue: [String]?
    public let callbackDeadlineSeconds: Double?
    public let callbackBudgetSeconds: Double?
    public let maxConcurrentCallbacks: Int?
}

public struct JobAccept: Codable, Sendable {
    public enum Tier: String, Codable, Sendable { case hot, deep }
    public let jobId: String
    public let attemptId: String
    public let tier: Tier
}

public struct JobReject: Codable, Sendable {
    public let jobId: String
    public let attemptId: String
    public let reason: RejectReason
    public let detail: String?
    public let retryAfterSeconds: Double?
}

public struct JobProgress: Codable, Sendable {
    public let jobId: String
    public let attemptId: String
    /// Monotonic within an attempt; a lower sequence is a duplicate.
    public let sequence: Int
    public let stage: String
    public let fraction: Double?
    public let summary: String?
}

public struct JobResult: Codable, Sendable {
    public let jobId: String
    public let attemptId: String
    public let result: JSONValue
    public let tokensUsed: Int?
}

public struct JobFailed: Codable, Sendable {
    public let jobId: String
    public let attemptId: String
    public let reason: FailureReason
    public let detail: String?
}

public struct JobCancel: Codable, Sendable {
    public enum Reason: String, Codable, Sendable {
        case superseded, deadline, shutdown
        case userCancelled = "user_cancelled"
        case localityLost = "locality_lost"
    }
    public let jobId: String
    public let attemptId: String
    public let reason: Reason
}

// MARK: - Tool callbacks

public struct ToolCall: Codable, Sendable {
    public let jobId: String
    public let attemptId: String
    public let callId: String
    public let provider: String
    public let tool: String
    public let arguments: JSONValue
}

public struct ToolResult: Codable, Sendable {
    public let jobId: String
    public let attemptId: String
    public let callId: String
    public let ok: Bool
    public let code: String
    public let message: String
    public let observed: JSONValue?
    public let sensitivity: Sensitivity
}

// MARK: - Approvals

public struct ApprovalRequest: Codable, Sendable {
    public let approvalId: String
    public let jobId: String?
    public let summary: String
    public let provider: String
    public let tool: String
    public let sensitivity: Sensitivity
    public let expiresAt: Date
}

public struct ApprovalResponse: Codable, Sendable {
    public let approvalId: String
    public let approved: Bool
    public let signature: String
}

// MARK: - Personal context

public struct PersonalCall: Codable, Sendable {
    public let callId: String
    public let tool: String
    public let arguments: JSONValue
    public let maxItems: Int
    public let deadlineSeconds: Double
}

public struct PersonalResult: Codable, Sendable {
    public let callId: String
    public let ok: Bool
    public let code: String
    public let message: String
    public let sensitivity: Sensitivity
    public let items: [JSONValue]
    public let truncated: Bool

    /// Explicit and public because the app builds these, and a synthesised
    /// memberwise initialiser on a public struct is only internal.
    public init(
        callId: String,
        ok: Bool,
        code: String,
        message: String,
        sensitivity: Sensitivity,
        items: [JSONValue],
        truncated: Bool
    ) {
        self.callId = callId
        self.ok = ok
        self.code = code
        self.message = message
        self.sensitivity = sensitivity
        self.items = items
        self.truncated = truncated
    }
}

// MARK: - Housekeeping

public struct Heartbeat: Codable, Sendable {
    public let sentAt: Date
    public init(sentAt: Date = Date()) { self.sentAt = sentAt }
}

public struct Ping: Codable, Sendable {
    public let sentAt: Date
}

public struct ConfigurationChanged: Codable, Sendable {
    /// Named settings only — never the values, which may be secret.
    public let changed: [String]
}

public struct TelemetryMessage: Codable, Sendable {
    public let telemetry: CompanionTelemetry
}

public struct LogEvent: Codable, Sendable {
    public enum Level: String, Codable, Sendable { case debug, info, warning, error }
    public let level: Level
    public let event: String
    /// Structural only. Never event titles, reminder notes, locations or
    /// Health values.
    public let detail: String?
    public let traceId: String?
}
