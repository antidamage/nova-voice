import Foundation

/// Every message type on the wire, by its `type` discriminator.
public enum CompanionMessageType: String, Codable, Sendable, CaseIterable {
    case authChallenge = "auth_challenge"
    case authResponse = "auth_response"
    case hello
    case helloAck = "hello_ack"
    case telemetry
    case heartbeat
    case ping
    case configurationChanged = "configuration_changed"
    case jobOffer = "job_offer"
    case jobAccept = "job_accept"
    case jobReject = "job_reject"
    case jobProgress = "job_progress"
    case jobResult = "job_result"
    case jobFailed = "job_failed"
    case jobCancel = "job_cancel"
    case toolCall = "tool_call"
    case toolResult = "tool_result"
    case approvalRequest = "approval_request"
    case approvalResponse = "approval_response"
    case personalCall = "personal_call"
    case personalResult = "personal_result"
    case logEvent = "log_event"
}

/// One decoded frame.
///
/// Unknown message types are rejected rather than ignored. A frame nobody
/// understands is a protocol mismatch, and continuing as though it had not
/// arrived is how two sides end up quietly disagreeing about state.
public enum CompanionMessage: Sendable {
    case authChallenge(AuthChallenge)
    case authResponse(AuthResponse)
    case hello(CompanionHello)
    case helloAck(HelloAck)
    case telemetry(TelemetryMessage)
    case heartbeat(Heartbeat)
    case ping(Ping)
    case configurationChanged(ConfigurationChanged)
    case jobOffer(JobOffer)
    case jobAccept(JobAccept)
    case jobReject(JobReject)
    case jobProgress(JobProgress)
    case jobResult(JobResult)
    case jobFailed(JobFailed)
    case jobCancel(JobCancel)
    case toolCall(ToolCall)
    case toolResult(ToolResult)
    case approvalRequest(ApprovalRequest)
    case approvalResponse(ApprovalResponse)
    case personalCall(PersonalCall)
    case personalResult(PersonalResult)
    case logEvent(LogEvent)

    public var type: CompanionMessageType {
        switch self {
        case .authChallenge: .authChallenge
        case .authResponse: .authResponse
        case .hello: .hello
        case .helloAck: .helloAck
        case .telemetry: .telemetry
        case .heartbeat: .heartbeat
        case .ping: .ping
        case .configurationChanged: .configurationChanged
        case .jobOffer: .jobOffer
        case .jobAccept: .jobAccept
        case .jobReject: .jobReject
        case .jobProgress: .jobProgress
        case .jobResult: .jobResult
        case .jobFailed: .jobFailed
        case .jobCancel: .jobCancel
        case .toolCall: .toolCall
        case .toolResult: .toolResult
        case .approvalRequest: .approvalRequest
        case .approvalResponse: .approvalResponse
        case .personalCall: .personalCall
        case .personalResult: .personalResult
        case .logEvent: .logEvent
        }
    }
}

public enum CompanionCodecError: Error, Equatable {
    case frameTooLarge(Int)
    case unknownMessageType(String)
    case missingType
}

/// Encodes and decodes companion frames.
///
/// The date strategy is fixed here rather than left to callers: every
/// timestamp on this wire is RFC 3339 in UTC, and a client that quietly used a
/// local-time format would produce deadlines the server reads as hours out.
public struct CompanionCodec: Sendable {
    public init() {}

    private static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }()

    private static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys]
        return encoder
    }()

    private struct TypeProbe: Decodable {
        let type: String
    }

    public func decode(_ data: Data) throws -> CompanionMessage {
        guard data.count <= CompanionProtocol.maxFrameBytes else {
            // Checked before parsing: a payload bomb should cost one length
            // comparison, not a full parse.
            throw CompanionCodecError.frameTooLarge(data.count)
        }
        let probe: TypeProbe
        do {
            probe = try Self.decoder.decode(TypeProbe.self, from: data)
        } catch {
            throw CompanionCodecError.missingType
        }
        guard let type = CompanionMessageType(rawValue: probe.type) else {
            throw CompanionCodecError.unknownMessageType(probe.type)
        }
        let decoder = Self.decoder
        switch type {
        case .authChallenge: return .authChallenge(try decoder.decode(AuthChallenge.self, from: data))
        case .authResponse: return .authResponse(try decoder.decode(AuthResponse.self, from: data))
        case .hello: return .hello(try decoder.decode(CompanionHello.self, from: data))
        case .helloAck: return .helloAck(try decoder.decode(HelloAck.self, from: data))
        case .telemetry: return .telemetry(try decoder.decode(TelemetryMessage.self, from: data))
        case .heartbeat: return .heartbeat(try decoder.decode(Heartbeat.self, from: data))
        case .ping: return .ping(try decoder.decode(Ping.self, from: data))
        case .configurationChanged:
            return .configurationChanged(try decoder.decode(ConfigurationChanged.self, from: data))
        case .jobOffer: return .jobOffer(try decoder.decode(JobOffer.self, from: data))
        case .jobAccept: return .jobAccept(try decoder.decode(JobAccept.self, from: data))
        case .jobReject: return .jobReject(try decoder.decode(JobReject.self, from: data))
        case .jobProgress: return .jobProgress(try decoder.decode(JobProgress.self, from: data))
        case .jobResult: return .jobResult(try decoder.decode(JobResult.self, from: data))
        case .jobFailed: return .jobFailed(try decoder.decode(JobFailed.self, from: data))
        case .jobCancel: return .jobCancel(try decoder.decode(JobCancel.self, from: data))
        case .toolCall: return .toolCall(try decoder.decode(ToolCall.self, from: data))
        case .toolResult: return .toolResult(try decoder.decode(ToolResult.self, from: data))
        case .approvalRequest:
            return .approvalRequest(try decoder.decode(ApprovalRequest.self, from: data))
        case .approvalResponse:
            return .approvalResponse(try decoder.decode(ApprovalResponse.self, from: data))
        case .personalCall: return .personalCall(try decoder.decode(PersonalCall.self, from: data))
        case .personalResult:
            return .personalResult(try decoder.decode(PersonalResult.self, from: data))
        case .logEvent: return .logEvent(try decoder.decode(LogEvent.self, from: data))
        }
    }

    /// Encode a payload together with its `type` discriminator.
    public func encode<T: Encodable>(_ value: T, as type: CompanionMessageType) throws -> Data {
        var object = try JSONSerialization.jsonObject(with: Self.encoder.encode(value))
        guard var dictionary = object as? [String: Any] else {
            throw CompanionCodecError.missingType
        }
        dictionary["type"] = type.rawValue
        object = dictionary
        return try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    }
}
