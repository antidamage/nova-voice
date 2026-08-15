import Foundation

/// A real WebSocket to Iridium's `/v1/companion`, plus the identity handshake.
///
/// The transport logic — backoff, queue bounds, correlation — lives above this
/// and is tested without a network. This is the thin part that cannot be:
/// URLSession, TLS client certificates, and the frames that must go in the
/// right order before the session is registered.
///
/// The client certificate is the whole point of the handshake. Holding a
/// household certificate proves only that the device is *a* household device;
/// signing the server's nonce proves it is the one whose name it is announcing.
public actor URLSessionCompanionSocket: NSObject, CompanionSocket {
    private let endpoint: URL
    private let identity: ClientIdentity
    private var task: URLSessionWebSocketTask?
    private var session: URLSession?

    /// The client certificate and its key, as a `SecIdentity` from the keychain.
    ///
    /// `@unchecked` because `SecIdentity` is a Core Foundation type Swift has
    /// no Sendable annotation for. It is immutable once created and the
    /// Security framework is documented as thread-safe, so sharing the handle
    /// is sound; the compiler simply cannot see that.
    public struct ClientIdentity: @unchecked Sendable {
        public let secIdentity: SecIdentity
        public let announcedId: String
        public let roles: [CompanionRole]

        public init(secIdentity: SecIdentity, announcedId: String, roles: [CompanionRole]) {
            self.secIdentity = secIdentity
            self.announcedId = announcedId
            self.roles = roles
        }
    }

    public init(endpoint: URL, identity: ClientIdentity) {
        self.endpoint = endpoint
        self.identity = identity
        super.init()
    }

    public func connect() async throws {
        let delegate = TLSDelegate(identity: identity.secIdentity)
        let configuration = URLSessionConfiguration.ephemeral
        // The server disables WebSocket pings because macOS URLSession tore
        // down otherwise-healthy streams on them, so liveness is the
        // application's heartbeat rather than the transport's.
        configuration.timeoutIntervalForRequest = 60
        let session = URLSession(
            configuration: configuration, delegate: delegate, delegateQueue: nil
        )
        let task = session.webSocketTask(with: endpoint)
        task.resume()
        self.session = session
        self.task = task
    }

    public func send(_ text: String) async throws {
        guard let task else { throw TransportError.notReady }
        try await task.send(.string(text))
    }

    public func receive() async throws -> String {
        guard let task else { throw TransportError.notReady }
        switch try await task.receive() {
        case .string(let text):
            return text
        case .data(let data):
            return String(decoding: data, as: UTF8.self)
        @unknown default:
            throw TransportError.notReady
        }
    }

    public func close() {
        task?.cancel(with: .goingAway, reason: nil)
        session?.invalidateAndCancel()
        task = nil
        session = nil
    }
}

/// Presents the household client certificate when the server asks for one.
private final class TLSDelegate: NSObject, URLSessionDelegate, @unchecked Sendable {
    private let identity: SecIdentity

    init(identity: SecIdentity) {
        self.identity = identity
    }

    func urlSession(
        _ session: URLSession,
        didReceive challenge: URLAuthenticationChallenge,
        completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void
    ) {
        guard challenge.protectionSpace.authenticationMethod
            == NSURLAuthenticationMethodClientCertificate
        else {
            // Server trust and everything else keep default handling. This
            // delegate exists only to answer the client-certificate request.
            completionHandler(.performDefaultHandling, nil)
            return
        }
        let credential = URLCredential(
            identity: identity, certificates: nil, persistence: .forSession
        )
        completionHandler(.useCredential, credential)
    }
}
