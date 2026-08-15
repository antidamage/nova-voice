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
        /// The household CA, anchored for server-trust evaluation. Nil falls
        /// back to system roots, which will reject Nova's own certificate.
        public let caCertificate: SecCertificate?

        public init(
            secIdentity: SecIdentity,
            announcedId: String,
            roles: [CompanionRole],
            caCertificate: SecCertificate? = nil
        ) {
            self.secIdentity = secIdentity
            self.announcedId = announcedId
            self.roles = roles
            self.caCertificate = caCertificate
        }
    }

    public init(endpoint: URL, identity: ClientIdentity) {
        self.endpoint = endpoint
        self.identity = identity
        super.init()
    }

    public func connect() async throws {
        let delegate = TLSDelegate(identity: identity.secIdentity, anchor: identity.caCertificate)
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

/// Presents the household client certificate, and trusts the household CA.
///
/// Both halves are necessary. Nova's server certificate is signed by the
/// household CA, which no device trusts by default, so without anchoring it
/// the connection fails TLS (-1200) even once the network path works.
///
/// The CA is used as an **additional anchor with the system roots disabled**,
/// so this trusts exactly one issuer rather than weakening validation
/// generally: a public certificate for the same address would now be rejected,
/// which is the correct behaviour for a service that is only ever the
/// household's own.
private final class TLSDelegate: NSObject, URLSessionDelegate, @unchecked Sendable {
    private let identity: SecIdentity
    private let anchor: SecCertificate?

    init(identity: SecIdentity, anchor: SecCertificate?) {
        self.identity = identity
        self.anchor = anchor
    }

    func urlSession(
        _ session: URLSession,
        didReceive challenge: URLAuthenticationChallenge,
        completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void
    ) {
        switch challenge.protectionSpace.authenticationMethod {
        case NSURLAuthenticationMethodClientCertificate:
            completionHandler(
                .useCredential,
                URLCredential(identity: identity, certificates: nil, persistence: .forSession)
            )

        case NSURLAuthenticationMethodServerTrust:
            guard let trust = challenge.protectionSpace.serverTrust, let anchor else {
                print("[tls] server trust challenge with no anchor; deferring to system roots")
                completionHandler(.performDefaultHandling, nil)
                return
            }
            SecTrustSetAnchorCertificates(trust, [anchor] as CFArray)
            // Without this the system roots stay in play alongside ours.
            SecTrustSetAnchorCertificatesOnly(trust, true)
            var error: CFError?
            if SecTrustEvaluateWithError(trust, &error) {
                completionHandler(.useCredential, URLCredential(trust: trust))
            } else {
                // Refused rather than accepted-anyway: the whole point of
                // anchoring is that an unexpected certificate is a real signal.
                //
                // Logged with the reason because the two ways this fails look
                // identical from outside — a chain that does not lead to the
                // household CA, and a certificate that does but names an
                // address it was never issued for — and the fixes are
                // completely different.
                print("[tls] server trust rejected for \(challenge.protectionSpace.host): "
                    + String(describing: error))
                completionHandler(.cancelAuthenticationChallenge, nil)
            }

        default:
            completionHandler(.performDefaultHandling, nil)
        }
    }
}
