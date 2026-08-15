import Foundation
import NovaCompanionKit
import OSLog
import UIKit

/// Console logging for the companion role.
///
/// The app can fail to connect for reasons that leave no trace anywhere else —
/// no endpoint, no identity, no on-device model — and the only surface showing
/// them was a screen nobody was looking at. `devicectl device process launch
/// --console` picks these up, which is the difference between diagnosing a
/// silent app in seconds and guessing at it.
private let log = Logger(subsystem: "nz.co.skull.NovaCompanion", category: "companion")

/// Holds the companion role: connect, serve jobs, reconnect when dropped.
///
/// Only the companion role. The satellite role is a separate lifecycle by
/// design — turning the microphone off must not tear down reasoning, and
/// losing the companion socket must not stop audio — so nothing here reaches
/// into audio and audio will not reach into here.
@MainActor
final class CompanionCoordinator: ObservableObject {
    struct Status {
        var connected = false
        var detail = "Not configured"
        var acceptedJobs = 0
        var lastError: String?
    }

    @Published private(set) var status = Status()

    private var task: Task<Void, Never>?
    private var backoff = ReconnectBackoff()
    private let engine = FoundationModelEngine()
    private let signer: KeychainSigner
    private let endpoints: [URL]
    private let announcedId: String
    private let identityPassphrase: String?

    init(configuration: CompanionConfiguration = .load()) {
        self.signer = KeychainSigner(label: configuration.keychainLabel)
        self.endpoints = configuration.orderedEndpoints
        self.announcedId = configuration.announcedId
        self.identityPassphrase = configuration.identityPassphrase
    }

    func start() {
        guard task == nil else { return }
        task = Task { [weak self] in await self?.run() }
    }

    func stop() {
        task?.cancel()
        task = nil
        status.connected = false
        status.detail = "Stopped"
    }

    private func run() async {
        guard !endpoints.isEmpty else {
            status.detail = "No endpoint configured"
            log.error("no endpoint configured; companion-config.json missing or unreadable")
            print("[companion] no endpoint configured")
            return
        }
        guard engine.isAvailable else {
            // Reported rather than retried: an unavailable model is a state of
            // the device, not a transient fault, and reconnecting in a loop
            // would burn battery to keep discovering the same thing.
            status.detail = "On-device model unavailable"
            let reason = engine.availabilityDescription
            log.error("on-device model unavailable: \(reason, privacy: .public)")
            print("[companion] model unavailable: \(reason)")
            return
        }

        importBundledIdentityIfNeeded()

        while !Task.isCancelled {
          for endpoint in endpoints {
            if Task.isCancelled { return }
            do {
                status.detail = "Connecting"
                print("[companion] connecting to \(endpoint) as \(announcedId)")
                let identity = URLSessionCompanionSocket.ClientIdentity(
                    secIdentity: try signer.secIdentity(),
                    announcedId: announcedId,
                    roles: [.companion],
                    caCertificate: Self.householdCA()
                )
                let socket = URLSessionCompanionSocket(endpoint: endpoint, identity: identity)
                try await socket.connect()

                let session = CompanionSession(
                    socket: socket,
                    signer: signer,
                    announcedId: announcedId,
                    workloads: FoundationModelEngine.supported,
                    engine: engine,
                    telemetry: Self.telemetry
                )
                try await session.handshake(
                    appVersion: Bundle.main.shortVersion,
                    osVersion: UIDevice.current.systemVersion
                )

                backoff.reset()
                status.connected = true
                status.lastError = nil
                status.detail = "Connected"
                print("[companion] connected; session established")

                try await session.serve()

                status.acceptedJobs = await session.acceptedJobs
                await socket.close()
                // A session that ran and ended is not a reason to try the next
                // endpoint: the one we had was working. Restart the search
                // from the preferred address instead.
                status.connected = false
                break
            } catch {
                status.lastError = String(describing: error).prefix(120).description
                log.error("companion session failed: \(String(describing: error), privacy: .public)")
                print("[companion] session failed via \(endpoint): \(error)")
            }
            status.connected = false
          }
            guard !Task.isCancelled else { return }
            let delay = backoff.next()
            status.detail = String(format: "Reconnecting in %.0fs", delay)
            try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
        }
    }

    /// Import the household identity on first launch, if one shipped with the
    /// build and the keychain does not already hold it.
    ///
    /// Shipping a `.p12` inside the app is a real trade-off: it puts a private
    /// key in a build artifact. It is acceptable only because this is a
    /// personally-signed build installed on one phone by its owner, and the
    /// identity is scoped to a household service that can revoke it. The file
    /// is untracked, and once imported the keychain copy is the one used.
    private func importBundledIdentityIfNeeded() {
        if (try? signer.secIdentity()) != nil { return }
        guard
            let url = Bundle.main.url(forResource: "companion-identity", withExtension: "p12"),
            let data = try? Data(contentsOf: url),
            let passphrase = identityPassphrase
        else {
            status.detail = "No household identity available"
            print("[companion] no bundled identity or passphrase available")
            return
        }
        do {
            try KeychainSigner.importIdentity(
                p12: data, passphrase: passphrase, label: signer.label
            )
        } catch {
            status.lastError = "identity import failed: \(error)"
            print("[companion] identity import failed: \(error)")
        }
    }

    /// The household CA shipped with the build, anchored so Nova's own
    /// certificate validates. It is a public certificate, not a secret, but it
    /// still names a specific deployment so it stays out of git.
    private static func householdCA() -> SecCertificate? {
        guard
            let url = Bundle.main.url(forResource: "household-ca", withExtension: "der"),
            let data = try? Data(contentsOf: url)
        else {
            print("[companion] no household CA bundled; server trust will fail")
            return nil
        }
        return SecCertificateCreateWithData(nil, data as CFData)
    }

    /// Device state the server ages to decide whether to trust this device.
    private static func telemetry() -> CompanionTelemetry {
        let device = UIDevice.current
        device.isBatteryMonitoringEnabled = true
        let thermal: ThermalState =
            switch ProcessInfo.processInfo.thermalState {
            case .nominal: .nominal
            case .fair: .fair
            case .serious: .serious
            case .critical: .critical
            @unknown default: .nominal
            }
        return CompanionTelemetry(
            battery: device.batteryLevel >= 0 ? Double(device.batteryLevel) : 1,
            charging: device.batteryState == .charging || device.batteryState == .full,
            lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled,
            thermalState: thermal,
            appState: UIApplication.shared.applicationState == .active ? .foreground : .background,
            models: ModelAvailability(hotAvailable: true, hotContextTokens: 4096)
        )
    }
}

/// Where to connect and which identity to present.
///
/// Read from a bundled `companion-config.json` rather than compiled in, so no
/// deployment address, device name or passphrase lands in git. The file is
/// untracked and written at build time by the machine that has those values.
///
/// (`INFOPLIST_KEY_*` build settings were the obvious route and do not work:
/// Xcode only maps them for Info.plist keys it already knows, so custom ones
/// are silently dropped — the build succeeds and the app starts unconfigured.)
struct CompanionConfiguration: Decodable {
    var endpoint: URL?
    /// Reached only when the LAN address is not. Tailscale gives one stable
    /// address that survives a Wi-Fi to cellular handoff, so the companion
    /// role can keep working away from the house — but the server still
    /// classifies that session as `tailnet` from its peer address, so it does
    /// not unlock the home-LAN-only routes. Reachability, not authorization.
    var tailnetEndpoint: URL?
    var announcedId: String
    var keychainLabel: String
    var identityPassphrase: String?

    private enum CodingKeys: String, CodingKey {
        case endpoint, tailnetEndpoint, announcedId, keychainLabel, identityPassphrase
    }

    /// The LAN address first, always. Preferring it is what makes "at home"
    /// the normal case rather than an accident of which address answered.
    var orderedEndpoints: [URL] {
        [endpoint, tailnetEndpoint].compactMap { $0 }
    }

    init(
        endpoint: URL?,
        tailnetEndpoint: URL? = nil,
        announcedId: String,
        keychainLabel: String,
        identityPassphrase: String? = nil
    ) {
        self.endpoint = endpoint
        self.tailnetEndpoint = tailnetEndpoint
        self.announcedId = announcedId
        self.keychainLabel = keychainLabel
        self.identityPassphrase = identityPassphrase
    }

    static func load() -> CompanionConfiguration {
        let fallback = CompanionConfiguration(
            endpoint: nil, announcedId: "companion-1", keychainLabel: "nova-companion"
        )
        guard
            let url = Bundle.main.url(forResource: "companion-config", withExtension: "json"),
            let data = try? Data(contentsOf: url),
            let decoded = try? JSONDecoder().decode(CompanionConfiguration.self, from: data)
        else {
            return fallback
        }
        return decoded
    }
}

extension Bundle {
    var shortVersion: String {
        infoDictionary?["CFBundleShortVersionString"] as? String ?? "0"
    }
}
