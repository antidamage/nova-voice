import Foundation
import NovaCompanionKit
import UIKit

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
    private let endpoint: URL?
    private let announcedId: String
    private let identityPassphrase: String?

    init(configuration: CompanionConfiguration = .load()) {
        self.signer = KeychainSigner(label: configuration.keychainLabel)
        self.endpoint = configuration.endpoint
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
        guard let endpoint else {
            status.detail = "No endpoint configured"
            return
        }
        guard engine.isAvailable else {
            // Reported rather than retried: an unavailable model is a state of
            // the device, not a transient fault, and reconnecting in a loop
            // would burn battery to keep discovering the same thing.
            status.detail = "On-device model unavailable"
            return
        }

        importBundledIdentityIfNeeded()

        while !Task.isCancelled {
            do {
                status.detail = "Connecting"
                let identity = URLSessionCompanionSocket.ClientIdentity(
                    secIdentity: try signer.secIdentity(),
                    announcedId: announcedId,
                    roles: [.companion]
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

                try await session.serve()

                status.acceptedJobs = await session.acceptedJobs
                await socket.close()
            } catch {
                status.lastError = String(describing: error).prefix(120).description
            }

            status.connected = false
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
            return
        }
        do {
            try KeychainSigner.importIdentity(
                p12: data, passphrase: passphrase, label: signer.label
            )
        } catch {
            status.lastError = "identity import failed: \(error)"
        }
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
    var announcedId: String
    var keychainLabel: String
    var identityPassphrase: String?

    private enum CodingKeys: String, CodingKey {
        case endpoint, announcedId, keychainLabel, identityPassphrase
    }

    init(
        endpoint: URL?,
        announcedId: String,
        keychainLabel: String,
        identityPassphrase: String? = nil
    ) {
        self.endpoint = endpoint
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
