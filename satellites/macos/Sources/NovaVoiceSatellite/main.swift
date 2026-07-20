import AVFoundation
import AudioToolbox
import CoreAudio
import Foundation
import NVSShim
import Security

private let frameBytes = 640
private let protocolVersion: UInt8 = 1

private struct ActivityGateConfiguration {
    let enabled: Bool
    let thresholdDB: Double
    let noiseMarginDB: Double
    let triggerFrames: Int
    let preRollFrames: Int
    let hangoverFrames: Int
    let calibrationFrames: Int

    static func load(_ environment: [String: String]) -> ActivityGateConfiguration {
        func value(_ shortKey: String) -> String? {
            environment["NOVA_VOICE_\(shortKey)"]
                ?? environment["NOVA_VOICE_SATELLITE_\(shortKey)"]
        }
        func boolean(_ key: String, fallback: Bool) -> Bool {
            guard let raw = value(key)?.lowercased() else { return fallback }
            if ["1", "true", "yes", "on"].contains(raw) { return true }
            if ["0", "false", "no", "off"].contains(raw) { return false }
            return fallback
        }
        func number(_ key: String, fallback: Double, range: ClosedRange<Double>) -> Double {
            guard let raw = value(key), let parsed = Double(raw) else { return fallback }
            return min(range.upperBound, max(range.lowerBound, parsed))
        }
        func frames(_ key: String, fallbackMS: Int, range: ClosedRange<Int>) -> Int {
            let parsed = value(key).flatMap(Int.init) ?? fallbackMS
            let bounded = min(range.upperBound, max(range.lowerBound, parsed))
            return max(1, bounded / 20)
        }
        func optionalFrames(_ key: String, fallbackMS: Int, range: ClosedRange<Int>) -> Int {
            let parsed = value(key).flatMap(Int.init) ?? fallbackMS
            let bounded = min(range.upperBound, max(range.lowerBound, parsed))
            return max(0, bounded / 20)
        }
        return ActivityGateConfiguration(
            enabled: boolean("LOCAL_VAD_ENABLED", fallback: true),
            thresholdDB: number("LOCAL_VAD_THRESHOLD_DB", fallback: -48, range: -80 ... -20),
            noiseMarginDB: number("LOCAL_VAD_NOISE_MARGIN_DB", fallback: 6, range: 3 ... 30),
            triggerFrames: frames("LOCAL_VAD_TRIGGER_MS", fallbackMS: 60, range: 20 ... 500),
            preRollFrames: frames("LOCAL_VAD_PRE_ROLL_MS", fallbackMS: 400, range: 20 ... 2_000),
            hangoverFrames: frames("LOCAL_VAD_HANGOVER_MS", fallbackMS: 800, range: 200 ... 3_000),
            calibrationFrames: optionalFrames(
                "LOCAL_VAD_CALIBRATION_MS", fallbackMS: 1_000, range: 0 ... 5_000
            )
        )
    }
}

private struct CapturedAudioFrame {
    let payload: Data
    let monotonicNanoseconds: UInt64
    let playbackActive: Bool
}

private final class LocalActivityGate: @unchecked Sendable {
    private let lock = NSLock()
    private var enabled: Bool
    private let thresholdDB: Double
    private let noiseMarginDB: Double
    private let triggerFrames: Int
    private let preRollFrames: Int
    private let hangoverFrames: Int
    private let calibrationFrames: Int
    private var preRoll: [CapturedAudioFrame] = []
    private var active = false
    private var speechFrames = 0
    private var silenceFrames = 0
    private var noiseFloorDB = -60.0
    private var lastLevelDB = -120.0
    private var calibrationRemaining = 0
    private var calibrationLevels: [Double] = []

    init(configuration: ActivityGateConfiguration) {
        enabled = configuration.enabled
        thresholdDB = configuration.thresholdDB
        noiseMarginDB = configuration.noiseMarginDB
        triggerFrames = configuration.triggerFrames
        preRollFrames = configuration.preRollFrames
        hangoverFrames = configuration.hangoverFrames
        calibrationFrames = configuration.calibrationFrames
        calibrationRemaining = configuration.calibrationFrames
    }

    func setEnabled(_ value: Bool) {
        lock.lock()
        enabled = value
        resetLocked()
        lock.unlock()
    }

    func accept(_ frame: CapturedAudioFrame) -> [CapturedAudioFrame] {
        lock.lock()
        defer { lock.unlock() }
        guard enabled else { return [frame] }

        let level = levelDB(frame.payload)
        lastLevelDB = level

        if !active && calibrationRemaining > 0 {
            appendPreRollLocked(frame)
            calibrationLevels.append(level)
            calibrationRemaining -= 1
            if calibrationRemaining == 0 {
                let sorted = calibrationLevels.sorted()
                noiseFloorDB = max(-100, min(-20, sorted[sorted.count / 2]))
                calibrationLevels.removeAll(keepingCapacity: false)
            }
            return []
        }

        let activationDB = max(thresholdDB, noiseFloorDB + noiseMarginDB)
        let activity = level >= activationDB

        if active {
            if activity {
                silenceFrames = 0
            } else {
                silenceFrames += 1
                learnNoiseFloorLocked(level)
            }
            if silenceFrames >= hangoverFrames {
                closeGateLocked()
            }
            return [frame]
        }

        appendPreRollLocked(frame)
        if activity {
            speechFrames += 1
        } else {
            speechFrames = 0
            learnNoiseFloorLocked(level)
        }
        guard speechFrames >= triggerFrames else { return [] }

        active = true
        speechFrames = 0
        silenceFrames = 0
        let buffered = preRoll
        preRoll.removeAll(keepingCapacity: true)
        return buffered
    }

    var health: [String: Any] {
        lock.lock()
        defer { lock.unlock() }
        return [
            "enabled": enabled,
            "active": active,
            "lastLevelDb": (lastLevelDB * 10).rounded() / 10,
            "noiseFloorDb": (noiseFloorDB * 10).rounded() / 10,
        ]
    }

    private func resetLocked() {
        closeGateLocked()
        noiseFloorDB = -60
        lastLevelDB = -120
        calibrationRemaining = calibrationFrames
        calibrationLevels.removeAll(keepingCapacity: false)
    }

    private func closeGateLocked() {
        preRoll.removeAll(keepingCapacity: true)
        active = false
        speechFrames = 0
        silenceFrames = 0
    }

    private func learnNoiseFloorLocked(_ level: Double) {
        let weight = level > noiseFloorDB ? 0.02 : 0.08
        noiseFloorDB = max(-100, min(-20, (1 - weight) * noiseFloorDB + weight * level))
    }

    private func appendPreRollLocked(_ frame: CapturedAudioFrame) {
        preRoll.append(frame)
        if preRoll.count > preRollFrames { preRoll.removeFirst() }
    }

    private func levelDB(_ payload: Data) -> Double {
        guard payload.count >= 2 else { return -120 }
        var sumSquares = 0.0
        var index = 0
        while index + 1 < payload.count {
            let bits = UInt16(payload[index]) | (UInt16(payload[index + 1]) << 8)
            let sample = Double(Int16(bitPattern: bits)) / 32_768.0
            sumSquares += sample * sample
            index += 2
        }
        let rms = sqrt(sumSquares / Double(payload.count / 2))
        return max(-120, 20 * log10(max(rms, 0.000_001)))
    }
}

private struct Configuration {
    let serverURL: URL
    let satelliteID: String
    let displayName: String
    let roomID: String
    let identityLabel: String
    let clientKeychainPath: String
    let clientKeychainPasswordPath: String
    let caCertificatePath: String
    let healthPath: URL
    let activityGate: ActivityGateConfiguration

    static func load() throws -> Configuration {
        let environment = ProcessInfo.processInfo.environment
        func required(_ key: String) throws -> String {
            guard let value = environment[key], !value.isEmpty else {
                throw NSError(domain: "NovaVoiceSatellite", code: 2,
                              userInfo: [NSLocalizedDescriptionKey: "Missing \(key)"])
            }
            return value
        }
        guard let url = URL(string: try required("NOVA_VOICE_SERVER_URL")),
              url.scheme == "wss" else {
            throw NSError(domain: "NovaVoiceSatellite", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "Server must use wss://"])
        }
        let health = environment["NOVA_VOICE_HEALTH_PATH"]
            ?? "~/Library/Application Support/NovaVoiceSatellite/health.json"
        return Configuration(
            serverURL: url,
            satelliteID: try required("NOVA_VOICE_SATELLITE_ID"),
            displayName: environment["NOVA_VOICE_DISPLAY_NAME"] ?? "Indium",
            roomID: try required("NOVA_VOICE_ROOM_ID"),
            identityLabel: environment["NOVA_VOICE_IDENTITY_LABEL"] ?? "indium",
            clientKeychainPath: ((environment["NOVA_VOICE_CLIENT_KEYCHAIN_PATH"]
                ?? "~/Library/Application Support/NovaVoiceSatellite/client.keychain-db") as NSString)
                .expandingTildeInPath,
            clientKeychainPasswordPath: ((environment["NOVA_VOICE_CLIENT_KEYCHAIN_PASSWORD_PATH"]
                ?? "~/Library/Application Support/NovaVoiceSatellite/client-keychain-password") as NSString)
                .expandingTildeInPath,
            caCertificatePath: ((environment["NOVA_VOICE_CA_CERTIFICATE_PATH"]
                ?? "~/Library/Application Support/NovaVoiceSatellite/ca.crt") as NSString)
                .expandingTildeInPath,
            healthPath: URL(fileURLWithPath: (health as NSString).expandingTildeInPath),
            activityGate: ActivityGateConfiguration.load(environment)
        )
    }
}

private extension Data {
    mutating func appendBigEndian<T: FixedWidthInteger>(_ value: T) {
        var encoded = value.bigEndian
        Swift.withUnsafeBytes(of: &encoded) { append(contentsOf: $0) }
    }
}

private struct WireFrame {
    let kind: UInt8
    let flags: UInt16
    let sequence: UInt64
    let monotonicNanoseconds: UInt64
    let payload: Data

    func encode() -> Data {
        var data = Data("NVAF".utf8)
        data.append(protocolVersion)
        data.append(kind)
        data.appendBigEndian(flags)
        data.appendBigEndian(sequence)
        data.appendBigEndian(monotonicNanoseconds)
        data.appendBigEndian(UInt32(payload.count))
        data.append(payload)
        return data
    }

    static func decode(_ data: Data) throws -> WireFrame {
        guard data.count >= 28, data.prefix(4) == Data("NVAF".utf8), data[4] == 1 else {
            throw NSError(domain: "NovaVoiceSatellite", code: 4,
                          userInfo: [NSLocalizedDescriptionKey: "Invalid audio frame"])
        }
        func integer<T: FixedWidthInteger>(_ range: Range<Int>, as: T.Type) -> T {
            data[range].reduce(T.zero) { ($0 << 8) | T($1) }
        }
        let length = Int(integer(24..<28, as: UInt32.self))
        guard data.count == 28 + length else {
            throw NSError(domain: "NovaVoiceSatellite", code: 5,
                          userInfo: [NSLocalizedDescriptionKey: "Invalid payload length"])
        }
        return WireFrame(
            kind: data[5],
            flags: integer(6..<8, as: UInt16.self),
            sequence: integer(8..<16, as: UInt64.self),
            monotonicNanoseconds: integer(16..<24, as: UInt64.self),
            payload: data.subdata(in: 28..<data.count)
        )
    }
}

private struct Hello: Encodable {
    let protocolVersion = 1
    let satelliteId: String
    let displayName: String
    let roomId: String
    let client = "macos-native"
    let supervisor = "launchd"
    let capturePolicy = "always"
    let capabilities = Capabilities()

    struct Capabilities: Encodable {
        let microphone = true
        let speaker = true
        // This satellite captures the microphone with a plain input-only HAL
        // unit rather than the Voice-Processing I/O aggregate.  That aggregate
        // seizes and ducks the Mac's default output for the whole time it is
        // connected, which stops all other audio on the machine.  Without it
        // there is no local acoustic echo cancellation, so we advertise these
        // as false; the server then runs its half-duplex policy and ignores our
        // microphone while it is streaming playback to us, which prevents echo
        // without monopolising the host's audio devices.
        let echoCancellation = false
        let noiseSuppression = false
        let automaticGainControl = false
        // Lets Iridium wait for CoreAudio's real render lifecycle instead of
        // guessing from TTS generation or network-delivery timing.
        let playbackEvents = true
        let localVad = true
    }
}

private final class TLSDelegate: NSObject, URLSessionDelegate {
    private let identityLabel: String
    private let keychainPath: String
    private let keychainPasswordPath: String
    private let caCertificatePath: String

    init(
        identityLabel: String,
        keychainPath: String,
        keychainPasswordPath: String,
        caCertificatePath: String
    ) {
        self.identityLabel = identityLabel
        self.keychainPath = keychainPath
        self.keychainPasswordPath = keychainPasswordPath
        self.caCertificatePath = caCertificatePath
    }

    private func certificate(from data: Data) -> SecCertificate? {
        if let certificate = SecCertificateCreateWithData(nil, data as CFData) {
            return certificate
        }
        // The household CA is distributed as PEM. Security expects DER, so
        // decode the PEM body before making it a trust anchor.
        guard let pem = String(data: data, encoding: .utf8) else { return nil }
        let body = pem
            .components(separatedBy: .newlines)
            .filter { !$0.hasPrefix("-----") }
            .joined()
        guard let der = Data(base64Encoded: body) else { return nil }
        return SecCertificateCreateWithData(nil, der as CFData)
    }

    func urlSession(
        _ session: URLSession,
        didReceive challenge: URLAuthenticationChallenge,
        completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void
    ) {
        if challenge.protectionSpace.authenticationMethod == NSURLAuthenticationMethodServerTrust {
            guard let trust = challenge.protectionSpace.serverTrust,
                  let caData = try? Data(contentsOf: URL(fileURLWithPath: caCertificatePath)),
                  let ca = certificate(from: caData) else {
                completionHandler(.cancelAuthenticationChallenge, nil)
                return
            }
            SecTrustSetAnchorCertificates(trust, [ca] as CFArray)
            SecTrustSetAnchorCertificatesOnly(trust, true)
            var error: CFError?
            if SecTrustEvaluateWithError(trust, &error) {
                completionHandler(.useCredential, URLCredential(trust: trust))
            } else {
                completionHandler(.cancelAuthenticationChallenge, nil)
            }
            return
        }
        guard challenge.protectionSpace.authenticationMethod
                == NSURLAuthenticationMethodClientCertificate else {
            completionHandler(.performDefaultHandling, nil)
            return
        }
        var keychain: SecKeychain?
        guard SecKeychainOpen(keychainPath, &keychain) == errSecSuccess,
              let keychain,
              let passwordData = try? Data(
                contentsOf: URL(fileURLWithPath: keychainPasswordPath)
              ),
              let passwordText = String(data: passwordData, encoding: .utf8) else {
            completionHandler(.cancelAuthenticationChallenge, nil)
            return
        }
        // Older provisioning runs wrote the OpenSSL-generated secret with a
        // trailing newline.  Trim it so those keychains remain recoverable;
        // new provisioning writes the secret without a newline.
        let password = Data(passwordText.trimmingCharacters(in: .whitespacesAndNewlines).utf8)
        let unlockStatus = password.withUnsafeBytes { bytes in
            SecKeychainUnlock(keychain, UInt32(password.count), bytes.baseAddress, true)
        }
        guard unlockStatus == errSecSuccess else {
            completionHandler(.cancelAuthenticationChallenge, nil)
            return
        }
        let query: [String: Any] = [
            kSecClass as String: kSecClassIdentity,
            kSecAttrLabel as String: identityLabel,
            kSecReturnRef as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
            kSecMatchSearchList as String: [keychain],
        ]
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
              let item else {
            completionHandler(.cancelAuthenticationChallenge, nil)
            return
        }
        // The query's kSecClassIdentity guarantees the Core Foundation type.
        // Swift 6.2 rejects a conditional cast here because SecIdentity is a
        // Core Foundation reference type rather than a Swift class hierarchy.
        let identity = unsafeBitCast(item, to: SecIdentity.self)
        var certificate: SecCertificate?
        SecIdentityCopyCertificate(identity, &certificate)
        let certificates = certificate.map { [$0] } ?? []
        completionHandler(
            .useCredential,
            URLCredential(identity: identity, certificates: certificates, persistence: .forSession)
        )
    }
}

private let inputCaptureCallback: AURenderCallback = {
    refCon, actionFlags, timeStamp, busNumber, frameCount, _ in
    let engine = Unmanaged<AudioEngine>.fromOpaque(refCon).takeUnretainedValue()
    return engine.renderCapturedInput(
        actionFlags: actionFlags,
        timeStamp: timeStamp,
        busNumber: busNumber,
        frameCount: frameCount
    )
}

private final class AudioEngine: @unchecked Sendable {
    // Capture uses a plain input-only HAL audio unit.  Enabling the higher
    // level AVAudioEngine voice-processing path builds a VPAUAggregateAudioDevice
    // that opens *and holds* the default output for as long as the satellite is
    // connected, ducking every other app on the Mac.  An input-only HAL unit
    // touches the microphone alone and leaves the speakers to the rest of the
    // system.
    private var captureUnit: AudioUnit?
    private var captureFormat: AVAudioFormat?
    private var converter: AVAudioConverter?
    private let targetFormat = AVAudioFormat(
        commonFormat: .pcmFormatInt16, sampleRate: 16_000, channels: 1, interleaved: true
    )!
    private let lock = NSLock()
    private var pending = Data()
    private var sendFrame: ((CapturedAudioFrame) -> Void)?
    private let activityGate: LocalActivityGate

    // Playback is opened on demand.  The output device is only touched while
    // Nova is actually speaking back through this satellite and is released a
    // short time after, so idle listening never blocks the host's audio.
    private let playbackLock = NSLock()
    private let playbackEventQueue = DispatchQueue(
        label: "nz.co.skull.NovaVoiceSatellite.playback-events",
        qos: .userInitiated
    )
    private var playbackEngine: AVAudioEngine?
    private var player: AVAudioPlayerNode?
    private var playbackFormat: AVAudioFormat?
    private var playbackRate: Double = 24_000
    private var lastPlaybackNs: UInt64 = 0
    private var idleTimer: DispatchSourceTimer?

    // Jitter buffer.  Chunks arrive at whatever rate the server's TTS
    // generates them; scheduling them straight onto the player makes every
    // network or generation stall audible.  Buffer a short pre-roll before
    // starting, and if the player ever drains mid-stream, pause and rebuffer
    // instead of stuttering chunk by chunk.
    private var pendingBuffers: [AVAudioPCMBuffer] = []
    private var pendingFrameCount: AVAudioFramePosition = 0
    private var buffering = true
    private var bufferTargetSeconds: Double = 0.5
    private var streamActive = false
    private var scheduledFrames: AVAudioFramePosition = 0
    private var completedFrames: AVAudioFramePosition = 0
    // The server advertises this per stream.  700 ms is a safe default for a
    // loaded TTS GPU; the value is clamped so a bad control frame cannot make
    // the satellite wait indefinitely.
    private var initialPrerollSeconds = 0.7
    private let rebufferSeconds = 0.3
    // Per-stream playback quality counters, surfaced in health.json so an
    // end-to-end test can read an objective stutter metric.
    private var streamUnderruns = 0
    private var streamScheduledFrames: AVAudioFramePosition = 0
    private var streamCompletedFrames: AVAudioFramePosition = 0
    private var streamEngineRebuilds = 0
    private var streamPlaybackID: String?
    private var streamPlaybackStarted = false
    private var streamPlaybackFinishedReported = false

    // Engine-death recovery.  A CoreAudio configuration change (a Continuity
    // iPhone microphone appearing, a display waking, a device sample-rate
    // flip) stops a running AVAudioEngine without any error surfacing here;
    // scheduled buffers then never render and their completion callbacks
    // never fire, so the response is silent while every counter looks clean.
    // Track scheduled-but-unplayed buffers so the engine can be rebuilt and
    // the audio rescheduled instead of lost.  The generation guard keeps
    // completion callbacks from a torn-down engine (player.stop() fires them)
    // out of the live counters.
    private var engineGeneration = 0
    private var inFlight: [(buffer: AVAudioPCMBuffer, frames: AVAudioFramePosition)] = []
    private var configChangeObserver: NSObjectProtocol?
    private var lastEngineAttemptNs: UInt64 = 0
    private var watchdogPrevCompleted: AVAudioFramePosition = 0
    private var watchdogStallTicks = 0
    private var engineErrorCode: String?

    // Called off-lock when rendering starts and when a finished stream's
    // scheduled audio has fully rendered. These are the only trustworthy
    // edges for synchronizing the dashboard animation to audible playback.
    var onPlaybackStarted: ((String) -> Void)?
    var onPlaybackFinished: ((String?, PlaybackStats) -> Void)?

    init(activityGateConfiguration: ActivityGateConfiguration) {
        activityGate = LocalActivityGate(configuration: activityGateConfiguration)
    }

    struct PlaybackStats {
        let underruns: Int
        let scheduledSeconds: Double
        let completedSeconds: Double
        let engineRebuilds: Int
    }

    var lastEngineError: String? {
        playbackLock.lock()
        defer { playbackLock.unlock() }
        return engineErrorCode
    }

    // The frame is tagged "playback active" while response audio is still
    // buffered, scheduled, or within a short tail of finishing.  The server
    // drops microphone frames carrying this flag from a satellite without
    // local echo cancellation, which suppresses the room echo of Nova's own
    // voice.  This must track *acoustic* playback, not chunk arrival — audio
    // is delivered faster than realtime and keeps playing well after the last
    // chunk arrives.
    var playbackActive: Bool {
        playbackLock.lock()
        defer { playbackLock.unlock() }
        if streamActive || pendingFrameCount > 0 || completedFrames < scheduledFrames {
            return true
        }
        guard lastPlaybackNs != 0 else { return false }
        let elapsed = DispatchTime.now().uptimeNanoseconds &- lastPlaybackNs
        return elapsed < 400_000_000
    }

    private func defaultInputDeviceID() -> AudioDeviceID {
        var deviceID = AudioDeviceID(0)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &deviceID
        )
        return deviceID
    }

    func start(sendFrame: @escaping (CapturedAudioFrame) -> Void) throws {
        self.sendFrame = sendFrame

        var description = AudioComponentDescription(
            componentType: kAudioUnitType_Output,
            componentSubType: kAudioUnitSubType_HALOutput,
            componentManufacturer: kAudioUnitManufacturer_Apple,
            componentFlags: 0,
            componentFlagsMask: 0
        )
        guard let component = AudioComponentFindNext(nil, &description) else {
            throw NSError(domain: "NovaVoiceSatellite", code: 6,
                          userInfo: [NSLocalizedDescriptionKey: "No HAL audio component"])
        }
        var unit: AudioUnit?
        try check(AudioComponentInstanceNew(component, &unit), "instantiate HAL unit")
        guard let unit else {
            throw NSError(domain: "NovaVoiceSatellite", code: 6,
                          userInfo: [NSLocalizedDescriptionKey: "Nil HAL unit"])
        }

        // Enable input (element 1), disable output (element 0): capture only.
        var enable: UInt32 = 1
        try check(AudioUnitSetProperty(unit, kAudioOutputUnitProperty_EnableIO,
                                       kAudioUnitScope_Input, 1, &enable, UInt32(MemoryLayout<UInt32>.size)),
                  "enable input")
        var disable: UInt32 = 0
        try check(AudioUnitSetProperty(unit, kAudioOutputUnitProperty_EnableIO,
                                       kAudioUnitScope_Output, 0, &disable, UInt32(MemoryLayout<UInt32>.size)),
                  "disable output")

        var device = defaultInputDeviceID()
        try check(AudioUnitSetProperty(unit, kAudioOutputUnitProperty_CurrentDevice,
                                       kAudioUnitScope_Global, 0, &device, UInt32(MemoryLayout<AudioDeviceID>.size)),
                  "bind input device")

        // Read the microphone's hardware format, then request a Float32 client
        // format at the same rate/channel count for the render.
        var hardware = AudioStreamBasicDescription()
        var hardwareSize = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        try check(AudioUnitGetProperty(unit, kAudioUnitProperty_StreamFormat,
                                       kAudioUnitScope_Input, 1, &hardware, &hardwareSize),
                  "read hardware format")
        let channels = max(1, hardware.mChannelsPerFrame)
        guard hardware.mSampleRate > 0, let clientFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: hardware.mSampleRate,
            channels: AVAudioChannelCount(channels),
            interleaved: false
        ) else {
            throw NSError(domain: "NovaVoiceSatellite", code: 6,
                          userInfo: [NSLocalizedDescriptionKey: "Cannot resolve capture format"])
        }
        var clientDescription = clientFormat.streamDescription.pointee
        try check(AudioUnitSetProperty(unit, kAudioUnitProperty_StreamFormat,
                                       kAudioUnitScope_Output, 1, &clientDescription,
                                       UInt32(MemoryLayout<AudioStreamBasicDescription>.size)),
                  "set client format")
        guard let converter = AVAudioConverter(from: clientFormat, to: targetFormat) else {
            throw NSError(domain: "NovaVoiceSatellite", code: 6,
                          userInfo: [NSLocalizedDescriptionKey: "Cannot configure 16 kHz capture"])
        }
        self.captureFormat = clientFormat
        self.converter = converter

        var callback = AURenderCallbackStruct(
            inputProc: inputCaptureCallback,
            inputProcRefCon: Unmanaged.passUnretained(self).toOpaque()
        )
        try check(AudioUnitSetProperty(unit, kAudioOutputUnitProperty_SetInputCallback,
                                       kAudioUnitScope_Global, 0, &callback,
                                       UInt32(MemoryLayout<AURenderCallbackStruct>.size)),
                  "install input callback")

        try check(AudioUnitInitialize(unit), "initialize HAL unit")
        self.captureUnit = unit
        try check(AudioOutputUnitStart(unit), "start capture")
    }

    fileprivate func renderCapturedInput(
        actionFlags: UnsafeMutablePointer<AudioUnitRenderActionFlags>,
        timeStamp: UnsafePointer<AudioTimeStamp>,
        busNumber: UInt32,
        frameCount: UInt32
    ) -> OSStatus {
        guard let unit = captureUnit, let captureFormat,
              let buffer = AVAudioPCMBuffer(pcmFormat: captureFormat, frameCapacity: frameCount) else {
            return noErr
        }
        buffer.frameLength = frameCount
        let status = AudioUnitRender(
            unit, actionFlags, timeStamp, busNumber, frameCount, buffer.mutableAudioBufferList
        )
        guard status == noErr else { return status }
        convert(buffer)
        return noErr
    }

    private func convert(_ input: AVAudioPCMBuffer) {
        guard let converter,
              let output = AVAudioPCMBuffer(
                pcmFormat: targetFormat,
                frameCapacity: AVAudioFrameCount(Double(input.frameLength) * 16_000
                    / input.format.sampleRate) + 32
              ) else { return }
        var consumed = false
        var conversionError: NSError?
        let status = converter.convert(to: output, error: &conversionError) { _, state in
            if consumed {
                state.pointee = .noDataNow
                return nil
            }
            consumed = true
            state.pointee = .haveData
            return input
        }
        guard status != .error, conversionError == nil,
              let samples = output.int16ChannelData?[0] else { return }
        let data = Data(bytes: samples, count: Int(output.frameLength) * 2)
        lock.lock()
        pending.append(data)
        while pending.count >= frameBytes {
            let frame = pending.prefix(frameBytes)
            pending.removeFirst(frameBytes)
            let captured = CapturedAudioFrame(
                payload: Data(frame),
                monotonicNanoseconds: DispatchTime.now().uptimeNanoseconds,
                playbackActive: playbackActive
            )
            for ready in activityGate.accept(captured) {
                sendFrame?(ready)
            }
        }
        lock.unlock()
    }

    func setLocalVadEnabled(_ enabled: Bool) {
        activityGate.setEnabled(enabled)
    }

    var localVadHealth: [String: Any] { activityGate.health }

    func setPlaybackRate(_ rate: Double) {
        playbackLock.lock()
        playbackRate = rate
        playbackLock.unlock()
    }

    func setPlaybackBufferMs(_ milliseconds: Double) {
        playbackLock.lock()
        initialPrerollSeconds = min(2.0, max(0.2, milliseconds / 1000.0))
        playbackLock.unlock()
    }

    // A response stream is starting: buffer the initial pre-roll before any
    // audio reaches the player so generation/network jitter stays inaudible.
    func beginPlaybackStream(playbackID: String?) {
        playbackLock.lock()
        defer { playbackLock.unlock() }
        streamActive = true
        buffering = true
        bufferTargetSeconds = initialPrerollSeconds
        streamUnderruns = 0
        streamScheduledFrames = 0
        streamCompletedFrames = 0
        streamEngineRebuilds = 0
        streamPlaybackID = playbackID
        streamPlaybackStarted = false
        streamPlaybackFinishedReported = false
    }

    // The response stream is complete: flush anything still held (short
    // replies may never reach the pre-roll threshold) and let the idle timer
    // release the device once the scheduled audio drains.
    @discardableResult
    func endPlaybackStream() -> PlaybackStats {
        playbackLock.lock()
        streamActive = false
        flushPendingLocked()
        let stats = PlaybackStats(
            underruns: streamUnderruns,
            scheduledSeconds: Double(streamScheduledFrames) / playbackRate,
            completedSeconds: Double(streamCompletedFrames) / playbackRate,
            engineRebuilds: streamEngineRebuilds
        )
        let finishedNow = !streamPlaybackFinishedReported
            && pendingFrameCount == 0
            && completedFrames >= scheduledFrames
        if finishedNow { streamPlaybackFinishedReported = true }
        let playbackID = streamPlaybackID
        let callback = onPlaybackFinished
        playbackLock.unlock()
        if finishedNow {
            playbackEventQueue.async { callback?(playbackID, stats) }
        }
        return stats
    }

    // A direct speech interruption must discard both pre-roll and already
    // scheduled response audio. Rebuilding on the next stream is preferable
    // to allowing buffered words to continue after the user says to stop.
    func cancelPlaybackStream() {
        playbackLock.lock()
        teardownPlaybackEngineLocked()
        idleTimer?.cancel()
        idleTimer = nil
        lastPlaybackNs = 0
        playbackLock.unlock()
    }

    func play(_ payload: Data) {
        guard payload.count.isMultiple(of: 2) else { return }
        playbackLock.lock()
        defer { playbackLock.unlock() }
        // Streams from servers that never send stream markers still play:
        // the first chunk after an idle player behaves as a stream start.
        if !streamActive && pendingFrameCount == 0 && completedFrames >= scheduledFrames {
            streamActive = true
            buffering = true
            bufferTargetSeconds = initialPrerollSeconds
            streamPlaybackID = nil
            streamPlaybackStarted = false
            streamPlaybackFinishedReported = false
        }
        if playbackEngine == nil || playbackFormat?.sampleRate != playbackRate {
            // While the output device is unavailable every chunk would retry
            // the build; once per 500 ms is plenty (the watchdog also
            // retries every second).
            let now = DispatchTime.now().uptimeNanoseconds
            let throttled = playbackEngine == nil && lastEngineAttemptNs != 0
                && now &- lastEngineAttemptNs < 500_000_000
            if !throttled {
                if playbackEngine == nil && inFlight.isEmpty {
                    buildEngineLocked()
                } else {
                    recoverEngineLocked()
                }
            }
        }
        // The server streams 16-bit PCM, but AVAudioEngine node graphs only
        // accept Float32 formats — connecting a player with an Int16 format
        // throws an uncatchable -10868 (FormatNotSupported) NSException that
        // crashes the whole process, taking microphone capture down with it.
        // Convert the incoming samples to Float32 before scheduling.
        let sampleCount = payload.count / 2
        guard let format = playbackFormat,
              let buffer = AVAudioPCMBuffer(
                pcmFormat: format, frameCapacity: AVAudioFrameCount(sampleCount)
              ), let channel = buffer.floatChannelData?[0] else { return }
        buffer.frameLength = AVAudioFrameCount(sampleCount)
        payload.withUnsafeBytes { raw in
            let source = raw.bindMemory(to: Int16.self)
            for index in 0..<sampleCount {
                channel[index] = Float(Int16(littleEndian: source[index])) / 32_768
            }
        }
        streamScheduledFrames += AVAudioFramePosition(buffer.frameLength)
        // A dead engine must never drop audio: hold everything in the pending
        // queue and let the watchdog rebuild, then flush.
        if buffering || player == nil {
            pendingBuffers.append(buffer)
            pendingFrameCount += AVAudioFramePosition(buffer.frameLength)
            let bufferedSeconds = Double(pendingFrameCount) / playbackRate
            if player != nil, bufferedSeconds >= bufferTargetSeconds {
                flushPendingLocked()
            }
        } else {
            scheduleLocked(buffer)
        }
        lastPlaybackNs = DispatchTime.now().uptimeNanoseconds
        startIdleTimerLocked()
    }

    private func flushPendingLocked() {
        // With no live engine the pending audio must survive: the watchdog
        // rebuilds and flushes again once the output device comes back.
        guard player != nil else { return }
        guard !pendingBuffers.isEmpty else {
            buffering = false
            return
        }
        let buffers = pendingBuffers
        pendingBuffers.removeAll(keepingCapacity: true)
        pendingFrameCount = 0
        buffering = false
        for buffer in buffers {
            scheduleLocked(buffer)
        }
    }

    private func scheduleLocked(_ buffer: AVAudioPCMBuffer) {
        guard let player else { return }
        let frames = AVAudioFramePosition(buffer.frameLength)
        scheduledFrames += frames
        inFlight.append((buffer, frames))
        let generation = engineGeneration
        player.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            guard let self else { return }
            self.playbackLock.lock()
            guard generation == self.engineGeneration else {
                // Stale callback from a torn-down or rebuilt engine
                // (player.stop() fires the handlers of unplayed buffers).
                self.playbackLock.unlock()
                return
            }
            self.completedFrames += frames
            self.streamCompletedFrames += frames
            if let index = self.inFlight.firstIndex(where: { $0.buffer === buffer }) {
                self.inFlight.remove(at: index)
            }
            self.lastPlaybackNs = DispatchTime.now().uptimeNanoseconds
            // The player drained while the server is still mid-stream: an
            // underrun.  Pause and rebuild a smaller cushion rather than
            // letting every following chunk stutter through individually.
            if self.completedFrames >= self.scheduledFrames, self.streamActive {
                self.buffering = true
                self.bufferTargetSeconds = self.rebufferSeconds
                self.streamUnderruns += 1
            }
            // The stream is over and its last scheduled frame has actually
            // rendered — report the final, trustworthy stats.
            var finished: (playbackID: String?, stats: PlaybackStats)?
            if !self.streamPlaybackFinishedReported, !self.streamActive,
               self.pendingFrameCount == 0,
               self.completedFrames >= self.scheduledFrames {
                self.streamPlaybackFinishedReported = true
                finished = (
                    self.streamPlaybackID,
                    PlaybackStats(
                        underruns: self.streamUnderruns,
                        scheduledSeconds: Double(self.streamScheduledFrames) / self.playbackRate,
                        completedSeconds: Double(self.streamCompletedFrames) / self.playbackRate,
                        engineRebuilds: self.streamEngineRebuilds
                    )
                )
            }
            let callback = self.onPlaybackFinished
            self.playbackLock.unlock()
            if let finished, let callback {
                self.playbackEventQueue.async {
                    callback(finished.playbackID, finished.stats)
                }
            }
        }
        var playerReady = player.isPlaying
        if !player.isPlaying {
            var error: NSError?
            if !NVSCatchException({ player.play() }, &error) {
                // The engine died between chunks (device reconfiguration).
                // The buffer is tracked in inFlight, so the watchdog's
                // rebuild replays it — nothing is lost.
                engineErrorCode = "playerStart"
            } else {
                playerReady = true
            }
        }
        if playerReady, !streamPlaybackStarted, let playbackID = streamPlaybackID {
            streamPlaybackStarted = true
            let callback = onPlaybackStarted
            playbackEventQueue.async {
                callback?(playbackID)
            }
        }
    }

    // Pure engine construction: never touches stream or buffer state, so a
    // mid-stream rebuild cannot lose markers or audio.  All AVAudioEngine
    // graph calls raise NSExceptions when the output device is mid-
    // reconfiguration (the -10868 crash took down microphone capture too),
    // so every one goes through the shim.
    private func buildEngineLocked() {
        engineGeneration += 1
        if let observer = configChangeObserver {
            NotificationCenter.default.removeObserver(observer)
            configChangeObserver = nil
        }
        if let player {
            _ = NVSCatchException({ player.stop() }, nil)
        }
        if let playbackEngine {
            _ = NVSCatchException({ playbackEngine.stop() }, nil)
        }
        player = nil
        playbackEngine = nil
        lastEngineAttemptNs = DispatchTime.now().uptimeNanoseconds
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: playbackRate,
            channels: 1,
            interleaved: false
        ) else { return }
        // Keep the format even if the engine fails below: incoming chunks
        // still convert into the pending queue instead of being dropped.
        playbackFormat = format
        let engine = AVAudioEngine()
        let node = AVAudioPlayerNode()
        engine.attach(node)
        var graphError: NSError?
        let connected = NVSCatchException({
            engine.connect(node, to: engine.mainMixerNode, format: format)
        }, &graphError)
        guard connected else {
            let reason = graphError?.localizedDescription ?? "unknown"
            engineErrorCode = "connect:" + String(reason.prefix(80))
            return
        }
        var startError: NSError?
        var startThrew: Error?
        let started = NVSCatchException({
            do { try engine.start() } catch { startThrew = error }
        }, &startError)
        guard started, startThrew == nil else {
            let underlying = startError?.localizedDescription
                ?? startThrew?.localizedDescription
                ?? "unknown"
            engineErrorCode = "start:" + String(underlying.prefix(80))
            return
        }
        playbackEngine = engine
        player = node
        engineErrorCode = nil
        // CoreAudio stops this engine outright on a configuration change (a
        // Continuity microphone appearing, a device sample-rate flip) and it
        // never restarts itself — recover instead of playing into the void.
        configChangeObserver = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange,
            object: engine,
            queue: nil
        ) { [weak self] _ in
            guard let self else { return }
            DispatchQueue.global(qos: .userInitiated).async {
                self.playbackLock.lock()
                defer { self.playbackLock.unlock() }
                guard self.playbackEngine === engine else { return }
                self.recoverEngineLocked()
            }
        }
    }

    // Rebuild the engine while preserving the stream: unplayed scheduled
    // buffers go back to the front of the pending queue and are rescheduled
    // on the fresh engine.  At worst the currently-playing 100 ms buffer
    // restarts — versus the rest of the response never being heard.
    private func recoverEngineLocked() {
        streamEngineRebuilds += 1
        let wasBuffering = buffering
        let unplayed = inFlight.map { $0.buffer }
        inFlight.removeAll(keepingCapacity: true)
        scheduledFrames = 0
        completedFrames = 0
        if !unplayed.isEmpty {
            pendingBuffers.insert(contentsOf: unplayed, at: 0)
            pendingFrameCount = pendingBuffers.reduce(into: AVAudioFramePosition(0)) {
                $0 += AVAudioFramePosition($1.frameLength)
            }
        }
        buildEngineLocked()
        guard player != nil else {
            // Device still unavailable; hold the audio, watchdog retries.
            buffering = true
            return
        }
        if !wasBuffering {
            flushPendingLocked()
        }
    }

    // Release the output device shortly after Nova stops speaking so idle
    // listening never keeps the Mac's speakers busy.
    private func startIdleTimerLocked() {
        if idleTimer != nil { return }
        let timer = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .utility))
        timer.schedule(deadline: .now() + 1, repeating: 1)
        timer.setEventHandler { [weak self] in
            guard let self else { return }
            self.playbackLock.lock()
            defer { self.playbackLock.unlock() }
            // Watchdog: audio is queued or scheduled but completions have
            // stopped advancing — the engine was silently stopped by a
            // configuration change this process missed, or it never started.
            // Rebuild and reschedule instead of playing into the void.
            if self.scheduledFrames > self.completedFrames {
                if self.completedFrames == self.watchdogPrevCompleted {
                    self.watchdogStallTicks += 1
                } else {
                    self.watchdogStallTicks = 0
                }
                self.watchdogPrevCompleted = self.completedFrames
                if self.watchdogStallTicks >= 2 {
                    self.watchdogStallTicks = 0
                    self.recoverEngineLocked()
                    return
                }
            } else if self.player == nil, self.pendingFrameCount > 0 {
                // Engine construction keeps failing while audio waits.
                self.watchdogStallTicks = 0
                self.recoverEngineLocked()
                return
            } else {
                self.watchdogStallTicks = 0
            }
            // Never tear the engine down mid-stream: a generation stall used
            // to trip this timer between chunks of one response, destroying
            // and rebuilding the engine mid-sentence (an audible stutter).
            // Release the device only once the stream is over and the
            // scheduled audio has drained.  A stuck stream (server died
            // mid-response) is reclaimed by the 20 s failsafe.
            let elapsed = DispatchTime.now().uptimeNanoseconds &- self.lastPlaybackNs
            let drained = !self.streamActive
                && self.pendingFrameCount == 0
                && self.completedFrames >= self.scheduledFrames
            let stuck = elapsed > 20_000_000_000
            if self.lastPlaybackNs != 0, elapsed > 1_500_000_000, drained || stuck {
                self.teardownPlaybackEngineLocked()
                self.idleTimer?.cancel()
                self.idleTimer = nil
            }
        }
        idleTimer = timer
        timer.resume()
    }

    private func teardownPlaybackEngineLocked() {
        engineGeneration += 1
        if let observer = configChangeObserver {
            NotificationCenter.default.removeObserver(observer)
            configChangeObserver = nil
        }
        if let player {
            _ = NVSCatchException({ player.stop() }, nil)
        }
        if let playbackEngine {
            _ = NVSCatchException({ playbackEngine.stop() }, nil)
        }
        player = nil
        playbackEngine = nil
        playbackFormat = nil
        pendingBuffers.removeAll(keepingCapacity: false)
        pendingFrameCount = 0
        buffering = true
        bufferTargetSeconds = initialPrerollSeconds
        streamActive = false
        streamPlaybackID = nil
        streamPlaybackStarted = false
        streamPlaybackFinishedReported = false
        scheduledFrames = 0
        completedFrames = 0
        inFlight.removeAll(keepingCapacity: false)
        watchdogPrevCompleted = 0
        watchdogStallTicks = 0
    }

    func stop() {
        if let unit = captureUnit {
            AudioOutputUnitStop(unit)
            AudioUnitUninitialize(unit)
            AudioComponentInstanceDispose(unit)
            captureUnit = nil
        }
        converter = nil
        captureFormat = nil
        lock.lock()
        pending.removeAll(keepingCapacity: false)
        lock.unlock()
        playbackLock.lock()
        teardownPlaybackEngineLocked()
        idleTimer?.cancel()
        idleTimer = nil
        lastPlaybackNs = 0
        playbackLock.unlock()
    }

    private func check(_ status: OSStatus, _ context: String) throws {
        guard status != noErr else { return }
        throw NSError(domain: "NovaVoiceSatellite", code: 6,
                      userInfo: [NSLocalizedDescriptionKey: "\(context) failed: \(status)"])
    }
}

private actor Satellite {
    private let configuration: Configuration
    private let audio: AudioEngine
    private var sequence: UInt64 = 0
    private var socket: URLSessionWebSocketTask?
    private var session: URLSession?
    private var delegate: TLSDelegate?
    private var pendingAudio: [CapturedAudioFrame] = []
    private var drainingAudio = false

    init(configuration: Configuration) {
        self.configuration = configuration
        audio = AudioEngine(activityGateConfiguration: configuration.activityGate)
    }

    func run() async {
        var delay: UInt64 = 1
        while !Task.isCancelled {
            do {
                try await connect()
                delay = 1
            } catch {
                let value = error as NSError
                // Health metadata may include a stable transport code but
                // never the server response, audio, or spoken content.
                writeHealth(connected: false, errorCode: "\(value.domain):\(value.code)")
            }
            audio.stop()
            socket?.cancel(with: .goingAway, reason: nil)
            session?.invalidateAndCancel()
            try? await Task.sleep(for: .seconds(delay))
            delay = min(30, delay * 2)
        }
    }

    private func connect() async throws {
        delegate = TLSDelegate(
            identityLabel: configuration.identityLabel,
            keychainPath: configuration.clientKeychainPath,
            keychainPasswordPath: configuration.clientKeychainPasswordPath,
            caCertificatePath: configuration.caCertificatePath
        )
        let session = URLSession(configuration: .ephemeral, delegate: delegate, delegateQueue: nil)
        self.session = session
        let socket = session.webSocketTask(with: configuration.serverURL)
        self.socket = socket
        socket.resume()
        let hello = Hello(
            satelliteId: configuration.satelliteID,
            displayName: configuration.displayName,
            roomId: configuration.roomID
        )
        let helloData = try JSONEncoder().encode(hello)
        try await socket.send(.string(String(decoding: helloData, as: UTF8.self)))
        // Do not open the microphone until the server has accepted this
        // identity and protocol policy.  mTLS authenticates the peer; the
        // explicit acknowledgement protects against stale/rejected endpoints.
        guard case let .string(acknowledgement) = try await socket.receive(),
              let ackData = acknowledgement.data(using: .utf8),
              let ack = try JSONSerialization.jsonObject(with: ackData) as? [String: Any],
              ack["type"] as? String == "hello",
              ack["protocolVersion"] as? Int == Int(protocolVersion),
              ack["satelliteId"] as? String == configuration.satelliteID,
              ack["capturePolicy"] as? String == "always" else {
            throw NSError(domain: "NovaVoiceSatellite", code: 8,
                          userInfo: [NSLocalizedDescriptionKey: "Invalid hello acknowledgement"])
        }
        audio.setLocalVadEnabled(
            ack["localVadEnabled"] as? Bool ?? configuration.activityGate.enabled
        )
        try audio.start { [weak self] data in
            Task { await self?.enqueueAudio(data) }
        }
        // Report again once the stream's scheduled audio has actually
        // rendered: completedSeconds ≈ scheduledSeconds is the only proof the
        // response was audible rather than scheduled into a dead engine.
        audio.onPlaybackStarted = { [weak self] playbackID in
            Task {
                await self?.sendPlaybackEvent(type: "playback_started", playbackID: playbackID)
            }
        }
        audio.onPlaybackFinished = { [weak self] playbackID, stats in
            self?.writeHealth(connected: true, errorCode: nil, playback: stats)
            if let playbackID {
                Task {
                    await self?.sendPlaybackEvent(
                        type: "playback_finished", playbackID: playbackID
                    )
                }
            }
        }
        writeHealth(connected: true, errorCode: nil)

        while true {
            switch try await socket.receive() {
            case let .string(value):
                if let data = value.data(using: .utf8),
                   let control = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    switch control["type"] as? String {
                    case "playback":
                        if let rate = control["sampleRate"] as? Double {
                            audio.setPlaybackRate(rate)
                        }
                        if let bufferMs = control["bufferMs"] as? Double {
                            audio.setPlaybackBufferMs(bufferMs)
                        }
                        audio.beginPlaybackStream(
                            playbackID: control["playbackId"] as? String
                        )
                    case "playback_done":
                        let stats = audio.endPlaybackStream()
                        writeHealth(connected: true, errorCode: nil, playback: stats)
                    case "playback_cancel":
                        audio.cancelPlaybackStream()
                    case "local_vad":
                        if let enabled = control["enabled"] as? Bool {
                            audio.setLocalVadEnabled(enabled)
                        }
                    default:
                        break
                    }
                }
            case let .data(data):
                let frame = try WireFrame.decode(data)
                if frame.kind == 2 { audio.play(frame.payload) }
            @unknown default:
                break
            }
        }
    }

    private func sendPlaybackEvent(type: String, playbackID: String) async {
        guard let socket,
              let data = try? JSONSerialization.data(withJSONObject: [
                  "type": type,
                  "playbackId": playbackID,
              ]) else { return }
        do {
            try await socket.send(.string(String(decoding: data, as: UTF8.self)))
        } catch {
            socket.cancel(with: .abnormalClosure, reason: nil)
        }
    }

    private func enqueueAudio(_ frame: CapturedAudioFrame) {
        pendingAudio.append(frame)
        guard !drainingAudio else { return }
        drainingAudio = true
        Task { [weak self] in
            await self?.drainAudio()
        }
    }

    private func drainAudio() async {
        defer { drainingAudio = false }
        while !pendingAudio.isEmpty {
            if Task.isCancelled {
                pendingAudio.removeAll(keepingCapacity: true)
                return
            }
            let frame = pendingAudio.removeFirst()
            guard await sendAudio(frame) else {
                pendingAudio.removeAll(keepingCapacity: true)
                return
            }
        }
    }

    private func sendAudio(_ captured: CapturedAudioFrame) async -> Bool {
        guard let socket, captured.payload.count == frameBytes else { return false }
        let flags: UInt16 = captured.playbackActive ? 1 : 0
        let frame = WireFrame(
            kind: 1,
            flags: flags,
            sequence: sequence,
            monotonicNanoseconds: captured.monotonicNanoseconds,
            payload: captured.payload
        )
        sequence &+= 1
        do {
            try await socket.send(.data(frame.encode()))
            return true
        } catch {
            socket.cancel(with: .abnormalClosure, reason: nil)
            return false
        }
    }

    private nonisolated func writeHealth(
        connected: Bool,
        errorCode: String?,
        playback: AudioEngine.PlaybackStats? = nil
    ) {
        // Objective playback-quality metric for the last response stream; an
        // end-to-end test reads this instead of needing a room microphone.
        let playbackValue: [String: Any]? = playback.map {
            [
                "underruns": $0.underruns,
                "scheduledSeconds": (($0.scheduledSeconds * 100).rounded()) / 100,
                "completedSeconds": (($0.completedSeconds * 100).rounded()) / 100,
                "engineRebuilds": $0.engineRebuilds,
                "at": ISO8601DateFormatter().string(from: Date()),
            ]
        }
        let value: [String: Any?] = [
            "satelliteId": configuration.satelliteID,
            "connected": connected,
            "lastErrorCode": errorCode,
            "lastEngineError": audio.lastEngineError,
            "localVad": audio.localVadHealth,
            "pid": ProcessInfo.processInfo.processIdentifier,
            "lastPlayback": playbackValue,
        ]
        let directory = configuration.healthPath.deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        if let data = try? JSONSerialization.data(
            withJSONObject: value.compactMapValues { $0 }, options: [.sortedKeys]
        ) {
            try? data.write(to: configuration.healthPath, options: .atomic)
        }
    }
}

private func microphoneAllowed() async -> Bool {
    switch AVCaptureDevice.authorizationStatus(for: .audio) {
    case .authorized:
        return true
    case .notDetermined:
        return await AVCaptureDevice.requestAccess(for: .audio)
    default:
        return false
    }
}

do {
    let configuration = try Configuration.load()
    guard await microphoneAllowed() else {
        throw NSError(domain: "NovaVoiceSatellite", code: 7,
                      userInfo: [NSLocalizedDescriptionKey: "Microphone permission denied"])
    }
    // Keep capture alive through display sleep while allowing the display to
    // turn off.  launchd still owns restart/recovery; this assertion only
    // expresses the always-on audio policy to power management.
    let activity = ProcessInfo.processInfo.beginActivity(
        options: [.idleSystemSleepDisabled],
        reason: "Nova Voice always-on household microphone"
    )
    defer { ProcessInfo.processInfo.endActivity(activity) }
    await Satellite(configuration: configuration).run()
} catch {
    FileHandle.standardError.write(Data("NovaVoiceSatellite startup failed\n".utf8))
    exit(1)
}
