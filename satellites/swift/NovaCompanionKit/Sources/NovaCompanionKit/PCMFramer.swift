import Foundation

/// Cuts a capture stream into the exact frames the server accepts.
///
/// Nova closes a satellite stream over a malformed frame, and "malformed"
/// here means any length other than **640 bytes** — 20 ms of 16 kHz mono
/// PCM16. `AVAudioEngine` does not deliver 640-byte buffers; it delivers
/// whatever size the hardware and the current route happen to produce, and it
/// changes that size when the route changes. So this sits between the two, and
/// getting it wrong does not degrade audio quality — it disconnects the
/// microphone.
///
/// It is a value type with no audio dependencies precisely so the awkward
/// cases below can be tested without a device, a microphone, or a room.
public struct PCMFramer: Sendable {
    /// 20 ms of 16 kHz mono PCM16. The server's `BYTES_PER_FRAME`.
    public static let bytesPerFrame = 640

    /// Bytes held over from the last buffer, waiting for enough to fill a frame.
    /// Never reaches a full frame — anything that does is emitted immediately.
    private var pending = Data()

    /// Samples abandoned by `reset`, for the diagnostics ring. Not an error
    /// count: discarding a partial frame at a discontinuity is correct.
    public private(set) var discardedBytes = 0
    public private(set) var emittedFrames = 0

    public init() {}

    /// Feed a capture buffer; take back whole frames.
    ///
    /// An odd byte count is not split mid-sample. PCM16 samples are two bytes,
    /// and a frame that begins half a sample out of phase is not quiet noise —
    /// every subsequent sample is reconstructed from the wrong byte pair, and
    /// the stream becomes loud static that still passes the length check.
    public mutating func append(_ data: Data) -> [Data] {
        pending.append(data)
        var frames: [Data] = []
        while pending.count >= Self.bytesPerFrame {
            frames.append(pending.prefix(Self.bytesPerFrame))
            pending.removeFirst(Self.bytesPerFrame)
        }
        emittedFrames += frames.count
        return frames
    }

    /// Throw away the partial frame at a discontinuity.
    ///
    /// Call this on a route change, an interruption, or a reconnect. Keeping
    /// the remainder would splice audio from before the gap onto audio from
    /// after it inside one frame — an audible click, and worse, a frame whose
    /// first milliseconds describe a moment that has passed. The server would
    /// accept it, because it is exactly 640 bytes.
    public mutating func reset() {
        discardedBytes += pending.count
        pending.removeAll(keepingCapacity: true)
    }

    /// How much is waiting. Only ever 0..<640.
    public var pendingBytes: Int { pending.count }
}

/// A capture buffer that arrived with an odd number of bytes.
///
/// Separated from the framer because the two failures need different
/// responses: a trailing half-sample is a *source* problem — a resampler or a
/// format conversion producing something that is not PCM16 — and silently
/// carrying it forward would misalign every frame after it.
public enum PCMAlignment {
    /// Split a buffer into the part safe to frame and any trailing half sample.
    ///
    /// Returning the remainder rather than dropping it lets a caller stitch it
    /// onto the next buffer when the source is merely chunking oddly, and
    /// notice it when the source is genuinely wrong.
    public static func aligned(_ data: Data) -> (whole: Data, remainder: Data) {
        guard data.count % 2 == 1 else { return (data, Data()) }
        return (data.dropLast(), data.suffix(1))
    }
}
