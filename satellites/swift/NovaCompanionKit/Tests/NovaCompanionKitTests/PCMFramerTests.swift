import Foundation
import XCTest

@testable import NovaCompanionKit

/// NPT-603. The server closes a satellite stream over a malformed frame, so
/// this is not an audio-quality concern — a framing bug disconnects the
/// microphone. Every case here is one the hardware actually produces:
/// buffers that do not divide evenly, route changes mid-stream, and the
/// half-sample tail that turns a stream into static while still passing the
/// length check.
final class PCMFramerTests: XCTestCase {
    private func audio(_ byteCount: Int, fill: UInt8 = 0xAB) -> Data {
        Data(repeating: fill, count: byteCount)
    }

    func testAnExactFrameComesStraightBack() {
        var framer = PCMFramer()
        let frames = framer.append(audio(640))

        XCTAssertEqual(frames.count, 1)
        XCTAssertEqual(frames[0].count, PCMFramer.bytesPerFrame)
        XCTAssertEqual(framer.pendingBytes, 0)
    }

    func testABufferSmallerThanAFrameEmitsNothingYet() {
        // The common case: hardware buffers are rarely 640 bytes.
        var framer = PCMFramer()

        XCTAssertTrue(framer.append(audio(320)).isEmpty)
        XCTAssertEqual(framer.pendingBytes, 320)
    }

    func testTwoHalfBuffersMakeExactlyOneFrame() {
        var framer = PCMFramer()
        _ = framer.append(audio(320))
        let frames = framer.append(audio(320))

        XCTAssertEqual(frames.count, 1)
        XCTAssertEqual(framer.pendingBytes, 0)
    }

    func testALargeBufferIsCutIntoWholeFramesWithTheRemainderHeld() {
        // 1600 bytes = two frames and a 320-byte tail.
        var framer = PCMFramer()
        let frames = framer.append(audio(1_600))

        XCTAssertEqual(frames.count, 2)
        XCTAssertTrue(frames.allSatisfy { $0.count == PCMFramer.bytesPerFrame })
        XCTAssertEqual(framer.pendingBytes, 320)
    }

    func testEveryByteIsAccountedForAcrossAwkwardBufferSizes() {
        // The property that matters: nothing is dropped and nothing is
        // duplicated, whatever sizes the hardware hands over.
        var framer = PCMFramer()
        var emitted = Data()
        var fed = 0

        for size in [7 * 2, 511 * 2, 1, 0, 999, 1_281, 320, 640] {
            let even = size - (size % 2)
            fed += even
            for frame in framer.append(audio(even)) {
                emitted.append(frame)
            }
        }

        XCTAssertEqual(emitted.count + framer.pendingBytes, fed)
        XCTAssertEqual(emitted.count % PCMFramer.bytesPerFrame, 0)
    }

    func testTheContentsOfAFrameAreContiguousSourceAudio() {
        // A framer that reordered or padded would still pass a length check.
        var framer = PCMFramer()
        var source = Data()
        for value in 0..<UInt8(255) {
            source.append(contentsOf: [value, value])
        }
        source.append(Data(repeating: 0x11, count: 640 - source.count % 640))

        var emitted = Data()
        for frame in framer.append(source) {
            emitted.append(frame)
        }

        XCTAssertEqual(emitted, source.prefix(emitted.count))
    }

    // MARK: - Discontinuities

    func testResetDiscardsThePartialFrameAtARouteChange() {
        // Keeping it would splice audio from before the gap onto audio from
        // after it, inside one frame the server would happily accept.
        var framer = PCMFramer()
        _ = framer.append(audio(320))
        framer.reset()

        XCTAssertEqual(framer.pendingBytes, 0)
        XCTAssertEqual(framer.discardedBytes, 320)
    }

    func testAudioAfterAResetStartsAFreshFrame() {
        var framer = PCMFramer()
        _ = framer.append(audio(320, fill: 0x01))
        framer.reset()
        let frames = framer.append(audio(640, fill: 0x02))

        XCTAssertEqual(frames.count, 1)
        // Entirely post-reset audio: no byte of the abandoned buffer survived.
        XCTAssertTrue(frames[0].allSatisfy { $0 == 0x02 })
    }

    func testResettingWithNothingPendingDiscardsNothing() {
        var framer = PCMFramer()
        framer.reset()

        XCTAssertEqual(framer.discardedBytes, 0)
    }

    func testTheFrameCountIsCumulativeAcrossResets() {
        // It is a diagnostic for "is this device actually streaming", so a
        // reset must not make a working session look idle.
        var framer = PCMFramer()
        _ = framer.append(audio(640))
        framer.reset()
        _ = framer.append(audio(640))

        XCTAssertEqual(framer.emittedFrames, 2)
    }

    // MARK: - Alignment

    func testAnOddBufferIsSplitRatherThanFramedMidSample() {
        // A frame beginning half a sample out of phase is not quiet noise:
        // every subsequent sample is rebuilt from the wrong byte pair and the
        // stream becomes loud static that still passes the length check.
        let (whole, remainder) = PCMAlignment.aligned(Data(repeating: 0x7F, count: 641))

        XCTAssertEqual(whole.count, 640)
        XCTAssertEqual(remainder.count, 1)
    }

    func testAnEvenBufferIsLeftAlone() {
        let (whole, remainder) = PCMAlignment.aligned(Data(repeating: 0x7F, count: 640))

        XCTAssertEqual(whole.count, 640)
        XCTAssertTrue(remainder.isEmpty)
    }

    func testAnEmptyBufferIsHandled() {
        let (whole, remainder) = PCMAlignment.aligned(Data())

        XCTAssertTrue(whole.isEmpty)
        XCTAssertTrue(remainder.isEmpty)
    }

    // MARK: - Backpressure

    func testTheOutboundQueueDropsOldestUnderBackpressure() {
        // A stalled socket must not grow memory without bound. Oldest-first,
        // because in a live microphone stream the newest audio is the audio
        // anyone still cares about.
        var queue = BoundedFrameQueue(capacity: 3)
        for index in 0..<10 {
            queue.append("frame-\(index)")
        }

        XCTAssertEqual(queue.count, 3)
        XCTAssertEqual(queue.dropped, 7)
        XCTAssertEqual(queue.frames, ["frame-7", "frame-8", "frame-9"])
    }
}
