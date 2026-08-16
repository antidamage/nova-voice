import Foundation
import XCTest

@testable import NovaCompanionKit

/// NPT-208. The ring exists to explain failures that leave no trace anywhere
/// else, so the tests are about the two properties that make it usable: it
/// cannot grow, and it cannot leak.
final class DiagnosticRingTests: XCTestCase {
    func testTheRingIsBoundedAndKeepsTheRecentPast() async {
        // A ring that grows is a log file, and a log file on a phone is
        // something nobody rotates.
        let ring = DiagnosticRing(capacity: 3)
        for index in 1...10 {
            await ring.record(.job, "accepted", detail: "job-\(index)")
        }

        let entries = await ring.snapshot()

        XCTAssertEqual(entries.count, 3)
        XCTAssertEqual(entries.map(\.detail), ["job-8", "job-9", "job-10"])
    }

    func testAnOverlongDetailIsTruncatedRatherThanDropped() async {
        // A caller that passes something long should lose the tail, not the
        // event — the event is the part that explains the fault.
        let ring = DiagnosticRing()
        await ring.record(.tool, "refused", detail: String(repeating: "x", count: 5_000))

        let entry = await ring.snapshot().first

        XCTAssertNotNil(entry)
        XCTAssertEqual(entry?.event, "refused")
        XCTAssertEqual(entry?.detail?.count, 200)
    }

    func testAnOverlongEventNameIsTruncated() async {
        let ring = DiagnosticRing()
        await ring.record(.connection, String(repeating: "e", count: 500))

        let entry = await ring.snapshot().first
        XCTAssertEqual(entry?.event.count, 64)
    }

    func testTheExportIsLegibleWithoutAViewer() async {
        // Read by a person deciding whether to bother reporting something.
        let ring = DiagnosticRing()
        await ring.record(
            .connection, "dropped", detail: "reason=timeout", at: Date(timeIntervalSince1970: 0)
        )

        let text = await ring.export()

        XCTAssertTrue(text.contains("[connection] dropped reason=timeout"))
        XCTAssertTrue(text.hasPrefix("1970-01-01T"))
    }

    func testTheExportIsNewestLast() async {
        let ring = DiagnosticRing()
        await ring.record(.job, "first")
        await ring.record(.job, "second")

        let lines = await ring.export().split(separator: "\n")

        XCTAssertTrue(lines.first?.contains("first") ?? false)
        XCTAssertTrue(lines.last?.contains("second") ?? false)
    }

    func testClearingLeavesNothingBehind() async {
        let ring = DiagnosticRing()
        await ring.record(.power, "unplugged")
        await ring.clear()

        let entries = await ring.snapshot()
        let exported = await ring.export()
        XCTAssertTrue(entries.isEmpty)
        XCTAssertEqual(exported, "")
    }

    func testAnEmptyRingExportsNothingRatherThanAHeader() async {
        // "Nothing happened" and "here is a report of nothing" read very
        // differently to someone about to send this on.
        let ring = DiagnosticRing()
        let exported = await ring.export()
        XCTAssertEqual(exported, "")
    }

    func testEveryCategoryIsStructuralRatherThanContentBearing() {
        // The categories are the whole leak-prevention story: there is no
        // "prompt", "transcript", "event" or "sample" category to put content
        // in, so a caller wanting to log content has nowhere to put it.
        let names = Set(DiagnosticRing.Category.allCases.map(\.rawValue))
        XCTAssertEqual(names, ["auth", "connection", "job", "tool", "power", "route"])
    }
}
