import Foundation
import XCTest

@testable import NovaCompanionKit

/// NPT-502/503/504. The parts of a personal-context read that can be wrong in
/// an interesting way, tested without a device, a calendar or a person: the
/// bounding, the ordering, the truncation, and what a permission state means.
final class PersonalScopeTests: XCTestCase {
    private let noon = Date(timeIntervalSince1970: 1_786_000_000)

    // MARK: - Bounding

    func testAWindowLongerThanTheContractIsTrimmedRatherThanRefused() {
        // A plan asking for a year is not an attack. Failing it would leave
        // the reasoning run with nothing when it could have had something.
        let scope = PersonalScope(
            start: noon, end: noon.addingTimeInterval(365 * 86_400), maxItems: 50
        )

        XCTAssertEqual(scope.start, noon)
        XCTAssertEqual(scope.end.timeIntervalSince(noon), PersonalScope.maximumWindow)
        XCTAssertTrue(scope.trimmed)
    }

    func testAReasonableWindowIsLeftAlone() {
        let scope = PersonalScope(
            start: noon, end: noon.addingTimeInterval(7 * 86_400), maxItems: 20
        )

        XCTAssertEqual(scope.end.timeIntervalSince(noon), 7 * 86_400)
        XCTAssertEqual(scope.maxItems, 20)
        XCTAssertFalse(scope.trimmed)
    }

    func testAnInvertedWindowIsCorrectedRatherThanReadAsEmpty() {
        // Reading it as an empty range would answer "you have no events" to a
        // question that was never asked properly.
        let scope = PersonalScope(
            start: noon.addingTimeInterval(86_400), end: noon, maxItems: 10
        )

        XCTAssertEqual(scope.start, noon)
        XCTAssertEqual(scope.end, noon.addingTimeInterval(86_400))
        XCTAssertTrue(scope.trimmed)
    }

    func testTheItemCapIsClampedAtBothEnds() {
        XCTAssertEqual(PersonalScope(start: noon, end: noon, maxItems: 10_000).maxItems, 200)
        XCTAssertEqual(PersonalScope(start: noon, end: noon, maxItems: 0).maxItems, 1)
        XCTAssertEqual(PersonalScope(start: noon, end: noon, maxItems: -5).maxItems, 1)
    }

    func testAClampedCapIsReportedAsTrimmed() {
        // Otherwise a caller asking for 1000 and getting 200 would believe it
        // had seen everything.
        let scope = PersonalScope(start: noon, end: noon, maxItems: 1_000)
        XCTAssertTrue(scope.trimmed)
    }

    // MARK: - Ordering and truncation

    func testResultsAreSortedBeforeTheCapIsApplied() {
        // Capping an unsorted set returns an arbitrary subset, and "your next
        // three events" made of three random ones is worse than an error.
        let scope = PersonalScope(start: noon, end: noon.addingTimeInterval(86_400), maxItems: 2)
        let records = [
            PersonalRecord(id: "c", title: "Third", start: noon.addingTimeInterval(300)),
            PersonalRecord(id: "a", title: "First", start: noon.addingTimeInterval(100)),
            PersonalRecord(id: "b", title: "Second", start: noon.addingTimeInterval(200)),
        ]

        let result = PersonalQueryResult.bounded(records, scope: scope)

        XCTAssertEqual(result.records.map(\.id), ["a", "b"])
        XCTAssertTrue(result.truncated)
    }

    func testUndatedItemsSortLastButAreNotDropped() {
        // A reminder with no due date is real, but it is never the answer to
        // "what's next".
        let scope = PersonalScope(start: noon, end: noon.addingTimeInterval(86_400), maxItems: 10)
        let records = [
            PersonalRecord(id: "undated", title: "Someday"),
            PersonalRecord(id: "dated", title: "Tomorrow", start: noon),
        ]

        let result = PersonalQueryResult.bounded(records, scope: scope)

        XCTAssertEqual(result.records.map(\.id), ["dated", "undated"])
        XCTAssertFalse(result.truncated)
    }

    func testItemsAtTheSameInstantHaveAStableOrder() {
        // Two events at 9am must not swap between polls; a caller comparing
        // successive reads would see phantom changes.
        let scope = PersonalScope(start: noon, end: noon.addingTimeInterval(86_400), maxItems: 10)
        let records = [
            PersonalRecord(id: "b", title: "Standup", start: noon),
            PersonalRecord(id: "a", title: "Standup", start: noon),
        ]

        let result = PersonalQueryResult.bounded(records, scope: scope)
        XCTAssertEqual(result.records.map(\.id), ["a", "b"])
    }

    func testAWindowThatWasTrimmedMarksItsResultTruncatedEvenWhenNothingWasDropped() {
        // The records returned are complete *for the trimmed window*, which is
        // not the same as complete for what was asked.
        let scope = PersonalScope(
            start: noon, end: noon.addingTimeInterval(365 * 86_400), maxItems: 50
        )
        let result = PersonalQueryResult.bounded(
            [PersonalRecord(id: "a", title: "Only one", start: noon)], scope: scope
        )

        XCTAssertEqual(result.records.count, 1)
        XCTAssertTrue(result.truncated)
    }

    func testAnEmptyResultIsNotTruncated() {
        let scope = PersonalScope(start: noon, end: noon.addingTimeInterval(86_400), maxItems: 10)
        let result = PersonalQueryResult.bounded([], scope: scope)

        XCTAssertTrue(result.records.isEmpty)
        XCTAssertFalse(result.truncated)
    }

    // MARK: - Permissions

    func testOnlyGrantedPermissionsCanBeRead() {
        XCTAssertTrue(PersonalPermission.authorised.usable)
        XCTAssertTrue(PersonalPermission.limited.usable)
        XCTAssertFalse(PersonalPermission.denied.usable)
        XCTAssertFalse(PersonalPermission.restricted.usable)
        XCTAssertFalse(PersonalPermission.notDetermined.usable)
    }

    func testADeniedPermissionIsNotWorthAskingAgain() {
        // iOS will not re-prompt once the owner has said no, so an app that
        // keeps trying achieves nothing except looking broken.
        XCTAssertTrue(PersonalPermission.notDetermined.worthRequesting)
        XCTAssertFalse(PersonalPermission.denied.worthRequesting)
        XCTAssertFalse(PersonalPermission.restricted.worthRequesting)
    }

    func testDenyingOnePermissionRemovesOnlyItsOwnTools() {
        // This is what keeps a single "no" from disabling the companion.
        let tools = availablePersonalTools(
            calendar: .authorised,
            reminders: .denied,
            location: .authorised,
            health: .restricted
        )

        XCTAssertEqual(
            tools, ["companion.calendar.list", "companion.location.current"]
        )
    }

    func testNoPermissionsMeansNoPersonalToolsAdvertised() {
        // Advertising a tool the device cannot serve costs Nova a round trip
        // and a refusal on every attempt.
        let tools = availablePersonalTools(
            calendar: .denied, reminders: .denied, location: .denied, health: .denied
        )
        XCTAssertTrue(tools.isEmpty)
    }

    func testAdvertisedToolNamesMatchTheServerManifest() {
        // These strings are a contract with providers/companion/provider.py.
        // A typo here is a tool that silently never becomes available.
        let tools = availablePersonalTools(
            calendar: .authorised, reminders: .authorised, location: .authorised,
            health: .authorised
        )

        XCTAssertEqual(
            tools,
            [
                "companion.calendar.list",
                "companion.reminders.list",
                "companion.location.current",
                "companion.health.summary",
            ]
        )
    }
}
