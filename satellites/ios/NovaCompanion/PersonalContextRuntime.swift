import EventKit
import Foundation
import NovaCompanionKit

/// Calendar and Reminders reads, as thin a layer over EventKit as it can be.
///
/// Everything that can be wrong in an interesting way — bounding a window,
/// ordering, truncation, what a permission state means — lives in
/// `NovaCompanionKit` where it is tested without a device. What is left here is
/// the part that genuinely cannot be tested off-device: asking iOS for
/// permission, and turning `EKEvent` into the small flat record that travels.
///
/// Only reads. Every create, update, delete and completion goes through Nova's
/// durable approval gate and comes back as an approved mutation with an exact
/// target; a second write path here would bypass that entirely.
@MainActor
final class PersonalContextRuntime {
    private let store: EKEventStore

    /// The store is injected so reads and writes share one. EventKit
    /// identifiers are only stable within a store, and an id handed to Nova
    /// from one store would not resolve in another when the approval came
    /// back naming it.
    init(store: EKEventStore) {
        self.store = store
    }

    // MARK: - Permissions

    /// Ask only when asking can achieve something.
    ///
    /// iOS will not re-prompt after a refusal, so requesting again on every
    /// launch achieves nothing except making the app look broken. A refused
    /// permission is reported and the matching tools are withdrawn.
    func requestAccess(to entity: EKEntityType) async -> PersonalPermission {
        let current = Self.permission(for: EKEventStore.authorizationStatus(for: entity))
        guard current.worthRequesting else { return current }
        do {
            let granted =
                entity == .event
                ? try await store.requestFullAccessToEvents()
                : try await store.requestFullAccessToReminders()
            return granted ? .authorised : .denied
        } catch {
            // A thrown request is not a refusal — the owner has not said no —
            // but it is not usable either, so it reports as restricted rather
            // than pretending the question was answered.
            return .restricted
        }
    }

    static func permission(for status: EKAuthorizationStatus) -> PersonalPermission {
        switch status {
        case .notDetermined: .notDetermined
        case .restricted: .restricted
        case .denied: .denied
        case .fullAccess: .authorised
        // Write-only access can create an event but cannot read one back, so
        // for a *read* runtime it is no more useful than a refusal.
        case .writeOnly: .denied
        @unknown default: .restricted
        }
    }

    var calendarPermission: PersonalPermission {
        Self.permission(for: EKEventStore.authorizationStatus(for: .event))
    }

    var remindersPermission: PersonalPermission {
        Self.permission(for: EKEventStore.authorizationStatus(for: .reminder))
    }

    // MARK: - Reads

    func events(start: Date, end: Date, maxItems: Int, calendars: [String] = []) throws
        -> PersonalQueryResult
    {
        guard calendarPermission.usable else { throw PersonalError.notPermitted }
        let scope = PersonalScope(start: start, end: end, maxItems: maxItems)
        let selected = Self.matching(store.calendars(for: .event), names: calendars)
        let predicate = store.predicateForEvents(
            withStart: scope.start, end: scope.end, calendars: selected
        )
        let records = store.events(matching: predicate).map(Self.record(from:))
        return PersonalQueryResult.bounded(records, scope: scope)
    }

    func reminders(
        maxItems: Int,
        list: String? = nil,
        includeCompleted: Bool = false,
        dueBefore: Date? = nil
    ) async throws -> PersonalQueryResult {
        guard remindersPermission.usable else { throw PersonalError.notPermitted }
        // Reminders have no natural window, so the scope carries the cap and a
        // nominal range. The bound that matters here is the count.
        let scope = PersonalScope(start: .distantPast, end: .distantFuture, maxItems: maxItems)
        let selected = Self.matching(
            store.calendars(for: .reminder), names: list.map { [$0] } ?? []
        )
        let predicate =
            includeCompleted
            ? store.predicateForReminders(in: selected)
            : store.predicateForIncompleteReminders(
                withDueDateStarting: nil, ending: dueBefore, calendars: selected
            )

        let fetched: [EKReminder] = await withCheckedContinuation { continuation in
            store.fetchReminders(matching: predicate) { reminders in
                continuation.resume(returning: reminders ?? [])
            }
        }
        return PersonalQueryResult.bounded(fetched.map(Self.record(from:)), scope: scope)
    }

    enum PersonalError: Error {
        case notPermitted
    }

    // MARK: - Transforms

    /// `nil` means every calendar, an empty match means none.
    ///
    /// The distinction matters: asking for a calendar that does not exist must
    /// return nothing, not silently widen to all of them.
    private static func matching(_ calendars: [EKCalendar], names: [String]) -> [EKCalendar]? {
        guard !names.isEmpty else { return nil }
        let wanted = Set(names.map { $0.lowercased() })
        return calendars.filter { wanted.contains($0.title.lowercased()) }
    }

    private static func record(from event: EKEvent) -> PersonalRecord {
        PersonalRecord(
            // `eventIdentifier` is stable across queries, which is what lets a
            // later mutation name exactly this occurrence.
            id: event.eventIdentifier ?? UUID().uuidString,
            title: event.title ?? "(untitled)",
            start: event.startDate,
            end: event.endDate,
            allDay: event.isAllDay,
            completed: nil,
            listName: event.calendar?.title
        )
    }

    private static func record(from reminder: EKReminder) -> PersonalRecord {
        // A reminder's due date is stored as components, so it is resolved
        // against the *current* calendar rather than assumed to be UTC —
        // reading it as UTC is how a 9am reminder becomes 9pm the day before.
        let due = reminder.dueDateComponents.flatMap {
            Calendar.current.date(from: $0)
        }
        return PersonalRecord(
            id: reminder.calendarItemIdentifier,
            title: reminder.title ?? "(untitled)",
            start: due,
            end: nil,
            allDay: reminder.dueDateComponents?.hour == nil,
            completed: reminder.isCompleted,
            listName: reminder.calendar?.title
        )
    }
}
