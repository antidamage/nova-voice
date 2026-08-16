import EventKit
import Foundation
import NovaCompanionKit

/// Turns Nova's `personal_call` frames into EventKit work and back.
///
/// Every mutation that arrives here has already been approved: Nova refuses
/// confirmable actions on the ordinary provider path, turns them into a durable
/// approval carrying the exact target, gets a signed decision, and only then
/// sends the call. So this layer does not ask again — asking twice would train
/// the owner to tap through prompts — but it does **revalidate**, because an
/// approval describes the item as it was when the owner read it, and an item
/// that has changed since is no longer the thing they agreed to.
@MainActor
struct PersonalCallHandler: CompanionSession.PersonalHandler {
    let runtime: PersonalContextRuntime
    let store: EKEventStore
    let location: HomeLocationRuntime

    func run(_ call: PersonalCall) async throws -> PersonalResult {
        let arguments = call.arguments
        switch call.tool {
        case "companion.calendar.list":
            return try listEvents(call, arguments)
        case "companion.reminders.list":
            return try await listReminders(call, arguments)
        case "companion.calendar.create":
            return try createEvent(call, arguments)
        case "companion.calendar.update":
            return try updateEvent(call, arguments)
        case "companion.calendar.delete":
            return try deleteEvent(call, arguments)
        case "companion.reminders.create":
            return try createReminder(call, arguments)
        case "companion.reminders.complete":
            return try setReminderCompleted(call, arguments, completed: true)
        case "companion.reminders.uncomplete":
            return try setReminderCompleted(call, arguments, completed: false)
        case "companion.reminders.delete":
            return try deleteReminder(call, arguments)
        case "companion.location.current":
            return await currentLocation(call, arguments)
        default:
            // Nova validated this against its manifest before sending, so an
            // unknown tool here means the two have drifted — worth saying
            // plainly rather than failing as a generic error.
            return failure(call, "invalid", "this device does not implement \(call.tool)")
        }
    }

    // MARK: - Reads

    private func listEvents(_ call: PersonalCall, _ arguments: JSONValue) throws -> PersonalResult {
        let result = try runtime.events(
            start: date(arguments["start"]) ?? Date(),
            end: date(arguments["end"]) ?? Date().addingTimeInterval(86_400),
            maxItems: arguments["maxItems"]?.intValue ?? call.maxItems,
            calendars: strings(arguments["calendars"])
        )
        return success(call, result)
    }

    private func listReminders(
        _ call: PersonalCall, _ arguments: JSONValue
    ) async throws -> PersonalResult {
        let result = try await runtime.reminders(
            maxItems: arguments["maxItems"]?.intValue ?? call.maxItems,
            list: arguments["list"]?.stringValue,
            includeCompleted: arguments["includeCompleted"]?.boolValue ?? false,
            dueBefore: date(arguments["dueBefore"])
        )
        return success(call, result)
    }

    // MARK: - Location

    /// Answers "is this at home", never "where is this".
    ///
    /// The coordinates stay on the phone. Nova's question is whether the owner
    /// is at the house, and the answer to that is a boolean plus how old and
    /// how precise it is — sending the point as well would put a household's
    /// exact location into job payloads and logs for no gain at all.
    private func currentLocation(
        _ call: PersonalCall, _ arguments: JSONValue
    ) async -> PersonalResult {
        let accuracy = arguments["accuracy"]?.stringValue ?? "coarse"
        let maxAge = arguments["maxAgeSeconds"]?.doubleValue ?? 900
        guard let fix = await location.reading(fine: accuracy == "fine", maxAge: maxAge) else {
            // Denied, restricted, or nothing recent enough. Reported as an
            // explicit no-answer rather than a default, because there is no
            // safe direction in which to guess where someone is.
            return failure(call, "unavailable", "no usable location is available")
        }
        return PersonalResult(
            callId: call.callId,
            ok: true,
            code: "ok",
            message: fix.isHome ? "at home" : "away from home",
            sensitivity: .location,
            items: [
                .object([
                    "isHome": .bool(fix.isHome),
                    "ageSeconds": .number(fix.ageSeconds),
                    "accuracy": .string(accuracy),
                ])
            ],
            truncated: false
        )
    }

    // MARK: - Calendar mutations

    private func createEvent(_ call: PersonalCall, _ arguments: JSONValue) throws -> PersonalResult {
        guard runtime.calendarPermission.usable else {
            return failure(call, "blocked", "calendar access is not granted")
        }
        guard let title = arguments["title"]?.stringValue,
              let start = date(arguments["start"]),
              let end = date(arguments["end"])
        else {
            return failure(call, "invalid", "an event needs a title, a start and an end")
        }
        let event = EKEvent(eventStore: store)
        event.title = title
        event.startDate = start
        event.endDate = end
        event.isAllDay = arguments["allDay"]?.boolValue ?? false
        event.notes = arguments["notes"]?.stringValue
        event.calendar =
            named(arguments["calendar"]?.stringValue, in: store.calendars(for: .event))
            ?? store.defaultCalendarForNewEvents
        guard event.calendar != nil else {
            return failure(call, "not_found", "no calendar to write to")
        }
        try store.save(event, span: .thisEvent, commit: true)
        return identified(call, id: event.eventIdentifier, message: "event created")
    }

    private func updateEvent(_ call: PersonalCall, _ arguments: JSONValue) throws -> PersonalResult {
        guard let id = arguments["eventId"]?.stringValue,
              let event = store.event(withIdentifier: id)
        else {
            return failure(call, "not_found", "that event no longer exists")
        }
        if let conflict = staleness(event.lastModifiedDate, arguments) { return conflict(call) }

        if let title = arguments["title"]?.stringValue { event.title = title }
        if let start = date(arguments["start"]) { event.startDate = start }
        if let end = date(arguments["end"]) { event.endDate = end }
        if let notes = arguments["notes"]?.stringValue { event.notes = notes }
        try store.save(event, span: .thisEvent, commit: true)
        return identified(call, id: event.eventIdentifier, message: "event updated")
    }

    private func deleteEvent(_ call: PersonalCall, _ arguments: JSONValue) throws -> PersonalResult {
        guard let id = arguments["eventId"]?.stringValue,
              let event = store.event(withIdentifier: id)
        else {
            // Already gone. Reported as success rather than not_found: the
            // owner approved "this event should not exist", and it does not.
            // Failing would invite a retry that can only fail the same way.
            return identified(call, id: nil, message: "the event was already gone")
        }
        if let conflict = staleness(event.lastModifiedDate, arguments) { return conflict(call) }
        try store.remove(event, span: .thisEvent, commit: true)
        return identified(call, id: id, message: "event deleted")
    }

    // MARK: - Reminder mutations

    private func createReminder(
        _ call: PersonalCall, _ arguments: JSONValue
    ) throws -> PersonalResult {
        guard runtime.remindersPermission.usable else {
            return failure(call, "blocked", "reminders access is not granted")
        }
        guard let title = arguments["title"]?.stringValue else {
            return failure(call, "invalid", "a reminder needs a title")
        }
        let reminder = EKReminder(eventStore: store)
        reminder.title = title
        reminder.notes = arguments["notes"]?.stringValue
        if let due = date(arguments["due"]) {
            reminder.dueDateComponents = Calendar.current.dateComponents(
                [.year, .month, .day, .hour, .minute], from: due
            )
        }
        reminder.calendar =
            named(arguments["list"]?.stringValue, in: store.calendars(for: .reminder))
            ?? store.defaultCalendarForNewReminders()
        guard reminder.calendar != nil else {
            return failure(call, "not_found", "no reminder list to write to")
        }
        try store.save(reminder, commit: true)
        return identified(call, id: reminder.calendarItemIdentifier, message: "reminder created")
    }

    private func setReminderCompleted(
        _ call: PersonalCall, _ arguments: JSONValue, completed: Bool
    ) throws -> PersonalResult {
        guard let reminder = reminder(from: arguments) else {
            return failure(call, "not_found", "that reminder no longer exists")
        }
        if reminder.isCompleted == completed {
            // Already in the requested state. Idempotent by nature, so this is
            // a success — a retry after a dropped reply must not look like a
            // failure and provoke a third attempt.
            return identified(
                call, id: reminder.calendarItemIdentifier, message: "already in that state"
            )
        }
        reminder.isCompleted = completed
        try store.save(reminder, commit: true)
        return identified(
            call,
            id: reminder.calendarItemIdentifier,
            message: completed ? "reminder completed" : "reminder reopened"
        )
    }

    private func deleteReminder(
        _ call: PersonalCall, _ arguments: JSONValue
    ) throws -> PersonalResult {
        guard let reminder = reminder(from: arguments) else {
            return identified(call, id: nil, message: "the reminder was already gone")
        }
        let id = reminder.calendarItemIdentifier
        try store.remove(reminder, commit: true)
        return identified(call, id: id, message: "reminder deleted")
    }

    // MARK: - Helpers

    private func reminder(from arguments: JSONValue) -> EKReminder? {
        guard let id = arguments["reminderId"]?.stringValue else { return nil }
        return store.calendarItem(withIdentifier: id) as? EKReminder
    }

    /// Has the item moved on since the owner read the proposal?
    ///
    /// An approval describes an item as it was. If it has changed since, the
    /// thing the owner agreed to is not the thing in front of us, and applying
    /// the change anyway would silently overwrite whatever happened in between.
    private func staleness(
        _ lastModified: Date?, _ arguments: JSONValue
    ) -> ((PersonalCall) -> PersonalResult)? {
        guard let expected = date(arguments["ifUnchangedSince"]),
              let lastModified,
              lastModified > expected
        else { return nil }
        return { call in
            self.failure(
                call, "conflict", "this changed after you approved it, so nothing was applied"
            )
        }
    }

    private func named(_ title: String?, in calendars: [EKCalendar]) -> EKCalendar? {
        guard let title else { return nil }
        return calendars.first { $0.title.lowercased() == title.lowercased() }
    }

    private func date(_ value: JSONValue?) -> Date? {
        guard let text = value?.stringValue else { return nil }
        return ISO8601DateFormatter().date(from: text)
            ?? ISO8601DateFormatter.withFractionalSeconds.date(from: text)
    }

    private func strings(_ value: JSONValue?) -> [String] {
        (value?.arrayValue ?? []).compactMap(\.stringValue)
    }

    private func success(_ call: PersonalCall, _ result: PersonalQueryResult) -> PersonalResult {
        PersonalResult(
            callId: call.callId,
            ok: true,
            code: "ok",
            message: "\(result.records.count) item(s)",
            sensitivity: .personal,
            items: result.records.map(Self.encode),
            truncated: result.truncated
        )
    }

    private func identified(
        _ call: PersonalCall, id: String?, message: String
    ) -> PersonalResult {
        PersonalResult(
            callId: call.callId,
            ok: true,
            code: "ok",
            message: message,
            sensitivity: .mutation,
            items: id.map { [JSONValue.object(["id": .string($0)])] } ?? [],
            truncated: false
        )
    }

    private func failure(_ call: PersonalCall, _ code: String, _ message: String) -> PersonalResult {
        PersonalResult(
            callId: call.callId,
            ok: false,
            code: code,
            message: message,
            sensitivity: .personal,
            items: [],
            truncated: false
        )
    }

    private static func encode(_ record: PersonalRecord) -> JSONValue {
        var object: [String: JSONValue] = [
            "id": .string(record.id),
            "title": .string(record.title),
            "allDay": .bool(record.allDay),
        ]
        let formatter = ISO8601DateFormatter()
        if let start = record.start { object["start"] = .string(formatter.string(from: start)) }
        if let end = record.end { object["end"] = .string(formatter.string(from: end)) }
        if let completed = record.completed { object["completed"] = .bool(completed) }
        if let list = record.listName { object["list"] = .string(list) }
        return .object(object)
    }
}

extension ISO8601DateFormatter {
    /// Nova emits fractional seconds in some payloads and not others, and the
    /// default formatter silently fails on the ones that have them — which
    /// surfaces as "an event needs a start" rather than as a parse problem.
    static let withFractionalSeconds: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()
}
