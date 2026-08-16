import Foundation

/// Bounding and normalising a personal-context query before it runs.
///
/// Kept here, away from EventKit, for two reasons. It is the part that can be
/// wrong in an interesting way — a window that silently becomes a decade, a
/// local time read as UTC, an all-day event that shifts a day — and it is the
/// only part that can be tested without a device, a calendar and a person.
///
/// The rule throughout: **clamp, never reject.** A request for a year of
/// events is not an attack and not a mistake worth failing; it is a plan that
/// asked for more than the contract allows, and the honest answer is the
/// allowed amount plus an explicit note that it was trimmed. Failing instead
/// would leave the reasoning run with nothing when it could have had something.
public struct PersonalScope: Sendable, Equatable {
    /// A window longer than this is trimmed. Two months covers "what's on next
    /// month" and every ordinary planning question, and stops one careless
    /// plan pulling a decade of records across the network.
    public static let maximumWindow: TimeInterval = 62 * 24 * 60 * 60
    public static let maximumItems = 200

    public let start: Date
    public let end: Date
    public let maxItems: Int
    /// True when the request asked for more than the contract allows. Carried
    /// through to the result so a caller can say "here is the first 200"
    /// rather than implying it saw everything.
    public let trimmed: Bool

    public init(start: Date, end: Date, maxItems: Int) {
        // An inverted window is a mistake, not a request for nothing: reading
        // it as an empty range would answer "you have no events" to a question
        // that was never asked properly.
        let (lower, upper) = start <= end ? (start, end) : (end, start)
        var boundedEnd = upper
        var wasTrimmed = start > end
        if upper.timeIntervalSince(lower) > Self.maximumWindow {
            boundedEnd = lower.addingTimeInterval(Self.maximumWindow)
            wasTrimmed = true
        }
        let boundedItems = min(max(1, maxItems), Self.maximumItems)
        if boundedItems != maxItems { wasTrimmed = true }

        self.start = lower
        self.end = boundedEnd
        self.maxItems = boundedItems
        self.trimmed = wasTrimmed
    }
}

/// One calendar event or reminder, in the shape that crosses to Nova.
///
/// Deliberately flat and deliberately small. Whatever EventKit hands back has
/// far more on it, and none of the rest has ever been needed to answer a
/// question — so it does not travel.
public struct PersonalRecord: Sendable, Equatable, Codable {
    /// Stable across queries so a later mutation can name exactly this item,
    /// and opaque so it carries nothing about the content.
    public let id: String
    public let title: String
    public let start: Date?
    public let end: Date?
    public let allDay: Bool
    public let completed: Bool?
    public let listName: String?

    public init(
        id: String,
        title: String,
        start: Date? = nil,
        end: Date? = nil,
        allDay: Bool = false,
        completed: Bool? = nil,
        listName: String? = nil
    ) {
        self.id = id
        self.title = title
        self.start = start
        self.end = end
        self.allDay = allDay
        self.completed = completed
        self.listName = listName
    }
}

public struct PersonalQueryResult: Sendable, Equatable {
    public let records: [PersonalRecord]
    public let truncated: Bool

    public init(records: [PersonalRecord], truncated: Bool) {
        self.records = records
        self.truncated = truncated
    }

    /// Apply the cap after sorting, and say so when anything was dropped.
    ///
    /// Sorting first matters: capping an unsorted set returns an arbitrary
    /// subset, and "your next three events" made of three random ones is worse
    /// than an error.
    public static func bounded(
        _ records: [PersonalRecord], scope: PersonalScope
    ) -> PersonalQueryResult {
        let sorted = records.sorted { left, right in
            switch (left.start, right.start) {
            case let (lhs?, rhs?): return lhs == rhs ? left.id < right.id : lhs < rhs
            // Undated items sort last. A reminder with no due date is real and
            // must not be dropped, but it is never the answer to "what's next".
            case (nil, _?): return false
            case (_?, nil): return true
            case (nil, nil): return left.id < right.id
            }
        }
        let kept = Array(sorted.prefix(scope.maxItems))
        return PersonalQueryResult(
            records: kept,
            truncated: scope.trimmed || kept.count < sorted.count
        )
    }
}

/// What the device can currently do, as iOS reports it.
///
/// Modelled as four states rather than a boolean because they need different
/// answers: `notDetermined` means ask, `denied` means the owner said no,
/// `restricted` means they cannot say yes, and `limited` means they said yes to
/// part of it. Collapsing those to "unavailable" would have Nova telling
/// someone to change a setting they are not allowed to change.
public enum PersonalPermission: String, Sendable, Codable, CaseIterable {
    case notDetermined = "not_determined"
    case denied
    case restricted
    case limited
    case authorised

    /// Can a read be attempted at all?
    public var usable: Bool {
        self == .authorised || self == .limited
    }

    /// Is there any point asking the owner again?
    ///
    /// `denied` is deliberately false: iOS will not re-prompt once the owner
    /// has said no, so an app that keeps trying achieves nothing except
    /// looking broken. The honest move is to send them to Settings.
    public var worthRequesting: Bool {
        self == .notDetermined
    }
}

/// Which personal capabilities to advertise, given what iOS currently allows.
///
/// Advertising a tool the device cannot serve costs Nova a round trip and a
/// refusal on every attempt, so a denied permission removes its tools from the
/// hello rather than leaving them to fail. Denying one must not affect the
/// others — that is what keeps a single "no" from disabling the companion.
public func availablePersonalTools(
    calendar: PersonalPermission,
    reminders: PersonalPermission,
    location: PersonalPermission,
    health: PersonalPermission
) -> [String] {
    var tools: [String] = []
    if calendar.usable { tools.append("companion.calendar.list") }
    if reminders.usable { tools.append("companion.reminders.list") }
    if location.usable { tools.append("companion.location.current") }
    if health.usable { tools.append("companion.health.summary") }
    return tools
}
