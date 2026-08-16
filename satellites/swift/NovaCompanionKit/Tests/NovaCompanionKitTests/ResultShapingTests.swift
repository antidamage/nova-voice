import Foundation
import XCTest

@testable import NovaCompanionKit

/// NPT-905. The shaping layer between a small on-device model and Nova's
/// domain types.
///
/// This is where a real bug lived: actions were emitted with `name` where the
/// domain wants `tool`, and without the `order` a `PlannedAction` requires.
/// Both parse fine on the device and are rejected on arrival, so every
/// companion plan failed and the only symptom was a fallback. A test catches
/// that in a second; a code review did not.
final class ResultShapingTests: XCTestCase {
    // MARK: - The action shape

    func testAnActionUsesTheFieldNamesTheDomainActuallyRequires() {
        let action = CompanionResultShaping.action(
            index: 0, provider: "nova", tool: "light_set", argumentsJSON: "{}"
        )

        guard case .object(let fields) = action else { return XCTFail("not an object") }
        // `order` and `depends_on` are required, not decorative.
        XCTAssertNotNil(fields["order"])
        XCTAssertNotNil(fields["depends_on"])

        guard case .object(let call)? = fields["call"] else { return XCTFail("no call") }
        // The one that bit: `tool`, never `name`.
        XCTAssertNotNil(call["tool"])
        XCTAssertNil(call["name"])
        XCTAssertEqual(call["tool"]?.stringValue, "light_set")
        XCTAssertEqual(call["provider"]?.stringValue, "nova")
    }

    func testActionsAreOrderedSequentiallyAndDependOnNothing() {
        // A model asked to invent an execution order invents dependencies with
        // it, and a wrong dependency deadlocks a plan rather than misordering
        // it. Spoken clauses run in the order they were spoken.
        for index in 0..<3 {
            let action = CompanionResultShaping.action(
                index: index, provider: "nova", tool: "light_set", argumentsJSON: "{}"
            )
            guard case .object(let fields) = action else { return XCTFail("not an object") }
            XCTAssertEqual(fields["order"]?.doubleValue, Double(index))
            XCTAssertEqual(fields["id"]?.stringValue, "companion-\(index)")
            XCTAssertEqual(fields["depends_on"]?.arrayValue?.count, 0)
        }
    }

    func testMalformedArgumentsBecomeNoArgumentsRatherThanABrokenPlan() {
        // The domain field is a mapping. An action with unparseable arguments
        // should be an action with none, not a plan that fails as a whole.
        for text in ["not json", "[1,2,3]", "\"a string\"", "", "null", "42"] {
            let arguments = CompanionResultShaping.decodeArguments(text)
            XCTAssertEqual(arguments.objectValue?.isEmpty, true, "for \(text)")
        }
    }

    func testWellFormedArgumentsSurvive() {
        let arguments = CompanionResultShaping.decodeArguments(#"{"state":"on","level":40}"#)

        XCTAssertEqual(arguments["state"]?.stringValue, "on")
        XCTAssertEqual(arguments["level"]?.intValue, 40)
    }

    // MARK: - Enum coercion

    func testAnUnrecognisedDecisionBecomesTheOneThatDoesNothing() {
        // Failing validation costs the whole attempt, so an answer outside the
        // enum is corrected rather than sent — to the member that does the
        // least if the model was wrong.
        XCTAssertEqual(
            CompanionResultShaping.oneOf(
                "do_the_thing",
                CompanionResultShaping.decisions,
                fallback: CompanionResultShaping.safeDecision
            ),
            "ignore"
        )
    }

    func testARecognisedValueIsKeptWhateverItsCasingAndPadding() {
        XCTAssertEqual(
            CompanionResultShaping.oneOf(
                "  EXECUTE \n", CompanionResultShaping.decisions, fallback: "ignore"
            ),
            "execute"
        )
    }

    func testTheMirroredEnumsMatchTheServersOwn() {
        // These are a copy of Nova's domain enums. A drift here is a companion
        // that silently fails validation on a value the server does have.
        XCTAssertEqual(CompanionResultShaping.decisions, ["execute", "reply", "clarify", "ignore"])
        XCTAssertTrue(CompanionResultShaping.speechActs.contains("desired_state"))
        XCTAssertTrue(CompanionResultShaping.goalStatuses.contains("needs_clarification"))
        XCTAssertTrue(CompanionResultShaping.emotions.contains("neutral"))
        // The safe members must themselves be members.
        XCTAssertTrue(
            CompanionResultShaping.decisions.contains(CompanionResultShaping.safeDecision)
        )
        XCTAssertTrue(
            CompanionResultShaping.speechActs.contains(CompanionResultShaping.safeSpeechAct)
        )
    }

    // MARK: - Numbers

    func testConfidenceIsClampedIntoRange() {
        XCTAssertEqual(CompanionResultShaping.clamp(1.7), 1)
        XCTAssertEqual(CompanionResultShaping.clamp(-0.3), 0)
        XCTAssertEqual(CompanionResultShaping.clamp(0.42), 0.42)
    }

    func testANotANumberConfidenceDoesNotSlipThrough() {
        // NaN compares false against everything, so a plain min/max would pass
        // it straight into a field typed 0...1.
        XCTAssertEqual(CompanionResultShaping.clamp(.nan), 0)
        XCTAssertEqual(CompanionResultShaping.clamp(.infinity), 0)
    }

    // MARK: - Optional strings

    func testTheModelsOwnFillerIsTreatedAsAbsent() {
        // "none" stored as somebody's name is worse than no name at all.
        for filler in ["none", "None", " null ", "n/a", "unknown", "unspecified", "  ", ""] {
            XCTAssertNil(CompanionResultShaping.tidy(filler), "for \(filler)")
        }
    }

    func testARealValueIsKeptAndTrimmed() {
        XCTAssertEqual(CompanionResultShaping.tidy("  Adeline \n"), "Adeline")
        // A name that merely contains a placeholder word is still a name.
        XCTAssertEqual(CompanionResultShaping.tidy("Noone"), "Noone")
    }

    // MARK: - Prompt rendering

    func testStateIsRenderedCompactlyAndDeterministically() {
        // The prompt budget is 4k tokens total, and an ordering that varied
        // between turns would make the prompt uncacheable and the output
        // irreproducible.
        let state = JSONValue.object([
            "temperature": .number(22),
            "on": .bool(true),
            "name": .string("lounge"),
        ])

        XCTAssertEqual(
            CompanionResultShaping.describe(state), "name=lounge on=yes temperature=22"
        )
    }

    func testAWholeNumberLosesItsDecimalTail() {
        XCTAssertEqual(CompanionResultShaping.describe(.number(22)), "22")
        XCTAssertEqual(CompanionResultShaping.describe(.number(22.5)), "22.5")
    }

    func testAnEnormousNumberDoesNotCrashThePrompt() {
        // `Int(exactly:)` guards a conversion that would otherwise trap, and a
        // prompt is not worth a crash.
        let rendered = CompanionResultShaping.describe(.number(1e30))
        XCTAssertFalse(rendered.isEmpty)
    }

    func testNullRendersAsUnknownRatherThanEmpty() {
        // An empty value in a prompt reads as "the field is blank"; the truth
        // is that nobody knows.
        XCTAssertEqual(CompanionResultShaping.describe(.null), "unknown")
    }
}
