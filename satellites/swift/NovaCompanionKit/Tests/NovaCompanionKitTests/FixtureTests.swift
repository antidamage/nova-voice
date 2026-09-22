import Foundation
import XCTest

@testable import NovaCompanionKit

/// Decode the committed protocol fixtures.
///
/// These are the *same* files the server's own tests round-trip, written from
/// the server's models. That is how the two implementations are kept in
/// agreement without a shared code generator: if either side changes the wire
/// incompatibly, one of these fails.
final class FixtureTests: XCTestCase {
    private let codec = CompanionCodec()

    /// Fixtures live in the repository, not in the test bundle, because they
    /// belong to the protocol rather than to this package. Walking up from the
    /// source file keeps them shared with the Python suite instead of copied.
    private static var fixtureDirectory: URL {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<6 { url.deleteLastPathComponent() }
        return url.appendingPathComponent("docs/companion-protocol")
    }

    private func fixture(_ name: String) throws -> Data {
        let url = Self.fixtureDirectory.appendingPathComponent("\(name).json")
        return try Data(contentsOf: url)
    }

    func testFixtureDirectoryIsFound() throws {
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: Self.fixtureDirectory.path),
            "fixtures not found at \(Self.fixtureDirectory.path)"
        )
    }

    func testEveryMessageTypeHasAFixtureThatDecodes() throws {
        for type in CompanionMessageType.allCases {
            let data = try fixture(type.rawValue)
            let message = try codec.decode(data)
            XCTAssertEqual(message.type, type, "fixture \(type.rawValue) decoded as \(message.type)")
        }
    }

    func testJobOfferCarriesItsEnvelope() throws {
        guard case .jobOffer(let offer) = try codec.decode(try fixture("job_offer")) else {
            return XCTFail("expected a job offer")
        }
        XCTAssertEqual(offer.envelope.workload, .interpret)
        XCTAssertEqual(offer.envelope.resultSchema, "interpret.v1")
        XCTAssertEqual(offer.envelope.locality, .homeLan)
        XCTAssertEqual(offer.callbackBudget, 12)
        XCTAssertEqual(offer.contextTokens, 4096)
        // The callback bounds travel with the offer, so the device knows what
        // it may call and for how long without inferring it from the prompt.
        XCTAssertEqual(offer.toolCatalogue, ["nova.light_set", "nova.scene_apply"])
        XCTAssertEqual(offer.callbackDeadlineSeconds, 8.0)
        XCTAssertEqual(offer.callbackBudgetSeconds, 30.0)
        XCTAssertEqual(offer.maxConcurrentCallbacks, 2)
        // The accept deadline is deliberately much shorter than completion:
        // before acceptance, falling back costs one round trip.
        XCTAssertLessThan(offer.envelope.acceptDeadline, offer.envelope.completeDeadline)
        XCTAssertEqual(offer.payload["roomId"]?.stringValue, "lounge")
    }

    /// An offer written before the callback bounds existed must still decode.
    ///
    /// The device and the server are updated separately — a 7-day provisioning
    /// profile guarantees the app is sometimes older than Iridium and sometimes
    /// newer — so a missing optional field has to mean "this server does not
    /// send it", not a frame the app throws away.
    func testAnOfferWithoutCallbackBoundsStillDecodes() throws {
        var object = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: try fixture("job_offer")) as? [String: Any]
        )
        for key in [
            "toolCatalogue", "callbackDeadlineSeconds", "callbackBudgetSeconds",
            "maxConcurrentCallbacks",
        ] {
            object.removeValue(forKey: key)
        }
        let data = try JSONSerialization.data(withJSONObject: object)
        guard case .jobOffer(let offer) = try codec.decode(data) else {
            return XCTFail("expected a job offer")
        }
        XCTAssertNil(offer.toolCatalogue)
        XCTAssertNil(offer.callbackDeadlineSeconds)
        XCTAssertEqual(offer.callbackBudget, 12)
    }

    func testTimestampsDecodeAsUTC() throws {
        guard case .jobOffer(let offer) = try codec.decode(try fixture("job_offer")) else {
            return XCTFail("expected a job offer")
        }
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "UTC")!
        let parts = calendar.dateComponents(
            [.year, .month, .day, .hour], from: offer.envelope.createdAt
        )
        XCTAssertEqual(parts.year, 2026)
        XCTAssertEqual(parts.month, 8)
        XCTAssertEqual(parts.day, 13)
        XCTAssertEqual(parts.hour, 9)
    }

    func testHelloAdvertisesTheWorkloadsTheServerKnows() throws {
        guard case .hello(let hello) = try codec.decode(try fixture("hello")) else {
            return XCTFail("expected a hello")
        }
        XCTAssertEqual(Set(hello.workloads), Set(CompanionWorkload.allCases))
        XCTAssertTrue(hello.telemetry.models.hotAvailable)
        XCTAssertEqual(hello.telemetry.models.hotContextTokens, 4096)
        XCTAssertEqual(hello.telemetry.permissions.health, .denied)
    }

    func testPersonalResultCarriesScopedItems() throws {
        guard case .personalResult(let result) = try codec.decode(try fixture("personal_result"))
        else {
            return XCTFail("expected a personal result")
        }
        XCTAssertEqual(result.sensitivity, .personal)
        XCTAssertEqual(result.items.count, 2)
        XCTAssertFalse(result.truncated)
        XCTAssertEqual(result.items.first?["id"]?.stringValue, "opaque-1")
    }

    func testApprovalRequestStatesExactlyWhatWillChange() throws {
        guard case .approvalRequest(let request) = try codec.decode(try fixture("approval_request"))
        else {
            return XCTFail("expected an approval request")
        }
        XCTAssertEqual(request.sensitivity, .mutation)
        XCTAssertEqual(request.tool, "reminders.create")
        XCTAssertFalse(request.summary.isEmpty)
    }

    // MARK: - Strictness

    func testUnknownMessageTypeIsRejected() throws {
        let data = try Data(
            contentsOf: Self.fixtureDirectory.appendingPathComponent("invalid/unknown_type.json")
        )
        XCTAssertThrowsError(try codec.decode(data)) { error in
            XCTAssertEqual(
                error as? CompanionCodecError,
                .unknownMessageType("take_over_the_house")
            )
        }
    }

    func testAFrameWithoutATypeIsRejected() throws {
        let data = Data(#"{"jobId":"x"}"#.utf8)
        XCTAssertThrowsError(try codec.decode(data)) { error in
            XCTAssertEqual(error as? CompanionCodecError, .missingType)
        }
    }

    func testMissingRequiredFieldsAreRejected() throws {
        let data = try Data(
            contentsOf: Self.fixtureDirectory.appendingPathComponent(
                "invalid/missing_required_field.json"
            )
        )
        XCTAssertThrowsError(try codec.decode(data))
    }

    func testAnOversizedFrameIsRejectedBeforeParsing() throws {
        let padding = String(repeating: "x", count: CompanionProtocol.maxFrameBytes)
        let data = Data(#"{"type":"heartbeat","sentAt":"2026-08-13T09:00:00Z","pad":"\#(padding)"}"#.utf8)
        XCTAssertThrowsError(try codec.decode(data)) { error in
            guard case .frameTooLarge = error as? CompanionCodecError else {
                return XCTFail("expected frameTooLarge, got \(error)")
            }
        }
    }

    // MARK: - Encoding

    func testEncodedFramesCarryTheirDiscriminator() throws {
        let data = try codec.encode(Heartbeat(sentAt: Date(timeIntervalSince1970: 0)), as: .heartbeat)
        guard case .heartbeat = try codec.decode(data) else {
            return XCTFail("a heartbeat did not survive a round trip")
        }
    }

    func testAnAuthResponseRoundTrips() throws {
        let response = AuthResponse(
            announcedId: "companion-1",
            roles: [.companion, .satellite],
            certificateChain: ["-----BEGIN CERTIFICATE-----\nPLACEHOLDER\n-----END CERTIFICATE-----\n"],
            signature: "MEUCIQD-placeholder"
        )
        let data = try codec.encode(response, as: .authResponse)
        guard case .authResponse(let decoded) = try codec.decode(data) else {
            return XCTFail("expected an auth response")
        }
        XCTAssertEqual(decoded.announcedId, "companion-1")
        XCTAssertEqual(decoded.roles, [.companion, .satellite])
        XCTAssertEqual(decoded.protocolVersion, CompanionProtocol.version)
    }
}

extension CompanionRole: @retroactive Equatable {}
