import Foundation
import XCTest

@testable import NovaCompanionKit

/// NPT-204. The ordering rule that decides whether a session is `home_lan`.
///
/// Worth testing precisely because the failure looks like something else: a
/// phone that reaches the tailnet address first while sitting on the home
/// Wi-Fi is correctly classified `tailnet` and correctly refused the home-LAN
/// routes, and every symptom points at routing rather than at ordering.
final class EndpointOrderTests: XCTestCase {
    private let lan = URL(string: "wss://voice.invalid:8766/v1/companion")!
    private let tailnet = URL(string: "wss://voice.tailnet.invalid:8766/v1/companion")!

    func testTheLanAddressIsAlwaysTriedFirst() {
        XCTAssertEqual(
            CompanionEndpoints.ordered(lan: lan, tailnet: tailnet), [lan, tailnet]
        )
    }

    func testTheTailnetAddressAloneIsStillUsable() {
        // Away from home, personal tools are still permitted over the tailnet;
        // only the reasoning-replacement routes are not.
        XCTAssertEqual(CompanionEndpoints.ordered(lan: nil, tailnet: tailnet), [tailnet])
    }

    func testAnUnconfiguredDeviceHasNowhereToGo() {
        // Better than defaulting to something: a wrong endpoint would produce
        // a confident connection failure instead of "not configured".
        XCTAssertTrue(CompanionEndpoints.ordered(lan: nil, tailnet: nil).isEmpty)
    }

    func testACleanDisconnectRestartsFromTheLanAddress() {
        // A session that ran and ended is not evidence the address was bad —
        // the server restarted, or the app was backgrounded. Carrying on down
        // the list would quietly demote the phone to `tailnet` for the rest of
        // the day over one ordinary reconnect.
        XCTAssertEqual(
            CompanionEndpoints.afterCleanDisconnect(lan: lan, tailnet: tailnet),
            [lan, tailnet]
        )
    }

    func testOnlyTheLanAddressIsAHomeCandidate() {
        XCTAssertTrue(CompanionEndpoints.isHomeCandidate(lan, lan: lan))
        XCTAssertFalse(CompanionEndpoints.isHomeCandidate(tailnet, lan: lan))
    }

    func testNothingIsAHomeCandidateWithoutALanAddressConfigured() {
        // The device must not be able to promote itself by having only one
        // address; locality is the server's call from the peer address.
        XCTAssertFalse(CompanionEndpoints.isHomeCandidate(tailnet, lan: nil))
    }
}
