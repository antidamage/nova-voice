import Foundation
import XCTest

@testable import NovaCompanionKit

/// The signed challenge material, pinned against the Python implementation.
///
/// A divergence here does not surface as a decoding error. It surfaces as a
/// signature that never verifies, from a phone, at the moment someone is
/// trying to use the thing — so it is worth a cross-language vector rather
/// than trusting that two prose descriptions were read the same way.
final class ChallengeTests: XCTestCase {
    private struct Vector: Decodable {
        let nonce: String
        let protocolVersion: Int
        let announcedId: String
        let roles: [String]
        let material: String
    }

    private static var vectorsURL: URL {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<6 { url.deleteLastPathComponent() }
        return url.appendingPathComponent("docs/companion-protocol/challenge-material.json")
    }

    private func vectors() throws -> [Vector] {
        try JSONDecoder().decode([Vector].self, from: try Data(contentsOf: Self.vectorsURL))
    }

    func testMaterialMatchesThePythonImplementation() throws {
        let cases = try vectors()
        XCTAssertFalse(cases.isEmpty)
        for vector in cases {
            let produced = CompanionChallenge.material(
                nonce: vector.nonce,
                protocolVersion: vector.protocolVersion,
                announcedId: vector.announcedId,
                roles: vector.roles
            )
            XCTAssertEqual(
                String(decoding: produced, as: UTF8.self),
                vector.material,
                "challenge material diverged for nonce \(vector.nonce)"
            )
        }
    }

    func testRoleOrderAndCaseDoNotChangeTheSignedBytes() {
        let first = CompanionChallenge.material(
            nonce: "n", announcedId: "companion-1", roles: ["satellite", "companion"]
        )
        let second = CompanionChallenge.material(
            nonce: "n", announcedId: "companion-1", roles: ["Companion", "SATELLITE"]
        )
        XCTAssertEqual(first, second)
    }

    func testEachInputChangesTheSignedBytes() {
        // All four are covered by one signature precisely so none of them can
        // be substituted after the fact.
        let base = CompanionChallenge.material(
            nonce: "n", announcedId: "companion-1", roles: ["companion"]
        )
        XCTAssertNotEqual(
            base,
            CompanionChallenge.material(
                nonce: "other", announcedId: "companion-1", roles: ["companion"]
            )
        )
        XCTAssertNotEqual(
            base,
            CompanionChallenge.material(
                nonce: "n", announcedId: "indium", roles: ["companion"]
            )
        )
        XCTAssertNotEqual(
            base,
            CompanionChallenge.material(
                nonce: "n", announcedId: "companion-1", roles: ["satellite"]
            )
        )
        XCTAssertNotEqual(
            base,
            CompanionChallenge.material(
                nonce: "n", protocolVersion: 2, announcedId: "companion-1", roles: ["companion"]
            )
        )
    }

    func testTypedRolesProduceTheSameBytesAsStrings() {
        XCTAssertEqual(
            CompanionChallenge.material(
                nonce: "n", announcedId: "companion-1", roles: [CompanionRole.companion]
            ),
            CompanionChallenge.material(
                nonce: "n", announcedId: "companion-1", roles: ["companion"]
            )
        )
    }
}
