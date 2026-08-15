import Foundation
import NovaCompanionKit
import Security

/// Signs the server's challenge with the household identity in the keychain.
///
/// The identity arrives as a `.p12` from `ops/issue-satellite-identity.sh` and
/// is imported once. It is stored `AfterFirstUnlock` deliberately: without
/// that, the app cannot reconnect after a reboot until someone physically
/// unlocks the phone, which for a companion that is meant to be reachable is
/// the difference between working and not.
struct KeychainSigner: ChallengeSigner {
    enum SignerError: Error {
        case identityNotFound
        case certificateUnavailable
        case signingFailed(String)
    }

    let label: String

    // MARK: - Import

    /// Import a `.p12` into the keychain. Idempotent: importing the same
    /// identity twice leaves one entry.
    static func importIdentity(p12: Data, passphrase: String, label: String) throws {
        var items: CFArray?
        let status = SecPKCS12Import(
            p12 as CFData,
            [kSecImportExportPassphrase as String: passphrase] as CFDictionary,
            &items
        )
        guard status == errSecSuccess,
            let entries = items as? [[String: Any]],
            let identity = entries.first?[kSecImportItemIdentity as String]
        else {
            throw SignerError.signingFailed("SecPKCS12Import failed: \(status)")
        }

        // macOS keychain APIs (SecKeychainOpen/Unlock) do not exist on iOS, so
        // the identity goes into the app's own keychain with an explicit
        // accessibility class rather than into a named keychain file.
        let attributes: [String: Any] = [
            kSecClass as String: kSecClassIdentity,
            kSecValueRef as String: identity,
            kSecAttrLabel as String: label,
            kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlock,
        ]
        let added = SecItemAdd(attributes as CFDictionary, nil)
        guard added == errSecSuccess || added == errSecDuplicateItem else {
            throw SignerError.signingFailed("SecItemAdd failed: \(added)")
        }
    }

    // MARK: - Lookup

    func secIdentity() throws -> SecIdentity {
        let query: [String: Any] = [
            kSecClass as String: kSecClassIdentity,
            kSecAttrLabel as String: label,
            kSecReturnRef as String: true,
        ]
        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        guard status == errSecSuccess, let found = item else {
            throw SignerError.identityNotFound
        }
        return found as! SecIdentity
    }

    // MARK: - ChallengeSigner

    func certificateChainPEM() throws -> [String] {
        var certificate: SecCertificate?
        guard SecIdentityCopyCertificate(try secIdentity(), &certificate) == errSecSuccess,
            let certificate
        else {
            throw SignerError.certificateUnavailable
        }
        let der = SecCertificateCopyData(certificate) as Data
        // The server parses PEM, so the DER is wrapped here rather than
        // shipping a second encoding of the same bytes over the wire.
        let base64 = der.base64EncodedString(options: [.lineLength64Characters, .endLineWithLineFeed])
        return [
            "-----BEGIN CERTIFICATE-----\n\(base64)\n-----END CERTIFICATE-----\n"
        ]
    }

    func sign(_ material: Data) throws -> Data {
        var privateKey: SecKey?
        guard SecIdentityCopyPrivateKey(try secIdentity(), &privateKey) == errSecSuccess,
            let privateKey
        else {
            throw SignerError.signingFailed("no private key for the stored identity")
        }

        // ECDSA over SHA-256, matching what the server verifies for the
        // EC P-256 keys `issue-satellite-identity.sh` mints.
        var error: Unmanaged<CFError>?
        guard
            let signature = SecKeyCreateSignature(
                privateKey,
                .ecdsaSignatureMessageX962SHA256,
                material as CFData,
                &error
            )
        else {
            let reason = error?.takeRetainedValue().localizedDescription ?? "unknown"
            throw SignerError.signingFailed(reason)
        }
        return signature as Data
    }
}
