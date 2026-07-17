#!/usr/bin/env bash
set -euo pipefail

P12_PATH="${1:-/tmp/indium.p12}"
CA_PATH="${2:-/tmp/ca.crt}"
P12_PASSWORD_FILE="${NOVA_VOICE_P12_PASSWORD_FILE:-}"
P12_PASSWORD="${NOVA_VOICE_P12_PASSWORD:-}"
IDENTITY_LABEL="${NOVA_VOICE_IDENTITY_LABEL:-indium}"
BASE="$HOME/Library/Application Support/NovaVoiceSatellite"
CLIENT_KEYCHAIN="$BASE/client.keychain-db"
CLIENT_PASSWORD_FILE="$BASE/client-keychain-password"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

if [[ -n "$P12_PASSWORD_FILE" ]]; then
  if [[ ! -r "$P12_PASSWORD_FILE" ]]; then
    echo "P12 password file is not readable: $P12_PASSWORD_FILE" >&2
    exit 2
  fi
  P12_PASSWORD="$(<"$P12_PASSWORD_FILE")"
fi
if [[ -z "$P12_PASSWORD" ]]; then
  echo "Set NOVA_VOICE_P12_PASSWORD_FILE (preferred) or NOVA_VOICE_P12_PASSWORD." >&2
  exit 2
fi

has_client_identity() {
  # ``security find-identity`` evaluates the system trust store and is a false
  # negative for this private-CA client. Query the same SecIdentity object that
  # the native satellite uses instead. Swift is already a prerequisite for
  # building the signed helper, and this short-lived probe is removed by the
  # enclosing trap.
  cat >"$WORK/identity-probe.swift" <<'SWIFT'
import Foundation
import Security

let base = NSString(string: NSHomeDirectory())
    .appendingPathComponent("Library/Application Support/NovaVoiceSatellite")
let keychainPath = (base as NSString).appendingPathComponent("client.keychain-db")
let passwordPath = (base as NSString).appendingPathComponent("client-keychain-password")
let label = ProcessInfo.processInfo.environment["NOVA_VOICE_IDENTITY_LABEL"] ?? "indium"
var keychain: SecKeychain?
guard SecKeychainOpen(keychainPath, &keychain) == errSecSuccess,
      let keychain,
      let passwordData = try? Data(contentsOf: URL(fileURLWithPath: passwordPath)),
      let passwordText = String(data: passwordData, encoding: .utf8) else {
    exit(1)
}
let password = Data(passwordText.trimmingCharacters(in: .whitespacesAndNewlines).utf8)
let unlock = password.withUnsafeBytes { bytes in
    SecKeychainUnlock(keychain, UInt32(password.count), bytes.baseAddress, true)
}
guard unlock == errSecSuccess else { exit(1) }
let query: [String: Any] = [
    kSecClass as String: kSecClassIdentity,
    kSecAttrLabel as String: label,
    kSecReturnRef as String: true,
    kSecMatchLimit as String: kSecMatchLimitOne,
    kSecMatchSearchList as String: [keychain],
]
var item: CFTypeRef?
guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess, item != nil else {
    exit(1)
}
SWIFT
  NOVA_VOICE_IDENTITY_LABEL="$IDENTITY_LABEL" \
    /usr/bin/swift "$WORK/identity-probe.swift" >/dev/null
}

install -d -m 0700 "$BASE"
install -m 0600 "$CA_PATH" "$BASE/ca.crt"

if [[ ! -f "$CLIENT_KEYCHAIN" ]]; then
  # ``openssl rand -hex`` emits a trailing newline.  Keep the on-disk secret
  # byte-for-byte identical to the value passed to ``security``; the Swift
  # client reads this file directly when unlocking the keychain.
  printf '%s' "$(openssl rand -hex 32)" >"$CLIENT_PASSWORD_FILE"
  chmod 0600 "$CLIENT_PASSWORD_FILE"
  client_password="$(cat "$CLIENT_PASSWORD_FILE")"
  security create-keychain -p "$client_password" "$CLIENT_KEYCHAIN"
  security unlock-keychain -p "$client_password" "$CLIENT_KEYCHAIN"
  security set-keychain-settings -lut 21600 "$CLIENT_KEYCHAIN"
fi
client_password="$(cat "$CLIENT_PASSWORD_FILE")"
security unlock-keychain -p "$client_password" "$CLIENT_KEYCHAIN"
# Re-import when a prior run created only the keychain shell or imported a
# certificate without its private key.  URLSession needs a SecIdentity, not a
# standalone certificate, for the mTLS challenge.
if ! has_client_identity; then
  security import "$P12_PATH" -k "$CLIENT_KEYCHAIN" -P "$P12_PASSWORD" -A >/dev/null
  security set-key-partition-list -S apple-tool:,apple: -s \
    -k "$client_password" "$CLIENT_KEYCHAIN" >/dev/null
fi
if ! has_client_identity; then
  echo "No usable client identity '$IDENTITY_LABEL' in $CLIENT_KEYCHAIN" >&2
  exit 1
fi

echo "Provisioned isolated client Keychain under $BASE"
