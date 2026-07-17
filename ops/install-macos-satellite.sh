#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MACOS_DIR="$SOURCE_DIR/satellites/macos"
APP="${NOVA_VOICE_APP_PATH:-$HOME/Applications/Nova Voice Satellite.app}"
IDENTITY="${NOVA_VOICE_CODESIGN_IDENTITY:-}"
STAGE="$(mktemp -d)/Nova Voice Satellite.app"
trap 'rm -rf "$(dirname -- "$STAGE")"' EXIT

swift build --package-path "$MACOS_DIR" -c release
if [[ -z "$IDENTITY" ]]; then
  echo "Set NOVA_VOICE_CODESIGN_IDENTITY to an approved Apple code-signing identity." >&2
  security find-identity -v -p codesigning >&2 || true
  exit 2
fi
if ! security find-identity -v -p codesigning 2>/dev/null \
  | grep -Fq "\"$IDENTITY\""; then
  echo "Configured code-signing identity is unavailable: $IDENTITY" >&2
  exit 2
fi
install -d "$STAGE/Contents/MacOS"
install -m 0755 "$MACOS_DIR/.build/release/NovaVoiceSatellite" \
  "$STAGE/Contents/MacOS/NovaVoiceSatellite"
install -m 0644 "$MACOS_DIR/Resources/Info.plist" "$STAGE/Contents/Info.plist"
codesign --force --options runtime \
  --entitlements "$MACOS_DIR/Resources/NovaVoiceSatellite.entitlements" \
  --sign "$IDENTITY" "$STAGE"
install -d "$(dirname -- "$APP")"
ditto "$STAGE" "$APP"
# This legacy LaunchAgent declares AssociatedBundleIdentifiers below.  Ensure
# Launch Services has registered the signed bundle before the agent starts, so
# macOS 15 can attribute its Local Network privacy request to this app.
LSREGISTER="/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
if [[ -x "$LSREGISTER" ]]; then
  "$LSREGISTER" -f "$APP"
fi

install -d "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
PLIST="$HOME/Library/LaunchAgents/nz.co.skull.NovaVoiceSatellite.plist"
install -m 0644 "$MACOS_DIR/Resources/nz.co.skull.NovaVoiceSatellite.plist" "$PLIST"
# The checked-in plist is a portable template.  Resolve the actual bundle and
# keychain paths at install time; LaunchAgent environment values do not expand
# shell variables, and a stale username would otherwise prevent startup or
# mTLS even though the app itself is correctly signed.
CLIENT_BASE="$HOME/Library/Application Support/NovaVoiceSatellite"
/usr/bin/plutil -remove ProgramArguments "$PLIST"
/usr/bin/plutil -insert ProgramArguments -array "$PLIST"
/usr/bin/plutil -insert ProgramArguments.0 -string \
  "$APP/Contents/MacOS/NovaVoiceSatellite" "$PLIST"
/usr/bin/plutil -replace EnvironmentVariables.NOVA_VOICE_CLIENT_KEYCHAIN_PATH -string \
  "$CLIENT_BASE/client.keychain-db" "$PLIST"
/usr/bin/plutil -replace EnvironmentVariables.NOVA_VOICE_CLIENT_KEYCHAIN_PASSWORD_PATH -string \
  "$CLIENT_BASE/client-keychain-password" "$PLIST"
/usr/bin/plutil -replace EnvironmentVariables.NOVA_VOICE_CA_CERTIFICATE_PATH -string \
  "$CLIENT_BASE/ca.crt" "$PLIST"
/usr/bin/plutil -replace StandardErrorPath -string \
  "$HOME/Library/Logs/NovaVoiceSatellite.log" "$PLIST"
launchctl bootout "gui/$(id -u)/nz.co.skull.NovaVoiceSatellite" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" \
  "$PLIST"
launchctl enable "gui/$(id -u)/nz.co.skull.NovaVoiceSatellite"
echo "Installed signed LaunchAgent. Allow Local Network, then approve microphone access in the GUI session."
