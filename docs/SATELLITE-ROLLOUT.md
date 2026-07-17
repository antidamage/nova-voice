# Satellite rollout

Nova Voice runs inference only on Iridium (address: see `PRIVATEREF.md#1.1`).
The Nova dashboard VM (hostname: see `PRIVATEREF.md#1.4`) runs on Indium and is
reached only by the provider adapter.  It is not a Voice deployment or
satellite target.

## Safe order

1. Keep Iridium in `development` + `shadow` mode and confirm its mutually
   authenticated `/health` response is healthy.
2. On Iridium, issue a one-device identity without replacing the household CA:

   ```sh
   sudo /opt/nova-voice/current/ops/issue-satellite-identity.sh \
     nocturnium /secure-transfer/nocturnium
   ```

   For Indium, pass a root-readable file containing a freshly generated
   transfer password as the third argument; the output also contains
   `indium.p12`. On macOS supply that same file through
   `NOVA_VOICE_P12_PASSWORD_FILE` when running the keychain provisioner.
   Transfer private key material only over an administrator-controlled path,
   then remove the transfer copy.
3. Install Nocturnium as the kiosk user's `systemd --user` service. Copy its
   PEM identity to `~/.config/nova-voice/tls/`, run
   `ops/configure-pipewire-aec.sh --restart` against the connected physical
   microphone and speakers, and only then start the service. Keep linger
   enabled. The service is always-on and is independent of the kiosk/browser.
4. On Indium, import `indium.p12` and `ca.crt` with
   `ops/provision-macos-keychains.sh`, build/install the signed native helper,
   explicitly set `NOVA_VOICE_CODESIGN_IDENTITY` to the approved Apple
   code-signing identity, then allow **Nova Voice Satellite** in **System
   Settings > Privacy & Security > Local Network**. Approve microphone access
   in the logged-in GUI session after it connects, and verify its
   `RunAtLoad`/`KeepAlive` LaunchAgent. The Local Network permission is a
   macOS 15 user-consent decision and must not be worked around by changing
   system-wide network privacy defaults. The helper is deliberately not a web
   mic or a LaunchDaemon.
5. Confirm exactly one authenticated stream appears per satellite, force-stop
   each client to prove supervisor recovery, then perform the WER, echo,
   duplicate-command, latency, and endurance gates in `docs/TESTING.md`.

Do not enable passive execution until the required shadow observation period
has passed. The satellite installers enable their supervisor but intentionally
do not start capture before its TLS identity and audio path are validated.

## Indium interactive signing gate

If a remote maintenance session reports `errSecInternalComponent` from
`codesign`, do not replace the existing helper. On Indium's logged-in GUI
session, unlock/approve the selected Apple Development identity and run the
staged installer with that exact identity, for example:

```sh
# Approved identity string: see PRIVATEREF.md#3.1
export NOVA_VOICE_CODESIGN_IDENTITY='Apple Development: <name> (<team id>)'
bash "$HOME/Library/Application Support/NovaVoiceSatellite/releases/20260714/ops/install-macos-satellite.sh"
```

Approve the resulting microphone prompt, then confirm the health file changes
to `connected: true`. This is an operating-system authorization step, not a
dashboard or Nova VM change.

On macOS 15, the Local Network alert may not repeat after an earlier denial.
If the health file records `NSURLErrorDomain:-1009`, use **System Settings >
Privacy & Security > Local Network** to allow **Nova Voice Satellite**, then
confirm its health file changes to `connected: true` before testing
microphone capture.
