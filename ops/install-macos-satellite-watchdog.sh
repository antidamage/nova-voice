#!/bin/sh
# Install the Nova Voice macOS satellite connection watchdog.
#
# Why this exists: the satellite process reconnects to the voice server by
# relying on URLSessionWebSocketTask.receive() throwing when the socket drops.
# URLSession WebSockets have no ping/keepalive and no read timeout, so a silent
# half-open drop (Wi-Fi blip, server restart, sleep/wake) can leave receive()
# blocked forever. The process stays alive but disconnected, and because it
# never crashes, launchd KeepAlive never relaunches it — the satellite can sit
# dead for hours (observed: connected at 02:05, silently offline all night).
#
# This watchdog is a launchd side-car that force-reconnects the satellite when
# it holds no established socket to the voice server across two consecutive
# checks. It touches only launchctl/lsof — it does NOT modify or re-sign the
# signed satellite .app, so it can be installed headlessly over SSH.
#
# The in-app root-cause fix (an application-level ping watchdog inside
# main.swift) is the preferred long-term fix but requires a signed rebuild.
# Keep this side-car regardless; it is cheap and also covers hard hangs.
set -eu

LABEL="nz.co.skull.NovaVoiceSatelliteWatchdog"
SAT_LABEL="nz.co.skull.NovaVoiceSatellite"
SERVER="${NOVA_VOICE_SERVER_HOSTPORT:-192.168.8.20:8766}"
SUPPORT_DIR="$HOME/Library/Application Support/NovaVoiceSatellite"
SCRIPT_PATH="$SUPPORT_DIR/watchdog.sh"
PLIST_PATH="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG_PATH="$HOME/Library/Logs/NovaVoiceSatelliteWatchdog.log"
DOMAIN="gui/$(id -u)"

mkdir -p "$SUPPORT_DIR" "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"

cat > "$SCRIPT_PATH" <<EOF
#!/bin/sh
# Nova Voice satellite connection watchdog (installed by
# ops/install-macos-satellite-watchdog.sh). Force-reconnects the satellite when
# it has held no established socket to the voice server across two consecutive
# checks. The two-strike state file avoids racing the app's own reconnect
# backoff (which tops out at 30s, well under one check interval).
LABEL="$SAT_LABEL"
SERVER="$SERVER"
DOMAIN="gui/\$(id -u)"
STATE="$SUPPORT_DIR/watchdog.miss"

PID=\$(launchctl print "\$DOMAIN/\$LABEL" 2>/dev/null | awk -F' = ' '/[[:space:]]pid = /{print \$2; exit}')

if [ -n "\$PID" ] && /usr/sbin/lsof -nP -p "\$PID" -iTCP -sTCP:ESTABLISHED 2>/dev/null | grep -q "\$SERVER"; then
  rm -f "\$STATE"
  exit 0
fi

# No server connection on this check.
if [ ! -f "\$STATE" ]; then
  # First miss: record it and give the app's own reconnect one interval.
  date +%s > "\$STATE"
  exit 0
fi

# Second consecutive miss: the app is wedged or dead. Force a clean relaunch.
rm -f "\$STATE"
logger -t nova-voice-watchdog "satellite (pid \${PID:-none}) has no \$SERVER socket across two checks; kickstarting"
if [ -n "\$PID" ]; then
  launchctl kickstart -k "\$DOMAIN/\$LABEL" 2>/dev/null || true
else
  launchctl kickstart "\$DOMAIN/\$LABEL" 2>/dev/null \\
    || launchctl bootstrap "\$DOMAIN" "\$HOME/Library/LaunchAgents/\$LABEL.plist" 2>/dev/null || true
fi
EOF
chmod +x "$SCRIPT_PATH"

cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/sh</string>
        <string>$SCRIPT_PATH</string>
    </array>
    <key>StartInterval</key>
    <integer>60</integer>
    <key>RunAtLoad</key>
    <true/>
    <key>ProcessType</key>
    <string>Background</string>
    <key>StandardOutPath</key>
    <string>/dev/null</string>
    <key>StandardErrorPath</key>
    <string>$LOG_PATH</string>
</dict>
</plist>
EOF

launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
launchctl bootstrap "$DOMAIN" "$PLIST_PATH"
launchctl kickstart "$DOMAIN/$LABEL"

echo "Installed watchdog: $PLIST_PATH"
echo "Watching satellite $SAT_LABEL for a live socket to $SERVER (check interval 60s)."
launchctl print "$DOMAIN/$LABEL" 2>/dev/null | grep -iE "state =|program = |run interval|last exit" | head
