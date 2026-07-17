#!/usr/bin/env bash
set -euo pipefail

# Read-only deployment inventory. It deliberately does not install packages,
# stop services, or touch Nova/Home Assistant. Run from a trusted workstation
# (real host/user values: see PRIVATEREF.md#1.1 and #2.1):
#   ./ops/inventory-iridium.sh <user>@<voice-server-host> [peer-host ...]

target=${1:?usage: ops/inventory-iridium.sh <user>@<voice-server-host> [peer-host ...]}
shift || true
ssh -o BatchMode=yes -o ConnectTimeout=8 "$target" 'bash -s' -- "$@" <<'REMOTE'
set -u
printf '%s\n' '--- identity ---'
hostname
uname -a
printf '%s\n' '--- operating system ---'
cat /etc/os-release
printf '%s\n' '--- cpu and memory ---'
lscpu | sed -n '1,18p'
free -h
printf '%s\n' '--- storage ---'
df -h /
printf '%s\n' '--- gpu ---'
nvidia-smi || true
printf '%s\n' '--- cuda ---'
(command -v nvcc && nvcc --version) || true
printf '%s\n' '--- audio ---'
(pactl list short sources; pactl list short sinks) 2>/dev/null || true
(wpctl status) 2>/dev/null || true
printf '%s\n' '--- services and listeners ---'
systemctl --no-pager --plain list-units --type=service --state=running | grep -Ei 'nova|voice|llama|wyoming' || true
ss -ltnp | grep -E '8765|8766' || true
printf '%s\n' '--- network names ---'
# Peer host names are passed as arguments; see PRIVATEREF.md#1 for the set.
[ "$#" -gt 0 ] && getent hosts "$@" || true
REMOTE
