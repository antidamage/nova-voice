#!/usr/bin/env bash
# Install the Nova TTS engine-switch machinery on the voice host. Run as root
# (deploy-nova-stack.ps1 invokes it with the staged source tree as $1).
#
# Installs the root switcher script, the per-engine drop-in profiles, and the
# request-watching path unit, then migrates any hand-written cutover drop-ins
# (zz-cutover-dots.conf / zz-vram-rebalance.conf) to the managed equivalents —
# preserving whichever engine is currently selected. Idempotent; does NOT
# restart voice services and does NOT change the selected engine.
set -euo pipefail
SRC="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

PROFILES=/etc/nova-voice/engine-profiles
VOICE_D=/etc/systemd/system/nova-voice.service.d
LLM_D=/etc/systemd/system/nova-voice-llm.service.d

install -d -m 0755 "$PROFILES" "$VOICE_D" "$LLM_D"
install -m 0644 "$SRC/deploy/systemd/engine-profiles/zz-engine-dots.conf" "$PROFILES/"
install -m 0644 "$SRC/deploy/systemd/engine-profiles/zz-engine-classic.conf" "$PROFILES/"
install -m 0644 "$SRC/deploy/systemd/engine-profiles/zz-engine-vram-dots.conf" "$PROFILES/"
install -m 0755 "$SRC/ops/switch-tts-engine.sh" /usr/local/bin/nova-switch-tts-engine
install -m 0644 "$SRC/deploy/systemd/nova-tts-engine-switch.service" /etc/systemd/system/
install -m 0644 "$SRC/deploy/systemd/nova-tts-engine-switch.path" /etc/systemd/system/

# Decide which engine the host currently has selected: the managed drop-in if
# present, else the legacy hand-written cutover drop-in, else Classic.
backend=classic
if [[ -f "$VOICE_D/zz-engine.conf" ]]; then
  grep -q 'NOVA_VOICE_TTS_BACKEND=dots' "$VOICE_D/zz-engine.conf" && backend=dots
elif grep -qs 'NOVA_VOICE_TTS_BACKEND=dots' "$VOICE_D/zz-cutover-dots.conf"; then
  backend=dots
fi

# (Re)install the managed drop-ins for that engine so profile updates (ports,
# VRAM tuning) propagate on every deploy, and retire the legacy files.
install -m 0644 "$PROFILES/zz-engine-${backend}.conf" "$VOICE_D/zz-engine.conf"
if [[ "$backend" == dots ]]; then
  install -m 0644 "$PROFILES/zz-engine-vram-dots.conf" "$LLM_D/zz-engine-vram.conf"
else
  rm -f "$LLM_D/zz-engine-vram.conf"
fi
rm -f "$VOICE_D/zz-cutover-dots.conf" "$LLM_D/zz-vram-rebalance.conf"

systemctl daemon-reload
systemctl enable --now nova-tts-engine-switch.path
echo ">>> engine-switch machinery installed (selected engine: $backend)"
