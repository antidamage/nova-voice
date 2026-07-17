#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run with sudo: sudo ops/install-iridium.sh" >&2
  exit 2
fi

SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${NOVA_VOICE_ROOT:-/opt/nova-voice}"
UV_BIN="${UV_BIN:-/usr/local/bin/uv}"

if [[ ! -x "$UV_BIN" ]]; then
  install -m 0755 /home/antidamage/.local/bin/uv "$UV_BIN"
fi

id nova-voice >/dev/null 2>&1 || useradd --system --home-dir "$ROOT" --shell /usr/sbin/nologin nova-voice
usermod -a -G video,render nova-voice
install -d -o nova-voice -g nova-voice \
  "$ROOT" "$ROOT/cache" "$ROOT/models" "$ROOT/runtime" \
  /var/lib/nova-voice /etc/nova-voice/tls
chown -R nova-voice:nova-voice "$SOURCE_DIR"

sudo -u nova-voice env UV_PROJECT_ENVIRONMENT="$ROOT/venv" \
  "$UV_BIN" sync --frozen --no-dev --inexact --project "$SOURCE_DIR"

install -m 0644 "$SOURCE_DIR/deploy/systemd/nova-voice.service" /etc/systemd/system/
install -m 0644 "$SOURCE_DIR/deploy/systemd/nova-voice-llm.service" /etc/systemd/system/
install -m 0644 "$SOURCE_DIR/deploy/systemd/nova-voice-tts.service" /etc/systemd/system/
install -m 0644 "$SOURCE_DIR/deploy/systemd/nova-voice-dfn.service" /etc/systemd/system/
install -d -m 0755 /etc/systemd/system/nova-voice.service.d
install -m 0644 "$SOURCE_DIR/deploy/systemd/nova-voice-tts-stream.conf" \
  /etc/systemd/system/nova-voice.service.d/tts-stream.conf
if [[ ! -f /etc/nova-voice/nova-voice.env ]]; then
  install -m 0640 -o root -g nova-voice "$SOURCE_DIR/.env.example" \
    /etc/nova-voice/nova-voice.env
fi
if [[ ! -f /etc/nova-voice/persona.yaml ]]; then
  install -m 0640 -o root -g nova-voice "$SOURCE_DIR/config/persona.example.yaml" \
    /etc/nova-voice/persona.yaml
fi

systemctl daemon-reload
systemctl enable nova-voice-llm.service nova-voice-tts.service nova-voice-dfn.service nova-voice.service
echo "Installed but not started. Provision TLS/model files and the isolated streaming TTS runtime, then run the preflight first."
