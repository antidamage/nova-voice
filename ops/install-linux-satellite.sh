#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_ROOT="${NOVA_VOICE_SATELLITE_ROOT:-$HOME/.local/opt/nova-voice}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"

if [[ ! -x "$UV_BIN" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

sudo apt-get install -y libportaudio2
sudo loginctl enable-linger "$USER"
install -d \
  "$INSTALL_ROOT" \
  "$HOME/.config/systemd/user" \
  "$HOME/.config/nova-voice/tls" \
  "$HOME/.config/pipewire/pipewire.conf.d" \
  "$HOME/.local/state/nova-voice"

env UV_PROJECT_ENVIRONMENT="$INSTALL_ROOT/venv" \
  "$UV_BIN" sync --frozen --no-dev --extra satellite --project "$SOURCE_DIR"
install -m 0644 "$SOURCE_DIR/deploy/systemd/nova-voice-satellite.service" \
  "$HOME/.config/systemd/user/nova-voice-satellite.service"
if [[ ! -f "$HOME/.config/pipewire/pipewire.conf.d/60-nova-voice-aec.conf" ]]; then
  install -m 0644 "$SOURCE_DIR/deploy/pipewire/60-nova-voice-aec.conf" \
    "$HOME/.config/pipewire/pipewire.conf.d/60-nova-voice-aec.conf"
fi
if [[ ! -f "$HOME/.config/nova-voice/satellite.env" ]]; then
  # EnvironmentFile does not expand $HOME.  Materialise this user's absolute
  # paths instead of shipping the example's placeholder /home path.
  sed "s|/home/satellite-user|$HOME|g" \
    "$SOURCE_DIR/config/satellite.env.example" \
    >"$HOME/.config/nova-voice/satellite.env"
  chmod 0600 "$HOME/.config/nova-voice/satellite.env"
fi

# Resolve host-specific PipeWire node names.  A missing audio session is a
# deployment warning, not a reason to leave the code-installed service half
# configured; the service remains stopped until TLS and audio are validated.
if command -v wpctl >/dev/null 2>&1; then
  if ! bash "$SOURCE_DIR/ops/configure-pipewire-aec.sh"; then
    echo "Warning: could not configure PipeWire AEC; pass explicit nodes to" >&2
    echo "ops/configure-pipewire-aec.sh once physical audio is connected." >&2
  fi
else
  echo "Warning: wpctl is unavailable; configure PipeWire AEC before starting." >&2
fi

systemctl --user daemon-reload
systemctl --user enable nova-voice-satellite.service
echo "Installed but not started. Provision the client TLS identity, validate AEC, then start."
