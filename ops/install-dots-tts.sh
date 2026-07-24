#!/usr/bin/env bash
# Provision the dots.tts custom-voice engine (Custom TTS engine) on the voice
# host. Idempotent and NON-disruptive: it creates the venv, stages the model and
# voices dir, installs the systemd unit + a scoped sudoers rule for the engine
# switch, but does NOT start the service or cut the orchestrator over. Going live
# (stop Qwen TTS, start dots, flip tts_backend=dots) is a separate, deliberate
# step because the two TTS engines cannot be VRAM-resident at once.
#
# Run on the voice host (fish is the login shell there; invoke via bash):
#   bash ops/install-dots-tts.sh
set -euo pipefail

DOTS_ROOT=/opt/nova-voice/dots
VENV="$DOTS_ROOT/venv"
SRC="$DOTS_ROOT/dots.tts"           # dots.tts source checkout (editable install)
MODEL_DIR=/opt/nova-voice/models/dots.tts-mf
VOICES_DIR=/opt/nova-voice/voices
DOTS_REPO=${DOTS_REPO:-https://github.com/rednote-hilab/dots.tts.git}
DOTS_REV=${DOTS_REV:-5ed719e}       # v0.2.1 (validated); override to bump
MODEL_REPO=${MODEL_REPO:-dots-studio/dots.tts-mf}
SERVICE_USER=${SERVICE_USER:-antidamage}
UNIT=nova-voice-dots-tts.service
CURRENT=/opt/nova-voice/current      # deploy symlink; services/dots_tts lives here

log() { printf '[install-dots-tts] %s\n' "$*"; }

command -v uv >/dev/null || { echo "uv is required"; exit 1; }

log "creating dirs"
sudo mkdir -p "$DOTS_ROOT" "$MODEL_DIR" "$VOICES_DIR"
sudo chown -R "$SERVICE_USER:$SERVICE_USER" "$DOTS_ROOT" "$MODEL_DIR" "$VOICES_DIR"

log "cloning dots.tts @ $DOTS_REV"
if [ ! -d "$SRC/.git" ]; then
  sudo -u "$SERVICE_USER" env GIT_LFS_SKIP_SMUDGE=1 git clone "$DOTS_REPO" "$SRC"
fi
sudo -u "$SERVICE_USER" git -C "$SRC" fetch --all --quiet || true
sudo -u "$SERVICE_USER" git -C "$SRC" checkout --quiet "$DOTS_REV"

log "building venv (python 3.12; dots.tts pins >=3.10,<3.13)"
sudo -u "$SERVICE_USER" bash -lc "
  set -e
  cd '$SRC'
  uv python install 3.12
  uv venv --python 3.12 '$VENV'
  uv pip install --python '$VENV' -e . -c constraints/recommended.txt
  # server-only deps not pulled by the library extras
  uv pip install --python '$VENV' 'fastapi' 'uvicorn' 'python-multipart'
"

log "staging model $MODEL_REPO -> $MODEL_DIR"
if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
  sudo -u "$SERVICE_USER" "$VENV/bin/python" - <<PY
from huggingface_hub import snapshot_download
snapshot_download("$MODEL_REPO", local_dir="$MODEL_DIR")
print("model staged")
PY
fi

log "installing systemd unit"
sudo install -m 0644 "$CURRENT/deploy/systemd/$UNIT" "/etc/systemd/system/$UNIT"

log "installing scoped sudoers rule for the engine switch"
SUDOERS=/etc/sudoers.d/nova-voice-engine-switch
sudo tee "$SUDOERS" >/dev/null <<EOF
# Allow the voice service user to swap the two mutually-exclusive TTS engine
# units (Classic Qwen <-> Custom dots) without a password, for the dashboard
# engine switch. Scoped to exactly these units and actions.
$SERVICE_USER ALL=(root) NOPASSWD: /usr/bin/systemctl start nova-voice-dots-tts.service, \\
  /usr/bin/systemctl stop nova-voice-dots-tts.service, \\
  /usr/bin/systemctl start nova-voice-tts.service, \\
  /usr/bin/systemctl stop nova-voice-tts.service, \\
  /usr/bin/systemctl restart nova-voice.service
EOF
sudo chmod 0440 "$SUDOERS"
sudo visudo -cf "$SUDOERS"

sudo systemctl daemon-reload
log "done. NOT started. To go live (deliberate; frees Qwen VRAM, ~7 min warmup):"
log "  sudo systemctl stop nova-voice-tts && sudo systemctl start $UNIT"
log "  # then set tts_backend=dots and restart nova-voice.service"
