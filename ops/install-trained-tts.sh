#!/usr/bin/env bash
# Provision the GPT-SoVITS trained-voice engine (Trained TTS engine) on the
# voice host. Idempotent and NON-disruptive: stages two SEPARATE environments,
# the trained-voices registry dir, and the systemd unit, but does NOT start
# the service or cut the orchestrator over. Going live (switch engine to
# "trained") is a separate, deliberate step because only one TTS engine can
# be VRAM-resident at once.
#
# Two environments, not one -- unlike dots.tts, services/trained_tts/server.py
# does NOT import GPT-SoVITS in-process. It's a thin FastAPI wrapper (fastapi/
# httpx/uvicorn only, no torch) that launches and proxies to GPT-SoVITS's own
# api_v2.py, which needs GPT-SoVITS's full (heavy, torch-based) environment:
#   $TRAINED_ROOT/venv               -- wrapper's own lightweight venv (uv)
#   $TRAINED_ROOT/GPT-SoVITS         -- GPT-SoVITS checkout
#   $TRAINED_ROOT/GPT-SoVITS's conda env -- GPT-SoVITS's own heavy environment
#
# GPT-SoVITS's own install.sh does not create a venv itself -- it installs
# PyTorch/deps into whatever conda environment is already active (verified
# against RVC-Boss/GPT-SoVITS's install.sh, 2026-07: `bash install.sh --device
# <CU126|CU128|ROCM|MPS|CPU> --source <HF|HF-Mirror|ModelScope>`), so this
# script creates and activates a dedicated conda env first, matching that
# expectation, rather than fighting it with uv/venv.
#
# The scoped sudoers rule for the engine switch is installed by
# ops/install-engine-switch.sh (derived from the engine registry), not here.
#
# Run on the voice host (invoke via bash):  bash ops/install-trained-tts.sh
set -euo pipefail

TRAINED_ROOT=/opt/nova-voice/gptsovits
WRAPPER_VENV="$TRAINED_ROOT/venv"           # our services/trained_tts/server.py
SRC="$TRAINED_ROOT/GPT-SoVITS"              # GPT-SoVITS checkout
CONDA_ENV_NAME=${CONDA_ENV_NAME:-nova-gptsovits}
GPTSOVITS_DEVICE=${GPTSOVITS_DEVICE:-CU126}  # RTX 2080 Ti (Turing) on iridium
GPTSOVITS_SOURCE=${GPTSOVITS_SOURCE:-HF}
VOICES_DIR=/opt/nova-voice/trained-voices
GPTSOVITS_REPO=${GPTSOVITS_REPO:-https://github.com/RVC-Boss/GPT-SoVITS.git}
GPTSOVITS_REV=${GPTSOVITS_REV:-main}         # pin to a tag/sha before production
SERVICE_USER=${SERVICE_USER:-antidamage}
UNIT=nova-voice-trained-tts.service
CURRENT=/opt/nova-voice/current              # deploy symlink; services/trained_tts lives here

log() { printf '[install-trained-tts] %s\n' "$*"; }

command -v uv >/dev/null || { echo "uv is required (for the wrapper venv)"; exit 1; }
command -v conda >/dev/null || { echo "conda is required (GPT-SoVITS's own install.sh expects it)"; exit 1; }

log "creating dirs"
sudo mkdir -p "$TRAINED_ROOT" "$VOICES_DIR"
sudo chown -R "$SERVICE_USER:$SERVICE_USER" "$TRAINED_ROOT" "$VOICES_DIR"

log "building the wrapper's own lightweight venv (no torch -- just serving glue)"
sudo -u "$SERVICE_USER" bash -lc "
  set -e
  uv python install 3.12
  uv venv --python 3.12 '$WRAPPER_VENV'
  uv pip install --python '$WRAPPER_VENV' 'fastapi' 'uvicorn' 'httpx' 'python-multipart'
"

log "cloning GPT-SoVITS @ $GPTSOVITS_REV"
if [ ! -d "$SRC/.git" ]; then
  sudo -u "$SERVICE_USER" env GIT_LFS_SKIP_SMUDGE=1 git clone "$GPTSOVITS_REPO" "$SRC"
fi
sudo -u "$SERVICE_USER" git -C "$SRC" fetch --all --quiet || true
sudo -u "$SERVICE_USER" git -C "$SRC" checkout --quiet "$GPTSOVITS_REV"

log "creating conda env '$CONDA_ENV_NAME' and running GPT-SoVITS's own installer"
log "  (--device $GPTSOVITS_DEVICE --source $GPTSOVITS_SOURCE; downloads several GB, expect a while)"
sudo -u "$SERVICE_USER" bash -lc "
  set -e
  eval \"\$(conda shell.bash hook)\"
  conda create -y -n '$CONDA_ENV_NAME' python=3.10
  conda activate '$CONDA_ENV_NAME'
  cd '$SRC'
  bash install.sh --device '$GPTSOVITS_DEVICE' --source '$GPTSOVITS_SOURCE'
"
GPTSOVITS_PYTHON_PATH="$(sudo -u "$SERVICE_USER" bash -lc "eval \"\$(conda shell.bash hook)\"; conda activate '$CONDA_ENV_NAME'; command -v python")"
log "GPT-SoVITS python: $GPTSOVITS_PYTHON_PATH"

log "installing systemd unit"
sudo install -m 0644 "$CURRENT/deploy/systemd/$UNIT" "/etc/systemd/system/$UNIT"
log "NOTE: if \$GPTSOVITS_PYTHON_PATH above differs from the unit's hardcoded"
log "  GPTSOVITS_PYTHON=.../GPT-SoVITS/venv/bin/python, edit the unit (conda envs"
log "  don't live at a fixed venv-style path) or symlink accordingly."

sudo systemctl daemon-reload
log "done. NOT started. To go live (deliberate; switches the resident TTS engine):"
log "  # from the dashboard: Voice Infrastructure -> engine picker -> Trained, or"
log "  sudo nova-switch-tts-engine trained"
