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
# Resolve conda to an absolute path rather than trusting it to be on PATH in the
# shells below. The sudo -u ... bash -lc blocks are NOT interactive, and Ubuntu's
# stock ~/.bashrc early-returns for non-interactive shells -- so the block
# `conda init` writes there never runs and `conda` is simply not found, even
# though it works fine in your own terminal. Search the usual install roots for
# the service user as well as root's PATH.
SERVICE_HOME="$(getent passwd "$SERVICE_USER" | cut -d: -f6)"
[[ -n "$SERVICE_HOME" ]] || { echo "could not resolve home directory for $SERVICE_USER" >&2; exit 1; }

CONDA_BIN="$(command -v conda || true)"
if [[ -z "$CONDA_BIN" ]]; then
  for candidate in \
    "$SERVICE_HOME/miniforge3/bin/conda" \
    "$SERVICE_HOME/miniconda3/bin/conda" \
    "$SERVICE_HOME/anaconda3/bin/conda" \
    /opt/miniforge3/bin/conda /opt/conda/bin/conda; do
    [[ -x "$candidate" ]] && { CONDA_BIN="$candidate"; break; }
  done
fi
[[ -n "$CONDA_BIN" ]] || {
  echo "conda is required (GPT-SoVITS's own install.sh expects it) and was not found." >&2
  echo "Install Miniforge (conda-forge only, no Anaconda Terms of Service gate):" >&2
  echo "  curl -fsSL -o /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" >&2
  echo "  bash /tmp/miniforge.sh -b -p \$HOME/miniforge3" >&2
  exit 1
}
log "using conda: $CONDA_BIN"

log "creating dirs"
sudo mkdir -p "$TRAINED_ROOT" "$VOICES_DIR"
sudo chown -R "$SERVICE_USER:$SERVICE_USER" "$TRAINED_ROOT" "$VOICES_DIR"

log "building the wrapper's own lightweight venv (no torch -- just serving glue)"
# Idempotent for real: `uv venv` hard-errors on an existing environment, so a
# re-run after any partial failure died here rather than continuing. Create only
# when missing; the pip install below is already idempotent and re-syncs either
# way. `uv python install` gets --force because a previous run may have left an
# unmanaged python3.12 shim in ~/.local/bin that it otherwise refuses to replace.
UV_BIN="$(command -v uv || echo "$SERVICE_HOME/.local/bin/uv")"
[[ -x "$UV_BIN" ]] || { echo "uv is required but was not found at $UV_BIN" >&2; exit 1; }
sudo -u "$SERVICE_USER" bash -lc "
  set -e
  '$UV_BIN' python install --force 3.12
  if [ ! -x '$WRAPPER_VENV/bin/python' ]; then
    '$UV_BIN' venv --python 3.12 '$WRAPPER_VENV'
  else
    echo '[install-trained-tts] wrapper venv already exists, reusing'
  fi
  '$UV_BIN' pip install --python '$WRAPPER_VENV' 'fastapi' 'uvicorn' 'httpx' 'python-multipart'
"

log "cloning GPT-SoVITS @ $GPTSOVITS_REV"
if [ ! -d "$SRC/.git" ]; then
  sudo -u "$SERVICE_USER" env GIT_LFS_SKIP_SMUDGE=1 git clone "$GPTSOVITS_REPO" "$SRC"
fi
sudo -u "$SERVICE_USER" git -C "$SRC" fetch --all --quiet || true
sudo -u "$SERVICE_USER" git -C "$SRC" checkout --quiet "$GPTSOVITS_REV"

log "creating conda env '$CONDA_ENV_NAME' and running GPT-SoVITS's own installer"
log "  (--device $GPTSOVITS_DEVICE --source $GPTSOVITS_SOURCE; downloads several GB, expect a while)"
# `conda create` before the hook: creating an env needs only the executable, and
# doing it first means the hook's `conda activate` has something to activate even
# on a box where the shell was never `conda init`-ed.
sudo -u "$SERVICE_USER" "$CONDA_BIN" create -y -n "$CONDA_ENV_NAME" python=3.10

# GPT-SoVITS's install.sh runs `conda install` WITHOUT --override-channels, so on
# an Anaconda-defaults conda it trips CondaToSNonInteractiveError. Pin this env's
# channel config to conda-forge so the `defaults` alias can't resolve to
# repo.anaconda.com -- no Terms of Service to accept, no upstream patch. (A
# Miniforge install is already conda-forge-only; this makes it true either way.)
ENV_PREFIX="$(sudo -u "$SERVICE_USER" "$CONDA_BIN" run -n "$CONDA_ENV_NAME" printenv CONDA_PREFIX | tr -d '\r')"
[[ -n "$ENV_PREFIX" ]] || { echo "could not resolve prefix for conda env $CONDA_ENV_NAME" >&2; exit 1; }
sudo -u "$SERVICE_USER" tee "$ENV_PREFIX/.condarc" >/dev/null <<'CONDARC'
channels:
  - conda-forge
default_channels:
  - conda-forge
CONDARC

# TERM=xterm because GPT-SoVITS's install.sh draws progress with `tput cuu1`
# (cursor-up). Over a TTY-less SSH command TERM is unset or "unknown", tput
# errors, and upstream's own error trap aborts the install -- after the 4.4 GB
# model download has already succeeded. xterm is the cheapest terminfo entry
# that actually has the cuu1 capability (TERM=dumb does not).
sudo -u "$SERVICE_USER" TERM=xterm bash -lc "
  set -e
  eval \"\$('$CONDA_BIN' shell.bash hook)\"
  conda activate '$CONDA_ENV_NAME'
  cd '$SRC'
  bash install.sh --device '$GPTSOVITS_DEVICE' --source '$GPTSOVITS_SOURCE'
"
GPTSOVITS_PYTHON_PATH="$(sudo -u "$SERVICE_USER" "$CONDA_BIN" run -n "$CONDA_ENV_NAME" command -v python | tr -d '\r')"
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
