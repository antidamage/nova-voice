#!/usr/bin/env bash
# Install the voice-training machinery on the voice host: the per-run template
# unit and the scoped sudoers rule the orchestrator needs to drive it.
#
# The orchestrator (nova-voice.service) has no sudo rights by default, and it
# needs exactly two privileged capabilities for training:
#
#   1. start/stop a training run, which must live in its OWN cgroup -- training
#      mode stops nova-voice.service to free the GPU, and with the default
#      KillMode=control-group a worker spawned as a child of the orchestrator
#      would be killed by that very step.
#   2. stop and restart the resident model units, which is what frees the GPU.
#
# Both are granted as a narrow allowlist of exact systemctl invocations rather
# than blanket sudo, mirroring ops/install-engine-switch.sh.
#
# Idempotent. Run on the voice host:  sudo bash ops/install-training.sh [SRC]
set -euo pipefail
SRC="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

ORCHESTRATOR_USER=${ORCHESTRATOR_USER:-nova-voice}
TRAINING_SETS_DIR=${TRAINING_SETS_DIR:-/opt/nova-voice/training-sets}
SUDOERS=/etc/sudoers.d/nova-voice-training
TEMPLATE=nova-voice-training@.service

log() { printf '[install-training] %s\n' "$*"; }

[[ -f "$SRC/deploy/systemd/$TEMPLATE" ]] || {
  echo "training template unit not found under $SRC/deploy/systemd" >&2
  exit 1
}

log "installing $TEMPLATE"
install -m 0644 "$SRC/deploy/systemd/$TEMPLATE" "/etc/systemd/system/$TEMPLATE"

log "installing the request watcher (the orchestrator cannot sudo: NoNewPrivileges=yes)"
install -m 0755 "$SRC/ops/training-switch.sh" /usr/local/bin/nova-training-switch
install -m 0644 "$SRC/deploy/systemd/nova-voice-training-switch.service" /etc/systemd/system/
install -m 0644 "$SRC/deploy/systemd/nova-voice-training-switch.path" /etc/systemd/system/

log "ensuring training-sets directory is writable by $ORCHESTRATOR_USER"
install -d -o "$ORCHESTRATOR_USER" -g "$ORCHESTRATOR_USER" -m 0755 "$TRAINING_SETS_DIR"

# Publishing a finished bundle copies it into the trained-voices catalogue. That
# directory belongs to the engine's service account (which serves from it), but
# the copy is performed by the orchestrator -- so it needs group write, or
# "Publish as voice" fails with a bare PermissionError. Group-writable with
# setgid, rather than a chown, so both accounts keep working.
TRAINED_VOICES_DIR=${TRAINED_VOICES_DIR:-/opt/nova-voice/trained-voices}
if [[ -d "$TRAINED_VOICES_DIR" ]]; then
  log "making $TRAINED_VOICES_DIR group-writable for $ORCHESTRATOR_USER"
  chgrp -R "$ORCHESTRATOR_USER" "$TRAINED_VOICES_DIR"
  chmod g+rwxs "$TRAINED_VOICES_DIR"
  find "$TRAINED_VOICES_DIR" -mindepth 1 -type d -exec chmod g+rwxs {} +
  find "$TRAINED_VOICES_DIR" -mindepth 1 -type f -exec chmod g+rw {} +
fi

# GPT-SoVITS writes inside its own checkout at runtime -- most importantly it
# lazily downloads the Faster-Whisper ASR model to a RELATIVE path,
# tools/asr/models/, resolved against the repo root. The checkout belongs to the
# provisioning user while training runs as the orchestrator account, so those
# directories have to be handed over explicitly or the ASR step dies with a
# PermissionError after the slicing step has already succeeded.
GPTSOVITS_SRC=${GPTSOVITS_SRC:-/opt/nova-voice/gptsovits/GPT-SoVITS}
if [[ -d "$GPTSOVITS_SRC" ]]; then
  log "handing GPT-SoVITS's runtime-writable directories to $ORCHESTRATOR_USER"
  for runtime_dir in tools/asr/models TEMP logs; do
    install -d -o "$ORCHESTRATOR_USER" -g "$ORCHESTRATOR_USER" -m 0755 \
      "$GPTSOVITS_SRC/$runtime_dir"
  done

  # GPT-SoVITS writes temp files to RELATIVE paths, resolved against the working
  # directory -- process_ckpt.my_save() does `torch.save(fea, "<timestamp>.pth")`
  # and only then moves the result into place. The pipeline therefore has to run
  # with the checkout as its cwd AND be able to write there, or every feature
  # save fails with "open file failed ... Permission denied" while the step still
  # exits 0, leaving an empty feature set that only surfaces stages later.
  # Group-write (plus setgid so new directories inherit the group) is enough;
  # ownership stays with the provisioning user.
  log "making the checkout group-writable for $ORCHESTRATOR_USER"
  chgrp -R "$ORCHESTRATOR_USER" "$GPTSOVITS_SRC"
  find "$GPTSOVITS_SRC" -type d -exec chmod g+rwxs {} +
  find "$GPTSOVITS_SRC" -type f -exec chmod g+rw {} +
fi

# No sudoers rule: NoNewPrivileges=yes makes sudo unusable from the orchestrator
# regardless of policy, and none is needed. Starting/stopping a run goes through
# the request watcher above, and the GPU handover is declared on the training
# unit itself (Conflicts= to stop the voice stack, OnSuccess=/OnFailure= to bring
# it back), so systemd performs both with its own privileges.
rm -f /etc/sudoers.d/nova-voice-training

systemctl daemon-reload
systemctl enable --now nova-voice-training-switch.path
log "done. Training runs launch as nova-voice-training@<set-id>.service"
