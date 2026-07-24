#!/usr/bin/env bash
# Nova TTS engine switcher (runs as root; installed as
# /usr/local/bin/nova-switch-tts-engine by ops/install-engine-switch.sh).
#
# Swaps the mutually exclusive Classic (Qwen vLLM, port 8091) and Custom
# (dots.tts, port 8095) engine units: installs the matching engine drop-in on
# nova-voice.service, applies/removes the LLM VRAM rebalance, swaps which TTS
# unit is enabled, and restarts the orchestrator. Progress is reported into a
# status file the orchestrator's GET /v1/engine relays to the dashboard.
#
# Invoked by nova-tts-engine-switch.service when the unprivileged orchestrator
# writes the request file, or manually:  nova-switch-tts-engine dots|classic
set -euo pipefail

REQUEST=/var/lib/nova-voice/engine-switch-request.json
STATUS=/var/lib/nova-voice/engine-switch-status.json
PROFILES=/etc/nova-voice/engine-profiles
VOICE_DROPIN_DIR=/etc/systemd/system/nova-voice.service.d
LLM_DROPIN_DIR=/etc/systemd/system/nova-voice-llm.service.d

target="${1:-}"
if [[ -z "$target" ]]; then
  if [[ ! -f "$REQUEST" ]]; then
    echo "no engine switch request pending" >&2
    exit 0
  fi
  target="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["target"])' "$REQUEST" 2>/dev/null || true)"
  # Consume the request no matter what it contained: PathExists= retriggers for
  # as long as the file exists, so a malformed request must not loop the unit.
  rm -f "$REQUEST"
fi
case "$target" in
  dots|custom) target=dots ;;
  classic|qwen|vllm) target=classic ;;
  *)
    echo "unknown engine target: '${target}'" >&2
    exit 1
    ;;
esac

write_status() {
  local phase="$1" error="${2:-}"
  ENGINE_TARGET="$target" SWITCH_PHASE="$phase" SWITCH_ERROR="$error" python3 - "$STATUS" <<'PY'
import datetime
import json
import os
import sys

body = {
    "target": "custom" if os.environ["ENGINE_TARGET"] == "dots" else "classic",
    "phase": os.environ["SWITCH_PHASE"],
    "updatedAt": datetime.datetime.now().astimezone().isoformat(),
}
if os.environ.get("SWITCH_ERROR"):
    body["error"] = os.environ["SWITCH_ERROR"]
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump(body, fh)
PY
  chmod 0644 "$STATUS"
}

fail() {
  write_status failed "$1"
  echo "$1" >&2
  exit 1
}

if [[ "$target" == dots ]]; then
  selected=nova-voice-dots-tts.service
  other=nova-voice-tts.service
  profile="$PROFILES/zz-engine-dots.conf"
  health_url=http://127.0.0.1:8095/health
else
  selected=nova-voice-tts.service
  other=nova-voice-dots-tts.service
  profile="$PROFILES/zz-engine-classic.conf"
  health_url=http://127.0.0.1:8091/health
fi
[[ -f "$profile" ]] || fail "engine profile missing: $profile"

echo ">>> switching Nova TTS engine to $target"
write_status preparing

install -d -m 0755 "$VOICE_DROPIN_DIR" "$LLM_DROPIN_DIR"
install -m 0644 "$profile" "$VOICE_DROPIN_DIR/zz-engine.conf"
if [[ "$target" == dots ]]; then
  install -m 0644 "$PROFILES/zz-engine-vram-dots.conf" "$LLM_DROPIN_DIR/zz-engine-vram.conf"
else
  rm -f "$LLM_DROPIN_DIR/zz-engine-vram.conf"
fi
# Hand-written cutover drop-ins from before the switch was managed.
rm -f "$VOICE_DROPIN_DIR/zz-cutover-dots.conf" "$LLM_DROPIN_DIR/zz-vram-rebalance.conf"
systemctl daemon-reload

write_status restarting
# Stop the orchestrator first so dependency propagation from the model units
# below cannot bounce it mid-swap; it is started last, once its engine exists.
systemctl stop nova-voice.service
systemctl disable --now "$other" || true
systemctl enable "$selected"
# The LLM's context size / GPU-layer split changes with the engine, so apply
# the (possibly changed) ExecStart now, while the orchestrator is down anyway.
systemctl restart nova-voice-llm.service
systemctl start "$selected"
# Start the orchestrator immediately rather than waiting for TTS warmup: STT
# and interpretation come back within a couple of minutes, and the TTS adapter
# health-gates itself until the engine is ready to speak.
systemctl start nova-voice.service

write_status warming
check_ready() {
  if [[ "$target" == dots ]]; then
    curl -sf -m 3 "$health_url" | grep -q '"ready" *: *true'
  else
    curl -sf -m 3 -o /dev/null "$health_url"
  fi
}
deadline=$((SECONDS + 900))
until check_ready; do
  if (( SECONDS >= deadline )); then
    fail "the $target TTS engine did not become ready within 15 minutes"
  fi
  sleep 5
done
write_status ready
echo ">>> engine switch to $target complete"
