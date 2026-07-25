#!/usr/bin/env bash
# Nova TTS engine switcher (runs as root; installed as
# /usr/local/bin/nova-switch-tts-engine by ops/install-engine-switch.sh).
#
# Swaps the mutually exclusive TTS engine units: installs the matching engine
# drop-in on nova-voice.service, applies/removes the LLM VRAM rebalance, stops
# every other engine unit and starts the selected one, and restarts the
# orchestrator. Progress is reported into a status file the orchestrator's
# GET /v1/engine relays to the dashboard.
#
# Every engine-specific fact (unit, drop-in profile, VRAM profile, health URL,
# readiness style) is read from the engine registry manifest at
# /etc/nova-voice/engine-registry.json, which is the JSON dump of
# nova_voice.tts_engines (the single source of truth shared with the app and
# the dashboard). Adding an engine there needs no edit here.
#
# Invoked by nova-tts-engine-switch.service when the unprivileged orchestrator
# writes the request file, or manually:  nova-switch-tts-engine <engine>
#   <engine> accepts an engine id (classic|custom|trained), a backend
#   (vllm|dots|gptsovits), or a legacy alias (qwen).
set -euo pipefail

REQUEST=/var/lib/nova-voice/engine-switch-request.json
STATUS=/var/lib/nova-voice/engine-switch-status.json
REGISTRY=/etc/nova-voice/engine-registry.json
PROFILES=/etc/nova-voice/engine-profiles
VOICE_DROPIN_DIR=/etc/systemd/system/nova-voice.service.d
LLM_DROPIN_DIR=/etc/systemd/system/nova-voice-llm.service.d

[[ -f "$REGISTRY" ]] || { echo "engine registry missing: $REGISTRY (run install-engine-switch.sh)" >&2; exit 1; }

# reg <command> [args] — read fields out of the registry manifest with stdlib
# python3 only (no venv/import of the app needed on the root side).
reg() {
  python3 - "$REGISTRY" "$@" <<'PY'
import json, sys
reg = json.load(open(sys.argv[1]))
cmd = sys.argv[2]
def by_id(i):
    return next((e for e in reg if e["id"] == i), None)
if cmd == "resolve":                     # id | backend | backend-value -> id
    tok = sys.argv[3]
    for e in reg:
        if tok == e["id"] or tok == e["backend"] or tok in e["backendValues"]:
            print(e["id"]); sys.exit(0)
    sys.exit(3)
elif cmd == "field":                     # field <id> <key>
    e = by_id(sys.argv[3])
    if e is None: sys.exit(3)
    v = e.get(sys.argv[4])
    print("" if v is None else v)
elif cmd == "units":                     # every engine's systemd unit
    print(" ".join(e["unit"] for e in reg))
else:
    sys.exit(2)
PY
}

raw_target="${1:-}"
if [[ -z "$raw_target" ]]; then
  if [[ ! -f "$REQUEST" ]]; then
    echo "no engine switch request pending" >&2
    exit 0
  fi
  raw_target="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["target"])' "$REQUEST" 2>/dev/null || true)"
  # Consume the request no matter what it contained: PathExists= retriggers for
  # as long as the file exists, so a malformed request must not loop the unit.
  rm -f "$REQUEST"
fi

target="$(reg resolve "$raw_target" || true)"
[[ -n "$target" ]] || { echo "unknown engine target: '${raw_target}'" >&2; exit 1; }

write_status() {
  local phase="$1" error="${2:-}"
  ENGINE_TARGET="$target" SWITCH_PHASE="$phase" SWITCH_ERROR="$error" python3 - "$STATUS" <<'PY'
import datetime
import json
import os
import sys

body = {
    "target": os.environ["ENGINE_TARGET"],
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

selected="$(reg field "$target" unit)"
profile="$PROFILES/$(reg field "$target" profile)"
vram_profile="$(reg field "$target" vramProfile)"
health_url="$(reg field "$target" healthUrl)"
health_style="$(reg field "$target" healthStyle)"
all_units="$(reg units)"
[[ -f "$profile" ]] || fail "engine profile missing: $profile"

echo ">>> switching Nova TTS engine to $target"
write_status preparing

install -d -m 0755 "$VOICE_DROPIN_DIR" "$LLM_DROPIN_DIR"
install -m 0644 "$profile" "$VOICE_DROPIN_DIR/zz-engine.conf"
# LLM VRAM rebalance: a single managed slot. Install the selected engine's
# profile if it has one (its resident model is large enough to need the LLM
# trimmed), otherwise clear the slot so the LLM runs at its full-GPU baseline.
if [[ -n "$vram_profile" ]]; then
  install -m 0644 "$PROFILES/$vram_profile" "$LLM_DROPIN_DIR/zz-engine-vram.conf"
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
# Stop EVERY other engine unit (not just one), so only the selected engine is
# GPU-resident regardless of how many engines exist.
for unit in $all_units; do
  [[ "$unit" == "$selected" ]] && continue
  systemctl disable --now "$unit" || true
done
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
  if [[ "$health_style" == ready-gate ]]; then
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
