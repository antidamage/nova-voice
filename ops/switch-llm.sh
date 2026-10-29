#!/usr/bin/env bash
# Nova interpretation-model switcher (runs as root; installed as
# /usr/local/bin/nova-switch-llm by ops/install-llm-switch.sh).
#
# Starts the selected llama-server unit while the current one keeps serving,
# waits for it to answer /health, hands the orchestrator the new route through
# the selection file and waits for its acknowledgement, then stops and disables
# every other model unit. The orchestrator is never restarted. Progress goes to
# a status file the orchestrator's GET /v1/llm relays to the dashboard.
#
# Every model fact (unit, port, health URL, whether it fits beside a TTS engine
# that trims the LLM's VRAM) is read from /etc/nova-voice/llm-registry.json, the
# JSON dump of nova_voice.llm_models. Adding a model needs no edit here.
#
# Invoked by nova-llm-switch.service when the orchestrator writes the request
# file, or manually:  nova-switch-llm <llm-id>
set -euo pipefail

REQUEST=/var/lib/nova-voice/llm-switch-request.json
STATUS=/var/lib/nova-voice/llm-switch-status.json
SELECTED=/var/lib/nova-voice/llm-selected.json
ROUTED=/var/lib/nova-voice/llm-routed.json
REGISTRY=/etc/nova-voice/llm-registry.json
ENGINE_REGISTRY=/etc/nova-voice/engine-registry.json
ENGINE_DROPIN=/etc/systemd/system/nova-voice.service.d/zz-engine.conf
VOICE_DROPIN_DIR=/etc/systemd/system/nova-voice.service.d

# reg <command> [args] -- read the registry with stdlib python3 only.
reg() {
  python3 - "$REGISTRY" "$@" <<'PY'
import json, sys
reg = json.load(open(sys.argv[1]))
cmd = sys.argv[2]
by_id = {e["id"]: e for e in reg}
if cmd == "known":
    sys.exit(0 if sys.argv[3] in by_id else 3)
elif cmd == "field":
    v = by_id[sys.argv[3]].get(sys.argv[4])
    print("" if v is None else (str(v).lower() if isinstance(v, bool) else v))
elif cmd == "units":
    print(" ".join(e["unit"] for e in reg))
else:
    sys.exit(2)
PY
}

raw_target="${1:-}"
if [[ -z "$raw_target" ]]; then
  if [[ ! -f "$REQUEST" ]]; then
    echo "no LLM switch request pending" >&2
    exit 0
  fi
  raw_target="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["target"])' "$REQUEST" 2>/dev/null || true)"
  # Consume the request whatever it held: PathExists= retriggers for as long as
  # the file exists, so a malformed request must not loop the unit.
  rm -f "$REQUEST"
fi
# Checked after the request is consumed, so a missing registry cannot loop the path unit.
[[ -f "$REGISTRY" ]] || { echo "LLM registry missing: $REGISTRY (run install-llm-switch.sh)" >&2; exit 1; }
reg known "$raw_target" || { echo "unknown LLM target: '${raw_target}'" >&2; exit 1; }
target="$raw_target"

write_status() {
  local phase="$1" error="${2:-}"
  LLM_TARGET="$target" SWITCH_PHASE="$phase" SWITCH_ERROR="$error" python3 - "$STATUS" <<'PY'
import datetime, json, os, sys
body = {
    "target": os.environ["LLM_TARGET"],
    "phase": os.environ["SWITCH_PHASE"],
    "updatedAt": datetime.datetime.now().astimezone().isoformat(),
}
if os.environ.get("SWITCH_ERROR"):
    body["error"] = os.environ["SWITCH_ERROR"]
with open(sys.argv[1] + ".tmp", "w", encoding="utf-8") as fh:
    json.dump(body, fh)
os.replace(sys.argv[1] + ".tmp", sys.argv[1])
PY
  chmod 0644 "$STATUS"
}

fail() {
  write_status failed "$1"
  echo "$1" >&2
  exit 1
}

selected="$(reg field "$target" unit)"
label="$(reg field "$target" label)"
health_url="$(reg field "$target" healthUrl)"
all_units="$(reg units)"

write_status preparing

# Refuse a model that is unusable beside the resident TTS engine: one that
# trims the LLM's GPU layers (Custom) would leave it decoding on the CPU.
if [[ "$(reg field "$target" fitsWithVramTrim)" != "true" && -f "$ENGINE_REGISTRY" && -f "$ENGINE_DROPIN" ]]; then
  backend="$(grep -oP 'NOVA_VOICE_TTS_BACKEND=\K\S+' "$ENGINE_DROPIN" || true)"
  trims="$(python3 - "$ENGINE_REGISTRY" "$backend" <<'PY'
import json, sys
for e in json.load(open(sys.argv[1])):
    if sys.argv[2] == e["backend"] or sys.argv[2] in e["backendValues"]:
        print("yes" if e.get("vramProfile") else "no")
PY
)"
  if [[ "$trims" == yes ]]; then
    fail "$label needs the whole GPU and the Custom voice engine is loaded; switch the voice engine first."
  fi
fi

echo ">>> switching Nova interpretation model to $target ($selected)"
write_status starting
systemctl enable "$selected"
systemctl start "$selected"
check_ready() { curl -sf -m 3 "$health_url" | grep -q '"status" *: *"ok"'; }
deadline=$((SECONDS + 300))
until check_ready; do
  if (( SECONDS >= deadline )) || systemctl is-failed --quiet "$selected"; then
    # Leave the old model selected and serving.
    systemctl disable --now "$selected" || true
    fail "$label did not become ready within 5 minutes; the previous model is still serving"
  fi
  sleep 2
done

# Wire the orchestrator's boot-time dependency to the selected model.
install -d -m 0755 "$VOICE_DROPIN_DIR"
cat >"$VOICE_DROPIN_DIR/zz-llm.conf" <<EOF
# Managed by nova-switch-llm -- do not hand-edit. Wants= (not Requires=) the
# selected interpretation model, so stopping a model never stops the orchestrator.
[Unit]
Wants=$selected
EOF
chmod 0644 "$VOICE_DROPIN_DIR/zz-llm.conf"
systemctl daemon-reload

write_status routing
rm -f "$ROUTED"
printf '{"llm": "%s"}\n' "$target" >"$SELECTED.tmp"
chmod 0644 "$SELECTED.tmp"
mv -f "$SELECTED.tmp" "$SELECTED"
if systemctl is-active --quiet nova-voice.service; then
  deadline=$((SECONDS + 30))
  until [[ -f "$ROUTED" ]] && grep -q "\"llm\": *\"$target\"" "$ROUTED"; do
    if (( SECONDS >= deadline )); then
      echo "orchestrator did not acknowledge the route within 30 s; continuing" >&2
      break
    fi
    sleep 1
  done
fi

write_status stopping
# A turn routed before the acknowledgement may still be decoding on the old
# model; give it the interpreter's 20 s request timeout (plus margin) to finish.
sleep 25
for unit in $all_units; do
  [[ "$unit" == "$selected" ]] && continue
  systemctl disable --now "$unit" || true
done

write_status ready
echo ">>> interpretation model switch to $target complete"
