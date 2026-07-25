#!/usr/bin/env bash
# Root-side start/stop for a training run, triggered by the request file the
# orchestrator writes (see nova-voice-training-switch.path).
#
# Why a request file rather than sudo: nova-voice.service runs with
# NoNewPrivileges=yes, which disables setuid outright -- sudo cannot work from
# there at all, by design. This mirrors ops/switch-tts-engine.sh, which solves
# the same problem the same way for the TTS engine switch.
#
# Handing the GPU over is NOT done here. The training unit declares Conflicts=
# against the voice stack and OnSuccess=/OnFailure=nova-voice.service, so systemd
# stops the resident models when a run starts and brings them back when it ends,
# however it ends. That keeps the restore path declarative and means a crashed
# run still gives the household its voice back.
set -euo pipefail

REQUEST=/var/lib/nova-voice/training-request.json
STATUS=/var/lib/nova-voice/training-request-status.json

write_status() {
  printf '{"action":"%s","setId":"%s","result":"%s","at":"%s"}\n' \
    "${2:-}" "${3:-}" "$1" "$(date -Is)" > "$STATUS"
  chmod 0644 "$STATUS"
}

[[ -f "$REQUEST" ]] || { echo "no training request pending" >&2; exit 0; }

action="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("action",""))' "$REQUEST" 2>/dev/null || true)"
set_id="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("setId",""))' "$REQUEST" 2>/dev/null || true)"

# Consume unconditionally: PathExists= re-triggers for as long as the file is
# there, so a malformed request must not spin the unit.
rm -f "$REQUEST"

# The instance name goes into a unit name, so accept only the same slug the
# training-set store produces. Anything else is refused rather than escaped.
if [[ ! "$set_id" =~ ^[a-z0-9_-]+$ ]]; then
  echo "refusing malformed training set id: '${set_id}'" >&2
  write_status "invalid-set-id" "$action" "$set_id"
  exit 1
fi

unit="nova-voice-training@${set_id}.service"
case "$action" in
  start)
    echo ">>> starting $unit"
    systemctl start "$unit"
    write_status "started" "$action" "$set_id"
    ;;
  stop)
    echo ">>> stopping $unit"
    systemctl stop "$unit"
    write_status "stopped" "$action" "$set_id"
    ;;
  *)
    echo "unknown training action: '${action}'" >&2
    write_status "unknown-action" "$action" "$set_id"
    exit 1
    ;;
esac
