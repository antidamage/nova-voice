#!/usr/bin/env bash
# Install the Nova TTS engine-switch machinery on the voice host. Run as root
# (deploy-nova-stack.ps1 invokes it with the staged source tree as $1).
#
# Everything engine-specific comes from the engine registry (nova_voice.tts_engines):
# this script dumps it to /etc/nova-voice/engine-registry.json (the root switcher
# reads that with stdlib python3), installs each engine's drop-in profiles,
# derives the scoped sudoers allowlist from the registry's unit list, and installs
# the request-watching path unit. It preserves whichever engine is currently
# selected. Idempotent; does NOT restart voice services or change the selected
# engine. Adding an engine to the registry needs no edit here.
set -euo pipefail
SRC="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

REGISTRY_DIR=/etc/nova-voice
REGISTRY="$REGISTRY_DIR/engine-registry.json"
PROFILES=/etc/nova-voice/engine-profiles
VOICE_D=/etc/systemd/system/nova-voice.service.d
LLM_D=/etc/systemd/system/nova-voice-llm.service.d
ORCH_PY=/opt/nova-voice/venv/bin/python
# The engine model units run as this user; the scoped sudoers below lets it swap
# them manually. Matches install-dots-tts.sh's established default.
SERVICE_USER=${SERVICE_USER:-antidamage}

install -d -m 0755 "$REGISTRY_DIR" "$PROFILES" "$VOICE_D" "$LLM_D"

# Dump the registry manifest — the single source of truth for the switcher, the
# profile list, and the sudoers allowlist below. Needs the orchestrator venv for
# the dependencies, but the *staged* source tree for the app package: the
# deployer runs this before the atomic promote, so $ROOT/current is still the
# previous release and would yield the previous release's engine registry (or,
# when the registry module itself is new, "No module named
# nova_voice.tts_engines"). PYTHONPATH puts the staged src ahead of whatever the
# venv resolves.
[[ -x "$ORCH_PY" ]] || { echo "orchestrator venv python not found at $ORCH_PY" >&2; exit 1; }
[[ -d "$SRC/src" ]] || { echo "staged source tree not found at $SRC/src" >&2; exit 1; }
PYTHONPATH="$SRC/src${PYTHONPATH:+:$PYTHONPATH}" "$ORCH_PY" -m nova_voice.tts_engines dump > "$REGISTRY"
chmod 0644 "$REGISTRY"

# reg <command> — read the just-written manifest with stdlib python3.
reg() {
  python3 - "$REGISTRY" "$@" <<'PY'
import json, sys
reg = json.load(open(sys.argv[1]))
cmd = sys.argv[2]
if cmd == "profiles":                 # every drop-in + vram profile basename
    names = []
    for e in reg:
        names.append(e["profile"])
        if e.get("vramProfile"):
            names.append(e["vramProfile"])
    print("\n".join(dict.fromkeys(names)))
elif cmd == "units":                  # every engine's systemd unit
    print("\n".join(e["unit"] for e in reg))
elif cmd == "id_for_backend":         # backend/value -> engine id
    tok = sys.argv[3]
    for e in reg:
        if tok == e["backend"] or tok in e["backendValues"]:
            print(e["id"]); break
elif cmd == "field":                  # field <id> <key>
    e = next((x for x in reg if x["id"] == sys.argv[3]), None)
    print("" if e is None else (e.get(sys.argv[4]) or ""))
PY
}

# Install every engine's drop-in profiles referenced by the manifest.
while IFS= read -r name; do
  [[ -z "$name" ]] && continue
  install -m 0644 "$SRC/deploy/systemd/engine-profiles/$name" "$PROFILES/"
done < <(reg profiles)

install -m 0755 "$SRC/ops/switch-tts-engine.sh" /usr/local/bin/nova-switch-tts-engine
install -m 0644 "$SRC/deploy/systemd/nova-tts-engine-switch.service" /etc/systemd/system/
install -m 0644 "$SRC/deploy/systemd/nova-tts-engine-switch.path" /etc/systemd/system/

# Decide which engine the host currently has selected: read the backend out of
# the managed drop-in if present (else the legacy hand-written cutover drop-in,
# else Classic) and map it to an engine id through the registry.
selected_backend=vllm
if [[ -f "$VOICE_D/zz-engine.conf" ]]; then
  selected_backend="$(grep -oP 'NOVA_VOICE_TTS_BACKEND=\K\S+' "$VOICE_D/zz-engine.conf" || echo vllm)"
elif grep -qs 'NOVA_VOICE_TTS_BACKEND=dots' "$VOICE_D/zz-cutover-dots.conf"; then
  selected_backend=dots
fi
selected_id="$(reg id_for_backend "$selected_backend")"
[[ -n "$selected_id" ]] || selected_id=classic

# (Re)install the managed drop-ins for that engine so profile updates (ports,
# VRAM tuning) propagate on every deploy, and retire the legacy files.
install -m 0644 "$PROFILES/$(reg field "$selected_id" profile)" "$VOICE_D/zz-engine.conf"
selected_vram="$(reg field "$selected_id" vramProfile)"
if [[ -n "$selected_vram" ]]; then
  install -m 0644 "$PROFILES/$selected_vram" "$LLM_D/zz-engine-vram.conf"
else
  rm -f "$LLM_D/zz-engine-vram.conf"
fi
rm -f "$VOICE_D/zz-cutover-dots.conf" "$LLM_D/zz-vram-rebalance.conf"

# Scoped sudoers for the engine switch, derived from the registry so it always
# covers exactly the current set of engine units (start/stop) plus the
# orchestrator restart — no manual edit when an engine is added.
SUDOERS=/etc/sudoers.d/nova-voice-engine-switch
{
  echo "# Managed by install-engine-switch.sh from the engine registry. Allow the"
  echo "# voice service user to swap the mutually-exclusive TTS engine units"
  echo "# without a password, for the dashboard engine switch. Scoped to exactly"
  echo "# these units and actions."
  cmds=""
  while IFS= read -r unit; do
    [[ -z "$unit" ]] && continue
    cmds+="/usr/bin/systemctl start $unit, /usr/bin/systemctl stop $unit, /usr/bin/systemctl disable $unit, /usr/bin/systemctl enable $unit, "
  done < <(reg units)
  cmds+="/usr/bin/systemctl restart nova-voice.service, /usr/bin/systemctl restart nova-voice-llm.service"
  echo "$SERVICE_USER ALL=(root) NOPASSWD: $cmds"
} > "$SUDOERS"
chmod 0440 "$SUDOERS"
visudo -cf "$SUDOERS"

systemctl daemon-reload
systemctl enable --now nova-tts-engine-switch.path
echo ">>> engine-switch machinery installed (selected engine: $selected_id)"
