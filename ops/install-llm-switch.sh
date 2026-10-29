#!/usr/bin/env bash
# Install the Nova interpretation-model switch machinery on the voice host. Run
# as root (deploy-nova-stack.ps1 invokes it with the staged source tree as $1).
#
# Dumps the model registry (nova_voice.llm_models) to
# /etc/nova-voice/llm-registry.json for the root switcher, installs the switcher
# and its request-watching path unit, and rewrites the managed zz-llm.conf
# drop-in for whichever model is currently selected. Idempotent; it does not
# start, stop or switch any model. Adding a model needs no edit here.
set -euo pipefail
SRC="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

REGISTRY=/etc/nova-voice/llm-registry.json
SELECTED=/var/lib/nova-voice/llm-selected.json
VOICE_D=/etc/systemd/system/nova-voice.service.d
ORCH_PY=/opt/nova-voice/venv/bin/python

# The staged source tree, not $ROOT/current: the deployer runs this before the
# atomic promote (see install-engine-switch.sh).
[[ -x "$ORCH_PY" ]] || { echo "orchestrator venv python not found at $ORCH_PY" >&2; exit 1; }
[[ -d "$SRC/src" ]] || { echo "staged source tree not found at $SRC/src" >&2; exit 1; }
install -d -m 0755 /etc/nova-voice "$VOICE_D"
PYTHONPATH="$SRC/src${PYTHONPATH:+:$PYTHONPATH}" "$ORCH_PY" -m nova_voice.llm_models dump > "$REGISTRY"
chmod 0644 "$REGISTRY"

install -m 0755 "$SRC/ops/switch-llm.sh" /usr/local/bin/nova-switch-llm
install -m 0644 "$SRC/deploy/systemd/nova-llm-switch.service" /etc/systemd/system/
install -m 0644 "$SRC/deploy/systemd/nova-llm-switch.path" /etc/systemd/system/

# The selected model: the selection file if it names a registered model, else
# the default (the first registry entry, LFM3), recorded so the orchestrator
# and the switcher agree from the first deploy on.
selected_unit="$(python3 - "$REGISTRY" "$SELECTED" <<'PY'
import json, sys
reg = json.load(open(sys.argv[1]))
try:
    llm = json.load(open(sys.argv[2])).get("llm")
except (OSError, ValueError):
    llm = None
entry = next((e for e in reg if e["id"] == llm), None)
if entry is None:
    entry = reg[0]
    with open(sys.argv[2], "w", encoding="utf-8") as fh:
        json.dump({"llm": entry["id"]}, fh)
print(entry["unit"])
PY
)"
chmod 0644 "$SELECTED"

cat >"$VOICE_D/zz-llm.conf" <<EOF
# Managed by nova-switch-llm -- do not hand-edit. Wants= (not Requires=) the
# selected interpretation model, so stopping a model never stops the orchestrator.
[Unit]
Wants=$selected_unit
EOF
chmod 0644 "$VOICE_D/zz-llm.conf"

systemctl daemon-reload
systemctl enable "$selected_unit"
systemctl enable --now nova-llm-switch.path
echo ">>> LLM switch machinery installed (selected model unit: $selected_unit)"
