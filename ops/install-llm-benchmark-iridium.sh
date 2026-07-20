#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run with sudo: sudo bash ops/install-llm-benchmark-iridium.sh" >&2
  exit 2
fi

SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${LLM_BENCHMARK_ROOT:-/opt/llm-benchmark}"
STATE_DIR="${LLM_BENCHMARK_STATE_DIR:-/var/lib/llm-benchmark}"
UV_BIN="${UV_BIN:-/home/antidamage/.local/bin/uv}"
PACKAGE_VERSION="${LLM_BENCHMARK_VERSION:-0.5.2}"

if [[ ! -x /usr/local/bin/ollama ]]; then
  echo "Ollama is required at /usr/local/bin/ollama" >&2
  exit 1
fi
if [[ ! -x "$UV_BIN" ]]; then
  echo "uv is required at $UV_BIN" >&2
  exit 1
fi
if ss -ltn | grep -qE ':11435\b'; then
  owner="$(systemctl show llm-benchmark-ollama.service -p MainPID --value 2>/dev/null || true)"
  if [[ -z "$owner" || "$owner" == "0" ]]; then
    echo "Port 11435 is already occupied by another process" >&2
    exit 1
  fi
fi

id llm-benchmark >/dev/null 2>&1 || useradd \
  --system --home-dir "$STATE_DIR" --create-home --shell /usr/sbin/nologin llm-benchmark
usermod -a -G video,render llm-benchmark

install -d -m 0755 "$ROOT" "$ROOT/config"
install -d -o llm-benchmark -g llm-benchmark -m 0750 \
  "$STATE_DIR" "$STATE_DIR/models"
install -d -o antidamage -g llm-benchmark -m 0775 \
  "$STATE_DIR/results" "$STATE_DIR/client"
usermod -a -G llm-benchmark antidamage

"$UV_BIN" venv --python 3.13 "$ROOT/venv"
"$UV_BIN" pip install --python "$ROOT/venv/bin/python" \
  "llm-benchmark==$PACKAGE_VERSION"

# Version 0.5.2 contains two fallback URLs fixed to Ollama's default port.
# The main client honors OLLAMA_HOST, and this mechanical rewrite keeps every
# fallback on the isolated benchmark port too.
PACKAGE_DIR="$(find "$ROOT/venv/lib" -type d -path '*/site-packages/llm_benchmark' -print -quit)"
if [[ -z "$PACKAGE_DIR" ]]; then
  echo "Installed llm_benchmark package was not found" >&2
  exit 1
fi
while IFS= read -r -d '' file; do
  sed -i -E \
    's#(localhost|127\.0\.0\.1):11434#127.0.0.1:11435#g' "$file"
done < <(grep -rlZE '(localhost|127\.0\.0\.1):11434' "$PACKAGE_DIR" || true)

install -m 0644 "$SOURCE_DIR/config/llm-benchmark-models.yml" \
  "$ROOT/config/models.yml"
install -m 0755 "$SOURCE_DIR/ops/llm-benchmark-iridium" \
  /usr/local/bin/llm-benchmark-iridium
install -m 0644 "$SOURCE_DIR/deploy/systemd/llm-benchmark-ollama.service" \
  /etc/systemd/system/llm-benchmark-ollama.service

systemctl daemon-reload
systemctl enable --now llm-benchmark-ollama.service

for _attempt in $(seq 1 20); do
  if curl -fsS http://127.0.0.1:11435/api/version >/dev/null; then
    break
  fi
  sleep 1
done
curl -fsS http://127.0.0.1:11435/api/version >/dev/null

echo "Installed llm-benchmark $PACKAGE_VERSION with Ollama on 127.0.0.1:11435"
echo "Run the local-only default suite with: llm-benchmark-iridium"
