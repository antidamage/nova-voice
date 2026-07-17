#!/usr/bin/env bash
set -euo pipefail

ROOT="${NOVA_VOICE_ROOT:-/opt/nova-voice}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_SOURCE="$ROOT/models/qwen3.5-4b-hf"
MODEL_F16="$ROOT/models/qwen3.5-4b-f16.gguf"
MODEL_Q4="$ROOT/models/qwen3.5-4b-q4_k_m.gguf"
CONVERT_ENV="$ROOT/runtime/convert-venv"
UV_BIN="${UV_BIN:-/usr/local/bin/uv}"

# shellcheck source=versions.env
source "$SCRIPT_DIR/versions.env"

cd "$ROOT"

install -d -o nova-voice -g nova-voice "$ROOT/models" "$ROOT/cache/huggingface"
sudo -H -u nova-voice env HF_HOME="$ROOT/cache/huggingface" \
  "$UV_BIN" tool run --from huggingface-hub hf download \
  "$QWEN_LLM_REPO" \
  --revision "$QWEN_LLM_REVISION" \
  --local-dir "$MODEL_SOURCE"

sudo -H -u nova-voice "$UV_BIN" venv --python 3.12 --clear "$CONVERT_ENV"
sudo -H -u nova-voice "$UV_BIN" pip install \
  --python "$CONVERT_ENV/bin/python" \
  --index-strategy unsafe-best-match \
  -r "$ROOT/runtime/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt"
sudo -H -u nova-voice "$CONVERT_ENV/bin/python" \
  "$ROOT/runtime/llama.cpp/convert_hf_to_gguf.py" \
  "$MODEL_SOURCE" \
  --outfile "$MODEL_F16" \
  --outtype f16
sudo -H -u nova-voice "$ROOT/runtime/llama.cpp/build/bin/llama-quantize" \
  "$MODEL_F16" \
  "$MODEL_Q4" \
  Q4_K_M

sha256sum "$MODEL_Q4" | tee "$MODEL_Q4.sha256"
rm -f "$MODEL_F16"
