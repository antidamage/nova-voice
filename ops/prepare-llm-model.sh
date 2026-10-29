#!/usr/bin/env bash
set -euo pipefail

ROOT="${NOVA_VOICE_ROOT:-/opt/nova-voice}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONVERT_ENV="$ROOT/runtime/convert-venv"
UV_BIN="${UV_BIN:-/usr/local/bin/uv}"

# shellcheck source=versions.env
source "$SCRIPT_DIR/versions.env"

cd "$ROOT"

install -d -o nova-voice -g nova-voice "$ROOT/models" "$ROOT/cache/huggingface"

# The resident model: LiquidAI publishes a ready Q4_K_M GGUF, so there is no
# conversion step. Download the file at the pinned revision and check it against
# the checksum recorded in versions.env.
prepare_lfm3() {
  sudo -H -u nova-voice env HF_HOME="$ROOT/cache/huggingface" \
    "$UV_BIN" tool run --from huggingface-hub hf download \
    "$LFM3_LLM_REPO" \
    "$LFM3_LLM_FILE" \
    --revision "$LFM3_LLM_REVISION" \
    --local-dir "$ROOT/models"

  # Absolute paths: sha256sum -c resolves each name against the caller's
  # working directory ($ROOT), not against where the download landed.
  local model_file="$ROOT/models/$LFM3_LLM_FILE"
  echo "$LFM3_LLM_SHA256  $model_file" \
    | sudo -H -u nova-voice sha256sum -c -
  # Record the checksum beside the model. The name stays relative because the
  # sidecar is read from that directory, not from $ROOT.
  echo "$LFM3_LLM_SHA256  $LFM3_LLM_FILE" \
    | sudo -H -u nova-voice tee "$model_file.sha256"
}

# The fallback, selected only before deployment: Qwen3.5-4B is distributed as an
# HF checkpoint and converted locally to Q4_K_M.
prepare_qwen() {
  local model_source="$ROOT/models/qwen3.5-4b-hf"
  local model_f16="$ROOT/models/qwen3.5-4b-f16.gguf"
  local model_q4="$ROOT/models/qwen3.5-4b-q4_k_m.gguf"

  sudo -H -u nova-voice env HF_HOME="$ROOT/cache/huggingface" \
    "$UV_BIN" tool run --from huggingface-hub hf download \
    "$QWEN_LLM_REPO" \
    --revision "$QWEN_LLM_REVISION" \
    --local-dir "$model_source"

  sudo -H -u nova-voice "$UV_BIN" venv --python 3.12 --clear "$CONVERT_ENV"
  sudo -H -u nova-voice "$UV_BIN" pip install \
    --python "$CONVERT_ENV/bin/python" \
    --index-strategy unsafe-best-match \
    -r "$ROOT/runtime/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt"
  sudo -H -u nova-voice "$CONVERT_ENV/bin/python" \
    "$ROOT/runtime/llama.cpp/convert_hf_to_gguf.py" \
    "$model_source" \
    --outfile "$model_f16" \
    --outtype f16
  sudo -H -u nova-voice "$ROOT/runtime/llama.cpp/build/bin/llama-quantize" \
    "$model_f16" \
    "$model_q4" \
    Q4_K_M

  sha256sum "$model_q4" | tee "$model_q4.sha256"
  rm -f "$model_f16"
}

case "${1:-lfm3}" in
  lfm3) prepare_lfm3 ;;
  qwen) prepare_qwen ;;
  *)
    echo "usage: $0 [lfm3|qwen]" >&2
    exit 2
    ;;
esac
