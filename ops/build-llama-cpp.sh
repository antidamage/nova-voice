#!/usr/bin/env bash
set -euo pipefail

ROOT="${NOVA_VOICE_ROOT:-/opt/nova-voice}"
SOURCE_DIR="$ROOT/runtime/llama.cpp"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=versions.env
source "$SCRIPT_DIR/versions.env"

install -d -o nova-voice -g nova-voice "$ROOT/runtime"
if [[ ! -d "$SOURCE_DIR/.git" ]]; then
  sudo -u nova-voice git clone https://github.com/ggml-org/llama.cpp.git "$SOURCE_DIR"
fi

sudo -u nova-voice git -C "$SOURCE_DIR" fetch --depth 1 origin "$LLAMA_CPP_COMMIT"
sudo -u nova-voice git -C "$SOURCE_DIR" checkout --detach "$LLAMA_CPP_COMMIT"
sudo -u nova-voice cmake \
  -S "$SOURCE_DIR" \
  -B "$SOURCE_DIR/build" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_F16=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DLLAMA_CURL=ON
sudo -u nova-voice cmake --build "$SOURCE_DIR/build" \
  --target llama-server llama-quantize \
  --parallel "$(nproc)"

"$SOURCE_DIR/build/bin/llama-server" --version
