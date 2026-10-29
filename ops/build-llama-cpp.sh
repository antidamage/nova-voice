#!/usr/bin/env bash
set -euo pipefail

ROOT="${NOVA_VOICE_ROOT:-/opt/nova-voice}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=versions.env
source "$SCRIPT_DIR/versions.env"

# Two trees, because the resident LLM and the retained fallback need different
# llama.cpp revisions: LFM3-1.7B needs LLAMA_CPP_LFM3_COMMIT, which the older
# pin in LLAMA_CPP_COMMIT predates and cannot load.
build_variant() {
  local name="$1" commit="$2"
  local source_dir="$ROOT/runtime/$name"

  install -d -o nova-voice -g nova-voice "$ROOT/runtime"
  if [[ ! -d "$source_dir/.git" ]]; then
    sudo -u nova-voice git clone https://github.com/ggml-org/llama.cpp.git "$source_dir"
  fi

  sudo -u nova-voice git -C "$source_dir" fetch --depth 1 origin "$commit"
  sudo -u nova-voice git -C "$source_dir" checkout --detach "$commit"
  sudo -u nova-voice cmake \
    -S "$source_dir" \
    -B "$source_dir/build" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_F16=ON \
    -DCMAKE_CUDA_ARCHITECTURES=75 \
    -DLLAMA_CURL=ON
  sudo -u nova-voice cmake --build "$source_dir/build" \
    --target llama-server llama-quantize \
    --parallel "$(nproc)"

  "$source_dir/build/bin/llama-server" --version
}

build_variant llama.cpp-lfm3 "$LLAMA_CPP_LFM3_COMMIT"
build_variant llama.cpp "$LLAMA_CPP_COMMIT"
