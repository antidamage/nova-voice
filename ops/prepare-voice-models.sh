#!/usr/bin/env bash
set -euo pipefail

ROOT="${NOVA_VOICE_ROOT:-/opt/nova-voice}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UV_BIN="${UV_BIN:-/usr/local/bin/uv}"
PYTHON="$ROOT/venv/bin/python"
STT_DIR="$ROOT/models/nemotron-speech-streaming-en-0.6b"
TTS_DIR="$ROOT/models/qwen3-tts-0.6b-customvoice"

# shellcheck source=versions.env
source "$SCRIPT_DIR/versions.env"

cd "$ROOT"

apt-get install -y libsndfile1 portaudio19-dev ffmpeg
sudo -H -u nova-voice "$UV_BIN" pip install \
  --python "$PYTHON" \
  --index-strategy unsafe-best-match \
  -r "$SCRIPT_DIR/requirements-inference.txt"

install -d -o nova-voice -g nova-voice \
  "$STT_DIR" "$TTS_DIR" "$ROOT/cache/huggingface"
sudo -H -u nova-voice env HF_HOME="$ROOT/cache/huggingface" \
  "$UV_BIN" tool run --from huggingface-hub hf download \
  "$NEMOTRON_STT_REPO" \
  --revision "$NEMOTRON_STT_REVISION" \
  --include '*.nemo' \
  --local-dir "$STT_DIR"
sudo -H -u nova-voice env HF_HOME="$ROOT/cache/huggingface" \
  "$UV_BIN" tool run --from huggingface-hub hf download \
  "$QWEN_TTS_REPO" \
  --revision "$QWEN_TTS_REVISION" \
  --local-dir "$TTS_DIR"

printf '%s\n' \
  "stt=$NEMOTRON_STT_REPO@$NEMOTRON_STT_REVISION" \
  "tts=$QWEN_TTS_REPO@$QWEN_TTS_REVISION" \
  >"$ROOT/models/voice-model-revisions.txt"
chown nova-voice:nova-voice "$ROOT/models/voice-model-revisions.txt"
