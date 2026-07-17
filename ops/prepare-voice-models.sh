#!/usr/bin/env bash
set -euo pipefail

ROOT="${NOVA_VOICE_ROOT:-/opt/nova-voice}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UV_BIN="${UV_BIN:-/usr/local/bin/uv}"
PYTHON="$ROOT/venv/bin/python"
STT_DIR="$ROOT/models/nemotron-speech-streaming-en-0.6b"
TTS_DIR="$ROOT/models/qwen3-tts-0.6b-customvoice"
WAKE_FEATURE_DIR="$ROOT/models/openwakeword"

# shellcheck source=versions.env
source "$SCRIPT_DIR/versions.env"

cd "$ROOT"

apt-get install -y libsndfile1 portaudio19-dev ffmpeg
sudo -H -u nova-voice "$UV_BIN" pip install \
  --python "$PYTHON" \
  --index-strategy unsafe-best-match \
  -r "$SCRIPT_DIR/requirements-inference.txt"
# openWakeWord's legacy tflite-runtime pin has no CPython 3.12 Linux wheel. The
# application maps that import to Google's current LiteRT package, so install
# only openWakeWord's package code here.
sudo -H -u nova-voice "$UV_BIN" pip install \
  --python "$PYTHON" \
  --no-deps \
  openwakeword==0.6.0

install -d -o nova-voice -g nova-voice \
  "$STT_DIR" "$TTS_DIR" "$WAKE_FEATURE_DIR" "$ROOT/cache/huggingface"
sudo -H -u nova-voice "$PYTHON" \
  "$SCRIPT_DIR/download_openwakeword_features.py" "$WAKE_FEATURE_DIR"
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
