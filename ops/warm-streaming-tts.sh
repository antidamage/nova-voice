#!/usr/bin/env bash
set -euo pipefail

base_url="${NOVA_TTS_WARMUP_BASE_URL:-http://127.0.0.1:8091}"

for _ in $(seq 1 90); do
  if curl --fail --silent --show-error --max-time 2 "$base_url/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

curl --fail --silent --show-error --max-time 120 \
  --header 'Content-Type: application/json' \
  --output /dev/null \
  --data-binary '{
    "model":"Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "input":"Nova voice is ready.",
    "voice":"serena",
    "language":"English",
    "instructions":"Speak naturally in a warm, conversational tone.",
    "stream":true,
    "stream_format":"audio",
    "response_format":"pcm"
  }' \
  "$base_url/v1/audio/speech"
