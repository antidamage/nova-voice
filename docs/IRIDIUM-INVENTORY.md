# Iridium deployment inventory

This is a redacted, read-only snapshot. Re-run `ops/inventory-iridium.sh`
before each model bake-off; it is intentionally not a provisioning script.

The verified host is Iridium (hostname/address: see `PRIVATEREF.md#1.1`) running Ubuntu 25.10
with a 12-thread i7-5820K, 15 GiB RAM, and an RTX 2080 Ti (11,264 MiB). The
NVIDIA driver is 580.159.03 and `nvidia-smi` reports CUDA 13.0; the installed
CUDA toolkit reports 12.4. PipeWire and WirePlumber are active, but the
current graph exposes only a dummy source/sink, so physical capture hardware
must be provisioned before a live local-mic test.

At the 2026-07-22 documentation-truth snapshot, all deployed resident model
services were healthy:

- `llama-server` on `127.0.0.1:8765`, Qwen3.5-4B Q4_K_M, context 4096,
  one slot, all layers on GPU, q8 KV cache (about 3.0 GiB GPU memory).
- `nova-voice` on mTLS `0.0.0.0:8766`, with Nemotron 0.6B STT resident
  (1,890 MiB GPU memory at the snapshot).
- `nova-voice-tts` on `127.0.0.1:8091`, with Qwen3-TTS 0.6B under vLLM-Omni.
  Its stage-0 talker and stage-1 codec workers used 2,602 MiB and 1,624 MiB.

The snapshot resident set uses 9,138 MiB of 11,264 MiB (3,022 MiB LLM, 4,226
MiB TTS workers, and 1,890 MiB Voice/STT), leaving 2,126 MiB for CUDA allocator
variance. The TTS runtime uses the pinned Turing-specific vLLM-Omni patch and
deployment configuration. The Nemotron adapter has a cache-aware streaming
interface, but the deployed turn path currently runs batch decoding after
central VAD finalizes the utterance. The 24-hour residency gate remains
mandatory; no model is swapped at runtime.

Iridium's dummy audio graph is expected while the primary microphone/speaker
paths are the Nocturnium and Indium satellites. Their physical audio,
certificate installation, AEC, and macOS microphone-consent checks must be
made on the respective hosts before capture is started. Home Assistant remains
outside this project and is not changed by the inventory step.
