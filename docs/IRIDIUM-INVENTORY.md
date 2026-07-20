# Iridium deployment inventory

This is a redacted, read-only snapshot. Re-run `ops/inventory-iridium.sh`
before each model bake-off; it is intentionally not a provisioning script.

The verified host is Iridium (hostname/address: see `PRIVATEREF.md#1.1`) running Ubuntu 25.10
with a 12-thread i7-5820K, 15 GiB RAM, and an RTX 2080 Ti (11,264 MiB). The
NVIDIA driver is 580.159.03 and `nvidia-smi` reports CUDA 13.0; the installed
CUDA toolkit reports 12.4. PipeWire and WirePlumber are active, but the
current graph exposes only a dummy source/sink, so physical capture hardware
must be provisioned before a live local-mic test.

At the snapshot, the two intended resident processes were healthy:

- `llama-server` on `127.0.0.1:8765`, Qwen3.5-4B Q4_K_M, context 4096,
  one slot, all layers on GPU, q8 KV cache (about 3.0 GiB GPU memory).
- `nova-voice` on mTLS `0.0.0.0:8766`, with Nemotron streaming 0.6B and
  Qwen3-TTS 0.6B loaded (5,978 MiB GPU memory after a full response turn).

The latest post-deployment resident set uses 9,000 MiB of 11,264 MiB (3,022
MiB LLM plus 5,978 MiB voice), leaving 2,264 MiB for CUDA allocator variance.
Qwen3-TTS uses FP32 because its official BF16 precision is unavailable on
Turing and the measured FP16 path generated invalid sampling probabilities.
The authenticated health check reports the native NeMo cache-aware streaming
path enabled. The 24-hour residency gate remains mandatory; no model is
swapped at runtime.

Iridium's dummy audio graph is expected while the primary microphone/speaker
paths are the Nocturnium and Indium satellites. Their physical audio,
certificate installation, AEC, and macOS microphone-consent checks must be
made on the respective hosts before capture is started. Home Assistant remains
outside this project and is not changed by the inventory step.
