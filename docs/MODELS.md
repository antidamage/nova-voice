# Model and runtime selection

Research checked 2026-07-14 against upstream model cards and documentation.
Selection is finalized only after the simultaneous Iridium benchmark.

## Deployment shortlist

| Role | Candidate | Runtime | Why |
| --- | --- | --- | --- |
| LLM | Qwen3.5-4B Q4_K_M | pinned llama.cpp server | current compact post-trained Qwen, non-thinking mode, small enough for shared VRAM, schema-constrained output |
| STT primary | NVIDIA Nemotron Speech Streaming EN 0.6B | pinned NeMo/Transformers | NVIDIA's recommended English-only cache-aware streaming model; punctuation and low-latency chunk choices |
| Speaker identity | NVIDIA TitaNet-Large (English) | pinned NeMo, CPU | 23M-parameter embedding model already supported by the NeMo runtime; enables local multi-template household profiles without using GPU residency |
| STT challenger | NVIDIA Nemotron 3.5 ASR Streaming 0.6B, `en-GB` | NeMo 26.06 or Transformers 5.13+ | native cache-aware 80-1120 ms streaming, explicit locale conditioning and Turing/Linux support |
| STT accuracy challenger | NVIDIA Parakeet TDT 0.6B v3 | pinned NeMo | strong published offline/leaderboard accuracy and Turing testing; buffered streaming latency must be measured |
| TTS quality candidate | Qwen3-TTS 12Hz 1.7B CustomVoice | qwen-tts, BF16 where supported | streaming, direct emotion/prosody instructions, stronger English content-consistency results than 0.6B |
| TTS selected on Iridium | Qwen3-TTS 12Hz 0.6B CustomVoice | vLLM-Omni 0.24, FP16 + Turing compatibility patch | true async PCM chunks from the unchanged checkpoint; warm short-phrase first audio measured at 206–266 ms |
| TTS blind challenger | current Chatterbox V3 and original English Chatterbox | pinned upstream runtime | independent naturalness comparison; deploy only if measured emotion control, latency, and residency win |
| Wake | transcript-first matching (no acoustic model) | in-process | wake words are matched in ASR transcripts; the openWakeWord acoustic chain was removed as unused |

This is a pre-deployment bake-off, not a runtime ensemble. Exactly one LLM, one
STT, and one TTS are pinned after testing. Qwen3-TTS's 0.6B model card reports
0.9B total parameters because its speech tokenizer/decoder is included. Budget
from measured loaded memory, not marketing names or weight size alone.

Current packaging constraint: `qwen-tts` 0.1.x pins Transformers 4.57.3, while
the Nemotron 3.5 Transformers integration requires 5.13+. Do not override either
pin in production without tests. Benchmark Transformers-only STT candidates in
an isolated environment; prefer the NeMo runtime for an STT selected to share
the production PyTorch process with Qwen TTS. Install only the compatible winner.

Wake detection is transcript-first only: the configured wake words are matched
in streaming ASR output. The optional openWakeWord/TFLite acoustic wake chain
(and the legacy `hey_nova.tflite` asset) was removed after it went unused in
production; see `docs/CUSTOM-WAKE-WORDS.md` for the custom-wake-word options
that replace it.

## Why these models

### Qwen3.5-4B

Qwen's official card describes a 4B post-trained model with a hybrid
Gated-DeltaNet/attention architecture, 262K native context, strong instruction
following, and agent benchmarks. Voice needs only 8K context. Thinking is
disabled with `enable_thinking=false` to minimize latency and prevent reasoning
text from complicating tool parsing.

Qwen3.6 is the newer family, but its current official open releases are 27B and
35B-A3B. Neither is an 11 GiB simultaneous voice-stack candidate. Until a
validated small Qwen3.6 checkpoint exists, Qwen3.5-4B is the current compact
choice rather than an outdated-family assumption.

Use a GGUF converted from a pinned official revision, omit the vision projector,
and quantize locally to Q4_K_M. Do not depend on an unofficial model name such as
`Qwen3.5-4B-Instruct`. Generate a small JSON Schema supported by llama.cpp's GBNF
conversion and constrain every interpretation response; do not make correctness
depend on a model-specific native tool parser. Pin the tested llama.cpp commit.
One inference slot is enough; household concurrency is scheduled above it.

Fallback selected only before deployment: Qwen3-4B-Instruct-2507 Q4_K_M. Use it
if Qwen3.5 GGUF/tool parsing is not reliable on Turing. Do not load both.

### Streaming STT bake-off

NVIDIA explicitly recommends `nvidia/nemotron-speech-streaming-en-0.6b` for
English-only transcription. It uses cache-aware FastConformer-RNNT and processes
new, non-overlapping frames. Benchmark its documented 160 ms and 560 ms operating
points first. It cannot be declared the winner until the primary user's natural
accent (see `PRIVATEREF.md#3.3`), far-field noise, negation, numbers, and Nova
aliases are measured.

The June 2026 multilingual `nvidia/nemotron-3.5-asr-streaming-0.6b` is the second
candidate. It supports `en-GB`, native 320 ms cache-aware streaming, punctuation,
and Turing/Linux. Set the locale explicitly rather than using auto detection.
Published FLEURS results are useful only within that model's latency curve; they
must not be compared directly with a different model's leaderboard WER.

`nvidia/parakeet-tdt-0.6b-v3` remains the accuracy challenger. Its official card
reports 6.34% average Open ASR Leaderboard WER and Turing T4 testing, but its
documented streaming example is buffered with two seconds of right context and
two-second chunks. Measure final-token latency and accuracy locally; do not infer
interactive performance from its offline real-time factor or leaderboard WER.

The winning STT gets independent per-satellite streaming state. Apply supported
phrase/context boosting for aliases only after a baseline run so its effect is
measured rather than assumed.

### Emotion-controlled TTS bake-off

Both Qwen3-TTS 12Hz CustomVoice checkpoints support streaming and an `instruct`
string such as "Speak in a very happy tone". That directly satisfies emotion
matching without a second TTS or fragile audio post-processing. The upstream
English content-consistency result is slightly better for 1.7B than 0.6B, so
benchmark 1.7B first for the user's highest-quality requirement. The deployed
speaker parameter is `Serena`, the selected checkpoint's warm, gentle female
preset; the checkpoint itself remains unchanged. Qwen recommends each preset's
native language for best quality, so English pronunciation remains a local
listening-test gate rather than an unsupported claim that this checkpoint has a
female English-native preset.

Do not reject 1.7B from weight-size arithmetic: activations, tokenizer/codec,
CUDA workspaces, fragmentation, ASR caches, and LLM KV memory determine the real
resident peak. If it cannot pass simultaneous residency and latency with safe
headroom, select 0.6B. That is a measured deployment choice, not a startup fallback.

Turing does not support BF16 and the main FlashAttention-2 CUDA implementation
supports Ampere/Ada/Hopper rather than RTX 2080. The official Qwen examples use
BF16; the original `qwen-tts` FP16 path on Iridium produced invalid sampling
probabilities, so the first correct baseline used FP32 with PyTorch SDPA. The
deployed path now uses vLLM-Omni's true async Code2Wav pipeline with the exact
same 0.6B CustomVoice checkpoint. It requires eager Code2Wav execution on
Turing and one explicit BF16-to-FP16 embedding cast at the AR boundary. Short
warm phrases delivered first PCM in 206–266 ms and completed synthesis in
1.43–1.64 seconds; a longer 7.76-second reply began after 1.01 seconds and was
fully generated in 5.00 seconds. The managed TTS unit gates Nova Voice startup
on health plus a short warm-up synthesis, so a reboot cannot race STT model
loading or pass the one-time compile delay to the first household request.

The selected speaker parameter is still `Serena`; neither the checkpoint nor
the overall voice model changed. NeMo STT restores on CPU, converts to FP16
before its CUDA transfer, and explicitly matches RNNT decoder dtypes. The three
resident services used about 8.6 GiB of the 10.57 GiB GPU after warm synthesis,
leaving roughly 1.9 GiB headroom without model unloading.

## Residency gate

Do not use provisional component ranges to choose the set. The harness records:

- idle residency after warm-up
- peak during simultaneous two-stream ASR, LLM interpretation, and streaming TTS
- fragmentation/high-water mark after repeated turns
- available headroom during a 24-hour endurance run
- queue latency and ASR starvation while TTS is active

Require at least 1 GiB or 10% of physical VRAM free at measured peak, whichever
is larger, with no OOM, model reload, CPU hot-path fallback, or unbounded queue.
If over budget:

1. reduce LLM context and KV precision
2. verify the ASR actually loaded FP16 rather than retained F32 weights
3. cap concurrent synthesis and stream caches
4. use a validated smaller quantization/runtime for the same selected model
5. choose the next pre-deployment candidate

Never solve VRAM pressure by unloading/reloading models between turns.

## Required benchmark corpus

- at least 100 Nova commands spoken by the primary user at near/far distances
- every entity/zone alias, numbers, temperatures, durations, and pronouns
- 100 negative contrasts: self-intention, quoted speech, TV/media, observations
- at least 20 clips for each supported emotion label
- quiet room, fan/air-con noise, music/TV, and Nova TTS playback
- short acknowledgement, two-sentence, and long social TTS responses

Measure WER, entity/value accuracy, speech-act accuracy, tool JSON validity,
emotion agreement, TTS real-time factor/first audio, total latency, VRAM, and
GPU starvation between simultaneous streams.

## Primary sources

- Qwen3.5-4B model card: https://huggingface.co/Qwen/Qwen3.5-4B
- Qwen3.6 official releases: https://github.com/QwenLM/Qwen3.6
- Qwen-Agent/tool calling: https://github.com/QwenLM/Qwen-Agent
- Qwen3-TTS upstream: https://github.com/QwenLM/Qwen3-TTS
- Qwen3-TTS 0.6B CustomVoice card: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
- Qwen3-TTS 1.7B CustomVoice card: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
- Chatterbox upstream/model card: https://huggingface.co/ResembleAI/chatterbox
- English Nemotron Streaming model card: https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b
- Nemotron 3.5 ASR model card: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
- Parakeet TDT 0.6B v3 card: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- llama.cpp JSON Schema/GBNF: https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md
- Pipecat WebSocket transport: https://docs.pipecat.ai/api-reference/client/js/transports/websocket
- Apple LaunchAgents and `KeepAlive`: https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/Chapters/CreatingLaunchdJobs.html
- Apple modern helper registration: https://developer.apple.com/documentation/servicemanagement/updating-helper-executables-from-earlier-versions-of-macos
- systemd upstream: https://github.com/systemd/systemd
- FlashAttention GPU support: https://github.com/Dao-AILab/flash-attention
