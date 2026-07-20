# Iridium local LLM benchmark

Iridium hosts an isolated Ollama backend for `llm-benchmark` on
`127.0.0.1:11435`. It does not replace or reconfigure Nova Voice's llama.cpp,
STT, TTS, or API services.

## Installation

From the deployed Nova Voice source tree on Iridium:

```bash
sudo bash ops/install-llm-benchmark-iridium.sh
```

The installer pins `llm-benchmark` 0.5.2 in `/opt/llm-benchmark/venv`, creates
`llm-benchmark-ollama.service`, and installs the `llm-benchmark-iridium`
wrapper. The backend is loopback-only and stores models/results beneath
`/var/lib/llm-benchmark`.

## Usage

The safe default benchmarks only `qwen2.5:0.5b`, automatically pulls it on the
first run, and never submits hardware identifiers or results:

```bash
llm-benchmark-iridium
```

Use another custom model list without enabling result submission:

```bash
LLM_BENCHMARK_MODELS_FILE=/path/to/models.yml llm-benchmark-iridium
```

Check the isolated backend and its result logs:

```bash
curl http://127.0.0.1:11435/api/version
ls -la /var/lib/llm-benchmark/results
```

The production voice models normally occupy most of the RTX 2080 Ti. Larger
benchmarks can contend for GPU memory and voice latency, so schedule them while
voice is idle. The installer never stops or restarts production Nova services.
