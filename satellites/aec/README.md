# Nova satellite AEC

The single acoustic echo canceller every Nova satellite uses, behind a narrow C
ABI so each platform *binds* it rather than growing its own. The contract it
implements — and the reasoning for it — is in `SPEC.md`, "The satellite AEC
contract". Read `include/nova_aec.h` for the API; it is documented at the point
of use.

Stage 1 of Nova's echo defence. Stage 2 (the post-STT transcript backstop in
`src/nova_voice/audio/echo.py`) stays enabled regardless: cancellation reduces
echo, it does not guarantee zero.

## What it is

`libnova_aec.a` = WebRTC **AEC3** — the same algorithm PipeWire's echo-canceller
uses, so satellites cancel identically wherever they run — plus a thin layer that
makes it usable from any host audio stack:

- **Any buffer size.** WebRTC works strictly in 10 ms blocks; this buffers
  internally, so a binding never has to reconcile CoreAudio's 512-frame buffers
  (or PipeWire's, or WebAudio's) with that.
- **The far end is the server's stream**, pushed at render time — never a host
  loopback/monitor device, because that is precisely the per-machine
  configuration this contract exists to avoid.
- **Trustworthy ERLE.** `nova_aec_erle_db()` measures cancellation here rather
  than returning WebRTC's own statistic, which in this vendored version reads
  ~0 dB while real cancellation is ~30 dB.

## Building

```sh
./build.sh            # build the static lib and run the self-test
./build.sh selftest   # re-run the self-test only
```

No package manager required — only a C/C++ toolchain, `python3`, `curl` and
`tar`. Indium has no Homebrew, so `meson`/`ninja` are installed into a local venv
under `build/` and `webrtc-audio-processing` is fetched and built there; abseil
arrives as its meson subproject. Nothing is installed on the host. First run
takes a few minutes; `build/` is git-ignored.

## The proof obligation

A satellite may only advertise `capabilities.echoCancellation = true` once it has
measured its own ERLE and cleared the floor — never because an audio device
exists. `build.sh` runs that proof and exits non-zero on failure, so it can gate
a deploy:

```
nova-aec selftest: rate=16000 chunk=512 delay=45ms blocks=1200
  offline ERLE    :  30.75 dB (floor 12.00 dB)
  rolling ERLE    :  24.00 dB  <- what nova_aec_erle_db reports live
  webrtc estimate :   0.18 dB  (diagnostics only; unreliable here)
  RESULT: PASS
```

The self-test needs no audio hardware: it synthesises a speech-like far end and
passes it through a simulated room (bulk delay, a second reflection, loudspeaker
compression) to make the echo, so results are deterministic and comparable across
platforms. There is no near-end talker, so *all* residual energy is failure to
cancel — a canceller that merely gates on far-end activity scores nothing.

Both ERLE figures are reported because they must agree: the live rolling value is
what a binding gates on, so if it diverged from the offline measurement the
capability decision would rest on a number that does not track reality.

## Adding a new environment

1. Link `build/libnova_aec.a` and include `include/nova_aec.h`.
2. Push far-end PCM in your render callback, at the moment audio goes to the
   device — not when it arrives from the network.
3. Process near-end PCM in your capture callback, in place.
4. Resample to one of 16000/32000/48000 Hz; that is the binding's job.
5. Estimate and report the bulk delay via `nova_aec_set_stream_delay_ms`. AEC3
   tracks delay itself, but seeding it shortens convergence markedly, and
   buffering differs per platform.
6. Serialise access to the handle — the far-end and near-end paths are separate
   entry points because they are driven from different real-time threads, and
   neither call blocks.
7. Run the self-test on that platform, and only advertise AEC if it passes.
