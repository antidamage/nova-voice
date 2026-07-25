/*
 * Nova satellite acoustic echo canceller -- portable C ABI.
 *
 * This is the single canceller implementation named in SPEC.md ("The satellite
 * AEC contract").  Every satellite environment binds *this* ABI rather than
 * growing its own canceller: the macOS satellite from Swift, a Linux capture
 * process, a WASM build for the browser.  The implementation behind it wraps
 * WebRTC AEC3, the same algorithm PipeWire's echo-canceller uses, so satellites
 * cancel identically wherever they run.
 *
 * The contract deliberately keeps platform concerns out:
 *
 *   - Frames are interleaved signed 16-bit PCM at the rate given to _create.
 *     Bindings own any resampling from their hardware rate.
 *   - Callers may pass ANY frame count. WebRTC works strictly in 10 ms blocks;
 *     this layer buffers internally so a binding never has to match CoreAudio's,
 *     PipeWire's, or WebAudio's differing buffer sizes to it.
 *   - The far end is the PCM the server sent for playback, pushed at render
 *     time. It is never a host loopback/monitor device, because that is exactly
 *     the per-machine configuration the contract exists to avoid.
 *   - ERLE is reported so a binding can prove cancellation before advertising
 *     capabilities.echoCancellation, instead of claiming it because a device
 *     exists.
 *
 * Thread safety: a handle is NOT internally synchronised, but the far-end and
 * near-end paths are separate entry points precisely because they are driven
 * from different real-time threads. A binding must serialise access to the
 * handle itself (a lock around each call is sufficient; neither call blocks).
 */

#ifndef NOVA_AEC_H
#define NOVA_AEC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nova_aec nova_aec;

/* Create a canceller. sample_rate must be one of 16000/32000/48000 (WebRTC's
 * supported native rates); channels must be 1 or 2. Returns NULL on failure. */
nova_aec *nova_aec_create(int sample_rate, int channels);

void nova_aec_destroy(nova_aec *aec);

/* Push far-end (loudspeaker-bound) audio at render time. Call this as the audio
 * is actually handed to the output device, not when it arrives from the network:
 * the filter aligns against real render timing. Returns 0 on success. */
int nova_aec_push_far(nova_aec *aec, const int16_t *frames, int frame_count);

/* Process near-end (microphone) audio in place, removing the far-end echo.
 * Returns 0 on success. Frames with no far-end reference yet pass through
 * unchanged, so a binding can run this unconditionally. */
int nova_aec_process_near(nova_aec *aec, int16_t *frames, int frame_count);

/* Report the residual bulk delay between a far-end frame being rendered and its
 * echo appearing at the microphone. AEC3 tracks delay itself, but seeding it
 * shortens convergence markedly; buffering differs per platform, so bindings
 * measure this rather than assuming it. */
void nova_aec_set_stream_delay_ms(nova_aec *aec, int delay_ms);

/* Echo return loss enhancement in dB -- how much echo is actually being removed.
 *
 * Measured here, from near-end energy before versus after processing, over a
 * rolling window of blocks that carried real echo. Deliberately NOT WebRTC's own
 * statistic: in the vendored version that value reads ~0 dB while true
 * cancellation is ~30 dB, so trusting it would make a healthy canceller look
 * broken and defeat the degradation monitoring it exists for. See
 * nova_aec_filter_erle_db if you want the library's opinion too.
 *
 * Negative means "not yet known" -- no far-end activity has been cancelled yet.
 * This is the value the contract requires a binding to clear a floor on before
 * advertising AEC, and to keep reporting so a filter that quietly stops adapting
 * becomes visible instead of surfacing as unexplained echo leakage. */
double nova_aec_erle_db(const nova_aec *aec);

/* WebRTC's own ERLE estimate, for diagnostics only. Negative when unavailable.
 * Do not gate the AEC capability on this; see nova_aec_erle_db. */
double nova_aec_filter_erle_db(const nova_aec *aec);

/* Total 10 ms blocks processed on the near-end path, for diagnostics. */
uint64_t nova_aec_blocks_processed(const nova_aec *aec);

#ifdef __cplusplus
}
#endif

#endif /* NOVA_AEC_H */
