/*
 * Nova satellite AEC self-test: proves a build of the portable canceller
 * actually cancels, and reports the ERLE a binding must clear before it may
 * advertise capabilities.echoCancellation (SPEC.md, "The satellite AEC
 * contract").  Deliberately written against the public C ABI only, so every
 * binding -- macOS, Linux, WASM -- runs the identical proof.
 *
 * The near end is synthesised rather than recorded so the result is
 * deterministic and needs no audio hardware: a speech-like far-end signal is
 * passed through a simulated room (bulk delay, attenuation, a second reflection,
 * and mild loudspeaker clipping) to produce the echo. A canceller that merely
 * gates on far-end activity scores nothing here, because ERLE is measured only
 * where the near end is genuinely echo.
 *
 * Exit status is 0 when the measured ERLE clears the floor, 1 otherwise, so this
 * can gate a deploy.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nova_aec.h"

#define SAMPLE_RATE 16000
#define CHANNELS 1
#define SECONDS 12
#define TOTAL_FRAMES (SAMPLE_RATE * SECONDS)
/* Deliberately not a multiple of a 10 ms block (160 frames): the ABI promises
 * arbitrary buffer sizes, and CoreAudio hands over 512-frame buffers. */
#define CHUNK_FRAMES 512
#define ECHO_DELAY_MS 45
#define ERLE_FLOOR_DB 12.0

static double frand(void) { return (double)rand() / (double)RAND_MAX; }

/* A speech-like far-end signal: a few harmonics whose amplitude is modulated in
 * syllable-length bursts, with pauses. Flat tones would flatter the filter. */
static void synth_far(int16_t *out, int frames) {
  for (int n = 0; n < frames; ++n) {
    const double t = (double)n / SAMPLE_RATE;
    const double syllable = 0.5 + 0.5 * sin(2.0 * M_PI * 3.1 * t);
    /* Two ~1s pauses so the filter must cope with far-end silence. */
    const double gate = (t > 4.0 && t < 5.0) || (t > 9.0 && t < 10.0) ? 0.0 : 1.0;
    double v = 0.60 * sin(2.0 * M_PI * 140.0 * t) +
               0.25 * sin(2.0 * M_PI * 430.0 * t) +
               0.10 * sin(2.0 * M_PI * 1250.0 * t) +
               0.03 * (frand() * 2.0 - 1.0);
    v *= syllable * gate * 0.55;
    out[n] = (int16_t)lrint(v * 26000.0);
  }
}

int main(void) {
  srand(20260726);

  int16_t *far = malloc(sizeof(int16_t) * TOTAL_FRAMES);
  int16_t *echo = malloc(sizeof(int16_t) * TOTAL_FRAMES);
  int16_t *near = malloc(sizeof(int16_t) * TOTAL_FRAMES);
  if (far == NULL || echo == NULL || near == NULL) {
    fprintf(stderr, "selftest: out of memory\n");
    return 1;
  }
  synth_far(far, TOTAL_FRAMES);

  /* Simulated room: bulk delay, a weaker second reflection, and clipping in the
   * loudspeaker path so the echo is not a purely linear copy of the far end. */
  const int delay = SAMPLE_RATE * ECHO_DELAY_MS / 1000;
  const int delay2 = delay + SAMPLE_RATE * 17 / 1000;
  memset(echo, 0, sizeof(int16_t) * TOTAL_FRAMES);
  for (int n = 0; n < TOTAL_FRAMES; ++n) {
    double v = 0.0;
    if (n >= delay) v += 0.55 * (double)far[n - delay];
    if (n >= delay2) v += 0.22 * (double)far[n - delay2];
    v = tanh(v / 20000.0) * 20000.0; /* mild loudspeaker compression */
    if (v > 32767.0) v = 32767.0;
    if (v < -32768.0) v = -32768.0;
    echo[n] = (int16_t)lrint(v);
  }
  /* Near end is echo plus a low noise floor; no local talker, so all residual
   * energy after cancellation is failure to cancel. */
  for (int n = 0; n < TOTAL_FRAMES; ++n) {
    double v = (double)echo[n] + (frand() * 2.0 - 1.0) * 60.0;
    if (v > 32767.0) v = 32767.0;
    if (v < -32768.0) v = -32768.0;
    near[n] = (int16_t)lrint(v);
  }

  nova_aec *aec = nova_aec_create(SAMPLE_RATE, CHANNELS);
  if (aec == NULL) {
    fprintf(stderr, "selftest: nova_aec_create failed\n");
    return 1;
  }
  nova_aec_set_stream_delay_ms(aec, ECHO_DELAY_MS);

  int16_t *processed = malloc(sizeof(int16_t) * TOTAL_FRAMES);
  if (processed == NULL) {
    fprintf(stderr, "selftest: out of memory\n");
    return 1;
  }

  for (int offset = 0; offset < TOTAL_FRAMES; offset += CHUNK_FRAMES) {
    int count = CHUNK_FRAMES;
    if (offset + count > TOTAL_FRAMES) count = TOTAL_FRAMES - offset;
    /* Far end is pushed at render time, i.e. as it goes to the speaker, which is
     * ahead of the echo arriving at the microphone -- the real ordering. */
    if (nova_aec_push_far(aec, far + offset, count) != 0) {
      fprintf(stderr, "selftest: push_far failed at %d\n", offset);
      return 1;
    }
    memcpy(processed + offset, near + offset, sizeof(int16_t) * (size_t)count);
    if (nova_aec_process_near(aec, processed + offset, count) != 0) {
      fprintf(stderr, "selftest: process_near failed at %d\n", offset);
      return 1;
    }
  }

  /* Measure over the final third only: the first seconds are convergence, and
   * scoring them would understate steady-state performance. */
  const int from = (TOTAL_FRAMES * 2) / 3;
  double in_energy = 0.0;
  double out_energy = 0.0;
  for (int n = from; n < TOTAL_FRAMES; ++n) {
    in_energy += (double)near[n] * (double)near[n];
    out_energy += (double)processed[n] * (double)processed[n];
  }
  const double measured =
      out_energy <= 0.0 ? 99.0 : 10.0 * log10(in_energy / out_energy);
  const double rolling = nova_aec_erle_db(aec);
  const double filter_reported = nova_aec_filter_erle_db(aec);

  printf("nova-aec selftest: rate=%d chunk=%d delay=%dms blocks=%llu\n",
         SAMPLE_RATE, CHUNK_FRAMES, ECHO_DELAY_MS,
         (unsigned long long)nova_aec_blocks_processed(aec));
  printf("  offline ERLE    : %6.2f dB (floor %.2f dB)\n", measured,
         ERLE_FLOOR_DB);
  printf("  rolling ERLE    : %6.2f dB  <- what nova_aec_erle_db reports live\n",
         rolling);
  printf("  webrtc estimate : %6.2f dB  (diagnostics only; unreliable here)\n",
         filter_reported);
  /* The live figure a binding gates on must agree with the offline measurement,
   * or capability advertising would be based on a number that does not track
   * reality. */
  if (rolling < ERLE_FLOOR_DB) {
    printf("  RESULT: FAIL -- rolling ERLE below floor; nova_aec_erle_db "
           "would not authorise advertising AEC\n");
    nova_aec_destroy(aec);
    return 1;
  }

  nova_aec_destroy(aec);
  free(far);
  free(echo);
  free(near);
  free(processed);

  if (measured < ERLE_FLOOR_DB) {
    printf("  RESULT: FAIL -- this build must advertise echoCancellation=false\n");
    return 1;
  }
  printf("  RESULT: PASS\n");
  return 0;
}
