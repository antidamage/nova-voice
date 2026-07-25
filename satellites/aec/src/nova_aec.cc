// WebRTC AEC3 behind Nova's portable satellite AEC ABI. See include/nova_aec.h
// for the contract and SPEC.md "The satellite AEC contract" for why it exists.

#include "nova_aec.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <memory>
#include <vector>

#include "modules/audio_processing/include/audio_processing.h"

namespace {

// WebRTC's APM consumes and produces strictly 10 ms frames.
constexpr int kBlockMs = 10;

bool SupportedRate(int rate) {
  return rate == 16000 || rate == 32000 || rate == 48000;
}

}  // namespace

struct nova_aec {
  webrtc::AudioProcessing *apm = nullptr;
  webrtc::StreamConfig config;
  int block_frames = 0;   // frames (per channel) in a 10 ms block
  int block_samples = 0;  // interleaved samples in a 10 ms block
  int channels = 1;
  // Callers hand us arbitrary buffer sizes; these hold the remainder between
  // calls so we only ever feed WebRTC whole blocks.
  std::deque<int16_t> far_pending;
  std::deque<int16_t> near_pending;
  // Near-end output that has been processed but not yet collected by the caller,
  // because their buffer did not land on a block boundary.
  std::deque<int16_t> near_ready;
  std::vector<int16_t> scratch;
  uint64_t blocks = 0;
  bool far_seen = false;
  // Rolling near-end energy before/after cancellation, exponentially forgotten
  // so a filter that stops adapting shows up rather than being averaged away by
  // hours of past success.
  double erle_in = 0.0;
  double erle_out = 0.0;

  ~nova_aec() { delete apm; }
};

extern "C" {

nova_aec *nova_aec_create(int sample_rate, int channels) {
  if (!SupportedRate(sample_rate) || channels < 1 || channels > 2) {
    return nullptr;
  }
  auto aec = std::make_unique<nova_aec>();
  aec->apm = webrtc::AudioProcessingBuilder().Create();
  if (aec->apm == nullptr) {
    return nullptr;
  }

  webrtc::AudioProcessing::Config config;
  // mobile_mode=false selects AEC3 rather than the mobile AECM.
  config.echo_canceller.enabled = true;
  config.echo_canceller.mobile_mode = false;
  // The satellite's own gate and the server's DSP own everything else; leaving
  // these off keeps this component a canceller and nothing more, so its effect
  // on the signal is attributable when diagnosing.
  config.gain_controller1.enabled = false;
  config.gain_controller2.enabled = false;
  config.noise_suppression.enabled = false;
  config.high_pass_filter.enabled = true;  // helps the filter converge
  aec->apm->ApplyConfig(config);

  aec->channels = channels;
  aec->config = webrtc::StreamConfig(sample_rate, channels);
  aec->block_frames = sample_rate / (1000 / kBlockMs);
  aec->block_samples = aec->block_frames * channels;
  aec->scratch.resize(static_cast<size_t>(aec->block_samples));
  return aec.release();
}

void nova_aec_destroy(nova_aec *aec) { delete aec; }

int nova_aec_push_far(nova_aec *aec, const int16_t *frames, int frame_count) {
  if (aec == nullptr || frames == nullptr || frame_count < 0) {
    return -1;
  }
  const int samples = frame_count * aec->channels;
  aec->far_pending.insert(aec->far_pending.end(), frames, frames + samples);
  while (static_cast<int>(aec->far_pending.size()) >= aec->block_samples) {
    for (int i = 0; i < aec->block_samples; ++i) {
      aec->scratch[static_cast<size_t>(i)] = aec->far_pending.front();
      aec->far_pending.pop_front();
    }
    const int status = aec->apm->ProcessReverseStream(
        aec->scratch.data(), aec->config, aec->config, aec->scratch.data());
    if (status != 0) {
      return status;
    }
    aec->far_seen = true;
  }
  return 0;
}

int nova_aec_process_near(nova_aec *aec, int16_t *frames, int frame_count) {
  if (aec == nullptr || frames == nullptr || frame_count < 0) {
    return -1;
  }
  const int samples = frame_count * aec->channels;
  // Before any far-end audio exists there is nothing to cancel. Passing through
  // rather than buffering keeps the mic path honest when Nova is silent, which
  // is most of the time, and lets a binding call this unconditionally.
  if (!aec->far_seen && aec->near_ready.empty() && aec->near_pending.empty()) {
    return 0;
  }

  aec->near_pending.insert(aec->near_pending.end(), frames, frames + samples);
  while (static_cast<int>(aec->near_pending.size()) >= aec->block_samples) {
    for (int i = 0; i < aec->block_samples; ++i) {
      aec->scratch[static_cast<size_t>(i)] = aec->near_pending.front();
      aec->near_pending.pop_front();
    }
    double before = 0.0;
    for (int i = 0; i < aec->block_samples; ++i) {
      const double sample = aec->scratch[static_cast<size_t>(i)];
      before += sample * sample;
    }
    const int status = aec->apm->ProcessStream(aec->scratch.data(), aec->config,
                                               aec->config, aec->scratch.data());
    if (status != 0) {
      return status;
    }
    double after = 0.0;
    for (int i = 0; i < aec->block_samples; ++i) {
      const double sample = aec->scratch[static_cast<size_t>(i)];
      after += sample * sample;
    }
    // Only blocks carrying real energy inform ERLE: during near-silence the
    // ratio is noise-on-noise and would drag a healthy figure toward 0 dB.
    const double floor = 1e4 * static_cast<double>(aec->block_samples);
    if (before > floor) {
      constexpr double kForget = 0.995;
      aec->erle_in = aec->erle_in * kForget + before;
      aec->erle_out = aec->erle_out * kForget + after;
    }
    ++aec->blocks;
    aec->near_ready.insert(aec->near_ready.end(), aec->scratch.begin(),
                           aec->scratch.end());
  }

  // Hand back exactly as many samples as the caller gave us. Early on, fewer
  // cancelled samples exist than were supplied (a block is still filling), so
  // the tail is left as captured rather than zeroed -- silence would be a
  // audible dropout, and this only happens in the first block.
  const int available =
      std::min<int>(samples, static_cast<int>(aec->near_ready.size()));
  for (int i = 0; i < available; ++i) {
    frames[i] = aec->near_ready.front();
    aec->near_ready.pop_front();
  }
  return 0;
}

void nova_aec_set_stream_delay_ms(nova_aec *aec, int delay_ms) {
  if (aec == nullptr || delay_ms < 0) {
    return;
  }
  aec->apm->set_stream_delay_ms(delay_ms);
}

double nova_aec_erle_db(const nova_aec *aec) {
  if (aec == nullptr || aec->erle_in <= 0.0) {
    return -1.0;
  }
  if (aec->erle_out <= 0.0) {
    return 99.0;  // cancelled to nothing measurable
  }
  return 10.0 * std::log10(aec->erle_in / aec->erle_out);
}

double nova_aec_filter_erle_db(const nova_aec *aec) {
  if (aec == nullptr) {
    return -1.0;
  }
  const auto stats = aec->apm->GetStatistics();
  if (!stats.echo_return_loss_enhancement.has_value()) {
    return -1.0;
  }
  return *stats.echo_return_loss_enhancement;
}

uint64_t nova_aec_blocks_processed(const nova_aec *aec) {
  return aec == nullptr ? 0 : aec->blocks;
}

}  // extern "C"
