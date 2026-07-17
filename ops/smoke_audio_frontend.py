from __future__ import annotations

import argparse

from nova_voice.audio.pcm import float32_to_pcm16
from nova_voice.audio.segmenter import SileroVad
from nova_voice.audio.wake import OpenWakeWordDetector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wake-model", required=True)
    parser.add_argument("--wake-feature-dir", required=True)
    args = parser.parse_args()

    vad = SileroVad()
    wake = OpenWakeWordDetector(args.wake_model, feature_model_dir=args.wake_feature_dir)
    silence = float32_to_pcm16([0.0] * 1_280)
    print(
        {
            "vadLoaded": True,
            "vadSilenceScore": vad.score(silence[: 512 * 2]),
            "wakeLoaded": True,
            "wakeSilenceDetected": wake.accept(silence),
        }
    )


if __name__ == "__main__":
    main()
