from __future__ import annotations

import argparse

from openwakeword.utils import download_models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target_directory")
    args = parser.parse_args()
    # An intentionally non-matching wake model name downloads only the feature
    # extraction and VAD assets required by a household custom model.
    download_models(
        model_names=["__features_only__"],
        target_directory=args.target_directory,
    )


if __name__ == "__main__":
    main()
