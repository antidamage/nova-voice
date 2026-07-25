"""Detached entry point for one training run.

Launched by the API with setsid + no controlling terminal, so the run outlives
the API process, a deploy, or the SSH session that started it. All progress goes
to the set's state.json and train.log; nothing is reported back through the
parent.

    python -m nova_voice.training.worker <set_id>
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

from nova_voice.training.mode import TrainingMode
from nova_voice.training.paths import (
    GPTSOVITS_PYTHON,
    GPTSOVITS_ROOT,
    TRAINING_MODE_STATE,
    TRAINING_ROOT,
)
from nova_voice.training.runner import TrainingRunner
from nova_voice.training.sets import TrainingSetStore


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: python -m nova_voice.training.worker <set_id>", file=sys.stderr)
        return 2

    store = TrainingSetStore(Path(TRAINING_ROOT))
    training_set = store.get(argv[0])
    if training_set is None:
        print(f"unknown training set: {argv[0]}", file=sys.stderr)
        return 1

    overrides: dict = {}
    for key, cast in (("batch_size", int), ("total_epoch", int), ("save_every_epoch", int)):
        raw = os.environ.get(f"NOVA_TRAINING_{key.upper()}")
        if raw:
            try:
                overrides[key] = cast(raw)
            except ValueError:
                pass

    runner = TrainingRunner(
        training_set,
        gptsovits_root=Path(GPTSOVITS_ROOT),
        python=GPTSOVITS_PYTHON,
        mode=TrainingMode(Path(TRAINING_MODE_STATE)),
        plan_overrides=overrides,
    )
    try:
        runner.train()
    except Exception:  # noqa: BLE001 - already recorded in state.json; log detail
        runner.log(traceback.format_exc())
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
