"""Generate GPT-SoVITS stage-1/stage-2 training configs.

GPT-SoVITS does not accept training parameters on the command line: both
trainers read a config file whose schema is version-specific and shifts across
the project's v1/v2/v2Pro/v3/v4 model lines. The webui does not hand-write those
files either -- it loads the template shipped in ``GPT_SoVITS/configs/``, mutates
a known set of keys, and writes a temporary copy. This module does exactly the
same thing, so the configs we train with are the ones upstream would have
produced, and a GPT-SoVITS upgrade that changes the schema is picked up for free
via its own templates.

Every key set below was read off ``webui.py``'s ``open1Ba`` (SoVITS/stage 2) and
``open1Bb`` (GPT/stage 1) functions rather than guessed. Two details that are
easy to get wrong and are verified here:

* the trainers take DIFFERENT flags -- ``s1_train.py --config_file`` and
  ``s2_train.py --config``. Neither accepts ``-c``.
* stage 1 needs ``_CUDA_VISIBLE_DEVICES`` and ``hz=25hz`` in the environment.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TrainingPlan:
    """Everything the trainers need for one voice, resolved to absolute paths."""

    exp_name: str
    exp_dir: Path
    """Per-voice experiment directory: holds extracted features and checkpoints.
    Both trainers resume from what they find here, so it must survive between
    runs -- that is the whole basis of stop/resume."""

    gptsovits_root: Path
    version: str = "v2"
    batch_size: int = 4
    total_epoch: int = 15
    save_every_epoch: int = 1
    """Checkpoint interval, in epochs. Also the granularity at which a stop
    request can take effect without losing work, so keep it small."""

    is_half: bool = True
    if_save_latest: bool = True
    if_save_every_weights: bool = True
    if_dpo: bool = False
    if_grad_ckpt: bool = False
    lora_rank: int = 32
    text_low_lr_rate: float = 0.4
    gpu_numbers: str = "0"

    @property
    def s1_output_dir(self) -> Path:
        return self.exp_dir / f"logs_s1_{self.version}"

    @property
    def s2_output_dir(self) -> Path:
        return self.exp_dir / f"logs_s2_{self.version}"


def _pretrained_dir(root: Path) -> Path:
    return root / "GPT_SoVITS" / "pretrained_models"


def _read_template(path: Path) -> str:
    """Read a shipped config template, or say which install is incomplete.

    A missing template means the GPT-SoVITS checkout is broken or a version
    renamed the file -- worth naming explicitly rather than surfacing a bare
    FileNotFoundError from deep inside config generation.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"GPT-SoVITS config template not found: {path}. The checkout is incomplete, "
            "or this version renamed it -- check GPT_SoVITS/configs/."
        )
    return path.read_text(encoding="utf-8")


def resolve_pretrained(root: Path, version: str = "v2") -> dict[str, Path]:
    """Locate the pretrained checkpoints the trainers warm-start from.

    Filenames move between GPT-SoVITS releases, so each is resolved by glob
    against what is actually installed rather than hardcoded, and a miss is
    reported with the directory listing instead of failing later inside the
    trainer with a bare FileNotFoundError.
    """
    base = _pretrained_dir(root)
    # Exact per-version filenames first, mirroring GPT-SoVITS's own config.py
    # (pretrained_gpt_name / pretrained_sovits_name). These MUST be version
    # correct: several s1 checkpoints ship side by side, and warm-starting v2
    # from v1's checkpoint fails deep inside training with "Error(s) in loading
    # state_dict for Text2SemanticLightningModule" rather than anything
    # obviously version-related. The globs are a fallback for layouts that move.
    exact = {
        "v1": {"s1": "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
               "s2G": "s2G488k.pth", "s2D": "s2D488k.pth"},
        "v2": {"s1": "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
               "s2G": "gsv-v2final-pretrained/s2G2333k.pth",
               "s2D": "gsv-v2final-pretrained/s2D2333k.pth"},
    }.get(version, {})
    patterns = {
        "s1": [exact.get("s1", ""), f"gsv-{version}final-pretrained/s1bert*.ckpt", "*s1bert*.ckpt"],
        "s2G": [exact.get("s2G", ""), f"gsv-{version}final-pretrained/s2G*.pth", "**/s2G*.pth"],
        "s2D": [exact.get("s2D", ""), f"gsv-{version}final-pretrained/s2D*.pth", "**/s2D*.pth"],
        "bert": ["chinese-roberta-wwm-ext-large"],
        "cnhubert": ["chinese-hubert-base"],
    }
    patterns = {key: [p for p in globs if p] for key, globs in patterns.items()}
    resolved: dict[str, Path] = {}
    for key, globs in patterns.items():
        for pattern in globs:
            hits = sorted(base.glob(pattern))
            if hits:
                resolved[key] = hits[0]
                break
    missing = [key for key in patterns if key not in resolved]
    if missing:
        available = "\n  ".join(sorted(p.name for p in base.glob("*"))) or "(empty)"
        raise FileNotFoundError(
            f"pretrained model(s) not found under {base}: {', '.join(missing)}.\n"
            f"Installed:\n  {available}"
        )
    return resolved


def build_s1_config(plan: TrainingPlan, out_path: Path) -> Path:
    """Write the stage-1 (GPT/AR) YAML, mirroring webui.open1Bb."""
    template = plan.gptsovits_root / "GPT_SoVITS" / "configs" / (
        "s1longer.yaml" if plan.version == "v1" else "s1longer-v2.yaml"
    )
    data = yaml.load(_read_template(template), Loader=yaml.FullLoader)
    pretrained = resolve_pretrained(plan.gptsovits_root, plan.version)

    batch_size = plan.batch_size
    if not plan.is_half:
        data["train"]["precision"] = "32"
        batch_size = max(1, batch_size // 2)

    data["train"]["batch_size"] = batch_size
    data["train"]["epochs"] = plan.total_epoch
    data["train"]["save_every_n_epoch"] = plan.save_every_epoch
    data["train"]["if_save_every_weights"] = plan.if_save_every_weights
    data["train"]["if_save_latest"] = plan.if_save_latest
    data["train"]["if_dpo"] = plan.if_dpo
    data["train"]["half_weights_save_dir"] = str(plan.exp_dir / "weights")
    data["train"]["exp_name"] = plan.exp_name
    data["pretrained_s1"] = str(pretrained["s1"])
    data["train_semantic_path"] = str(plan.exp_dir / "6-name2semantic.tsv")
    data["train_phoneme_path"] = str(plan.exp_dir / "2-name2text.txt")
    data["output_dir"] = str(plan.s1_output_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")
    return out_path


def build_s2_config(plan: TrainingPlan, out_path: Path) -> Path:
    """Write the stage-2 (SoVITS decoder) JSON, mirroring webui.open1Ba."""
    name = "s2.json" if plan.version not in {"v2Pro", "v2ProPlus"} else f"s2{plan.version}.json"
    template = plan.gptsovits_root / "GPT_SoVITS" / "configs" / name
    data = json.loads(_read_template(template))
    pretrained = resolve_pretrained(plan.gptsovits_root, plan.version)

    batch_size = plan.batch_size
    if not plan.is_half:
        data["train"]["fp16_run"] = False
        batch_size = max(1, batch_size // 2)

    data["train"]["batch_size"] = batch_size
    data["train"]["epochs"] = plan.total_epoch
    data["train"]["text_low_lr_rate"] = plan.text_low_lr_rate
    data["train"]["pretrained_s2G"] = str(pretrained["s2G"])
    data["train"]["pretrained_s2D"] = str(pretrained["s2D"])
    data["train"]["if_save_latest"] = plan.if_save_latest
    data["train"]["if_save_every_weights"] = plan.if_save_every_weights
    data["train"]["save_every_epoch"] = plan.save_every_epoch
    data["train"]["gpu_numbers"] = plan.gpu_numbers
    data["train"]["grad_ckpt"] = plan.if_grad_ckpt
    data["train"]["lora_rank"] = plan.lora_rank
    data["model"]["version"] = plan.version
    data["data"]["exp_dir"] = data["s2_ckpt_dir"] = str(plan.exp_dir)
    data["save_weight_dir"] = str(plan.exp_dir / "weights")
    data["name"] = plan.exp_name
    data["version"] = plan.version

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out_path


def s1_command(plan: TrainingPlan, config_path: Path, python: str) -> list[str]:
    # -s: run without site-packages injection, exactly as the webui invokes it.
    return [python, "-s", str(plan.gptsovits_root / "GPT_SoVITS" / "s1_train.py"),
            "--config_file", str(config_path)]


def s2_command(plan: TrainingPlan, config_path: Path, python: str) -> list[str]:
    return [python, "-s", str(plan.gptsovits_root / "GPT_SoVITS" / "s2_train.py"),
            "--config", str(config_path)]


def training_env(plan: TrainingPlan, base: dict[str, str]) -> dict[str, str]:
    """Environment every GPT-SoVITS subprocess needs.

    PYTHONPATH is the important one: GPT-SoVITS's scripts import across their own
    tree (``from tools.my_utils import ...``), but running a script by path puts
    only *that script's* directory on sys.path, so the imports fail with
    ``ModuleNotFoundError: No module named 'tools'``. The webui gets away with it
    because it does ``sys.path.insert(0, now_dir)`` for itself and then spawns
    children that inherit it; we set it explicitly for the same effect.

    ``hz`` is read by stage 1's dataset module.
    """
    # BOTH roots are needed, for different scripts: tools/*.py import across the
    # repo root (``from tools.my_utils import ...``) while
    # GPT_SoVITS/prepare_datasets/*.py import as if GPT_SoVITS/ were the root
    # (``import text``). Supplying only one gets you through the slicer and then
    # fails at feature extraction.
    existing = base.get("PYTHONPATH", "")
    roots = [str(plan.gptsovits_root), str(plan.gptsovits_root / "GPT_SoVITS")]
    if existing:
        roots.append(existing)
    return {
        **base,
        "PYTHONPATH": os.pathsep.join(roots),
        "_CUDA_VISIBLE_DEVICES": plan.gpu_numbers,
        "CUDA_VISIBLE_DEVICES": plan.gpu_numbers,
        "hz": "25hz",
        "PYTHONUNBUFFERED": "1",
        # RESUME depends on this. PyTorch >=2.6 defaults torch.load to
        # weights_only=True, and the Lightning checkpoints these trainers write
        # embed a pathlib.PosixPath in their hyperparameters -- so continuing a
        # run fails with "Weights only load failed ... Unsupported global: GLOBAL
        # pathlib.PosixPath". The restriction exists to stop untrusted
        # checkpoints executing code on load; these are produced by this same
        # pipeline, on this host, from the owner's own samples, so there is no
        # untrusted input to protect against. Scoped to the training
        # subprocesses -- it is not set for the serving stack, which does load
        # third-party checkpoints.
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
    }
