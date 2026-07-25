"""Run one voice's fine-tune, start to packaged bundle.

Runs as a detached worker process (see worker.py) rather than inside the API, so
training survives an API restart, a deploy, or a dropped SSH session -- it is a
multi-hour job and anything shorter-lived would lose it.

Three behaviours shape the design:

* **Stop must still produce something usable.** A stop is cooperative, never a
  kill: the running stage continues to its next checkpoint, then the remaining
  stages are skipped and whatever checkpoints exist are packaged. Bundles from a
  stopped run are marked incomplete so they can be tested but never mistaken for
  finished.
* **Resume must be free.** GPT-SoVITS's own trainers already resume from the
  newest checkpoint in the experiment directory, so the only requirement is that
  we never clear it. Starting a set that has checkpoints continues it.
* **The GPU must come back.** Training mode is entered once and left in a
  finally, so a crash or an exception still restores the voice stack.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from nova_voice.training.config_builder import (
    TrainingPlan,
    build_s1_config,
    build_s2_config,
    resolve_pretrained,
    s1_command,
    s2_command,
    training_env,
)
from nova_voice.training.mode import TrainingMode
from nova_voice.training.sets import TrainingSet, TrainingState

# Poll interval while a training stage runs, in seconds. Also the worst-case
# delay between a stop request landing and the runner noticing it.
POLL_SECONDS = 5.0


class StopRequested(Exception):
    """Raised to unwind to packaging when a stop is honoured."""


class TrainingRunner:
    def __init__(
        self,
        training_set: TrainingSet,
        *,
        gptsovits_root: Path,
        python: str,
        mode: TrainingMode,
        plan_overrides: dict | None = None,
    ) -> None:
        self.set = training_set
        self.root = gptsovits_root
        self.python = python
        self.mode = mode
        self.plan_overrides = plan_overrides or {}
        self._log = training_set.log_path
        self._stopped = False

    # --- logging -----------------------------------------------------------
    def log(self, message: str) -> None:
        self._log.parent.mkdir(parents=True, exist_ok=True)
        with self._log.open("a", encoding="utf-8") as fh:
            fh.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")

    def _stage(self, stage: str, message: str, **extra) -> None:
        self.log(f"=== {stage}: {message}")
        self.set.update_state(stage=stage, message=message, **extra)

    # --- stop handling -----------------------------------------------------
    @property
    def stop_requested(self) -> bool:
        if self._stopped:
            return True
        if self.set.stop_path.exists():
            self._stopped = True
            self.set.update_state(status="stopping",
                                  message="Stop requested; finishing at the next checkpoint")
            self.log("stop requested -- will wind up at the next checkpoint")
        return self._stopped

    def _run(self, cmd: list[str], env: dict[str, str], *, stoppable: bool) -> bool:
        """Run a pipeline step. Returns False if it was stopped rather than finishing."""
        self.log(f"$ {' '.join(cmd)}")
        with self._log.open("a", encoding="utf-8") as sink:
            process = subprocess.Popen(cmd, cwd=str(self.root), env=env,
                                       stdout=sink, stderr=subprocess.STDOUT)
            self.set.update_state(pid=process.pid)
            while True:
                try:
                    process.wait(timeout=POLL_SECONDS)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if stoppable and self.stop_requested:
                    self.log("terminating stage; its last checkpoint will be used")
                    process.terminate()
                    try:
                        process.wait(timeout=180)
                    except subprocess.TimeoutExpired:
                        self.log("stage did not exit in 180s -- killing")
                        process.kill()
                        process.wait()
                    return False
        if process.returncode != 0:
            # A non-zero code right after a stop is the stop taking effect.
            if stoppable and self._stopped:
                return False
            raise RuntimeError(f"step failed ({process.returncode}): {' '.join(cmd)}")
        return True

    # --- pipeline ----------------------------------------------------------
    def prepare(self, env: dict[str, str]) -> None:
        """Slice + transcribe. Skipped when a dataset already exists (resume)."""
        transcripts = list(self.set.asr_dir.glob("*.list"))
        if transcripts and any(self.set.sliced_dir.iterdir()):
            self.log("dataset already prepared -- reusing it (resume)")
            return

        self._stage("slice", "Slicing samples into training segments")
        self._run([
            self.python, str(self.root / "tools" / "slice_audio.py"),
            str(self.set.raw_dir), str(self.set.sliced_dir),
            "-34", "4000", "300", "10", "500", "0.25", "0.25", "0", "1",
        ], env, stoppable=False)

        self._stage("asr", "Transcribing segments (Faster-Whisper)")
        self._run([
            self.python, str(self.root / "tools" / "asr" / "fasterwhisper_asr.py"),
            "-i", str(self.set.sliced_dir), "-o", str(self.set.asr_dir),
            "-l", self.set.language, "-p", "float16",
        ], env, stoppable=False)

    def extract_features(self, env: dict[str, str], plan: TrainingPlan) -> None:
        transcripts = list(self.set.asr_dir.glob("*.list"))
        if not transcripts:
            raise RuntimeError("ASR produced no transcript -- cannot train")
        pretrained = resolve_pretrained(self.root, plan.version)

        # Clear stale shards before extracting. These scripts guard their work
        # with `if os.path.exists(<their shard>) == False:` -- so a zero-byte or
        # truncated shard left behind by an interrupted run makes them skip
        # everything and exit 0, and the failure only surfaces much later as an
        # empty feature set. Extraction is cheap next to training, so always
        # redo it rather than trusting a shard we did not see completed.
        for stale in list(plan.exp_dir.glob("2-name2text-*.txt")) + \
                list(plan.exp_dir.glob("6-name2semantic-*.tsv")):
            stale.unlink()

        common = {
            **env,
            "inp_text": str(transcripts[0]),
            "inp_wav_dir": str(self.set.sliced_dir),
            "exp_name": plan.exp_name,
            "opt_dir": str(plan.exp_dir),
            "i_part": "0",
            "all_parts": "1",
            "is_half": "True" if plan.is_half else "False",
        }
        steps = [
            ("1-get-text.py", {"bert_pretrained_dir": str(pretrained["bert"])}),
            ("2-get-hubert-wav32k.py", {"cnhubert_base_dir": str(pretrained["cnhubert"])}),
        ]
        # 2-get-sv.py extracts speaker-verification embeddings, which only the
        # v2Pro/v2ProPlus model lines consume -- the webui gates it behind
        # `if "Pro" in version` for exactly that reason. Running it on plain v2
        # fails inside torch.load, because the ERes2NetV2 checkpoint it wants is
        # not part of a v2 install at all.
        if "Pro" in plan.version:
            steps.append(("2-get-sv.py", {}))
        # s2config_path is the SHIPPED TEMPLATE, matching webui.open1abc -- this
        # step only reads model hyperparameters from it, and it runs before the
        # generated training config exists. Pointing it at our generated file
        # would reference something not yet written.
        s2_template = "s2.json" if "Pro" not in plan.version else f"s2{plan.version}.json"
        steps.append(
            ("3-get-semantic.py", {
                "pretrained_s2G": str(pretrained["s2G"]),
                "s2config_path": str(self.root / "GPT_SoVITS" / "configs" / s2_template),
            }),
        )
        for index, (script, extra) in enumerate(steps, start=1):
            self._stage("features", f"Extracting features ({index}/{len(steps)}: {script})")
            self._run([self.python, str(self.root / "GPT_SoVITS" / "prepare_datasets" / script)],
                      {**common, **extra}, stoppable=False)

        # The prepare scripts shard their output by worker: each writes
        # <name>-<i_part>.<ext>, and the caller is expected to concatenate the
        # shards into the unsuffixed file the trainers actually read. The webui
        # does this inline after each step; skipping it leaves stage 1 failing
        # with FileNotFoundError on 6-name2semantic.tsv. Note the semantic file
        # gets a header row -- the dataset reader parses it as a TSV.
        _merge_parts(plan.exp_dir, "2-name2text", ".txt", header=None)
        _merge_parts(plan.exp_dir, "6-name2semantic", ".tsv", header="item_name\tsemantic_audio")

        semantic = plan.exp_dir / "6-name2semantic.tsv"
        if semantic.stat().st_size <= len("item_name\tsemantic_audio\n"):
            raise RuntimeError(
                "semantic feature extraction produced no data -- every sample was skipped. "
                "This usually means the clips are too short or silent after slicing; check "
                f"{plan.exp_dir / '2-name2text.txt'} and the sliced audio."
            )

    def train(self) -> None:
        self.set.ensure_dirs()
        if self.set.sample_count() == 0:
            raise RuntimeError("no samples uploaded for this set")

        plan = TrainingPlan(
            exp_name=self.set.id,
            exp_dir=self.set.exp_dir,
            gptsovits_root=self.root,
            **self.plan_overrides,
        )
        env = training_env(plan, dict(os.environ))
        resuming = self.set.resumable()

        self.set.update_state(
            status="preparing", stage="", message="Preparing", error="",
            started_at=time.strftime("%Y-%m-%dT%H:%M:%S"), finished_at="",
            total_epochs=plan.total_epoch,
        )
        self.log(f"--- run start (resume={resuming}) ---")

        # The GPU handover happens as late as possible and is always undone.
        self.mode.enter()
        self.log("training mode: voice stack stopped, GPU released")
        try:
            self.prepare(env)
            if self.stop_requested:
                raise StopRequested
            self.extract_features(env, plan)
            if self.stop_requested:
                raise StopRequested

            # The trainers torch.save() weights here mid-run; if it does not
            # exist (or is not writable) the failure surfaces as a bare
            # "open file failed ... Permission denied" from inside torch.
            (plan.exp_dir / "weights").mkdir(parents=True, exist_ok=True)
            plan.s1_output_dir.mkdir(parents=True, exist_ok=True)
            plan.s2_output_dir.mkdir(parents=True, exist_ok=True)

            s1_config = build_s1_config(plan, plan.exp_dir / "tmp_s1.yaml")
            s2_config = build_s2_config(plan, plan.exp_dir / "tmp_s2.json")

            self.set.update_state(status="training")
            self._stage("s1", "Stage 1: GPT/AR prosody model (the long one)")
            if not self._run(s1_command(plan, s1_config, self.python), env, stoppable=True):
                raise StopRequested

            self._stage("s2", "Stage 2: SoVITS decoder")
            if not self._run(s2_command(plan, s2_config, self.python), env, stoppable=True):
                raise StopRequested

            self.package(complete=True)
        except StopRequested:
            self.log("stopped early -- packaging what has trained so far")
            self.package(complete=False)
        except Exception as error:  # noqa: BLE001 - surfaced to the dashboard
            self.log(f"FAILED: {error}")
            self.set.update_state(status="failed", error=str(error), pid=None,
                                  finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
            raise
        finally:
            # Always give the GPU back, even on failure. Losing a training run is
            # recoverable; leaving the household without a voice assistant is not.
            self.set.stop_path.unlink(missing_ok=True)
            self.mode.leave()
            self.log("training mode left: voice stack restored")

    # --- packaging ---------------------------------------------------------
    def package(self, *, complete: bool) -> None:
        self._stage("package", "Packaging bundle")
        bundle = self.set.bundle_dir
        bundle.mkdir(parents=True, exist_ok=True)

        gpt_ckpt = _newest(self.set.exp_dir, "*.ckpt")
        sovits = _newest(self.set.exp_dir, "*.pth")
        from_pretrained = False
        if sovits is None:
            # Stage 2 never saved (an early stop during stage 1). The pretrained
            # decoder still speaks -- stage 1's prosody with generic timbre --
            # which is enough to audition the voice and decide whether to keep going.
            pretrained = resolve_pretrained(self.root)
            sovits = pretrained["s2G"]
            from_pretrained = True

        if gpt_ckpt is None:
            raise RuntimeError(
                "no stage-1 checkpoint was written yet -- let training reach its first "
                "checkpoint before stopping (save_every_epoch controls how often that is)"
            )

        shutil.copy2(gpt_ckpt, bundle / "gpt.ckpt")
        shutil.copy2(sovits, bundle / "sovits.pth")

        reference = _newest(self.set.sliced_dir, "*.wav")
        if reference is not None:
            shutil.copy2(reference, bundle / "reference.wav")
            transcripts = list(self.set.asr_dir.glob("*.list"))
            text = _reference_text(transcripts[0], reference.name) if transcripts else ""
            if text:
                (bundle / "reference.txt").write_text(text, encoding="utf-8")

        (bundle / "meta.json").write_text(json.dumps({
            "id": self.set.id,
            "name": self.set.name or self.set.id,
            "language": self.set.language,
            "source": "gptsovits",
            "complete": complete,
            "sovits_from_pretrained": from_pretrained,
            "gpt_checkpoint": gpt_ckpt.name,
            "sovits_checkpoint": sovits.name,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

        self.set.update_state(
            status="ready", stage="package", pid=None,
            message="Bundle ready" if complete else "Partial bundle ready (stopped early)",
            has_bundle=True, complete=complete,
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        self.log(f"bundle ready (complete={complete}) at {bundle}")


def _merge_parts(exp_dir: Path, stem: str, suffix: str, *, header: str | None) -> Path:
    """Concatenate <stem>-<i>.<suffix> shards into <stem><suffix>, mirroring webui.

    Shards are removed afterwards, as upstream does, so a re-run cannot silently
    concatenate stale output from a previous pass on top of fresh output.
    """
    target = exp_dir / f"{stem}{suffix}"
    lines: list[str] = [header] if header else []
    for part in sorted(exp_dir.glob(f"{stem}-*{suffix}")):
        content = part.read_text(encoding="utf-8").strip("\n")
        if content:
            lines += content.split("\n")
        part.unlink()
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _newest(directory: Path, pattern: str) -> Path | None:
    if not directory.is_dir():
        return None
    matches = sorted(directory.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _reference_text(transcript: Path, wav_name: str) -> str:
    """Pull the reference clip's own transcript out of the ASR .list file.

    GPT-SoVITS needs the exact text of reference.wav as its inference prompt; a
    mismatched prompt degrades every generation, so this is looked up rather
    than left for a human to fill in.
    """
    try:
        for line in transcript.read_text(encoding="utf-8").splitlines():
            parts = line.split("|")
            if len(parts) >= 4 and Path(parts[0]).name == wav_name:
                return parts[3].strip()
    except OSError:
        pass
    return ""
