"""Voice training: set lifecycle, stop semantics, resume detection, training mode."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from nova_voice.training.config_builder import TrainingPlan, build_s1_config, build_s2_config
from nova_voice.training.mode import ModeSnapshot, TrainingMode
from nova_voice.training.service import TrainingError, TrainingService
from nova_voice.training.sets import TrainingSetStore, normalize_set_id


@pytest.fixture()
def service(tmp_path: Path) -> TrainingService:
    svc = TrainingService(root=str(tmp_path / "sets"))
    svc.mode = TrainingMode(tmp_path / "training-mode.json")
    return svc


def test_set_id_normalizes_like_the_voice_registry():
    assert normalize_set_id("Johnny Silverhand") == "johnny-silverhand"
    assert normalize_set_id("  MiXeD_Case-1 ") == "mixed_case-1"
    with pytest.raises(ValueError):
        normalize_set_id("!!!")


def test_create_and_upload_samples(service: TrainingService):
    service.create("johnny", "Johnny", "en")
    result = service.add_samples("johnny", [
        ("a.wav", io.BytesIO(b"RIFF")),
        ("b.flac", io.BytesIO(b"fLaC")),
        ("notes.txt", io.BytesIO(b"skip me")),
    ])
    assert result == {"accepted": 2, "skipped": 1, "total": 2}

    # A second upload appends rather than replacing -- 100+ files may arrive in
    # several batches from the browser.
    again = service.add_samples("johnny", [("c.wav", io.BytesIO(b"RIFF"))])
    assert again["total"] == 3


def test_state_is_camelcase_on_the_wire_but_round_trips_on_disk(service: TrainingService):
    """The dashboard reads camelCase; the state file must still reload."""
    service.create("wire", "Wire", "en")
    live = service.store.get("wire")
    live.update_state(status="training", total_epochs=12, has_bundle=True)

    wire = live.summary()["state"]
    assert "totalEpochs" in wire and "hasBundle" in wire
    assert "total_epochs" not in wire

    reloaded = service.store.get("wire").read_state()
    assert reloaded.total_epochs == 12, "on-disk state must survive a reload"
    assert reloaded.has_bundle is True
    assert reloaded.status == "training"


def test_duplicate_set_is_rejected(service: TrainingService):
    service.create("johnny", "Johnny", "en")
    with pytest.raises(TrainingError):
        service.create("johnny", "Johnny", "en")


def test_start_requires_samples(service: TrainingService):
    service.create("empty", "Empty", "en")
    with pytest.raises(TrainingError, match="upload some samples"):
        service.start("empty")


def test_start_explains_a_home_directory_interpreter(service: TrainingService, monkeypatch, tmp_path: Path):
    """ProtectHome=yes hides /home from the service; a bare PermissionError at
    the first pipeline step sends you chasing file permissions that are fine."""
    import nova_voice.training.service as service_module

    service.create("homed", "Homed", "en")
    service.add_samples("homed", [("a.wav", io.BytesIO(b"RIFF"))])
    monkeypatch.setattr(service_module, "GPTSOVITS_ROOT", str(tmp_path))
    monkeypatch.setattr(service_module, "GPTSOVITS_PYTHON", "/home/someone/env/bin/python")

    with pytest.raises(TrainingError, match="ProtectHome"):
        service.start("homed")


def test_start_reports_a_missing_interpreter(service: TrainingService, monkeypatch, tmp_path: Path):
    import nova_voice.training.service as service_module

    service.create("noexe", "NoExe", "en")
    service.add_samples("noexe", [("a.wav", io.BytesIO(b"RIFF"))])
    monkeypatch.setattr(service_module, "GPTSOVITS_ROOT", str(tmp_path))
    monkeypatch.setattr(service_module, "GPTSOVITS_PYTHON", str(tmp_path / "missing" / "python"))

    with pytest.raises(TrainingError, match="not executable"):
        service.start("noexe")


def test_stop_requires_a_running_set(service: TrainingService):
    service.create("idle", "Idle", "en")
    with pytest.raises(TrainingError, match="not running"):
        service.stop("idle")


def test_stop_writes_the_cooperative_stop_file(service: TrainingService):
    training_set = service.create("run", "Run", "en")
    store = TrainingSetStore(service.store.root)
    live = store.get("run")
    live.update_state(status="training")

    service.stop("run")
    assert live.stop_path.exists(), "stop must be a request the runner can honour"
    assert live.read_state().status == "stopping"


def test_resumable_reflects_existing_checkpoints(service: TrainingService):
    service.create("resume", "Resume", "en")
    live = service.store.get("resume")
    assert live.resumable() is False

    (live.exp_dir / "logs_s1").mkdir(parents=True, exist_ok=True)
    (live.exp_dir / "logs_s1" / "epoch=3.ckpt").write_bytes(b"ckpt")
    assert live.resumable() is True, "a set with checkpoints must continue, not restart"


def test_adding_samples_invalidates_the_prepared_dataset(service: TrainingService):
    """Reusing a dataset is what makes resume fast, but it must be keyed on the
    SAMPLES -- otherwise adding recordings and pressing Resume silently trains
    on the old set again and republishes the same voice."""
    import io as _io

    service.create("grow", "Grow", "en")
    live = service.store.get("grow")
    service.add_samples("grow", [("a.wav", _io.BytesIO(b"RIFF0"))])

    # Simulate a completed prepare + training run.
    (live.sliced_dir / "seg.wav").write_bytes(b"RIFF")
    (live.asr_dir / "sliced.list").write_text("seg.wav|x|EN|hello\n", encoding="utf-8")
    (live.exp_dir / "logs_s1").mkdir(parents=True, exist_ok=True)
    (live.exp_dir / "logs_s1" / "e15.ckpt").write_bytes(b"ckpt")
    live.record_dataset_fingerprint()

    assert live.dataset_matches_samples() is True
    assert service.store.get("grow").summary()["resumable"] is True
    assert service.store.get("grow").summary()["samplesChanged"] is False

    # New audio arrives.
    service.add_samples("grow", [("b.wav", _io.BytesIO(b"RIFF-longer-content"))])

    fresh = service.store.get("grow")
    assert fresh.dataset_matches_samples() is False, "new samples must invalidate the dataset"
    summary = fresh.summary()
    assert summary["samplesChanged"] is True
    assert summary["resumable"] is False, "must not offer Resume when it would ignore new audio"


def test_publish_refuses_an_incomplete_bundle(service: TrainingService):
    service.create("partial", "Partial", "en")
    with pytest.raises(TrainingError, match="incomplete"):
        service.publish("partial")


def test_publish_copies_the_bundle(service: TrainingService, tmp_path: Path):
    service.create("done", "Done", "en")
    live = service.store.get("done")
    live.bundle_dir.mkdir(parents=True, exist_ok=True)
    for name in ("gpt.ckpt", "sovits.pth", "reference.wav"):
        (live.bundle_dir / name).write_bytes(b"x")

    import nova_voice.training.service as service_module

    target_root = tmp_path / "trained-voices"
    original = service_module.TRAINED_VOICES_DIR
    service_module.TRAINED_VOICES_DIR = str(target_root)
    try:
        result = service.publish("done")
    finally:
        service_module.TRAINED_VOICES_DIR = original

    assert result["voiceId"] == "done"
    assert (target_root / "done" / "gpt.ckpt").is_file()


def test_publish_reports_a_permission_problem_actionably(service: TrainingService, tmp_path: Path):
    """The catalogue belongs to the engine's account while publish runs as the
    orchestrator, so a permissions mismatch must not surface as a bare 500."""
    import nova_voice.training.service as service_module

    service.create("blocked", "Blocked", "en")
    live = service.store.get("blocked")
    live.bundle_dir.mkdir(parents=True, exist_ok=True)
    for name in ("gpt.ckpt", "sovits.pth", "reference.wav"):
        (live.bundle_dir / name).write_bytes(b"x")

    original = service_module.TRAINED_VOICES_DIR
    service_module.TRAINED_VOICES_DIR = str(tmp_path / "catalogue")

    def deny(*args, **kwargs):
        raise PermissionError(13, "Permission denied")

    original_mkdir = Path.mkdir
    Path.mkdir = deny  # type: ignore[method-assign]
    try:
        with pytest.raises(TrainingError, match="could not publish"):
            service.publish("blocked")
    finally:
        Path.mkdir = original_mkdir  # type: ignore[method-assign]
        service_module.TRAINED_VOICES_DIR = original


def test_samples_cannot_change_mid_run(service: TrainingService):
    service.create("busy", "Busy", "en")
    service.store.get("busy").update_state(status="training")
    with pytest.raises(TrainingError, match="while this set is training"):
        service.add_samples("busy", [("a.wav", io.BytesIO(b"RIFF"))])


class TestTrainingMode:
    """The GPU handover. Getting restore wrong leaves the household voiceless."""

    def test_reentering_keeps_the_original_snapshot(self, tmp_path: Path, monkeypatch):
        state = tmp_path / "mode.json"
        mode = TrainingMode(state)
        monkeypatch.setattr(mode, "_stop_stack", lambda: None)

        first = ModeSnapshot(orchestrator_active=True, llm_active=True,
                             engine_unit="nova-voice-dots-tts.service",
                             engine_active=True, entered_at="t0")
        monkeypatch.setattr(ModeSnapshot, "capture", classmethod(lambda cls: first))
        mode.enter()

        # Second enter must NOT record the already-stopped stack as the thing to
        # restore, or leaving training mode would bring nothing back up.
        stopped = ModeSnapshot(orchestrator_active=False, llm_active=False,
                               engine_unit=None, engine_active=False, entered_at="t1")
        monkeypatch.setattr(ModeSnapshot, "capture", classmethod(lambda cls: stopped))
        again = mode.enter()

        assert again.orchestrator_active is True
        assert mode.read_snapshot().entered_at == "t0"

    def test_wait_for_gpu_blocks_until_the_models_are_gone(self, tmp_path: Path, monkeypatch):
        """systemd stops the conflicting units asynchronously, so a run that
        starts GPU work immediately OOMs against models that are still resident."""
        import nova_voice.training.mode as mode_module

        mode = TrainingMode(tmp_path / "mode.json")
        calls = {"n": 0}

        def fake_is_active(unit: str) -> bool:
            calls["n"] += 1
            # Still shutting down for the first few polls, then gone.
            return calls["n"] < 4

        monkeypatch.setattr(mode_module, "_is_active", fake_is_active)
        monkeypatch.setattr(mode_module, "gpu_free_mib", lambda: 9000)
        monkeypatch.setattr(mode_module.time, "sleep", lambda _: None)
        monkeypatch.setattr(mode_module, "selected_engine_unit", lambda: None)

        assert mode.wait_for_gpu(timeout=30, poll=0) is True
        assert calls["n"] >= 4, "must keep polling while a unit is still active"

    def test_wait_for_gpu_gives_up_rather_than_hanging(self, tmp_path: Path, monkeypatch):
        import nova_voice.training.mode as mode_module

        mode = TrainingMode(tmp_path / "mode.json")
        monkeypatch.setattr(mode_module, "_is_active", lambda unit: True)
        monkeypatch.setattr(mode_module, "selected_engine_unit", lambda: None)
        monkeypatch.setattr(mode_module.time, "sleep", lambda _: None)

        assert mode.wait_for_gpu(timeout=0.01, poll=0) is False

    def test_leave_is_a_noop_when_not_active(self, tmp_path: Path):
        mode = TrainingMode(tmp_path / "mode.json")
        mode.leave()  # must not raise

    def test_leave_clears_state(self, tmp_path: Path, monkeypatch):
        state = tmp_path / "mode.json"
        mode = TrainingMode(state)
        monkeypatch.setattr(mode, "_stop_stack", lambda: None)
        monkeypatch.setattr(
            ModeSnapshot, "capture",
            classmethod(lambda cls: ModeSnapshot(True, True, "u", True, "t0")),
        )
        mode.enter()
        assert mode.active is True

        # leave() must NOT shell out. Restoring the voice stack is declared on
        # the training unit (OnSuccess=/OnFailure=nova-voice.service) so that a
        # crash or a kill still restores it; the training unit also runs with
        # NoNewPrivileges=yes and so could not run a privileged command anyway.
        calls: list[list[str]] = []
        monkeypatch.setattr("nova_voice.training.mode._run",
                            lambda cmd, check=True: calls.append(cmd) or _ok())
        mode.leave()
        assert mode.active is False, "the dashboard's training-mode flag must clear"
        assert calls == [], "restore is systemd's job, not a subprocess call"


def _ok():
    class R:
        returncode = 0
        stdout = ""
        stderr = ""
    return R()


def test_feature_shards_are_merged_with_the_semantic_header(tmp_path: Path):
    """The prepare scripts shard output per worker; the trainers read the
    unsuffixed file. The semantic TSV additionally needs its header row."""
    from nova_voice.training.runner import _merge_parts

    (tmp_path / "2-name2text-0.txt").write_text("a|x\nb|y\n", encoding="utf-8")
    merged = _merge_parts(tmp_path, "2-name2text", ".txt", header=None)
    assert merged.read_text(encoding="utf-8") == "a|x\nb|y\n"
    assert not (tmp_path / "2-name2text-0.txt").exists(), "shards must be consumed"

    (tmp_path / "6-name2semantic-0.tsv").write_text("one\t1\n", encoding="utf-8")
    (tmp_path / "6-name2semantic-1.tsv").write_text("two\t2\n", encoding="utf-8")
    merged = _merge_parts(tmp_path, "6-name2semantic", ".tsv", header="item_name\tsemantic_audio")
    assert merged.read_text(encoding="utf-8").splitlines() == [
        "item_name\tsemantic_audio", "one\t1", "two\t2",
    ]


def test_reference_scoring_rejects_whisper_repetition():
    """Faster-Whisper loops on some clips and repeats a phrase; a plain word
    count prefers exactly those, and a prompt whose text does not match its
    audio degrades every generation."""
    from nova_voice.training.runner import _reference_score

    varied = "the sun came up over the badlands and nobody said a word"
    looped = " ".join(["guess it's your lucky day"] * 7)
    assert _reference_score(varied) > _reference_score(looped)
    assert _reference_score("") == 0.0


def test_speaker_verification_step_is_v2pro_only(tmp_path: Path):
    """2-get-sv.py extracts embeddings only the Pro model lines consume; the
    webui gates it behind `if "Pro" in version` and it hard-fails on plain v2
    because the checkpoint it loads is not installed."""
    from nova_voice.training.config_builder import TrainingPlan
    from nova_voice.training.runner import TrainingRunner

    def steps_for(version: str) -> list[str]:
        exp = tmp_path / f"exp-{version}"
        exp.mkdir(parents=True, exist_ok=True)
        # The merge step runs after extraction and asserts the semantic file has
        # content, so give it a plausible shard to consume.
        (exp / "6-name2semantic-0.tsv").write_text("clip\t1 2 3\n", encoding="utf-8")
        (exp / "2-name2text-0.txt").write_text("clip|x\n", encoding="utf-8")
        plan = TrainingPlan(exp_name="x", exp_dir=exp,
                            gptsovits_root=Path("/tmp/gs"), version=version)
        captured: list[str] = []

        runner = TrainingRunner.__new__(TrainingRunner)
        runner.root = plan.gptsovits_root
        runner.python = "python"
        runner._stopped = False
        runner._log = tmp_path / "x.log"
        runner.set = None

        # Capture the script list without executing anything.
        def fake_run(cmd, env, *, stoppable):
            captured.append(Path(cmd[1]).name)
            return True

        runner._run = fake_run  # type: ignore[assignment]
        runner._stage = lambda *a, **k: None  # type: ignore[assignment]
        return captured, runner, plan

    captured, runner, plan = steps_for("v2")
    import nova_voice.training.runner as runner_module

    fake_pretrained = {k: Path("/tmp") / k for k in ("bert", "cnhubert", "s2G", "s2D", "s1")}
    original = runner_module.resolve_pretrained
    runner_module.resolve_pretrained = lambda *a, **k: fake_pretrained
    try:
        class FakeSet:
            asr_dir = Path("/tmp")
            sliced_dir = Path("/tmp")

            def glob(self, pattern):
                return []

        runner.set = type("S", (), {
            "asr_dir": type("D", (), {"glob": lambda self, p: [Path("/tmp/t.list")]})(),
            "sliced_dir": Path("/tmp"),
        })()
        runner.extract_features({}, plan)
        assert "2-get-sv.py" not in captured, "v2 must not run the Pro-only step"

        captured_pro, runner_pro, plan_pro = steps_for("v2Pro")
        runner_pro.set = runner.set
        runner_pro.extract_features({}, plan_pro)
        assert "2-get-sv.py" in captured_pro, "v2Pro must run it"
    finally:
        runner_module.resolve_pretrained = original


class TestConfigBuilder:
    """Configs are generated from GPT-SoVITS's own templates, so the trainers
    get the schema their version expects."""

    def _templates(self, tmp_path: Path) -> Path:
        root = tmp_path / "GPT-SoVITS"
        configs = root / "GPT_SoVITS" / "configs"
        configs.mkdir(parents=True)
        (configs / "s1longer-v2.yaml").write_text(
            "train:\n  batch_size: 1\n  epochs: 1\n  precision: '16-mixed'\n", encoding="utf-8")
        (configs / "s2.json").write_text(json.dumps({
            "train": {"batch_size": 1, "epochs": 1, "fp16_run": True},
            "data": {}, "model": {}, "s2_ckpt_dir": "",
        }), encoding="utf-8")

        pre = root / "GPT_SoVITS" / "pretrained_models"
        (pre / "gsv-v2final-pretrained").mkdir(parents=True)
        (pre / "s1bert25hz-x.ckpt").write_bytes(b"x")
        (pre / "gsv-v2final-pretrained" / "s2G.pth").write_bytes(b"x")
        (pre / "gsv-v2final-pretrained" / "s2D.pth").write_bytes(b"x")
        (pre / "chinese-roberta-wwm-ext-large").mkdir()
        (pre / "chinese-hubert-base").mkdir()
        return root

    def test_s1_and_s2_configs_carry_the_plan(self, tmp_path: Path):
        root = self._templates(tmp_path)
        plan = TrainingPlan(exp_name="johnny", exp_dir=tmp_path / "exp",
                            gptsovits_root=root, batch_size=6, total_epoch=9,
                            save_every_epoch=2)

        import yaml
        s1 = yaml.safe_load(build_s1_config(plan, tmp_path / "s1.yaml").read_text(encoding="utf-8"))
        assert s1["train"]["batch_size"] == 6
        assert s1["train"]["epochs"] == 9
        assert s1["train"]["save_every_n_epoch"] == 2
        assert s1["output_dir"].endswith("logs_s1_v2")

        s2 = json.loads(build_s2_config(plan, tmp_path / "s2.json").read_text(encoding="utf-8"))
        assert s2["train"]["batch_size"] == 6
        assert s2["train"]["save_every_epoch"] == 2
        assert s2["data"]["exp_dir"] == str(plan.exp_dir)

    def test_training_env_enables_resume_from_our_own_checkpoints(self, tmp_path: Path):
        """torch>=2.6 defaults weights_only=True, which rejects the PosixPath in
        a Lightning checkpoint -- so without this, resume is impossible."""
        from nova_voice.training.config_builder import training_env

        root = self._templates(tmp_path)
        plan = TrainingPlan(exp_name="j", exp_dir=tmp_path / "exp", gptsovits_root=root)
        env = training_env(plan, {})
        assert env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"
        # Both roots must be importable; only one gets you through the slicer.
        assert str(root) in env["PYTHONPATH"]
        assert str(root / "GPT_SoVITS") in env["PYTHONPATH"]

    def test_non_half_halves_batch_size(self, tmp_path: Path):
        root = self._templates(tmp_path)
        plan = TrainingPlan(exp_name="j", exp_dir=tmp_path / "exp", gptsovits_root=root,
                            batch_size=8, is_half=False)
        import yaml
        s1 = yaml.safe_load(build_s1_config(plan, tmp_path / "s1.yaml").read_text(encoding="utf-8"))
        assert s1["train"]["batch_size"] == 4
        assert s1["train"]["precision"] == "32"

    def test_missing_pretrained_names_what_is_missing(self, tmp_path: Path):
        root = self._templates(tmp_path)
        # Templates present, pretrained weights gone: the error must name the
        # weights, not fail somewhere deeper with a bare path.
        import shutil as _shutil
        _shutil.rmtree(root / "GPT_SoVITS" / "pretrained_models")
        (root / "GPT_SoVITS" / "pretrained_models").mkdir()
        plan = TrainingPlan(exp_name="j", exp_dir=tmp_path / "exp", gptsovits_root=root)
        with pytest.raises(FileNotFoundError, match="pretrained model"):
            build_s1_config(plan, tmp_path / "s1.yaml")

    def test_v2_uses_the_v2_pretrained_checkpoints_not_v1(self, tmp_path: Path):
        """Several s1 checkpoints ship side by side. Warm-starting v2 from v1's
        fails deep inside training with a state_dict mismatch, so the version
        mapping has to be exact rather than a first-glob-wins guess."""
        from nova_voice.training.config_builder import resolve_pretrained

        root = self._templates(tmp_path)
        pre = root / "GPT_SoVITS" / "pretrained_models"
        # The v1 checkpoint sits at the top level and would win a naive glob.
        (pre / "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt").write_bytes(b"v1")
        v2 = pre / "gsv-v2final-pretrained"
        (v2 / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt").write_bytes(b"v2")
        (v2 / "s2G2333k.pth").write_bytes(b"v2")
        (v2 / "s2D2333k.pth").write_bytes(b"v2")

        resolved = resolve_pretrained(root, "v2")
        assert resolved["s1"].read_bytes() == b"v2", "v2 must not warm-start from the v1 checkpoint"
        assert resolved["s2G"].name == "s2G2333k.pth"

    def test_missing_template_names_the_broken_install(self, tmp_path: Path):
        root = self._templates(tmp_path)
        (root / "GPT_SoVITS" / "configs" / "s1longer-v2.yaml").unlink()
        plan = TrainingPlan(exp_name="j", exp_dir=tmp_path / "exp", gptsovits_root=root)
        with pytest.raises(FileNotFoundError, match="config template not found"):
            build_s1_config(plan, tmp_path / "s1.yaml")
