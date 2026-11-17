"""CPU-only Task 7 orchestration, telemetry and resume-budget regressions."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from arbim.rl.online.telemetry import (
    OnlineTelemetry, append_native_ratio_records,
    reconcile_online_metrics, reconcile_native_ratio_csv, native_ratio_summary,
)
from arbim.training.train_online_rl100_dp2d import (
    _budget_reached, _new_run_dir, _frozen_config, _file_sha256,
    normalize_online_config, validate_config,
)


def profile():
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(
        (root / "configs/online/rl100_dp2d_hinge_chest.yaml").read_text()
    )
    cfg["policy"]["checkpoint_path"] = "/tmp/verified/last.ckpt"
    return cfg


def test_ppo_updates_are_the_only_training_budget():
    cfg = profile()
    cfg["ppo"]["max_updates"] = 10000
    validate_config(cfg)
    assert not _budget_reached(cfg, updates=9999)
    assert _budget_reached(cfg, updates=10000)
    # Total environmental interactions and Eval steps do not affect stopping.
    cfg["env"]["max_episode_steps"] = 992
    cfg["evaluation"]["every_updates"] = 50
    assert not _budget_reached(cfg, updates=9999)


def test_ppo_budget_requires_positive_updates():
    cfg = profile()
    cfg["ppo"]["max_updates"] = None
    with pytest.raises(ValueError, match="ppo.max_updates"):
        validate_config(cfg)


def test_episode_limit_is_unified_and_old_budgets_are_rejected():
    cfg = profile()
    cfg["env"]["max_episode_steps"] = 992
    validate_config(cfg)
    cfg["env"]["max_episode_steps"] = 0
    with pytest.raises(ValueError, match="env.max_episode_steps"):
        validate_config(cfg)


def test_old_run_yaml_can_be_migrated_without_changing_episode_limit():
    cfg = profile()
    old = copy.deepcopy(cfg)
    cap = old["env"].pop("max_episode_steps")
    old["rollout"].update(max_env_steps=cap, max_updates=old["ppo"].pop("max_updates"),
                          max_total_env_steps=None)
    old["evaluation"]["max_env_steps"] = cap
    old["ppo"]["max_train_steps"] = 100000
    normalized, original = normalize_online_config(old)
    assert normalized == cfg
    assert original == old
    validate_config(normalized)
    old["evaluation"]["max_env_steps"] += 32
    with pytest.raises(ValueError, match="episode limits differ"):
        normalize_online_config(old)
    old["evaluation"]["max_env_steps"] = cap
    old["rollout"]["max_total_env_steps"] = 10000
    with pytest.raises(ValueError, match="cannot be mapped"):
        normalize_online_config(old)


def test_run_name_template_and_non_overwrite(tmp_path):
    cfg = profile()
    cfg["output"].update(root=str(tmp_path), run_name="run_{timestep}")
    run_dir = _new_run_dir(cfg)
    assert run_dir.parent == tmp_path / "hinge_chest"
    assert run_dir.name.startswith("run_")
    assert len(run_dir.name) == len("run_YYYYMMDD_HHMMSS")
    cfg["output"]["run_name"] = run_dir.name
    with pytest.raises(FileExistsError):
        _new_run_dir(cfg)


def test_frozen_config_preserves_user_input():
    cfg = profile()
    cfg["checkpoint"]["resume"] = "/tmp/last.ckpt"
    before = copy.deepcopy(cfg)
    frozen = _frozen_config(cfg)
    assert cfg == before
    assert frozen["checkpoint"]["resume"] is None


def test_online_metrics_reconcile_uncommitted_records(tmp_path):
    path = tmp_path / "metrics.jsonl"
    data = [
        {"event": "ppo", "updates": 1, "env_steps": 32},
        {"event": "evaluation", "updates": 1, "env_steps": 32},
        {"event": "rollout", "updates": 1, "env_steps": 64},
        {"event": "ppo", "updates": 2, "env_steps": 64},
    ]
    path.write_text("".join(json.dumps(x) + "\n" for x in data))
    reconcile_online_metrics(path, updates=1, env_steps=32)
    kept = [json.loads(x) for x in path.read_text().splitlines()]
    archived = [json.loads(x) for x in
                (tmp_path / "metrics.uncommitted.jsonl").read_text().splitlines()]
    assert kept == data[:2]
    assert archived == data[2:]


def test_local_online_telemetry_without_tensorboard(tmp_path):
    cfg = {"jsonl": True, "tensorboard": {"enabled": False}}
    sink = OnlineTelemetry(
        cfg, tmp_path, resumed=False, resume_updates=0, resume_env_steps=0
    )
    sink.event(
        "ppo",
        {"event": "ppo", "updates": 1, "env_steps": 32, "actor_loss": 0.5},
        summary="[PPO] update=1 actor=0.5",
        groups={"ppo": {"actor_loss": 0.5}},
    )
    sink.close()
    assert json.loads((tmp_path / "metrics.jsonl").read_text().strip()) == {
        "event": "ppo", "updates": 1, "env_steps": 32, "actor_loss": 0.5
    }
    assert "[PPO]" in (tmp_path / "train.log").read_text()


def test_append_upstream_ratio_records_across_updates(tmp_path):
    ppo = SimpleNamespace(_ratio_records=[
        {"idx": 0, "phase": "actor", "denoise_step": 0,
         "ratio_mean": 1.1, "ratio_q05": 0.9, "ratio_q50": 1.0,
         "ratio_q95": 1.2, "delta_mean": 0.01},
    ], _delta_cache=[1])
    assert native_ratio_summary(ppo)["ratio_mean"] == pytest.approx(1.1)
    append_native_ratio_records(ppo, tmp_path, updates=1, env_steps=32)
    assert not ppo._ratio_records
    ppo._ratio_records = [
        {"idx": 1, "phase": "actor", "denoise_step": 1,
         "ratio_mean": 1.2, "ratio_q05": 1.0, "ratio_q50": 1.1,
         "ratio_q95": 1.3, "delta_mean": 0.02},
    ]
    append_native_ratio_records(ppo, tmp_path, updates=2, env_steps=64)
    rows = (tmp_path / "diagnostics" / "ratio_stats.csv").read_text().splitlines()
    assert len(rows) == 3
    assert rows[1].startswith("1,32,")
    assert rows[2].startswith("2,64,")


def test_ratio_resume_archives_uncommitted_updates(tmp_path):
    from csv import DictReader
    folder = tmp_path / "diagnostics"
    folder.mkdir()
    path = folder / "ratio_stats.csv"
    path.write_text(
        "update,env_steps,ratio_mean\n"
        "1,32,1.1\n"
        "2,64,1.2\n", encoding="utf-8"
    )
    reconcile_native_ratio_csv(path, updates=1, env_steps=32)
    with path.open() as handle:
        kept = list(DictReader(handle))
    with (folder / "ratio.uncommitted.csv").open() as handle:
        discarded = list(DictReader(handle))
    assert kept == [{"update": "1", "env_steps": "32", "ratio_mean": "1.1"}]
    assert discarded == [{"update": "2", "env_steps": "64", "ratio_mean": "1.2"}]


def test_initial_il_checksum_is_content_sensitive(tmp_path):
    import hashlib
    path = tmp_path / "last.ckpt"
    path.write_bytes(b"original IL")
    assert _file_sha256(path) == hashlib.sha256(b"original IL").hexdigest()
    path.write_bytes(b"changed IL")
    assert _file_sha256(path) != hashlib.sha256(b"original IL").hexdigest()


def test_tensorboard_step_is_ppo_updates_not_environment_steps(tmp_path, monkeypatch):
    import arbim.rl.online.telemetry as telemetry

    seen = []

    class FakeLogger:
        def __init__(self, cfg, run_dir, *, resumed, resume_step):
            seen.append(("init", resume_step, cfg["logging"]["tensorboard"]["log_dir"]))

        def log(self, group, step, **metrics):
            seen.append((group, step, metrics))

        def close(self):
            pass

    monkeypatch.setattr(telemetry, "MetricLogger", FakeLogger)
    logger = telemetry.OnlineTelemetry(
        {"jsonl": False, "tensorboard": {"enabled": True}},
        tmp_path, resumed=False, resume_updates=0, resume_env_steps=0,
    )
    logger.event(
        "ppo", {"event": "ppo", "updates": 3, "episodes": 6, "env_steps": 5952},
        summary="[PPO] update=3",
        groups={
            "ppo": {"actor_loss": 0.2},
            "progress": {"env_steps": 5952, "ppo_updates": 3},
        },
    )
    logger.close()
    assert ("ppo", 3, {"actor_loss": 0.2}) in seen
    assert ("progress", 3, {"env_steps": 5952, "ppo_updates": 3}) in seen
    assert seen[0][1] == 0
    assert seen[0][2].endswith("/tensorboard")


def test_legacy_resume_preserves_old_tensorboard_event_axis(tmp_path, monkeypatch):
    import arbim.rl.online.telemetry as telemetry

    observed = {}

    class FakeLogger:
        def __init__(self, cfg, run_dir, *, resumed, resume_step):
            observed["tb"] = cfg["logging"]["tensorboard"]["log_dir"]
            observed["purge"] = resume_step

        def close(self):
            pass

    monkeypatch.setattr(telemetry, "MetricLogger", FakeLogger)
    (tmp_path / "tensorboard").mkdir()
    sink = telemetry.OnlineTelemetry(
        {"jsonl": True, "tensorboard": {"enabled": True}},
        tmp_path, resumed=True, resume_updates=2, resume_env_steps=3968,
        migrate_legacy_tb=True,
    )
    sink.close()
    assert observed["purge"] == 2
    assert observed["tb"].endswith("tensorboard_ppo_updates")
