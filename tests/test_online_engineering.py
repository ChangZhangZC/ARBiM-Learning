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
    _budget_reached, _new_run_dir, _frozen_config, validate_config,
)


def profile():
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(
        (root / "configs/online/rl100_dp2d_hinge_chest.yaml").read_text()
    )
    cfg["policy"]["checkpoint_path"] = "/tmp/verified/last.ckpt"
    return cfg


def test_total_budget_terminates_at_complete_update_boundary():
    cfg = profile()
    cfg["rollout"].update(max_updates=None, max_total_env_steps=10000)
    validate_config(cfg)
    assert not _budget_reached(cfg, updates=10, env_steps=9920)
    assert _budget_reached(cfg, updates=11, env_steps=10912)
    cfg["rollout"].update(max_updates=9)
    assert _budget_reached(cfg, updates=9, env_steps=8928)


def test_budget_requires_at_least_one_limit():
    cfg = profile()
    cfg["rollout"].update(max_updates=None, max_total_env_steps=None)
    with pytest.raises(ValueError, match="max_updates"):
        validate_config(cfg)


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
