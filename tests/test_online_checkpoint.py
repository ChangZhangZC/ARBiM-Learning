"""Online checkpoints: compact V2, strict resume, legacy V1 and policy snapshots."""
import copy
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from arbim.rl.online.checkpoint import (
    read_online_checkpoint, restore_online_checkpoint, save_inference_snapshot,
    save_online_checkpoint, prune_policy_snapshots, online_resume_contract,
)


class FakePPO:
    def __init__(self):
        self._policy = torch.nn.Linear(2, 2)
        self.critic = torch.nn.Linear(2, 1)
        self.optimizer_actor = torch.optim.Adam(self._policy.parameters(), lr=1e-4)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.kl_scheduler_actor = SimpleNamespace(current_lr=1e-4)
        self.kl_scheduler_critic = SimpleNamespace(current_lr=1e-3)
        self.iteration = 2
        self._policy.n_action_steps = 1
        self._policy.num_inference_steps = 10
        self.old_synced = False

    def set_old_policy(self):
        self.old_synced = True


def profile():
    return {
        "env": {"task": "hinge_chest", "frequency": 30},
        "policy": {"checkpoint_path": "/tmp/example/best.ckpt",
                   "device": "cpu", "weights": "raw"},
        "ppo": {"gamma": 0.99, "epsilon": 0.2},
        "reward": {"mode": "env_raw"},
        "rollout": {"max_env_steps": 32, "seed": 42,
                    "max_updates": 10, "max_total_env_steps": None},
        "evaluation": {"every_updates": 1, "num_episodes": 2},
        "checkpoint": {"every_updates": 1},
        "output": {"root": "/tmp/online", "run_name": "one"},
        "logging": {"tensorboard": {"enabled": False}},
    }


def test_online_training_state_round_trip(tmp_path, monkeypatch):
    native_util = ModuleType("rl_100.common.pytorch_util")
    calls = []

    def fake_optimizer_to(optimizer, device):
        calls.append((optimizer, device))
        return optimizer

    native_util.optimizer_to = fake_optimizer_to
    monkeypatch.setitem(sys.modules, "rl_100", ModuleType("rl_100"))
    monkeypatch.setitem(sys.modules, "rl_100.common", ModuleType("rl_100.common"))
    monkeypatch.setitem(sys.modules, "rl_100.common.pytorch_util", native_util)
    config = profile()
    original = "/tmp/example/best.ckpt"
    first = FakePPO()
    path = tmp_path / "checkpoints" / "last.ckpt"
    save_online_checkpoint(path, first, config=config, updates=2,
                           episode_count=4, env_steps=50, original_checkpoint=original)
    payload = read_online_checkpoint(path, config=config, original_checkpoint=original)
    assert payload["format_version"] == 2
    assert "config" not in payload
    assert len(payload["contract_digest"]) == 64
    other = FakePPO()
    counts = restore_online_checkpoint(other, payload)
    assert counts == (2, 4, 50)
    assert other.old_synced
    assert [optimizer for optimizer, _ in calls] == [
        other.optimizer_actor, other.optimizer_critic
    ]
    assert all(device == torch.device("cpu") for _, device in calls)
    assert all(torch.equal(a, b) for a, b in zip(
        first._policy.parameters(), other._policy.parameters()
    ))


def test_snapshot_matches_existing_inference_role(tmp_path):
    ppo = FakePPO()
    path = tmp_path / "step.ckpt"
    save_inference_snapshot(path, ppo, il_contract={"policy": {"horizon": 5}}, env_steps=10)
    data = torch.load(path, map_location="cpu", weights_only=True)
    assert data["format_version"] == 2
    assert data["checkpoint_role"] == "policy_snapshot"
    assert data["selected_weights"] == "raw"
    assert data["contract"]["policy"]["n_action_steps"] == 1
    assert data["contract"]["policy"]["num_inference_steps"] == 10


def test_contract_allows_budget_logs_eval_and_device_changes(tmp_path):
    config = profile()
    path = tmp_path / "last.ckpt"
    save_online_checkpoint(path, FakePPO(), config=config,
                           updates=1, episode_count=1, env_steps=32,
                           original_checkpoint="/tmp/example/best.ckpt")
    changed = copy.deepcopy(config)
    changed["rollout"]["max_updates"] = 20
    changed["rollout"]["max_total_env_steps"] = 10000
    changed["evaluation"]["every_updates"] = 5
    changed["logging"]["tensorboard"]["enabled"] = True
    changed["policy"]["device"] = "cuda:1"
    assert read_online_checkpoint(
        path, config=changed, original_checkpoint="/tmp/example/best.ckpt"
    )["updates"] == 1
    assert online_resume_contract(config) == online_resume_contract(changed)


@pytest.mark.parametrize("section,key,value", [
    ("ppo", "gamma", 0.98),
    ("env", "frequency", 60),
    ("rollout", "seed", 43),
    ("policy", "weights", "ema"),
])
def test_checkpoint_rejects_changed_algorithm_contract(tmp_path, section, key, value):
    config = profile()
    path = tmp_path / "last.ckpt"
    save_online_checkpoint(path, FakePPO(), config=config,
                           updates=1, episode_count=1, env_steps=32,
                           original_checkpoint="/tmp/example/best.ckpt")
    changed = copy.deepcopy(config)
    changed[section][key] = value
    with pytest.raises(ValueError, match="contract"):
        read_online_checkpoint(
            path, config=changed, original_checkpoint="/tmp/example/best.ckpt"
        )


def test_legacy_v1_checkpoint_preserves_strict_full_configuration(tmp_path):
    config = profile()
    path = tmp_path / "last.ckpt"
    save_online_checkpoint(path, FakePPO(), config=config,
                           updates=1, episode_count=1, env_steps=32,
                           original_checkpoint="/tmp/example/best.ckpt")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["format_version"] = 1
    payload["config"] = config
    payload.pop("contract_digest")
    torch.save(payload, path)
    assert read_online_checkpoint(
        path, config=config, original_checkpoint="/tmp/example/best.ckpt"
    )["updates"] == 1
    changed = copy.deepcopy(config)
    changed["rollout"]["max_updates"] += 1
    with pytest.raises(ValueError, match="V1 Resume"):
        read_online_checkpoint(
            path, config=changed, original_checkpoint="/tmp/example/best.ckpt"
        )


def test_snapshot_pruning_only_affects_step_files(tmp_path):
    folder = tmp_path / "checkpoints"
    folder.mkdir()
    (folder / "last.ckpt").write_bytes(b"keep")
    for step in (1, 2, 3, 4, 5):
        (folder / f"step_{step:08d}.ckpt").write_bytes(b"snapshot")
    prune_policy_snapshots(folder, 2)
    assert [x.name for x in sorted(folder.glob("step_*.ckpt"))] == [
        "step_00000004.ckpt", "step_00000005.ckpt"
    ]
    assert (folder / "last.ckpt").read_bytes() == b"keep"
