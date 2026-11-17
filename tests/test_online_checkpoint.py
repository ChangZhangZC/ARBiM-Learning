"""Training checkpoints and inference snapshots use distinct formats."""
import sys
from types import ModuleType, SimpleNamespace

import torch

from arbim.rl.online.checkpoint import (
    read_online_checkpoint, restore_online_checkpoint,
    save_inference_snapshot, save_online_checkpoint,
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


def test_online_training_state_round_trip(tmp_path, monkeypatch):
    # This isolated FakePPO test validates the native helper call contract
    # without requiring the RL-100 submodule to be importable. Real Online PPO
    # still requires the pinned rl_100 package at runtime.
    native_util = ModuleType("rl_100.common.pytorch_util")
    calls = []

    def fake_optimizer_to(optimizer, device):
        calls.append((optimizer, device))
        return optimizer

    native_util.optimizer_to = fake_optimizer_to
    monkeypatch.setitem(sys.modules, "rl_100", ModuleType("rl_100"))
    monkeypatch.setitem(sys.modules, "rl_100.common", ModuleType("rl_100.common"))
    monkeypatch.setitem(sys.modules, "rl_100.common.pytorch_util", native_util)
    config = {"test": "fixed"}
    original = "/tmp/example/best.ckpt"
    first = FakePPO()
    path = tmp_path / "checkpoints" / "last.ckpt"
    save_online_checkpoint(path, first, config=config, updates=2,
                           episode_count=4, env_steps=50, original_checkpoint=original)
    payload = read_online_checkpoint(path, config=config, original_checkpoint=original)
    other = FakePPO()
    counts = restore_online_checkpoint(other, payload)
    assert counts == (2, 4, 50)
    assert other.old_synced
    assert [optimizer for optimizer, _ in calls] == [other.optimizer_actor, other.optimizer_critic]
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


def test_checkpoint_rejects_wrong_resume_contract(tmp_path):
    path = tmp_path / "last.ckpt"
    save_online_checkpoint(path, FakePPO(), config={"a": 1},
                           updates=1, episode_count=1, env_steps=4,
                           original_checkpoint="/tmp/a.ckpt")
    try:
        read_online_checkpoint(path, config={"a": 2}, original_checkpoint="/tmp/a.ckpt")
    except ValueError as exc:
        assert "contract" in str(exc)
    else:
        raise AssertionError("Resume accepted a modified config")
