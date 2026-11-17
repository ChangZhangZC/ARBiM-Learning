"""Collect actual one-step environment rewards and full denoising chains."""
import numpy as np
import pytest
import torch

from arbim.rl.online import rollout
from arbim.rl.online.policy_adapter import OnlineSample
from arbim.rl.rewards.duobench_hinge_chest import HingeChestReward


class FakePolicy(torch.nn.Module):
    def __init__(self, h=4):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(()))
        self.n_action_steps = h
        self.n_obs_steps = 2
        self.horizon = 5
        self.action_dim = 16


class FakeBuffer:
    def __init__(self):
        self.records = []

    def store(self, *args):
        self.records.append(args)


class FakeAdapter:
    def __init__(self, spec, *, terminal_step, truncated=False):
        self.spec = spec
        self.step_count = 0
        self.terminal_step = terminal_step
        self.truncated = truncated

    def obs(self):
        return {
            "observation.state": np.zeros(16, dtype=np.float32),
            **{k: np.zeros((2, 3, 3), dtype=np.uint8)
               for k in self.spec["camera_keys"]},
        }

    def reset(self, *, seed=None):
        self.step_count = 0
        return self.obs(), {"stage": 0, "success": False}

    def step(self, action):
        assert np.asarray(action).shape == (16,)
        self.step_count += 1
        end = self.step_count == self.terminal_step
        terminated, truncated = (end and not self.truncated), (end and self.truncated)
        return self.obs(), 1.0, terminated, truncated, {
            "stage": 1 if end else 0,
            "success": terminated,
        }


@pytest.fixture
def patch_policy(monkeypatch):
    spec = {
        "state_key": "observation.state",
        "camera_keys": ("head", "left", "right"),
    }

    def fake_prepare(batch, spec, n_obs_steps, *, device):
        return {"agent_pos": torch.zeros((1, n_obs_steps, 16))}

    def fake_sample(policy, observation):
        h = policy.n_action_steps
        return OnlineSample(
            action=torch.zeros((1, h, 16)),
            all_x=torch.zeros((3, 1, 4, 16)),
            a_logprob=torch.zeros((2, 1, 4, 16)),
        )

    monkeypatch.setattr(rollout, "prepare_observation", fake_prepare)
    monkeypatch.setattr(rollout, "sample_online", fake_sample)
    return spec


def test_full_chunk_reward_and_success_end(patch_policy):
    policy = FakePolicy()
    buffer = FakeBuffer()
    result = rollout.collect_online_episode(
        FakeAdapter(patch_policy, terminal_step=4), policy, patch_policy, buffer,
        HingeChestReward(), gamma=0.5, max_env_steps=8,
    )
    assert result["episode_steps"] == 4
    assert result["success"] and len(buffer.records) == 1
    record = buffer.records[0]
    assert record[1].shape == (3, 4, 16)
    assert record[3] == 1.875
    assert record[5:] == (True, True)


def test_true_terminal_partial_chunk_has_no_bootstrap(patch_policy):
    policy = FakePolicy()
    buffer = FakeBuffer()
    result = rollout.collect_online_episode(
        FakeAdapter(patch_policy, terminal_step=2), policy, patch_policy, buffer,
        HingeChestReward(), gamma=0.5, max_env_steps=8,
    )
    assert result["episode_steps"] == 2
    assert buffer.records[0][3] == 1.5
    assert buffer.records[0][-1] is True


def test_partial_truncation_rejected_until_variable_span_supported(patch_policy):
    with pytest.raises(RuntimeError, match="Partial truncated"):
        rollout.collect_online_episode(
            FakeAdapter(patch_policy, terminal_step=2, truncated=True),
            FakePolicy(), patch_policy, FakeBuffer(), HingeChestReward(),
            gamma=0.99, max_env_steps=8,
        )
