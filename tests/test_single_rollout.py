"""Task 5.3 control-flow tests: fake Policy/Env, no simulator required."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from arbim.rollout import single


@pytest.fixture
def spec():
    return {
        "state_key": "observation.state", "state_dim": 16, "action_dim": 16,
        "camera_keys": (
            "observation.images.head", "observation.images.left_wrist",
            "observation.images.right_wrist",
        ),
    }


class FakePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.zeros(()))
        self.n_obs_steps = 2
        self.n_action_steps = 16
        self.horizon = 17
        self.no_pre_action = True
        self.action_dim = 16
        self.inferences = []

    def predict_action(self, observation, *, deterministic=False):
        self.inferences.append((observation["agent_pos"].detach().cpu().numpy().copy(), deterministic))
        index = len(self.inferences) - 1
        actions = (torch.arange(1, 17, dtype=torch.float32).reshape(1, 16, 1)
                   .expand(1, 16, 16).clone() + 100 * index)
        return {"action": actions, "action_pred": actions.clone()}


class FakeAdapter:
    def __init__(self, spec, end_at=None, outcome="success"):
        self.spec, self.end_at, self.outcome = spec, end_at, outcome
        self.reset_count = 0
        self.actions = []
        self.seeds = []
        self.tick = 0

    def _observation(self):
        value = self.base + self.tick
        state = np.zeros(16, dtype=np.float32)
        state[0] = value
        return {
            "observation.state": state,
            **{key: np.full((2, 3, 3), value % 255, dtype=np.uint8)
               for key in self.spec["camera_keys"]},
        }

    def reset(self, *, seed=None):
        self.reset_count += 1
        self.seeds.append(seed)
        self.tick = 0
        self.base = 1000 * self.reset_count
        return self._observation(), {"success": False, "stage": 0}

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        self.tick += 1
        done = self.tick == self.end_at
        info = {"success": done and self.outcome == "success", "stage": self.tick}
        return (self._observation(), 0.25,
                done and self.outcome == "terminated",
                done and self.outcome == "truncated", info)


@pytest.fixture(autouse=True)
def replace_preparation(monkeypatch, spec):
    def fake_prepare(window, saved_spec, n_obs_steps, *, device):
        assert saved_spec is spec
        assert window["observation.state"].shape == (1, n_obs_steps, 16)
        obs = {"agent_pos": torch.as_tensor(window["observation.state"], device=device)}
        for i, key in enumerate(spec["camera_keys"]):
            assert window[key].shape == (1, n_obs_steps, 2, 3, 3)
            obs[f"rgb_{i}"] = torch.as_tensor(window[key], device=device).permute(0, 1, 4, 2, 3).float() / 255
        return obs
    monkeypatch.setattr(single, "prepare_observation", fake_prepare)


def test_history_initial_padding_and_updates_every_step(spec):
    policy, adapter = FakePolicy(), FakeAdapter(spec)
    result = single.run_single_episode(adapter, policy, spec, execute_steps=2, max_env_steps=3, seed=7)
    assert result["episode_steps"] == 3
    assert result["inference_count"] == 2
    assert adapter.seeds == [7]
    np.testing.assert_array_equal(policy.inferences[0][0][0, :, 0], [1000, 1000])
    np.testing.assert_array_equal(policy.inferences[1][0][0, :, 0], [1001, 1002])


def test_chunk_order_and_unnormalized_values(spec):
    adapter = FakeAdapter(spec)
    result = single.run_single_episode(adapter, FakePolicy(), spec, execute_steps=3, max_env_steps=5)
    assert [action[0] for action in adapter.actions] == [1, 2, 3, 101, 102]
    assert result["stop_reason"] == "max_env_steps"
    assert result["episode_return"] == pytest.approx(1.25)
    assert result["max_steps_reached"] is True
    assert result["inference_count"] == 2
    assert result["mean_inference_ms"] >= 0


@pytest.mark.parametrize("outcome,success,terminated,truncated,reason", [
    ("success", True, False, False, "success"),
    ("terminated", False, True, False, "terminated"),
    ("truncated", False, False, True, "truncated"),
])
def test_stop_mid_chunk(spec, outcome, success, terminated, truncated, reason):
    adapter = FakeAdapter(spec, end_at=3, outcome=outcome)
    result = single.run_single_episode(adapter, FakePolicy(), spec, max_env_steps=20)
    assert len(adapter.actions) == 3
    assert result["success"] is success
    assert result["terminated"] is terminated
    assert result["truncated"] is truncated
    assert result["stop_reason"] == reason
    assert result["final_stage"] == 3
    assert result["inference_count"] == 1


def test_reset_does_not_retain_old_history(spec):
    adapter, policy = FakeAdapter(spec), FakePolicy()
    single.run_single_episode(adapter, policy, spec, execute_steps=1, max_env_steps=2)
    single.run_single_episode(adapter, policy, spec, execute_steps=1, max_env_steps=1)
    np.testing.assert_array_equal(policy.inferences[-1][0][0, :, 0], [2000, 2000])
    assert adapter.reset_count == 2


@pytest.mark.parametrize("settings", [
    {"execute_steps": 17}, {"execute_steps": 0}, {"execute_steps": True},
    {"max_env_steps": 0}, {"max_env_steps": True},
    {"deterministic": 1}, {"seed": -1},
])
def test_invalid_parameters_rejected_before_reset(spec, settings):
    adapter = FakeAdapter(spec)
    with pytest.raises(ValueError):
        single.run_single_episode(adapter, FakePolicy(), spec, **settings)
    assert adapter.reset_count == 0
