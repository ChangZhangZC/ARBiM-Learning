"""Reward interface preserves separate environment diagnostics."""
import pytest

from arbim.data.online_adapter import discounted_chunk_reward
from arbim.rl.rewards.duobench_hinge_chest import HingeChestReward, build_reward_evaluator


def test_reward_is_not_shaped_by_arbitrary_stage():
    evaluator = build_reward_evaluator("env_raw")
    result = evaluator.evaluate(0.25, {"success": False, "stage": 2},
                                terminated=False, truncated=False)
    assert result.train_reward == result.env_reward == 0.25
    assert result.metrics["stage"] == 2
    assert not result.metrics["success"]


def test_discounted_chunk_reward_uses_single_environment_gamma():
    assert discounted_chunk_reward([1, 2, 3], 0.5) == 2.75
    with pytest.raises(ValueError):
        discounted_chunk_reward([], 0.99)


def test_unknown_reward_is_not_implicitly_implemented():
    with pytest.raises(ValueError, match="env_raw"):
        build_reward_evaluator("stage_reward")
