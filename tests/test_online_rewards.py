"""Reward interface preserves separate environment diagnostics."""
import pytest

from arbim.data.online_adapter import discounted_chunk_reward
from arbim.rl.rewards.duobench_hinge_chest import HingeChestReward, build_reward_evaluator
from arbim.rl.rewards.stage_increment import StageIncrementReward


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


def _info(stage):
    return {"stage": stage, "max_stage": 3, "success": stage == 3}


def test_stage_increment_rewards_once_and_raw_diagnostics_unchanged():
    evaluator = build_reward_evaluator("stage_increment")
    evaluator.reset(_info(0))
    stages = [0, 1, 1, 2, 2, 3]
    results = [
        evaluator.evaluate(stage / 3, _info(stage),
                           terminated=stage == 3, truncated=False)
        for stage in stages
    ]
    assert [result.train_reward for result in results] == [0, 1, 0, 1, 0, 5]
    assert [result.env_reward for result in results] == [0, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 1]
    assert sum(result.train_reward for result in results) == 7


def test_stage_skips_and_episode_reset():
    evaluator = StageIncrementReward()
    evaluator.reset(_info(0))
    assert evaluator.evaluate(2 / 3, _info(2), terminated=False, truncated=False).train_reward == 2
    assert evaluator.evaluate(1, _info(3), terminated=True, truncated=False).train_reward == 5
    evaluator.reset(_info(0))
    assert evaluator.evaluate(1, _info(3), terminated=True, truncated=False).train_reward == 7


def test_stage_reward_rejects_invalid_progress():
    evaluator = StageIncrementReward()
    with pytest.raises(RuntimeError, match="reset"):
        evaluator.evaluate(0, _info(0), terminated=False, truncated=False)
    evaluator.reset(_info(0))
    with pytest.raises(ValueError, match="success/stage"):
        evaluator.evaluate(0, {"stage": 1, "success": True}, terminated=False, truncated=False)
    with pytest.raises(ValueError, match="stage"):
        evaluator.evaluate(0, _info(4), terminated=False, truncated=False)
    with pytest.raises(ValueError, match="stage"):
        evaluator.evaluate(0, {"stage": 1.5, "success": False}, terminated=False, truncated=False)
    assert evaluator.evaluate(2 / 3, _info(2), terminated=False, truncated=False).train_reward == 2
    with pytest.raises(ValueError, match="nondecreasing"):
        evaluator.evaluate(1 / 3, _info(1), terminated=False, truncated=False)
