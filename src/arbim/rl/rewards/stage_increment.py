"""One-time Hinge-Chest milestones and success reward for Online PPO."""
from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from typing import Any

from arbim.rl.rewards.base import RewardEvaluator, RewardResult

MILESTONE_REWARD = 1.0
SUCCESS_REWARD = 5.0
MAX_STAGE = 3


class StageIncrementReward(RewardEvaluator):
    """Pay once for reaching stages 1 and 2, and once for final success."""

    def __init__(self) -> None:
        self._previous_stage: int | None = None
        self._success_awarded = False

    @staticmethod
    def _read_stage(info: Mapping[str, Any]) -> int:
        if not isinstance(info, Mapping):
            raise TypeError("DuoBench info must be a mapping.")
        stage = info.get("stage")
        max_stage = info.get("max_stage", MAX_STAGE)
        if (isinstance(stage, bool) or not isinstance(stage, numbers.Integral)
                or not 0 <= int(stage) <= MAX_STAGE):
            raise ValueError(f"Invalid Hinge-Chest stage: {stage!r}.")
        if (isinstance(max_stage, bool) or not isinstance(max_stage, numbers.Integral)
                or int(max_stage) != MAX_STAGE):
            raise ValueError(f"Expected Hinge-Chest max_stage={MAX_STAGE}, got {max_stage!r}.")
        return int(stage)

    def reset(self, info: Mapping[str, Any]) -> None:
        """Start a fresh episode without re-awarding the initial state."""
        self._previous_stage = self._read_stage(info)
        initial_success = bool(info.get("success", False))
        if initial_success != (self._previous_stage == MAX_STAGE):
            raise ValueError("Hinge-Chest initial success/stage disagree.")
        self._success_awarded = initial_success

    def evaluate(
        self, reward: float, info: Mapping[str, Any],
        *, terminated: bool, truncated: bool,
    ) -> RewardResult:
        if self._previous_stage is None:
            raise RuntimeError("StageIncrementReward.reset(info) must be called after env.reset().")
        stage = self._read_stage(info)
        raw_reward = float(reward)
        if not math.isfinite(raw_reward):
            raise ValueError("Environment reward must be finite.")
        if stage < self._previous_stage:
            raise ValueError("Hinge-Chest stage must be nondecreasing within an episode.")
        success = bool(info.get("success", False))
        if success != (stage == MAX_STAGE):
            raise ValueError("Hinge-Chest success/stage disagree.")

        new_milestones = min(stage, MAX_STAGE - 1) - min(self._previous_stage, MAX_STAGE - 1)
        train_reward = MILESTONE_REWARD * new_milestones
        if success and not self._success_awarded:
            train_reward += SUCCESS_REWARD

        self._previous_stage = stage
        self._success_awarded |= success
        return RewardResult(
            train_reward=float(train_reward),
            env_reward=raw_reward,
            metrics={
                "success": success,
                "stage": stage,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            },
        )
