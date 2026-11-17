"""Hinge-Chest online V1: use the native environment reward unmodified."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from arbim.rl.rewards.base import RewardEvaluator, RewardResult


class HingeChestReward(RewardEvaluator):
    def evaluate(
        self, reward: float, info: Mapping[str, Any],
        *, terminated: bool, truncated: bool,
    ) -> RewardResult:
        if not isinstance(info, Mapping):
            raise TypeError("DuoBench info must be a mapping.")
        r = float(reward)
        if not math.isfinite(r):
            raise ValueError("Environment reward must be finite.")
        stage = info.get("stage")
        if stage is not None and (isinstance(stage, bool) or not isinstance(stage, (int, float))):
            raise ValueError("Unexpected Hinge-Chest stage type.")
        return RewardResult(
            train_reward=r,
            env_reward=r,
            metrics={
                "success": bool(info.get("success", False)),
                "stage": int(stage) if stage is not None else None,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            },
        )


def build_reward_evaluator(name: str) -> RewardEvaluator:
    if name != "env_raw":
        raise ValueError("Task 6 V1 supports reward.mode='env_raw' only.")
    return HingeChestReward()
