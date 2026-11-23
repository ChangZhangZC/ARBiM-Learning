"""Hinge-Chest online V1: use the native environment reward unmodified."""
from __future__ import annotations

import math
import numbers

import numpy as np
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
        if stage is not None and (isinstance(stage, bool) or not isinstance(stage, (numbers.Real, np.integer, np.floating))):
            raise ValueError("Unexpected Hinge-Chest stage type.")
        if stage is not None and not math.isfinite(float(stage)):
            raise ValueError("Hinge-Chest stage must be finite.")
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
    if name == "env_raw":
        return HingeChestReward()
    if name == "stage_increment":
        from arbim.rl.rewards.stage_increment import StageIncrementReward
        return StageIncrementReward()
    raise ValueError("Supported reward modes: env_raw, stage_increment.")
