"""Reward interface: keep raw environment signals independent of training reward."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RewardResult:
    train_reward: float
    env_reward: float
    metrics: dict[str, Any]


class RewardEvaluator:
    def reset(self, info: Mapping[str, Any]) -> None:
        """Optional per-episode hook; stateless rewards need no reset."""

    def evaluate(
        self, reward: float, info: Mapping[str, Any],
        *, terminated: bool, truncated: bool,
    ) -> RewardResult:
        raise NotImplementedError
