"""Online window and transition assembly; no robot, policy or reward changes."""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np


def snapshot_observation(observation: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, np.ndarray]:
    keys = (spec["state_key"], *spec["camera_keys"])
    return {key: np.asarray(observation[key]).copy() for key in keys}


def initial_history(observation: Mapping[str, Any], spec: Mapping[str, Any], n_obs_steps: int):
    if type(n_obs_steps) is not int or n_obs_steps < 1:
        raise ValueError("n_obs_steps must be positive.")
    return deque(
        (snapshot_observation(observation, spec) for _ in range(n_obs_steps)),
        maxlen=n_obs_steps,
    )


def observation_window(history: deque, spec: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if not history or len(history) != history.maxlen:
        raise ValueError("Observation history must contain the complete window.")
    return {
        key: np.stack([frame[key] for frame in history], axis=0)
        for key in (spec["state_key"], *spec["camera_keys"])
    }


def discounted_chunk_reward(rewards: list[float], gamma: float) -> float:
    if not rewards or not 0 < gamma <= 1:
        raise ValueError("Chunk rewards must be nonempty, and gamma in (0,1].")
    values = np.asarray(rewards, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Chunk contains nonfinite reward.")
    return float(sum(float(reward) * gamma ** i for i, reward in enumerate(values)))
