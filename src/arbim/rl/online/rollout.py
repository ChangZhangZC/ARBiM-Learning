"""Collect genuine on-policy DDIM trajectories through existing DuoBenchAdapter."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from arbim.data.online_adapter import (
    discounted_chunk_reward, initial_history,
    observation_window, snapshot_observation,
)
from arbim.policy.inference_rl100_dp2d import prepare_observation
from arbim.rl.online.policy_adapter import sample_online
from arbim.rl.rewards.base import RewardEvaluator


def collect_online_episode(
    adapter: Any, policy: Any, spec: Mapping[str, Any], buffer: Any,
    evaluator: RewardEvaluator, *, gamma: float, max_env_steps: int,
    seed: int | None = None, fix_encoder: bool = True,
) -> dict[str, Any]:
    """Collect full DDIM trajectories while executing only available env steps.

    The final chunk may be shorter than the policy horizon. Record its actual
    execution span for variable-discount GAE outside the pinned native PPO.
    """
    span = policy.n_action_steps
    if type(max_env_steps) is not int or max_env_steps < 1:
        raise ValueError("max_env_steps must be a positive integer.")
    if adapter.spec is not spec and adapter.spec != spec:
        raise ValueError("DuoBench checkpoint/environment specs disagree.")
    device = next(policy.parameters()).device
    raw_obs, initial_info = adapter.reset(seed=seed)
    evaluator.reset(initial_info)
    history = initial_history(raw_obs, spec, policy.n_obs_steps)
    total_steps, transitions = 0, 0
    return_raw, return_train = 0.0, 0.0
    success = bool(initial_info.get("success", False))
    final_stage = initial_info.get("stage")
    ended = success
    while total_steps < max_env_steps and not ended:
        before = observation_window(history, spec)
        prepared = prepare_observation(
            {key: value[None] for key, value in before.items()},
            spec, policy.n_obs_steps, device=device,
        )
        sampled = sample_online(policy, prepared, fix_encoder=fix_encoder)
        if sampled.action.shape[0] != 1:
            raise ValueError("Task 6 V1 uses a single DuoBench environment.")
        commands = sampled.action[0].detach().cpu().numpy()
        rewards: list[float] = []
        terminated = truncated = False
        for command in commands[:max_env_steps - total_steps]:
            raw_obs, reward, terminated, truncated, info = adapter.step(command)
            result = evaluator.evaluate(reward, info, terminated=terminated, truncated=truncated)
            history.append(snapshot_observation(raw_obs, spec))
            rewards.append(result.train_reward)
            total_steps += 1
            return_raw += result.env_reward
            return_train += result.train_reward
            success = success or bool(result.metrics["success"])
            final_stage = result.metrics.get("stage", final_stage)
            # Native Hinge-Chest termination is its success event. Do NOT
            # carry over Task 5's optional post_success_steps evaluation.
            ended = bool(terminated or truncated or success)
            if ended:
                break

        if not rewards:
            raise RuntimeError("Online episode attempted an empty action chunk.")
        after = observation_window(history, spec)
        chunk_r = discounted_chunk_reward(rewards, gamma)
        buffer.store(
            before, sampled.all_x[:, 0], sampled.a_logprob[:, 0],
            chunk_r, after,
            bool(ended or total_steps >= max_env_steps),
            bool(terminated or success),
            executed_steps=len(rewards),
        )
        transitions += 1
        if total_steps >= max_env_steps:
            ended = True

    return {
        "episode_steps": total_steps, "transitions": transitions,
        "success": bool(success), "final_stage": final_stage,
        "env_return": return_raw, "train_return": return_train,
        "terminated": bool(terminated) if transitions else False,
        "truncated": bool(truncated) if transitions else False,
    }
