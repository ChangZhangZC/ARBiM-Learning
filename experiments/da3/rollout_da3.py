"""Experimental copy of the native RL100 DP-2D rollout.

Preserves policy history, action-chunk scheduling, and termination logic.
The only added hook captures the current raw observation once per new chunk.
"""
from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from arbim.policy.inference_rl100_dp2d import (
    prepare_observation, predict,
)


def run_single_episode_da3(
    adapter: Any,
    policy: Any,
    spec: Mapping[str, Any],
    *,
    execute_steps: int = 16,
    max_env_steps: int = 1000,
    post_success_steps: int = 0,
    seed: int | None = None,
    deterministic: bool = True,
    on_frame: Callable[[Mapping[str, Any], int], None] | None = None,
    on_chunk: Callable[[Mapping[str, Any], int, int], None] | None = None,
) -> dict[str, Any]:
    """Reset once, predict native unnormalized chunks, and step until stopped.

    The caller owns the policy/environment lifetime and inference RNG seed.
    Observation history is fresh for every call, even when reusing the adapter.
    """
    if type(execute_steps) is not int or not 1 <= execute_steps <= policy.n_action_steps:
        raise ValueError("execute_steps must be in [1, policy.n_action_steps].")
    if type(max_env_steps) is not int or max_env_steps < 1:
        raise ValueError("max_env_steps must be a positive integer.")
    if type(post_success_steps) is not int or post_success_steps < 0:
        raise ValueError("post_success_steps must be a nonnegative integer.")
    if type(policy.n_obs_steps) is not int or policy.n_obs_steps < 1:
        raise ValueError("policy.n_obs_steps must be a positive integer.")
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ValueError("seed must be a nonnegative integer or null.")
    if type(deterministic) is not bool:
        raise ValueError("deterministic must be boolean.")
    if on_frame is not None and not callable(on_frame):
        raise TypeError("on_frame must be callable or None.")
    if on_chunk is not None and not callable(on_chunk):
        raise TypeError("on_chunk must be callable or None.")

    keys = (spec["state_key"], *spec["camera_keys"])
    device = next(policy.parameters()).device
    started = time.perf_counter()
    observation, info = adapter.reset(seed=seed)
    if on_frame is not None:
        on_frame(observation, 0)

    def snapshot(frame: Mapping[str, Any]) -> dict[str, np.ndarray]:
        return {key: np.asarray(frame[key]).copy() for key in keys}

    history = deque((snapshot(observation) for _ in range(policy.n_obs_steps)),
                    maxlen=policy.n_obs_steps)
    success = bool(info.get("success", False))
    first_success_step = 0 if success else None
    terminated = truncated = False
    episode_steps = 0
    episode_return = 0.0
    inference_count = 0
    inference_seconds = 0.0
    final_stage = info.get("stage")
    stop_reason = "success" if success else "max_env_steps"

    while episode_steps < max_env_steps and (first_success_step is None or
           episode_steps - first_success_step < post_success_steps):
        should_stop = False
        window = {
            key: np.stack([frame[key] for frame in history], axis=0)[None]
            for key in keys
        }
        prepared = prepare_observation(window, spec, policy.n_obs_steps, device=device)
        infer_started = time.perf_counter()
        action_chunk = predict(policy, prepared, deterministic=deterministic)["action"]
        # Moving the result to CPU synchronizes GPU inference before timing it.
        actions = action_chunk[0].detach().cpu().numpy()
        inference_seconds += time.perf_counter() - infer_started
        inference_count += 1
        if actions.shape != (policy.n_action_steps, spec["action_dim"]):
            raise ValueError(f"Unexpected native Action Chunk shape: {actions.shape}.")

        # Capture the same fresh environment observation used for this chunk.
        # This callback never changes the policy's history or action values.
        if on_chunk is not None:
            raw = getattr(adapter, "last_raw_observation", None)
            if not isinstance(raw, Mapping):
                raise ValueError("DuoBench raw observation unavailable at chunk inference.")
            on_chunk(raw, episode_steps, inference_count - 1)

        for action in actions[:execute_steps]:
            observation, reward, terminated, truncated, info = adapter.step(action)
            episode_steps += 1
            episode_return += float(reward)
            if on_frame is not None:
                on_frame(observation, episode_steps)
            history.append(snapshot(observation))
            step_success = bool(info["success"])
            if step_success and first_success_step is None:
                first_success_step = episode_steps
            success = success or step_success
            final_stage = info.get("stage", final_stage)
            grace_complete = (first_success_step is not None and
                              episode_steps - first_success_step >= post_success_steps)
            # HingeChest's native terminated=True comes from success itself.
            # The pinned RCS simulation can continue stepping after this signal.
            if grace_complete or truncated or (terminated and not step_success) or episode_steps >= max_env_steps:
                if truncated:
                    stop_reason = "truncated"
                elif terminated and not step_success:
                    stop_reason = "terminated"
                elif grace_complete:
                    stop_reason = "success"
                else:
                    stop_reason = "max_env_steps"
                should_stop = True
                break
        if should_stop:
            break

    if isinstance(final_stage, np.generic):
        final_stage = final_stage.item()
    return {
        "success": success,
        "first_success_step": first_success_step,
        "post_success_steps_executed": (episode_steps - first_success_step if first_success_step is not None else 0),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "max_steps_reached": stop_reason == "max_env_steps",
        "stop_reason": stop_reason,
        "episode_steps": episode_steps,
        "episode_return": episode_return,
        "final_stage": final_stage,
        "inference_count": inference_count,
        "mean_inference_ms": 1000.0 * inference_seconds / inference_count if inference_count else 0.0,
        "elapsed_time": time.perf_counter() - started,
    }
