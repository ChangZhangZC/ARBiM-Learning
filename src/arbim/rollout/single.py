"""Single-episode RL100 DP-2D rollout in the DuoBench environment.

Keeps observation history and chunk scheduling outside the native Policy and
DuoBench adapter. Recording and multi-episode statistics belong to Task 5.4.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from arbim.policy.inference_rl100_dp2d import (
    load_inference_policy, prepare_observation, predict, seed_inference,
)


def run_single_episode(
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


def main() -> None:
    """Minimal non-interactive entrypoint for Task 5.3 smoke tests."""
    import yaml
    from arbim.envs.duobench import DuoBenchAdapter, create_duobench_env

    parser = argparse.ArgumentParser(description="One RL100 DP-2D DuoBench Hinge-Chest episode")
    parser.add_argument("--config", type=Path, default=Path("configs/rollout/hinge_chest.yaml"))
    parser.add_argument("--checkpoint", type=Path, help="Override policy.checkpoint_path")
    parser.add_argument("--device", help="Override policy.device")
    parser.add_argument("--max-env-steps", type=int, help="Override rollout.max_env_steps")
    args = parser.parse_args()
    with args.config.expanduser().open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, Mapping):
        raise ValueError("Rollout YAML must be a mapping.")
    env_cfg, policy_cfg, rollout_cfg = (config.get(key) for key in ("env", "policy", "rollout"))
    if not all(isinstance(value, Mapping) for value in (env_cfg, policy_cfg, rollout_cfg)):
        raise ValueError("Rollout YAML requires env, policy and rollout mappings.")
    checkpoint = args.checkpoint or policy_cfg.get("checkpoint_path")
    if not checkpoint:
        raise ValueError("Set policy.checkpoint_path or --checkpoint.")
    device = args.device or policy_cfg.get("device", "cuda:0")
    policy, spec, _ = load_inference_policy(
        checkpoint, device=device, weights=policy_cfg.get("weights", "raw"),
        n_action_steps=policy_cfg.get("n_action_steps"),
        num_inference_steps=policy_cfg.get("num_inference_steps"),
    )
    seed = rollout_cfg.get("seed")
    env = create_duobench_env(env_cfg)
    try:
        seed_inference(seed)
        adapter = DuoBenchAdapter(
            env, spec, env_cfg["camera_map"],
            binarize_state_gripper=env_cfg.get("binarize_state_gripper", True),
            gripper_threshold=env_cfg.get("gripper_threshold", 0.9),
        )
        result = run_single_episode(
            adapter, policy, spec,
            execute_steps=rollout_cfg.get("execute_steps", policy.n_action_steps),
            max_env_steps=(args.max_env_steps if args.max_env_steps is not None
                           else rollout_cfg.get("max_env_steps", 1000)),
            post_success_steps=rollout_cfg.get("post_success_steps", 0),
            seed=seed, deterministic=policy_cfg.get("deterministic", True),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
