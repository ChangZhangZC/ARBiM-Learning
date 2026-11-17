"""Separate exact Online resume state from version-2 inference snapshots."""
from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

from arbim.training.train_rl100_dp2d import _rng_state, _restore_rng


def _atomic_save(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_online_checkpoint(
    path: str | Path, ppo: Any, *, config: Mapping[str, Any],
    updates: int, episode_count: int, env_steps: int,
    original_checkpoint: str,
) -> None:
    """Call only AFTER a successful PPO update, with an emptied on-policy buffer."""
    _atomic_save(path, {
        "format_version": 1,
        "checkpoint_role": "online_training",
        "original_checkpoint": original_checkpoint,
        "config": copy.deepcopy(dict(config)),
        "updates": int(updates), "episodes": int(episode_count),
        "env_steps": int(env_steps),
        "policy": ppo._policy.state_dict(),
        "critic": ppo.critic.state_dict(),
        "optimizer_actor": ppo.optimizer_actor.state_dict(),
        "optimizer_critic": ppo.optimizer_critic.state_dict(),
        "ppo_iteration": ppo.iteration,
        "actor_adaptive_lr": ppo.kl_scheduler_actor.current_lr,
        "critic_adaptive_lr": ppo.kl_scheduler_critic.current_lr,
        "rng": _rng_state(),
    })


def read_online_checkpoint(
    path: str | Path, *, config: Mapping[str, Any],
    original_checkpoint: str,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or payload.get("checkpoint_role") != "online_training":
        raise ValueError("Not a supported completed-batch Online PPO checkpoint.")
    if payload.get("config") != dict(config) or payload.get("original_checkpoint") != original_checkpoint:
        raise ValueError("Online Resume contract differs from the saved run configuration.")
    for key in ("updates", "episodes", "env_steps"):
        if type(payload.get(key)) is not int or payload[key] < 0:
            raise ValueError(f"Invalid Online checkpoint counter: {key}.")
    return payload


def restore_online_checkpoint(ppo: Any, payload: Mapping[str, Any]) -> tuple[int, int, int]:
    # Do not reconstruct optimizer: native optimizer owns policy.model parameters.
    ppo._policy.load_state_dict(payload["policy"], strict=True)
    device = next(ppo._policy.parameters()).device
    ppo._policy.to(device)  # Native normalizer may re-create buffers on CPU.
    ppo.critic.load_state_dict(payload["critic"], strict=True)
    ppo.optimizer_actor.load_state_dict(payload["optimizer_actor"])
    ppo.optimizer_critic.load_state_dict(payload["optimizer_critic"])
    from rl_100.common.pytorch_util import optimizer_to
    optimizer_to(ppo.optimizer_actor, device)
    optimizer_to(ppo.optimizer_critic, device)
    ppo.iteration = int(payload["ppo_iteration"])
    ppo.kl_scheduler_actor.current_lr = float(payload["actor_adaptive_lr"])
    ppo.kl_scheduler_critic.current_lr = float(payload["critic_adaptive_lr"])
    ppo.set_old_policy()
    _restore_rng(payload["rng"])
    return payload["updates"], payload["episodes"], payload["env_steps"]


def save_inference_snapshot(
    path: str | Path, ppo: Any, *, il_contract: Mapping[str, Any], env_steps: int,
) -> None:
    """Existing ARBiM inference loader accepts policy_snapshot, format_version=2."""
    _atomic_save(path, {
        "format_version": 2, "checkpoint_role": "policy_snapshot",
        "selected_weights": "raw", "policy": ppo._policy.state_dict(),
        "global_step": int(env_steps), "contract": copy.deepcopy(dict(il_contract)),
    })
