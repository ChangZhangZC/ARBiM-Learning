"""Separate resumable Online PPO state from stand-alone inference snapshots.

V2 stores a strict algorithm-contract digest instead of a duplicate YAML config.
V1 checkpoints remain readable with their original full-config contract.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

from arbim.training.train_rl100_dp2d import _rng_state, _restore_rng


def _atomic_save(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name, suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def online_resume_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Hold algorithm/env/rollout semantics fixed; allow operational budget changes.

    The run directory's config_resolved.yaml is the authoritative full config.
    """
    rollout = config["rollout"]
    return {
        "env": copy.deepcopy(config["env"]),
        "policy": copy.deepcopy(config["policy"]),
        "ppo": copy.deepcopy(config["ppo"]),
        "reward": copy.deepcopy(config["reward"]),
        "rollout": {
            "max_env_steps": rollout["max_env_steps"],
            "seed": rollout["seed"],
        },
    }


def _contract_digest(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        online_resume_contract(config), sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_online_checkpoint(
    path: str | Path, ppo: Any, *, config: Mapping[str, Any],
    updates: int, episode_count: int, env_steps: int,
    original_checkpoint: str,
) -> None:
    """Call only AFTER a successful PPO update, with an emptied on-policy buffer."""
    _atomic_save(path, {
        "format_version": 2,
        "checkpoint_role": "online_training",
        "original_checkpoint": original_checkpoint,
        "contract_digest": _contract_digest(config),
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
    version = payload.get("format_version")
    if version not in (1, 2) or payload.get("checkpoint_role") != "online_training":
        raise ValueError("Not a supported completed-batch Online PPO checkpoint.")
    if payload.get("original_checkpoint") != original_checkpoint:
        raise ValueError("Online Resume original IL checkpoint path differs.")
    if version == 1:
        # Preserve the exact contract of existing Task 6 smoke checkpoints.
        if payload.get("config") != dict(config):
            raise ValueError("Online V1 Resume requires its original complete configuration.")
    elif payload.get("contract_digest") != _contract_digest(config):
        raise ValueError("Online Resume algorithm/environment contract differs.")
    for key in ("updates", "episodes", "env_steps"):
        if type(payload.get(key)) is not int or payload[key] < 0:
            raise ValueError(f"Invalid Online checkpoint counter: {key}.")
    return payload


def restore_online_checkpoint(ppo: Any, payload: Mapping[str, Any]) -> tuple[int, int, int]:
    # Optimizers already reference the live policy.model parameters.
    ppo._policy.load_state_dict(payload["policy"], strict=True)
    device = next(ppo._policy.parameters()).device
    ppo._policy.to(device)  # Native normalizer can recreate buffers on CPU.
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
    contract = copy.deepcopy(dict(il_contract))
    contract["policy"]["n_action_steps"] = int(ppo._policy.n_action_steps)
    contract["policy"]["num_inference_steps"] = int(ppo._policy.num_inference_steps)
    if "ddim_inference_steps" in contract["policy"]:
        contract["policy"]["ddim_inference_steps"] = int(ppo._policy.num_inference_steps)
    _atomic_save(path, {
        "format_version": 2, "checkpoint_role": "policy_snapshot",
        "selected_weights": "raw", "policy": ppo._policy.state_dict(),
        "global_step": int(env_steps), "contract": contract,
    })


def prune_policy_snapshots(folder: str | Path, keep_last: int) -> None:
    """Retain only recent policy-only snapshots; never remove last.ckpt or IL."""
    if type(keep_last) is not int or keep_last < 1:
        raise ValueError("checkpoint.keep_last must be positive.")
    directory = Path(folder)
    snapshots = sorted(directory.glob("step_*.ckpt"))
    for stale in snapshots[:-keep_last]:
        stale.unlink()
