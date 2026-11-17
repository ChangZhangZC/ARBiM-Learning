"""Separate resumable Online PPO state from stand-alone inference snapshots.

V2 stores a strict algorithm-contract digest instead of a duplicate YAML config.
V1 checkpoints remain readable with their original full-config contract.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
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
    policy = copy.deepcopy(config["policy"])
    policy.pop("device", None)  # runtime device is not an algorithm change
    ppo = copy.deepcopy(config["ppo"])
    ppo.pop("max_updates", None)  # training budget can increase on Resume
    return {
        "env": copy.deepcopy(config["env"]),  # includes common episode limit
        "policy": policy,
        "ppo": ppo,
        "reward": copy.deepcopy(config["reward"]),
        "rollout": {"seed": config["rollout"]["seed"]},
    }


def _digest(contract: Mapping[str, Any]) -> str:
    payload = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _contract_digest(config: Mapping[str, Any]) -> str:
    return _digest(online_resume_contract(config))


def _legacy_contract_digest(config: Mapping[str, Any]) -> str:
    """Original Task 7 V2 contract, needed to read pre-reorganization checkpoints."""
    policy = copy.deepcopy(config["policy"])
    policy.pop("device", None)
    return _digest({
        "env": copy.deepcopy(config["env"]),
        "policy": policy,
        "ppo": copy.deepcopy(config["ppo"]),
        "reward": copy.deepcopy(config["reward"]),
        "rollout": {
            "max_env_steps": config["rollout"]["max_env_steps"],
            "seed": config["rollout"]["seed"],
        },
    })


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
    legacy_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    version = payload.get("format_version")
    if version not in (1, 2) or payload.get("checkpoint_role") != "online_training":
        raise ValueError("Not a supported completed-batch Online PPO checkpoint.")
    if payload.get("original_checkpoint") != original_checkpoint:
        raise ValueError("Online Resume original IL checkpoint path differs.")
    if version == 1:
        # Previous V1 checkpoints embedded the full pre-reorganization YAML.
        expected = dict(legacy_config) if legacy_config is not None else dict(config)
        if payload.get("config") != expected:
            raise ValueError("Online V1 Resume requires its original complete configuration.")
    elif payload.get("contract_digest") != _contract_digest(config):
        # Allow the original Task 7 V2 digest during schema migration only.
        if legacy_config is None or payload["contract_digest"] != _legacy_contract_digest(legacy_config):
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
    path: str | Path, ppo: Any, *, il_contract: Mapping[str, Any],
    updates: int, env_steps: int, episodes: int,
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
        "global_step": int(updates), "ppo_updates": int(updates),
        "env_steps": int(env_steps), "episodes": int(episodes),
        "contract": contract,
    })


def prune_policy_snapshots(folder: str | Path, keep_last: int) -> None:
    """Retain only recent policy-only snapshots; never remove last.ckpt or IL."""
    if type(keep_last) is not int or keep_last < 1:
        raise ValueError("checkpoint.keep_last must be positive.")
    directory = Path(folder)
    # Old Task 7 snapshots use environment steps; new ones use PPO update ids.
    # Treat legacy files as earlier than newly created update snapshots.
    legacy = sorted(
        (p for p in directory.glob("step_*.ckpt")
         if re.fullmatch(r"step_\\d+\\.ckpt", p.name)),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    current = sorted(
        (p for p in directory.glob("update_*.ckpt")
         if re.fullmatch(r"update_\\d+\\.ckpt", p.name)),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    for stale in (legacy + current)[:-keep_last]:
        stale.unlink()
