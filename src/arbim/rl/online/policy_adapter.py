"""Native RL1002D stochastic sampling contract. No PPO mathematics lives here."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class OnlineSample:
    action: torch.Tensor          # [B, H_exec, D], unnormalized environment actions
    all_x: torch.Tensor           # [K+1, B, T, D], normalized denoising trajectory
    a_logprob: torch.Tensor       # [K, B, T, D], log probabilities of those transitions


def validate_online_policy(policy: Any, ratio_mode: str) -> None:
    if ratio_mode not in ("chunk_joint", "per_action_step"):
        raise ValueError("ratio_mode must be chunk_joint or per_action_step.")
    if getattr(policy, "is_flow", False) or getattr(policy, "w_pc", False):
        raise ValueError("Task 6 supports native RGB DDIM RL1002D, not Flow or point clouds.")
    if not getattr(policy, "no_pre_action", False):
        raise ValueError("Task 6 requires the existing no_pre_action=True checkpoint.")
    if getattr(policy, "use_aug", False):
        raise ValueError("Disable stochastic image augmentation for online log-prob recomputation.")
    h = policy.n_action_steps
    t = policy.horizon - policy.n_obs_steps + 1
    if type(h) is not int or not 1 <= h <= t:
        raise ValueError("Invalid effective n_action_steps or prediction horizon.")
    if ratio_mode == "per_action_step" and h > 1:
        raise ValueError(
            "Pinned RL-100 Online PPO per_action_step broadcasts [B,H,1] "
            "against [B,1] incorrectly for H>1. Use chunk_joint or H=1 "
            "until an explicit upstream-fix decision is approved."
        )
    if getattr(policy, "num_inference_steps", 0) < 1:
        raise ValueError("A positive DDIM num_inference_steps is required.")


@torch.no_grad()
def sample_online(
    policy: Any, observation: Mapping[str, torch.Tensor], *, fix_encoder: bool = True
) -> OnlineSample:
    """Preserve the exact sampled denoising transitions used by the environment action."""
    if "agent_pos" not in observation:
        raise KeyError("Online observation requires agent_pos.")
    action, all_x, logprob = policy.all_step_action_logprob(
        dict(observation), fix_encoder=fix_encoder, use_cm=False
    )
    b = observation["agent_pos"].shape[0]
    t = policy.horizon - policy.n_obs_steps + 1
    d = policy.action_dim
    k = policy.num_inference_steps
    expected = ((b, policy.n_action_steps, d), (k + 1, b, t, d), (k, b, t, d))
    actual = (tuple(action.shape), tuple(all_x.shape), tuple(logprob.shape))
    if actual != expected:
        raise ValueError(f"Native RL1002D sampling contract mismatch: {actual} != {expected}.")
    if not all(torch.isfinite(x).all().item() for x in (action, all_x, logprob)):
        raise FloatingPointError("Online DDIM sampling produced NaN/Inf.")
    return OnlineSample(action, all_x, logprob)
