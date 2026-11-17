"""Build the pinned RL-100 Online PPO without copying or rewriting its loss."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

import torch

from arbim.rl.online.policy_adapter import validate_online_policy


def build_online_ppo(policy: Any, settings: Mapping[str, Any]):
    """Frozen shared RGB encoder + native ValueMLP + native denoising PPO.

    ValueMLP receives the already normalized and encoded observation via
    RL1002D.obs2latent(). Skipping offline Q/V means the value starts random.
    """
    from rl_100.unidpg.net import ValueMLP
    from rl_100.unidpg.uni_ppo import BehaviorProximalPolicyOptimization

    cfg = dict(settings)
    ratio_mode = cfg["ratio_mode"]
    validate_online_policy(policy, ratio_mode)
    if cfg.get("fix_encoder") is not True or cfg.get("share_encoder") is not True:
        raise ValueError("Task 6 V1 requires fix_encoder=True and share_encoder=True.")
    if cfg.get("use_bc") or cfg.get("value_recon") or cfg.get("recon"):
        raise ValueError("BC/Recon objectives are outside Task 6 V1.")
    if cfg["mini_batch_size"] > cfg["batch_size"]:
        raise ValueError("PPO mini_batch_size cannot exceed batch_size.")
    if cfg.get("distill_phase", "off") != "off":
        raise ValueError("Task 6 V1 excludes online distillation.")
    device = next(policy.parameters()).device
    feature_dim = int(policy.obs_feature_dim) * int(policy.n_obs_steps)
    critic = ValueMLP(
        None, feature_dim, int(cfg["value_hidden_dim"]),
        int(cfg["value_depth"]), n_obs_steps=policy.n_obs_steps,
        fix_encoder=True,
    ).to(device)
    ppo_args = SimpleNamespace(
        batch_size=int(cfg["batch_size"]),
        mini_batch_size=int(cfg["mini_batch_size"]),
        K_epochs=int(cfg["K_epochs"]),
        epsilon=float(cfg["epsilon"]),
        gamma=float(cfg["gamma"]),
        lamda=float(cfg["gae_lambda"]),
        lr_a=float(cfg["actor_lr"]),
        lr_c=float(cfg["critic_lr"]),
        min_lr_actor=float(cfg["min_lr_actor"]),
        max_lr_actor=float(cfg["max_lr_actor"]),
        min_lr_critic=float(cfg["min_lr_critic"]),
        max_lr_critic=float(cfg["max_lr_critic"]),
        adaptive_kl=float(cfg["adaptive_kl"]),
        fix_encoder=True, share_encoder=True, encoder_lr_scale=1.0,
        actor_grad=True, use_bc=False, recon=False, value_recon=False,
        per_step_recon=False, use_adv_norm=bool(cfg["advantage_normalize"]),
        use_grad_clip=True, is_clip_value=bool(cfg["clip_value"]),
        use_lr_decay=bool(cfg["linear_lr_decay"]),
        v_encoder=False, iterations=0,
        max_train_steps=int(cfg["max_train_steps"]),
        enable_ratio_logging=bool(cfg.get("enable_ratio_logging", False)),
        ratio_log_every_updates=int(cfg.get("ratio_log_every_updates", 10)),
        ratio_plot_on_final_flush=False,
    )
    native_cfg = SimpleNamespace(
        n_action_steps=policy.n_action_steps,
        no_pre_action=policy.no_pre_action,
        chunk_as_single_action=ratio_mode == "chunk_joint",
        distill_phase="off",
        ppo=ppo_args,
    )
    bridge = BehaviorProximalPolicyOptimization(
        policy=policy, device=device,
        policy_lr=ppo_args.lr_a, clip_ratio=ppo_args.epsilon,
        entropy_weight=0.0, decay=1.0, omega=0.5,
        batch_size=ppo_args.batch_size, is_iql=False,
        ratio_strategy="last", num_inference_steps=policy.num_inference_steps,
        fix_encoder=True, cfg=native_cfg,
    )
    bridge.transfer2online(critic=critic, dynamics=None, cfg=native_cfg)
    bridge._policy.obs_encoder.requires_grad_(False)
    bridge._policy.obs_encoder.eval()
    bridge._old_policy.requires_grad_(False)
    bridge._old_policy.eval()
    return bridge


def update_online_ppo(ppo: Any, buffer: Any, *, total_steps: int) -> dict[str, float]:
    if buffer.count < 1:
        raise ValueError("Online PPO update needs at least one transition.")
    # Native loop uses args.batch_size as an exact range. Collect complete
    # episodes to preserve GAE boundaries, then adapt its current batch length.
    ppo.args.batch_size = buffer.count
    actor, critic, bc, distill = ppo.dp_align_update_no_share(buffer, total_steps)
    metrics = dict(actor_loss=float(actor), critic_loss=float(critic),
                   bc_loss=float(bc), distill_loss=float(distill))
    if not all(torch.isfinite(torch.tensor(v)).item() for v in metrics.values()):
        raise FloatingPointError(f"Nonfinite native PPO metrics: {metrics}")
    return metrics
