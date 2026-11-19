"""Build the pinned RL-100 Online PPO without copying or rewriting its loss."""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Mapping

import torch

from arbim.rl.online.policy_adapter import validate_online_policy


def build_online_ppo(policy: Any, settings: Mapping[str, Any]):
    """Configure native PPO Encoder modes without changing RL-100 loss math.

    Shared Critic consumes actor features; independent Critic owns a deepcopy
    of the pretrained IL RGB Encoder and learns its own ValueMLP head.
    The ValueMLP head starts random because no offline Q/V is available.
    """
    from rl_100.unidpg.net import ValueMLP
    from rl_100.unidpg.uni_ppo import BehaviorProximalPolicyOptimization
    from arbim.rl.online.telemetry import RatioDiagnosticsMixin

    class PPOWithRatioDiagnostics(RatioDiagnosticsMixin, BehaviorProximalPolicyOptimization):
        pass


    cfg = dict(settings)
    ratio_mode = cfg["ratio_mode"]
    validate_online_policy(policy, ratio_mode)
    fix_encoder = cfg.get("fix_encoder")
    share_encoder = cfg.get("share_encoder")
    if type(fix_encoder) is not bool or type(share_encoder) is not bool:
        raise ValueError("fix_encoder and share_encoder must be booleans.")
    if cfg.get("use_bc"):
        raise ValueError("Online BC remains outside the current scope.")
    recon = cfg.get("recon", False)
    value_recon = cfg.get("value_recon", False)
    per_step_recon = cfg.get("per_step_recon", False)
    for name, enabled in (("recon", recon), ("value_recon", value_recon),
                          ("per_step_recon", per_step_recon)):
        if type(enabled) is not bool:
            raise ValueError(f"ppo.{name} must be boolean.")
    if recon and fix_encoder:
        raise ValueError("ppo.recon requires fix_encoder=false; native PPO skips Actor aux loss otherwise.")
    if value_recon and share_encoder:
        raise ValueError("ppo.value_recon requires share_encoder=false in native RL-100.")
    if per_step_recon and not recon:
        raise ValueError("ppo.per_step_recon requires ppo.recon=true.")
    encoder = policy.obs_encoder
    supports_aux = bool(getattr(encoder, "use_recon", False) or getattr(encoder, "use_vib", False))
    if (recon or value_recon) and not supports_aux:
        raise ValueError(
            "Recon/VIB requires a matching IL checkpoint built with vision.use_recon "
            "or vision.use_vib; changing only Online YAML cannot add native heads."
        )
    if cfg["mini_batch_size"] > cfg["batch_size"]:
        raise ValueError("PPO mini_batch_size cannot exceed batch_size.")
    if cfg.get("distill_phase", "off") != "off":
        raise ValueError("Task 6 V1 excludes online distillation.")
    if cfg["linear_lr_decay"] and (
        type(cfg.get("max_train_steps")) is not int or cfg["max_train_steps"] < 1
    ):
        raise ValueError("Native linear LR decay requires a positive decision-step horizon.")
    device = next(policy.parameters()).device
    feature_dim = int(policy.obs_feature_dim) * int(policy.n_obs_steps)
    critic = ValueMLP(
        None if share_encoder else deepcopy(policy.obs_encoder),
        feature_dim, int(cfg["value_hidden_dim"]),
        int(cfg["value_depth"]), n_obs_steps=policy.n_obs_steps,
        # RL-100 ValueLearner initializes its independent Value Encoder as
        # trainable even if the Actor Encoder is frozen.
        fix_encoder=False,
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
        fix_encoder=fix_encoder, share_encoder=share_encoder, encoder_lr_scale=1.0,
        actor_grad=True, use_bc=False, recon=recon, value_recon=value_recon,
        per_step_recon=per_step_recon, use_adv_norm=bool(cfg["advantage_normalize"]),
        use_grad_clip=True, is_clip_value=bool(cfg["clip_value"]),
        use_lr_decay=bool(cfg["linear_lr_decay"]),
        v_encoder=False, iterations=0,
        max_train_steps=int(cfg.get("max_train_steps", 100000)),  # native scheduler, counts chunk decisions
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
    ppo_class = PPOWithRatioDiagnostics if cfg.get("enable_ratio_logging", False) else BehaviorProximalPolicyOptimization
    bridge = ppo_class(
        policy=policy, device=device,
        policy_lr=ppo_args.lr_a, clip_ratio=ppo_args.epsilon,
        entropy_weight=0.0, decay=1.0, omega=0.5,
        batch_size=ppo_args.batch_size, is_iql=False,
        ratio_strategy="last", num_inference_steps=policy.num_inference_steps,
        fix_encoder=fix_encoder, cfg=native_cfg,
    )
    bridge.transfer2online(critic=critic, dynamics=None, cfg=native_cfg)
    # Pinned upstream initializes the Critic KL scheduler with Actor LR.
    # Correct the adapter-owned initialization without editing RL-100.
    bridge.kl_scheduler_critic.current_lr = ppo_args.lr_c
    # transfer2online() constructs native optimizers before these mode flags.
    # Explicitly restore gradients if the incoming IL model had frozen params.
    bridge._policy.obs_encoder.requires_grad_(not fix_encoder)
    bridge._policy.obs_encoder.eval()  # native RL-100 keeps Online encoder in eval mode
    if not share_encoder:
        bridge.critic._obs_encoder.requires_grad_(True)
        bridge.critic._obs_encoder.eval()
        if bridge.critic._obs_encoder is bridge._policy.obs_encoder:
            raise RuntimeError("Independent Critic must own a distinct Encoder.")
    bridge._old_policy.requires_grad_(False)
    bridge._old_policy.eval()
    return bridge


def _variable_span_advantage(ppo: Any, buffer: Any) -> dict[str, torch.Tensor]:
    """Use the native precomputed interface when transitions span fewer env steps.

    A complete chunk has gamma**H; a partial chunk has gamma**executed_steps.
    Keep upstream's per-decision GAE lambda and terminal/dw semantics.
    """
    s, _, _, rewards, next_s, dw, done = buffer.numpy_to_tensor()
    with torch.no_grad():
        vs, vs_next = ppo._compute_critic_values_in_chunks(
            s, next_s, use_obs2latent=bool(ppo.args.share_encoder),
        )
        spans = torch.as_tensor(
            buffer.executed_spans(), dtype=rewards.dtype, device=rewards.device,
        ).view(-1, 1)
        discount = torch.pow(torch.as_tensor(ppo.args.gamma, device=rewards.device), spans)
        deltas = rewards + discount * (1.0 - dw) * vs_next - vs
        adv = torch.zeros_like(deltas)
        gae = torch.zeros_like(deltas[0])
        for idx in reversed(range(buffer.count)):
            gae = deltas[idx] + discount[idx] * ppo.args.lamda * gae * (1.0 - done[idx])
            adv[idx] = gae
        return {"adv": adv, "v_target": adv + vs, "vs": vs}


def count_ppo_steps(batch_size: int, mini_batch_size: int, epochs: int) -> int:
    """Native BatchSampler(drop_last=False) minibatches across all PPO epochs."""
    return epochs * ((batch_size + mini_batch_size - 1) // mini_batch_size)


def update_online_ppo(ppo: Any, buffer: Any, *, total_steps: int) -> dict[str, float]:
    if buffer.count < 1:
        raise ValueError("Online PPO update needs at least one transition.")
    # Native loop uses args.batch_size as an exact range. Collect complete
    # episodes to preserve GAE boundaries, then adapt its current batch length.
    ppo.args.batch_size = buffer.count
    has_partial = getattr(buffer, "has_partial_spans", lambda: False)()
    precomputed = _variable_span_advantage(ppo, buffer) if has_partial else None
    actor, critic, bc, distill = ppo.dp_align_update_no_share(
        buffer, total_steps, precomputed=precomputed,
    )
    metrics = dict(actor_loss=float(actor), critic_loss=float(critic),
                   bc_loss=float(bc), distill_loss=float(distill))
    if not all(torch.isfinite(torch.tensor(v)).item() for v in metrics.values()):
        raise FloatingPointError(f"Nonfinite native PPO metrics: {metrics}")
    return metrics
