"""Native RL100 DP-2D Online PPO on DuoBench Hinge-Chest.

PPO, DDIM sampling and the normalizer stay upstream-native. Orchestration,
telemetry, checkpoint retention and experiment budgets live outside RL-100.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from arbim.policy.inference_rl100_dp2d import load_inference_policy
from arbim.rl.online.buffer import OnlinePPOBuffer
from arbim.rl.online.checkpoint import (
    read_online_checkpoint, restore_online_checkpoint,
    save_inference_snapshot, save_online_checkpoint, prune_policy_snapshots,
)
from arbim.rl.online.policy_adapter import validate_online_policy
from arbim.rl.online.ppo_bridge import build_online_ppo, update_online_ppo, count_ppo_steps
from arbim.rl.online.rollout import collect_online_episode
from arbim.rl.online.telemetry import (
    OnlineTelemetry, native_ratio_summary, append_native_ratio_records,
)
from arbim.rl.rewards.duobench_hinge_chest import build_reward_evaluator


DEFAULT_CONFIG = Path("configs/online/rl100_dp2d_hinge_chest.yaml")


def validate_config(cfg: Mapping[str, Any]) -> None:
    for section in ("env", "policy", "ppo", "reward", "rollout", "evaluation", "checkpoint", "output"):
        if not isinstance(cfg.get(section), Mapping):
            raise ValueError(f"Missing Task 6 configuration section: {section}.")
    if cfg["env"].get("task") != "hinge_chest":
        raise ValueError("Online Task 6 V1 targets DuoBench hinge_chest.")
    if not cfg["policy"].get("checkpoint_path"):
        raise ValueError("Set policy.checkpoint_path or pass --checkpoint.")
    if cfg["reward"].get("mode") != "env_raw":
        raise ValueError("Task 6 V1 supports only native env_raw reward.")
    ppo = cfg["ppo"]
    if ("max_steps" in ppo) == ("max_updates" in ppo):
        raise ValueError("Set exactly one of ppo.max_steps (PPO minibatches) or legacy ppo.max_updates.")
    for key in ("batch_size", "mini_batch_size", "K_epochs", "value_hidden_dim", "value_depth"):
        if type(ppo.get(key)) is not int or ppo[key] < 1:
            raise ValueError(f"ppo.{key} must be a positive integer.")
    budget_key = "max_steps" if "max_steps" in ppo else "max_updates"
    if type(ppo[budget_key]) is not int or ppo[budget_key] < 1:
        raise ValueError(f"ppo.{budget_key} must be a positive integer.")
    for key in ("actor_lr", "critic_lr", "min_lr_actor", "max_lr_actor", "min_lr_critic",
                "max_lr_critic", "adaptive_kl", "gamma", "gae_lambda", "epsilon"):
        value = ppo.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not np.isfinite(value) or value <= 0):
            raise ValueError(f"ppo.{key} must be a positive finite number.")
    if ppo["gamma"] > 1 or ppo["gae_lambda"] > 1:
        raise ValueError("ppo.gamma and ppo.gae_lambda must be <= 1.")
    for flag in ("fix_encoder", "share_encoder"):
        if type(ppo.get(flag)) is not bool:
            raise ValueError(f"ppo.{flag} must be boolean.")
    if type(ppo.get("linear_lr_decay")) is not bool:
        raise ValueError("ppo.linear_lr_decay must be boolean.")
    if ppo["value_depth"] < 2:
        raise ValueError("ppo.value_depth must be >= 2.")
    if ppo.get("distill_phase", "off") != "off":
        raise ValueError("Task 6 V1 excludes distillation.")
    if any(ppo.get(k) for k in ("use_bc", "recon", "value_recon")):
        raise ValueError("Task 6 V1 excludes BC/reconstruction objectives.")
    if type(cfg["env"].get("max_episode_steps")) is not int or cfg["env"]["max_episode_steps"] < 1:
        raise ValueError("env.max_episode_steps must be a positive integer.")
    if any(key in cfg["rollout"] for key in ("max_env_steps", "max_updates", "max_total_env_steps")):
        raise ValueError("Legacy rollout budget keys are unsupported; use env.max_episode_steps and ppo.max_updates.")
    if any(key in cfg["evaluation"] for key in ("max_env_steps",)):
        raise ValueError("Use env.max_episode_steps; evaluation.max_env_steps is no longer supported.")
    if "max_train_steps" in ppo:
        raise ValueError("The native LR schedule horizon is derived from the PPO update budget; do not set ppo.max_train_steps.")
    rollout = cfg["rollout"]
    if type(rollout.get("seed")) is not int or rollout["seed"] < 0:
        raise ValueError("rollout.seed must be a nonnegative integer.")
    evaluation = cfg["evaluation"]
    for key in ("every_updates", "num_episodes"):
        if type(evaluation.get(key)) is not int or evaluation[key] < 1:
            raise ValueError(f"evaluation.{key} must be positive.")
    for key in ("enabled", "record_video"):
        if type(evaluation.get(key)) is not bool:
            raise ValueError(f"evaluation.{key} must be boolean.")
    if evaluation.get("seed") is not None and (
        type(evaluation["seed"]) is not int or evaluation["seed"] < 0
    ):
        raise ValueError("evaluation.seed must be nonnegative or null.")
    checkpoints = cfg["checkpoint"]
    if type(checkpoints.get("every_updates")) is not int or checkpoints["every_updates"] < 1:
        raise ValueError("checkpoint.every_updates must be positive.")
    if type(checkpoints.get("keep_last", 3)) is not int or checkpoints.get("keep_last", 3) < 1:
        raise ValueError("checkpoint.keep_last must be positive.")
    if type(checkpoints.get("save_on_finish", True)) is not bool:
        raise ValueError("checkpoint.save_on_finish must be boolean.")
    logging_cfg = cfg.get("logging", {})
    if not isinstance(logging_cfg, Mapping):
        raise ValueError("logging must be a mapping.")
    for name in ("progress_bar", "jsonl"):
        if type(logging_cfg.get(name, True)) is not bool:
            raise ValueError(f"logging.{name} must be boolean.")
    for name in ("tensorboard", "ratio_diagnostics"):
        part = logging_cfg.get(name, {})
        if not isinstance(part, Mapping) or type(part.get("enabled", False)) is not bool:
            raise ValueError(f"logging.{name}.enabled must be boolean.")
    ratio_freq = logging_cfg.get("ratio_diagnostics", {}).get("every_updates", 5)
    if type(ratio_freq) is not int or ratio_freq < 1:
        raise ValueError("logging.ratio_diagnostics.every_updates must be positive.")


def normalize_online_config(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Read old Task 6/7 YAML only for legacy Resume; emit the new canonical schema.

    Do not reinterpret an old environment-step training budget as PPO updates.
    """
    cfg = copy.deepcopy(dict(config))
    rollout = cfg["rollout"]
    evaluation = cfg["evaluation"]
    legacy = (
        "max_env_steps" in rollout or "max_updates" in rollout
        or "max_total_env_steps" in rollout or "max_env_steps" in evaluation
        or "max_train_steps" in cfg["ppo"]
    )
    if not legacy:
        return cfg, None
    original = copy.deepcopy(cfg)
    training_limit = rollout.pop("max_env_steps", None)
    eval_limit = evaluation.pop("max_env_steps", None)
    if training_limit is not None and eval_limit is not None and training_limit != eval_limit:
        raise ValueError("Legacy rollout/evaluation episode limits differ; cannot unify silently.")
    episode_limit = training_limit if training_limit is not None else eval_limit
    if episode_limit is None and "max_episode_steps" not in cfg["env"]:
        raise ValueError("Legacy run is missing an episode environment-step limit.")
    if episode_limit is not None and cfg["env"].get("max_episode_steps", episode_limit) != episode_limit:
        raise ValueError("Conflicting legacy and new episode-step limits.")
    cfg["env"]["max_episode_steps"] = episode_limit or cfg["env"]["max_episode_steps"]
    old_updates = rollout.pop("max_updates", None)
    if old_updates is not None and cfg["ppo"].get("max_updates", old_updates) != old_updates:
        raise ValueError("Conflicting old and new PPO update budgets.")
    cfg["ppo"]["max_updates"] = cfg["ppo"].get("max_updates", old_updates)
    if rollout.pop("max_total_env_steps", None) is not None:
        raise ValueError("Legacy total environment-step budget cannot be mapped to PPO updates.")
    cfg["ppo"].pop("max_train_steps", None)  # unused with linear_lr_decay=False
    return cfg, original


def _frozen_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Operational resume path is never part of the saved run configuration."""
    result = copy.deepcopy(dict(config))
    result["checkpoint"]["resume"] = None
    return result


def _save_config(path: Path, cfg: Mapping[str, Any]) -> None:
    path.write_text(yaml.safe_dump(dict(cfg), sort_keys=False, allow_unicode=True), encoding="utf-8")


def _new_run_dir(cfg: Mapping[str, Any]) -> Path:
    template = cfg["output"].get("run_name") or "run_{timestep}"
    if not isinstance(template, str):
        raise ValueError("output.run_name must be a string or null.")
    name = template.replace("{timestep}", datetime.now().strftime("%Y%m%d_%H%M%S"))
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError("output.run_name must resolve to a single directory name.")
    run_dir = (Path(cfg["output"]["root"]).expanduser()
               / cfg["env"]["task"] / name).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir



def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_run_meta(run_dir: Path, cfg: Mapping[str, Any], il_path: str) -> None:
    from arbim.training.train_rl100_dp2d import _git_revision
    root = Path(__file__).resolve().parents[3]
    metadata = {
        "git_revision": _git_revision(root),
        "upstream_revisions": {
            "rl100": _git_revision(root / "third_party" / "rl100"),
            "duobench": _git_revision(root / "third_party" / "duobench"),
        },
        "initial_il_checkpoint": il_path,
        "initial_il_sha256": _file_sha256(il_path),
        "policy_device": cfg["policy"]["device"],
        "task": cfg["env"]["task"],
        "created_at": datetime.now().isoformat(),
    }
    (run_dir / "run_meta.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _budget_reached(cfg: Mapping[str, Any], updates: int, ppo_steps: int = 0) -> bool:
    """Stop only at complete native PPO rollout/update boundaries."""
    ppo = cfg["ppo"]
    return (ppo_steps >= int(ppo["max_steps"]) if "max_steps" in ppo
            else updates >= int(ppo["max_updates"]))


def _linear_lr_total_steps(cfg: Mapping[str, Any], action_span: int) -> int:
    """Native LR decay horizon in *action-chunk decisions*, not env steps.

    Whole episodes can overshoot the requested PPO batch. Use the largest
    full-episode buffer size implied by the configured episode cap so that
    native linear LR cannot become negative before max_updates is reached.
    Early task termination can leave a positive final LR.
    """
    if type(action_span) is not int or action_span < 1:
        raise ValueError("Effective action span must be positive.")
    episode_steps = int(cfg["env"]["max_episode_steps"])
    transitions_per_episode = math.ceil(episode_steps / action_span)
    transitions_per_update = (
        math.ceil(int(cfg["ppo"]["batch_size"]) / transitions_per_episode)
        * transitions_per_episode
    )
    ppo = cfg["ppo"]
    if "max_steps" in ppo:
        min_steps_per_update = count_ppo_steps(
            int(ppo["batch_size"]), int(ppo["mini_batch_size"]), int(ppo["K_epochs"]),
        )
        maximum_updates = math.ceil(ppo["max_steps"] / min_steps_per_update)
    else:
        maximum_updates = int(ppo["max_updates"])
    return transitions_per_update * maximum_updates


def _mean_eval_stage(folder: Path) -> float | None:
    stages = []
    for line in (folder / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line).get("final_stage")
        if isinstance(value, (int, float)) and math.isfinite(value):
            stages.append(float(value))
    return sum(stages) / len(stages) if stages else None


def run(config: dict[str, Any], *, legacy_config: dict[str, Any] | None = None) -> Path:
    from arbim.envs.duobench import DuoBenchAdapter, create_duobench_env, validate_dataset_spec
    from arbim.rollout.eval import evaluate_episodes
    from arbim.training.train_rl100_dp2d import _seed_everything, _rng_state, _restore_rng
    from arbim.policy.inference_rl100_dp2d import seed_inference

    cfg, inferred_legacy = normalize_online_config(config)
    if legacy_config is None:
        legacy_config = inferred_legacy
    validate_config(cfg)
    il_path = str(Path(cfg["policy"]["checkpoint_path"]).expanduser().resolve())
    if not Path(il_path).is_file():
        raise FileNotFoundError(f"Validated Hinge-Chest IL checkpoint not found: {il_path}")
    cfg["policy"]["checkpoint_path"] = il_path
    resume = cfg["checkpoint"].get("resume")
    resume_path = Path(resume).expanduser().resolve() if resume else None
    if resume_path is not None and resume_path.is_dir():
        resume_path = resume_path / "checkpoints" / "last.ckpt"
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"Online Resume checkpoint not found: {resume_path}")
    frozen = _frozen_config(cfg)

    seed = int(cfg["rollout"]["seed"])
    _seed_everything(seed)
    policy, spec, _ = load_inference_policy(
        il_path, device=cfg["policy"]["device"], weights=cfg["policy"].get("weights", "raw"),
        n_action_steps=cfg["policy"].get("n_action_steps"),
        num_inference_steps=cfg["policy"].get("num_inference_steps"),
    )
    validate_online_policy(policy, cfg["ppo"]["ratio_mode"])
    span = policy.n_action_steps
    # Partial final chunks are handled by variable-span GAE in the bridge.
    if cfg["ppo"]["mini_batch_size"] > cfg["ppo"]["batch_size"]:
        raise ValueError("PPO mini_batch_size > batch_size.")
    validate_dataset_spec(spec, cfg["env"]["camera_map"])

    original_payload = torch.load(il_path, map_location="cpu", weights_only=True)
    il_contract = original_payload["contract"]
    log_cfg = cfg.get("logging", {})
    ratio_cfg = log_cfg.get("ratio_diagnostics", {})
    settings = dict(cfg["ppo"])
    if settings["linear_lr_decay"]:
        settings["max_train_steps"] = _linear_lr_total_steps(cfg, span)
    settings["enable_ratio_logging"] = bool(ratio_cfg.get("enabled", False))
    settings["ratio_log_every_updates"] = int(ratio_cfg.get("every_updates", 5))
    ppo = build_online_ppo(policy, settings)
    eval_policy = ppo._policy
    evaluator = build_reward_evaluator(cfg["reward"]["mode"])
    buffer = OnlinePPOBuffer(
        spec=spec, n_obs_steps=policy.n_obs_steps,
        prediction_steps=policy.horizon - policy.n_obs_steps + 1,
        num_inference_steps=policy.num_inference_steps,
        action_dim=policy.action_dim, device=next(policy.parameters()).device,
        action_span=span,
    )

    if resume_path is not None:
        if resume_path.name != "last.ckpt" or resume_path.parent.name != "checkpoints":
            raise ValueError("Resume must use checkpoints/last.ckpt.")
        run_dir = resume_path.parent.parent
        meta_path = run_dir / "run_meta.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("initial_il_sha256") and meta["initial_il_sha256"] != _file_sha256(il_path):
                raise ValueError("Original IL checkpoint contents changed since Online run creation.")
        legacy_frozen = _frozen_config(legacy_config) if legacy_config is not None else None
        payload = read_online_checkpoint(
            resume_path, config=frozen, original_checkpoint=il_path,
            legacy_config=legacy_frozen,
        )
        updates, episodes, env_steps = restore_online_checkpoint(ppo, payload)
        decision_steps = int(payload.get("decision_steps", 0))
        ppo_steps = int(payload.get("ppo_steps", 0))
        if _budget_reached(cfg, updates, ppo_steps):
            raise ValueError("Online run already reached its configured training budget.")
        previous_config = yaml.safe_load(
            (run_dir / "config_resolved.yaml").read_text(encoding="utf-8")
        )
        pending_config = previous_config if previous_config != frozen else None
    else:
        run_dir = _new_run_dir(cfg)
        _save_config(run_dir / "config_resolved.yaml", frozen)
        _write_run_meta(run_dir, frozen, il_path)
        updates, episodes, env_steps, decision_steps, ppo_steps = 0, 0, 0, 0, 0
        pending_config = None

    if _budget_reached(cfg, updates, ppo_steps):
        raise ValueError("Online run already reached its configured training budget.")

    logging_settings = {
        "jsonl": log_cfg.get("jsonl", True),
        "tensorboard": {"enabled": log_cfg.get("tensorboard", {}).get("enabled", False)},
    }
    # Telemetry must not perturb stochastic DDIM or environment sampling.
    rng = _rng_state()
    try:
        logger = OnlineTelemetry(
            logging_settings, run_dir, resumed=resume_path is not None,
            resume_updates=updates, resume_env_steps=env_steps,
            migrate_legacy_tb=legacy_config is not None,
        )
    finally:
        _restore_rng(rng)
    logger.message(f"[ONLINE] run={run_dir}")
    logger.message("[LOG] train.log contains ARBiM summaries; native PPO tqdm/stdout is console-only.")
    logger.message(
        f"[DDIM] clip_std_min={getattr(policy.noise_scheduler, 'clip_std_min', None)} "
        f"clip_std_max={getattr(policy.noise_scheduler, 'clip_std_max', None)}"
    )
    logger.message(
        f"[INIT] actor=RL1002D H={span} K={policy.num_inference_steps} "
        f"device={cfg['policy']['device']} update={updates}/{cfg['ppo']['max_updates']} "
        f"env_steps={env_steps} decision_steps={decision_steps} ppo_steps={ppo_steps} "
        f"native_lr_horizon={ppo.args.max_train_steps if cfg['ppo']['linear_lr_decay'] else 'adaptive_kl'}"
    )

    stop_requested = False
    old_sigint = signal.getsignal(signal.SIGINT)

    def _request_stop(signum: int, frame: Any) -> None:
        nonlocal stop_requested
        if stop_requested:
            raise KeyboardInterrupt("Second Ctrl+C forced interruption.")
        stop_requested = True
        logger.message("[STOP] Requested; finish current on-policy batch/update, then checkpoint.")

    signal.signal(signal.SIGINT, _request_stop)
    environment = None
    progress = None
    run_start = time.perf_counter()
    try:
        rng = _rng_state()
        try:
            environment = create_duobench_env(cfg["env"])
        finally:
            # Reconstructing MuJoCo must not advance the resumed policy RNG.
            _restore_rng(rng)
        adapter = DuoBenchAdapter(
            environment, spec, cfg["env"]["camera_map"],
            binarize_state_gripper=cfg["env"].get("binarize_state_gripper", True),
            gripper_threshold=cfg["env"].get("gripper_threshold", 0.9),
        )
        minibatch_budget = "max_steps" in cfg["ppo"]
        total_budget = cfg["ppo"]["max_steps" if minibatch_budget else "max_updates"]
        if log_cfg.get("progress_bar", True) and sys.stderr.isatty():
            from tqdm import tqdm
            progress = tqdm(total=total_budget, initial=ppo_steps if minibatch_budget else updates,
                            desc="PPO Mini-batches" if minibatch_budget else "PPO Updates",
                            unit="step" if minibatch_budget else "update", dynamic_ncols=True)
        while not _budget_reached(cfg, updates, ppo_steps):
            batch_rollouts = []
            batch_rollout_elapsed = 0.0
            while buffer.count < cfg["ppo"]["batch_size"]:
                ppo._policy.eval()
                start = time.perf_counter()
                result = collect_online_episode(
                    adapter, ppo._policy, spec, buffer, evaluator,
                    gamma=float(cfg["ppo"]["gamma"]),
                    max_env_steps=cfg["env"]["max_episode_steps"],
                    seed=seed + episodes,
                    fix_encoder=cfg["ppo"]["fix_encoder"],
                )
                if result["transitions"] < 1:
                    raise RuntimeError("Online episode yielded no transitions.")
                episodes += 1
                env_steps += result["episode_steps"]
                elapsed = time.perf_counter() - start
                record = {
                    "event": "rollout", "updates": updates, "episodes": episodes,
                    "env_steps": env_steps, "decision_steps": decision_steps + buffer.count,
                    **result, "elapsed_s": elapsed,
                }
                logger.event(
                    "rollout", record,
                    summary=(f"[ROLL] ep={episodes} env_steps={result['episode_steps']} "
                             f"return={result['env_return']:.2f} "
                             f"stage={result['final_stage']} success={int(result['success'])}"),
                    # Individual episodes are kept in JSONL. TensorBoard uses
                    # per-PPO-update aggregates to avoid duplicate x-axis steps.
                    groups=None,
                )
                batch_rollouts.append(result)
                batch_rollout_elapsed += elapsed
                # Keep collecting complete episodes until the PPO batch is valid.
                # Stopping early can produce a one-sample advantage/std NaN.

            ppo._policy.train()
            # Native RL-100 leaves its RGB Encoder in eval mode even when
            # fix_encoder=False: eval() changes BN/dropout, not gradient flow.
            ppo._policy.obs_encoder.eval()
            update_start = time.perf_counter()
            # RL-100 native lr_decay(total_steps) expects POLICY DECISIONS.
            # One decision corresponds to one chunk transition, not 32 env steps.
            current_batch_transitions = buffer.count
            decision_steps += current_batch_transitions
            lr_steps = (
                min(decision_steps, ppo.args.max_train_steps)
                if cfg["ppo"]["linear_lr_decay"] else decision_steps
            )
            batch_ppo_steps = count_ppo_steps(
                current_batch_transitions, cfg["ppo"]["mini_batch_size"], cfg["ppo"]["K_epochs"],
            )
            metrics = update_online_ppo(ppo, buffer, total_steps=lr_steps)
            ppo_steps += batch_ppo_steps
            update_elapsed = time.perf_counter() - update_start
            ratio_metrics = native_ratio_summary(ppo) if settings["enable_ratio_logging"] else {}
            updates += 1
            if settings["enable_ratio_logging"]:
                append_native_ratio_records(ppo, run_dir, updates=updates, env_steps=env_steps)
            buffer.clear()
            ppo._policy.eval()
            ppo.set_old_policy()
            ppo._old_policy.requires_grad_(False)
            ppo._old_policy.eval()
            actor_lr = float(ppo.optimizer_actor.param_groups[0]["lr"])
            critic_lr = float(ppo.optimizer_critic.param_groups[0]["lr"])
            batch_size = len(batch_rollouts)
            batch_env_steps = sum(item["episode_steps"] for item in batch_rollouts)
            batch_stage = [float(item["final_stage"]) for item in batch_rollouts
                           if isinstance(item["final_stage"], (int, float))]
            rollout_averages = {
                "return_raw": sum(item["env_return"] for item in batch_rollouts) / batch_size,
                "return_train": sum(item["train_return"] for item in batch_rollouts) / batch_size,
                "success_rate": sum(bool(item["success"]) for item in batch_rollouts) / batch_size,
                "episode_steps": batch_env_steps / batch_size,
                "transitions": sum(item["transitions"] for item in batch_rollouts) / batch_size,
                **({"stage": sum(batch_stage) / len(batch_stage)} if batch_stage else {}),
            }
            record = {
                "event": "ppo", "updates": updates, "episodes": episodes, "env_steps": env_steps,
                "decision_steps": decision_steps, "ppo_steps": ppo_steps,
                "requested_batch_size": cfg["ppo"]["batch_size"],
                "actual_batch_size": current_batch_transitions,
                **metrics, "actor_lr": actor_lr, "critic_lr": critic_lr,
                "update_s": update_elapsed, **ratio_metrics,
            }
            logger.event(
                "ppo", record,
                summary=(f"[PPO] rollout_update={updates} ppo_steps={ppo_steps}/{total_budget if minibatch_budget else 'n/a'} transitions={current_batch_transitions} "
                         f"decisions={decision_steps} env_steps={env_steps} "
                         f"actor={metrics['actor_loss']:.4f} critic={metrics['critic_loss']:.2f} "
                         f"lr={actor_lr:.2e}/{critic_lr:.2e}"),
                groups={
                    "ppo": {"actor_loss": metrics["actor_loss"], "critic_loss": metrics["critic_loss"]},
                    "lr": {"actor": actor_lr, "critic": critic_lr},
                    "rollout": rollout_averages,
                    "progress": {"ppo_updates": updates, "episodes": episodes,
                                 "env_steps": env_steps, "decision_steps": decision_steps,
                                 "actual_batch_size": current_batch_transitions,
                                 "ppo_steps": ppo_steps},
                    "system": {
                        "rollout_s": batch_rollout_elapsed,
                        "env_steps_per_s": batch_env_steps / max(batch_rollout_elapsed, 1e-9),
                        "ppo_update_s": update_elapsed,
                    },
                    "diagnostics": ratio_metrics,
                },
            )
            finished = _budget_reached(cfg, updates, ppo_steps)
            if progress is not None:
                progress.update(batch_ppo_steps if minibatch_budget else 1)
            checkpoint_cfg = cfg["checkpoint"]
            if (
                updates % checkpoint_cfg["every_updates"] == 0
                or stop_requested
                or (finished and checkpoint_cfg.get("save_on_finish", True))
            ):
                checkpoint_dir = run_dir / "checkpoints"
                save_online_checkpoint(
                    checkpoint_dir / "last.ckpt", ppo, config=frozen, updates=updates,
                    episode_count=episodes, env_steps=env_steps, original_checkpoint=il_path,
                    decision_steps=decision_steps, ppo_steps=ppo_steps,
                )
                if pending_config is not None:
                    # Write the migrated YAML only after a valid new-format
                    # checkpoint exists. A crash before this boundary leaves
                    # the old YAML + old checkpoint resumable.
                    entry = {
                        "changed_at": datetime.now().isoformat(),
                        "at_updates": updates, "at_env_steps": env_steps,
                        "old_ppo_budget": pending_config.get("ppo", {}).get(
                            "max_updates", pending_config.get("rollout", {}).get("max_updates")
                        ),
                        "new_ppo_budget": frozen["ppo"]["max_updates"],
                        "old_logging": pending_config.get("logging"),
                        "new_logging": frozen.get("logging"),
                    }
                    with (run_dir / "config_history.jsonl").open("a", encoding="utf-8") as out:
                        out.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
                    _save_config(run_dir / "config_resolved.yaml", frozen)
                    pending_config = None
                snapshot = checkpoint_dir / f"update_{updates:06d}.ckpt"
                save_inference_snapshot(
                    snapshot, ppo, il_contract=il_contract,
                    updates=updates, env_steps=env_steps, episodes=episodes,
                )
                prune_policy_snapshots(checkpoint_dir, checkpoint_cfg.get("keep_last", 3))
                logger.event(
                    "checkpoint",
                    {"event": "checkpoint", "updates": updates, "episodes": episodes,
                     "env_steps": env_steps, "path": str(snapshot)},
                    summary=f"[SAVE] update={updates} env_steps={env_steps} last.ckpt + {snapshot.name}",
                )
            evaluation_cfg = cfg["evaluation"]
            if not stop_requested and evaluation_cfg["enabled"] and (
                updates % evaluation_cfg["every_updates"] == 0 or finished
            ):
                folder = run_dir / "evaluation" / f"update_{updates:06d}"
                eval_seed = evaluation_cfg.get("seed")
                if eval_seed is None:
                    eval_seed = seed + 100000  # constant across all PPO checkpoints
                rng = _rng_state()
                start = time.perf_counter()
                try:
                    seed_inference(eval_seed)
                    summary = evaluate_episodes(
                        adapter, eval_policy, spec,
                        num_episodes=evaluation_cfg["num_episodes"], output_dir=folder,
                        camera_map=cfg["env"]["camera_map"], execute_steps=span,
                        max_env_steps=cfg["env"]["max_episode_steps"], post_success_steps=0,
                        seed=eval_seed, deterministic=True,
                        record_video=evaluation_cfg["record_video"],
                        video_fps=evaluation_cfg["video_fps"],
                        video_source=evaluation_cfg["video_source"],
                        verbose=False,
                    )
                finally:
                    _restore_rng(rng)
                eval_elapsed = time.perf_counter() - start
                mean_stage = _mean_eval_stage(folder)
                record = {
                    "event": "evaluation", "updates": updates, "episodes": episodes,
                    "env_steps": env_steps, "seed": eval_seed,
                    **summary, "mean_stage": mean_stage, "elapsed_s": eval_elapsed,
                }
                logger.event(
                    "evaluation", record,
                    summary=(f"[EVAL] episodes={summary['num_episodes']} "
                             f"success={summary['success_rate']:.1%} "
                             f"return={summary['mean_episode_return']:.2f} "
                             f"stage={mean_stage if mean_stage is not None else 'n/a'}"),
                    groups={
                        "eval": {
                            "success_rate": summary["success_rate"],
                            "mean_return": summary["mean_episode_return"],
                            "mean_episode_steps": summary["mean_episode_steps"],
                            **({"mean_stage": mean_stage} if mean_stage is not None else {}),
                        },
                        "system": {"eval_s": eval_elapsed},
                    },
                )
            if stop_requested:
                break
        elapsed = time.perf_counter() - run_start
        logger.message(f"[DONE] rollout_updates={updates} ppo_steps={ppo_steps} env_steps={env_steps} elapsed_s={elapsed:.1f}")
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        if progress is not None:
            progress.close()
        if environment is not None:
            environment.close()
        logger.close()
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="RL100 DP-2D Online PPO on DuoBench Hinge-Chest")
    parser.add_argument("--config", type=Path, help=f"Online YAML (default: {DEFAULT_CONFIG})")
    parser.add_argument("--checkpoint", type=Path, help="Override validated Hinge-Chest IL checkpoint.")
    parser.add_argument("--device", help="Override runtime GPU/CPU device.")
    parser.add_argument("--resume", type=Path, help="Existing run directory or checkpoints/last.ckpt.")
    parser.add_argument("--output-root", type=Path, help="Output root for a new run.")
    parser.add_argument("--run-name", help="New run directory name; supports {timestep}.")
    budget_flags = parser.add_mutually_exclusive_group()
    budget_flags.add_argument("--max-steps", type=int, help="Override ppo.max_steps (PPO mini-batches).")
    budget_flags.add_argument("--max-updates", type=int, help="Legacy: override ppo.max_updates (whole PPO cycles).")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation.")
    args = parser.parse_args()
    if args.resume is not None:
        if args.config is not None or args.checkpoint is not None or args.output_root is not None or args.run_name is not None:
            parser.error("Resume uses the saved run config; no --config/--checkpoint/--output-root/--run-name.")
        resume = args.resume.expanduser().resolve()
        if resume.is_file():
            if resume.name != "last.ckpt" or resume.parent.name != "checkpoints":
                parser.error("--resume file must be checkpoints/last.ckpt")
            run_dir = resume.parent.parent
        else:
            run_dir = resume
        resume_ckpt = run_dir / "checkpoints" / "last.ckpt"
        config_path = run_dir / "config_resolved.yaml"
    else:
        resume_ckpt = None
        config_path = args.config or DEFAULT_CONFIG
    with config_path.expanduser().open(encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError("Online YAML must be a mapping.")
    config, legacy_source = normalize_online_config(loaded)
    if resume_ckpt is not None:
        config["checkpoint"]["resume"] = str(resume_ckpt)
    if args.checkpoint is not None:
        config["policy"]["checkpoint_path"] = str(args.checkpoint)
    if args.device is not None:
        config["policy"]["device"] = args.device
    if args.output_root is not None:
        config["output"]["root"] = str(args.output_root)
    if args.run_name is not None:
        config["output"]["run_name"] = args.run_name
    if args.max_steps is not None:
        config["ppo"].pop("max_updates", None)
        config["ppo"]["max_steps"] = args.max_steps
    if args.max_updates is not None:
        config["ppo"].pop("max_steps", None)
        config["ppo"]["max_updates"] = args.max_updates
    if args.no_eval:
        config["evaluation"]["enabled"] = False
    run(config, legacy_config=legacy_source)


if __name__ == "__main__":
    main()
