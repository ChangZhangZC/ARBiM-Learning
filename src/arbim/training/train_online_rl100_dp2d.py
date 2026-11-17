"""Task 6 native RL1002D Online PPO on DuoBench Hinge-Chest.

Example:
  python -m arbim.training.train_online_rl100_dp2d \
      --config configs/online/rl100_dp2d_hinge_chest.yaml \
      --checkpoint /path/to/validated/hinge_chest/best.ckpt

No Offline IQL, Q, dynamics, CM, selective-arm updates or PPO loss rewrite.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from arbim.policy.inference_rl100_dp2d import load_inference_policy
from arbim.rl.online.buffer import OnlinePPOBuffer
from arbim.rl.online.checkpoint import (
    read_online_checkpoint, restore_online_checkpoint,
    save_inference_snapshot, save_online_checkpoint,
)
from arbim.rl.online.policy_adapter import validate_online_policy
from arbim.rl.online.ppo_bridge import build_online_ppo, update_online_ppo
from arbim.rl.online.rollout import collect_online_episode
from arbim.rl.rewards.duobench_hinge_chest import build_reward_evaluator


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
    for key in ("batch_size", "mini_batch_size", "K_epochs", "value_hidden_dim", "value_depth", "max_train_steps"):
        if type(ppo.get(key)) is not int or ppo[key] < 1:
            raise ValueError(f"ppo.{key} must be a positive integer.")
    for key in ("actor_lr", "critic_lr", "min_lr_actor", "max_lr_actor", "min_lr_critic",
                "max_lr_critic", "adaptive_kl", "gamma", "gae_lambda", "epsilon"):
        value = ppo.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"ppo.{key} must be a positive finite number.")
    if ppo["gamma"] > 1 or ppo["gae_lambda"] > 1:
        raise ValueError("ppo.gamma and ppo.gae_lambda must be <= 1.")
    if ppo.get("fix_encoder") is not True or ppo.get("share_encoder") is not True:
        raise ValueError("Task 6 V1 freezes and shares the native RGB encoder.")
    if ppo.get("linear_lr_decay") is True:
        raise ValueError("Task 6 V1 keeps native adaptive KL scheduler; linear decay requires separate testing.")
    if ppo.get("value_depth", 0) < 2:
        raise ValueError("ppo.value_depth must be >= 2.")
    r = cfg["rollout"]
    for key in ("max_env_steps", "max_updates"):
        if type(r.get(key)) is not int or r[key] < 1:
            raise ValueError(f"rollout.{key} must be positive.")
    ev = cfg["evaluation"]
    for key in ("every_updates", "num_episodes", "max_env_steps"):
        if type(ev.get(key)) is not int or ev[key] < 1:
            raise ValueError(f"evaluation.{key} must be positive.")
    if type(ev.get("enabled")) is not bool:
        raise ValueError("evaluation.enabled must be boolean.")
    if type(ev.get("record_video")) is not bool:
        raise ValueError("evaluation.record_video must be boolean.")
    if (type(cfg["checkpoint"].get("every_updates")) is not int
            or cfg["checkpoint"]["every_updates"] < 1):
        raise ValueError("checkpoint.every_updates must be positive.")


def _frozen_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resume path is operational, not part of the training algorithm contract."""
    result = copy.deepcopy(dict(config))
    result["checkpoint"]["resume"] = None
    return result


def _save_config(path: Path, cfg: Mapping[str, Any]) -> None:
    path.write_text(yaml.safe_dump(dict(cfg), sort_keys=False, allow_unicode=True), encoding="utf-8")


def run(config: dict[str, Any]) -> Path:
    from arbim.envs.duobench import DuoBenchAdapter, create_duobench_env
    from arbim.rollout.eval import evaluate_episodes
    from arbim.training.train_rl100_dp2d import _seed_everything

    validate_config(config)
    cfg = copy.deepcopy(config)
    policy_cfg, rollout_cfg = cfg["policy"], cfg["rollout"]
    il_path = str(Path(policy_cfg["checkpoint_path"]).expanduser().resolve())
    if not Path(il_path).is_file():
        raise FileNotFoundError(f"Validated Hinge-Chest IL checkpoint not found: {il_path}")
    cfg["policy"]["checkpoint_path"] = il_path
    resume = cfg["checkpoint"].get("resume")
    resume_path = Path(resume).expanduser().resolve() if resume else None
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"Online Resume checkpoint not found: {resume_path}")
    frozen = _frozen_config(cfg)
    seed = int(rollout_cfg["seed"])
    _seed_everything(seed)

    policy, spec, model_cfg = load_inference_policy(
        il_path, device=policy_cfg["device"], weights=policy_cfg.get("weights", "raw"),
        n_action_steps=policy_cfg.get("n_action_steps"),
        num_inference_steps=policy_cfg.get("num_inference_steps"),
    )
    validate_online_policy(policy, cfg["ppo"]["ratio_mode"])
    span = policy.n_action_steps
    if rollout_cfg["max_env_steps"] % span:
        raise ValueError("rollout.max_env_steps must be divisible by effective n_action_steps.")
    if cfg["evaluation"]["max_env_steps"] % span:
        raise ValueError("evaluation.max_env_steps must be divisible by effective n_action_steps.")
    if cfg["ppo"]["mini_batch_size"] > cfg["ppo"]["batch_size"]:
        raise ValueError("PPO mini_batch_size > batch_size.")
    if cfg["env"].get("camera_map") is None:
        raise ValueError("DuoBench camera mapping is required.")
    from arbim.envs.duobench import validate_dataset_spec
    validate_dataset_spec(spec, cfg["env"]["camera_map"])

    # Keep the original IL snapshot contract for inference-loader compatibility.
    original_payload = torch.load(il_path, map_location="cpu", weights_only=True)
    il_contract = original_payload["contract"]
    ppo = build_online_ppo(policy, cfg["ppo"])
    eval_policy = ppo._policy
    evaluator = build_reward_evaluator(cfg["reward"]["mode"])
    buffer = OnlinePPOBuffer(
        spec=spec, n_obs_steps=policy.n_obs_steps,
        prediction_steps=policy.horizon - policy.n_obs_steps + 1,
        num_inference_steps=policy.num_inference_steps,
        action_dim=policy.action_dim, device=next(policy.parameters()).device,
    )
    if resume_path:
        if resume_path.name != "last.ckpt" or resume_path.parent.name != "checkpoints":
            raise ValueError("Resume must reference the online run's checkpoints/last.ckpt.")
        run_dir = resume_path.parent.parent
        payload = read_online_checkpoint(
            resume_path, config=frozen, original_checkpoint=il_path
        )
        updates, episodes, env_steps = restore_online_checkpoint(ppo, payload)
    else:
        name = cfg["output"].get("run_name") or (
            "online_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        )
        if not isinstance(name, str) or not name or "/" in name or "\\" in name or name in (".", ".."):
            raise ValueError("output.run_name must be a simple path component.")
        run_dir = Path(cfg["output"]["root"]).expanduser() / "hinge_chest" / name
        run_dir.mkdir(parents=True, exist_ok=False)
        _save_config(run_dir / "config_resolved.yaml", frozen)
        updates, episodes, env_steps = 0, 0, 0
    if updates >= rollout_cfg["max_updates"]:
        raise ValueError("Online run already reached rollout.max_updates.")
    metrics_path = run_dir / "metrics.jsonl"
    environment = create_duobench_env(cfg["env"])
    try:
        adapter = DuoBenchAdapter(
            environment, spec, cfg["env"]["camera_map"],
            binarize_state_gripper=cfg["env"].get("binarize_state_gripper", True),
            gripper_threshold=cfg["env"].get("gripper_threshold", 0.9),
        )
        with metrics_path.open("a", encoding="utf-8") as log:
            while updates < rollout_cfg["max_updates"]:
                # Complete episodes only: native GAE must not propagate across
                # unrelated episode boundaries or across policy versions.
                while buffer.count < cfg["ppo"]["batch_size"]:
                    ppo._policy.eval()
                    result = collect_online_episode(
                        adapter, ppo._policy, spec, buffer, evaluator,
                        gamma=float(cfg["ppo"]["gamma"]),
                        max_env_steps=rollout_cfg["max_env_steps"],
                        seed=seed + episodes,
                    )
                    if result["transitions"] < 1:
                        raise RuntimeError("Online episode yielded no transitions.")
                    episodes += 1
                    env_steps += result["episode_steps"]
                    line = {"event": "rollout", "episodes": episodes, "env_steps": env_steps, **result}
                    log.write(json.dumps(line, allow_nan=False) + "\n")
                    log.flush()
                    print("[ONLINE]", json.dumps(line, allow_nan=False), flush=True)
                ppo._policy.train()
                ppo._policy.obs_encoder.eval()
                metrics = update_online_ppo(ppo, buffer, total_steps=env_steps)
                updates += 1
                buffer.clear()
                ppo._policy.eval()
                ppo.set_old_policy()
                event = {"event": "ppo", "updates": updates, "episodes": episodes,
                         "env_steps": env_steps, **metrics}
                log.write(json.dumps(event, allow_nan=False) + "\n")
                log.flush()
                print("[ONLINE]", json.dumps(event, allow_nan=False), flush=True)

                # Persist only at empty-buffer / completed-update boundaries.
                if updates % cfg["checkpoint"]["every_updates"] == 0 or updates == rollout_cfg["max_updates"]:
                    save_online_checkpoint(
                        run_dir / "checkpoints" / "last.ckpt", ppo,
                        config=frozen, updates=updates, episode_count=episodes,
                        env_steps=env_steps, original_checkpoint=il_path,
                    )
                    save_inference_snapshot(
                        run_dir / "checkpoints" / f"step_{env_steps:08d}.ckpt",
                        ppo, il_contract=il_contract, env_steps=env_steps,
                    )
                if cfg["evaluation"]["enabled"] and (
                    updates % cfg["evaluation"]["every_updates"] == 0
                    or updates == rollout_cfg["max_updates"]
                ):
                    evaluation_dir = run_dir / "evaluation" / f"update_{updates:06d}"
                    from arbim.policy.inference_rl100_dp2d import seed_inference
                    # Keep evaluation RNG isolated from subsequent on-policy sampling.
                    from arbim.training.train_rl100_dp2d import _rng_state, _restore_rng
                    rng = _rng_state()
                    try:
                        seed_inference(seed + 100000 + updates)
                        summary = evaluate_episodes(
                            adapter, eval_policy, spec,
                            num_episodes=cfg["evaluation"]["num_episodes"],
                            output_dir=evaluation_dir,
                            camera_map=cfg["env"]["camera_map"],
                            execute_steps=span,
                            max_env_steps=cfg["evaluation"]["max_env_steps"],
                            post_success_steps=0,
                            seed=seed + 100000 + updates,
                            deterministic=True,
                            record_video=cfg["evaluation"]["record_video"],
                            video_fps=cfg["evaluation"]["video_fps"],
                            video_source=cfg["evaluation"]["video_source"],
                        )
                    finally:
                        _restore_rng(rng)
                    log.write(json.dumps(
                        {"event": "evaluation", "updates": updates, **summary},
                        allow_nan=False,
                    ) + "\n")
                    log.flush()
    finally:
        environment.close()
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Native RL100 DP-2D Online PPO on Hinge-Chest")
    parser.add_argument("--config", type=Path, default=Path("configs/online/rl100_dp2d_hinge_chest.yaml"))
    parser.add_argument("--checkpoint", type=Path, help="Override verified Hinge-Chest IL checkpoint.")
    parser.add_argument("--device", help="Override GPU device.")
    parser.add_argument("--resume", type=Path, help="Resume existing Online checkpoints/last.ckpt.")
    args = parser.parse_args()
    with args.config.expanduser().open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Online YAML must contain a mapping.")
    if args.checkpoint:
        config.setdefault("policy", {})["checkpoint_path"] = str(args.checkpoint)
    if args.device:
        config.setdefault("policy", {})["device"] = args.device
    if args.resume:
        config.setdefault("checkpoint", {})["resume"] = str(args.resume)
    result = run(config)
    print(f"[ONLINE] completed run: {result}", flush=True)


if __name__ == "__main__":
    main()
