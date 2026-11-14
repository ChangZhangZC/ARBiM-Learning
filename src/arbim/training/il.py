"""Minimal native RL-100 DP-2D imitation-learning entry for LeRobot v3."""

from __future__ import annotations

import argparse
import copy
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from arbim.policy.rl100_dp import build_rl100_dp


def prepare_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensors and scale uint8 RGB; native RL1002D normalizes state/actions."""
    obs = {}
    for key, value in batch["obs"].items():
        if not torch.is_tensor(value):
            raise TypeError(f"Expected tensor observation {key!r}.")
        value = value.to(device, non_blocking=True)
        if key == "agent_pos":
            obs[key] = value.float()
        elif value.dtype == torch.uint8:
            obs[key] = value.float().div_(255.0)
        elif value.is_floating_point():
            obs[key] = value.float()
        else:
            raise TypeError(f"Unsupported RGB dtype for {key!r}: {value.dtype}.")
    action = batch["action"].to(device, non_blocking=True).float()
    is_pad = batch["action_is_pad"].to(device, non_blocking=True).bool()
    if action.ndim != 3 or is_pad.shape != action.shape[:2]:
        raise ValueError("Expected action [B,H,D] and action_is_pad [B,H].")
    obs_steps = obs["agent_pos"].shape[1]
    if is_pad[:, obs_steps - 1 :].any():
        raise ValueError("Native RL1002D loss does not mask padded supervised actions.")
    return {"obs": obs, "action": action, "action_is_pad": is_pad}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _rng_state() -> dict[str, Any]:
    n = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (n[0], torch.from_numpy(n[1].astype(np.int64)), n[2], n[3], n[4]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, keys.numpy().astype(np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])



def _contract(cfg: dict[str, Any], dataset: Any, total_steps: int,
              steps_per_epoch: int) -> dict[str, Any]:
    """Freeze every input that can change model updates or sampling order."""
    return {
        "dataset": copy.deepcopy(cfg["dataset"]),
        "dataset_spec": copy.deepcopy(dataset.spec),
        "train_episodes": tuple(dataset._train_episodes),
        "val_episodes": tuple(dataset._val_episodes),
        "policy": copy.deepcopy(cfg["policy"]),
        "vision": copy.deepcopy(cfg["vision"]),
        "schedulers": copy.deepcopy(cfg["schedulers"]),
        "optimizer": copy.deepcopy(cfg["optimizer"]),
        "lr_scheduler": copy.deepcopy(cfg["lr_scheduler"]),
        "ema": copy.deepcopy(cfg["ema"]),
        "dataloader": copy.deepcopy(cfg["dataloader"]),
        "validation_seed": cfg["validation"]["seed"],
        "seed": cfg["training"]["seed"],
        "max_steps": cfg["training"]["max_steps"],
        "max_epochs": cfg["training"]["max_epochs"],
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
    }


def save_checkpoint(
    path: str | Path, policy: torch.nn.Module, optimizer: torch.optim.Optimizer,
    scheduler: Any, ema: Any, *, global_step: int,
    contract: dict[str, Any], best_val_loss: float | None,
    best_step: int | None,
) -> None:
    """Atomically persist the native model and every resumable training state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save({
            "format_version": 2,
            "checkpoint_role": "training",
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict(),
            "ema_policy": ema.averaged_model.state_dict(),
            "ema_optimization_step": ema.optimization_step,
            "ema_decay": ema.decay,
            "global_step": global_step,
            "completed_epochs": global_step // contract["steps_per_epoch"],
            "batch_offset": global_step % contract["steps_per_epoch"],
            "best_val_loss": best_val_loss,
            "best_step": best_step,
            "contract": contract,
            "rng": _rng_state(),
        }, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path, policy: torch.nn.Module, optimizer: torch.optim.Optimizer,
    scheduler: Any, ema: Any, *, contract: dict[str, Any], device: torch.device,
) -> tuple[int, float | None, int | None]:
    """Strictly restore training at a completed optimizer step."""
    from rl_100.common.pytorch_util import optimizer_to

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 2 or payload.get("checkpoint_role") != "training":
        raise ValueError("Not a version-2 resumable training checkpoint.")
    if payload.get("contract") != contract:
        raise ValueError("Checkpoint contract mismatch: data, model or training schedule changed.")
    step = payload["global_step"]
    if type(step) is not int or not 0 <= step <= contract["total_steps"]:
        raise ValueError("Invalid checkpoint global_step.")
    if (payload["completed_epochs"], payload["batch_offset"]) != divmod(
        step, contract["steps_per_epoch"]
    ):
        raise ValueError("Checkpoint sampler cursor mismatch.")
    policy.load_state_dict(payload["policy"], strict=True)
    ema.averaged_model.load_state_dict(payload["ema_policy"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    optimizer_to(optimizer, device)
    scheduler.load_state_dict(payload["lr_scheduler"])
    ema.optimization_step = int(payload["ema_optimization_step"])
    ema.decay = float(payload["ema_decay"])
    _restore_rng(payload["rng"])
    return step, payload["best_val_loss"], payload["best_step"]


@contextmanager
def _validation_without_random_shifts(policy: Any):
    """Native compute_loss checks use_aug even in eval mode."""
    old_value = policy.use_aug
    policy.use_aug = False
    try:
        yield
    finally:
        policy.use_aug = old_value


def _loader(dataset: Any, cfg: dict[str, Any], *, seed: int,
            batch_sampler: Any = None) -> DataLoader:
    options = dict(cfg)
    if options.get("num_workers", 0) == 0:
        options.pop("prefetch_factor", None)
        options["persistent_workers"] = False
    generator = torch.Generator().manual_seed(seed)
    if batch_sampler is not None:
        options.pop("batch_size", None)
        return DataLoader(dataset, batch_sampler=batch_sampler, generator=generator, **options)
    return DataLoader(dataset, shuffle=False, generator=generator, **options)


def _validate(policy: Any, loader: DataLoader, device: torch.device, seed: int) -> float:
    """Full held-out validation with fixed diffusion noise and isolated RNG."""
    rng = _rng_state()
    was_training = policy.training
    total_loss = 0.0
    num_samples = 0
    try:
        _seed_everything(seed)
        policy.eval()
        with torch.no_grad(), _validation_without_random_shifts(policy):
            for sample in loader:
                ready = prepare_batch(sample, device)
                loss, _ = policy.compute_loss(ready)
                if not torch.isfinite(loss).all():
                    raise FloatingPointError("Nonfinite validation loss.")
                n = ready["action"].shape[0]
                total_loss += float(loss.item()) * n
                num_samples += n
    finally:
        policy.train(was_training)
        _restore_rng(rng)
    if num_samples == 0:
        raise ValueError("Validation dataset produced no batches.")
    return total_loss / num_samples


def _save_best(path: Path, policy: Any, step: int, loss: float,
               contract: dict[str, Any]) -> None:
    """Inference-oriented model-only checkpoint; never used for Resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save({
            "format_version": 2,
            "checkpoint_role": "best_policy",
            "policy": policy.state_dict(),
            "global_step": step,
            "val_loss": loss,
            "contract": contract,
            "selected_weights": "raw",
        }, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint_periodic(run_dir: Path, cfg: dict[str, Any], step: int,
                         policy: Any, optimizer: Any, scheduler: Any, ema: Any,
                         contract: dict[str, Any], best_loss: float | None,
                         best_step: int | None) -> None:
    """Always update last; keep at most N archived step snapshots."""
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        checkpoint_dir / "last.ckpt", policy, optimizer, scheduler, ema,
        global_step=step, contract=contract,
        best_val_loss=best_loss, best_step=best_step,
    )
    # Snapshot is a hard link to the one atomic serialization, when supported.
    # Each archived snapshot stays independent when last.ckpt is replaced.
    import os
    import shutil
    snapshot = checkpoint_dir / f"step_{step:08d}.ckpt"
    if snapshot.exists():
        raise FileExistsError(f"Refusing to overwrite existing step snapshot: {snapshot}")
    try:
        os.link(checkpoint_dir / "last.ckpt", snapshot)
    except OSError:
        temporary = snapshot.with_name(snapshot.name + ".tmp")
        try:
            shutil.copy2(checkpoint_dir / "last.ckpt", temporary)
            temporary.replace(snapshot)
        finally:
            temporary.unlink(missing_ok=True)
    files = sorted(checkpoint_dir.glob("step_*.ckpt"))
    for old in files[:-cfg["checkpoint"]["keep_last"]]:
        old.unlink()


def _git_revision(directory: Path) -> str | None:
    import subprocess
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _write_run_metadata(run_dir: Path, cfg: dict[str, Any],
                        dataset: Any, contract: dict[str, Any]) -> None:
    import json

    root = Path(__file__).resolve().parents[3]
    metadata = {
        "git_revision": _git_revision(root),
        "upstream_revisions": {
            "rl100": _git_revision(root / "third_party" / "rl100"),
            "lerobot": _git_revision(root / "third_party" / "lerobot"),
        },
        "dataset_spec": dataset.spec,
        "train_episodes": list(dataset._train_episodes),
        "val_episodes": list(dataset._val_episodes),
        "preprocessor": {
            "camera_key_mapping": dataset.spec["camera_obs_keys"],
            "rgb_transport": "uint8 -> float32 [0,1] in prepare_batch",
            "state_action_normalizer": "native RL1002D.LinearNormalizer (limits)",
            "rgb_resize_crop_imagenet": "native RL100 MultiImageObsEncoder",
        },
        "postprocessor": {
            "action_unnormalization": "native RL1002D.predict_action",
            "robot_control_mapping": "environment adapter; not part of IL training",
        },
        "total_steps": contract["total_steps"],
        "steps_per_epoch": contract["steps_per_epoch"],
    }
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (run_dir / "run_meta.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run(config: dict[str, Any], *, resume: str | Path | None = None,
        stop_after: int | None = None) -> Path:
    """Native RL1002D.compute_loss, with reproducible step-addressable orchestration."""
    import sys
    import time
    from datetime import timedelta
    from tqdm import tqdm

    from rl_100.model.common.lr_scheduler import get_scheduler
    from rl_100.model.diffusion.ema_model import EMAModel
    from arbim.data.lerobot_adapter import LeRobotDP2DDataset, build_shape_meta
    from arbim.training.run_utils import (
        StepBatchSampler, check_config, is_due, resolve_run_directory,
        resolve_training_budget,
    )
    from arbim.training.telemetry import MetricLogger

    check_config(config)
    train_cfg = config["training"]
    run_dir = resolve_run_directory(config, resume)
    device = torch.device(train_cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing silent CPU fallback.")
    _seed_everything(int(train_cfg["seed"]))
    train_dataset = LeRobotDP2DDataset(**config["dataset"], split="train")
    val_dataset = (train_dataset.get_validation_dataset()
                   if config["dataset"]["val_ratio"] > 0 else None)
    total_steps, steps_per_epoch = resolve_training_budget(config, len(train_dataset))
    contract = _contract(config, train_dataset, total_steps, steps_per_epoch)
    if stop_after is not None and (type(stop_after) is not int or
                                  not 1 <= stop_after <= total_steps):
        raise ValueError("stop_after must be within the fixed training budget.")
    target_step = stop_after if stop_after is not None else total_steps
    policy = build_rl100_dp(build_shape_meta(train_dataset.spec), {
        **config,
        "policy": {**config["policy"], "horizon": train_dataset.horizon,
                   "n_obs_steps": train_dataset.n_obs_steps},
    })
    # Register native normalizer parameters before creating optimizer and EMA.
    policy.set_normalizer(train_dataset.get_normalizer(mode="limits"))
    policy.to(device)
    ema = EMAModel(model=copy.deepcopy(policy), **config["ema"])
    ema.averaged_model.to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), **config["optimizer"])
    scheduler = get_scheduler(
        config["lr_scheduler"]["name"], optimizer=optimizer,
        num_warmup_steps=config["lr_scheduler"]["warmup_steps"],
        num_training_steps=total_steps,
    )
    start_step, best_val_loss, best_step = 0, None, None
    if resume is not None:
        start_step, best_val_loss, best_step = load_checkpoint(
            run_dir / "checkpoints" / "last.ckpt",
            policy, optimizer, scheduler, ema, contract=contract, device=device,
        )
        if best_step is not None and not (run_dir / "checkpoints" / "best.ckpt").is_file():
            raise FileNotFoundError("Resume checkpoint refers to a missing best.ckpt")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        _write_run_metadata(run_dir, config, train_dataset, contract)

    sampler = StepBatchSampler(
        len(train_dataset), config["dataloader"]["batch_size"],
        int(train_cfg["seed"]), start_step, target_step,
    )
    train_loader = _loader(
        train_dataset, config["dataloader"], seed=int(train_cfg["seed"]),
        batch_sampler=sampler,
    )
    val_loader = (_loader(
        val_dataset, config["dataloader"],
        seed=int(config["validation"]["seed"]),
    ) if val_dataset is not None else None)
    # Do not let telemetry imports/initialization change training noise.
    rng = _rng_state()
    try:
        logger = MetricLogger(config, run_dir, resumed=resume is not None,
                              resume_step=start_step)
    finally:
        _restore_rng(rng)

    print(f"[IL] run={run_dir}", flush=True)
    print(f"[IL] train_windows={len(train_dataset)} steps_per_epoch={steps_per_epoch} "
          f"budget={total_steps} resume_step={start_step}", flush=True)
    checkpoint_cfg = config["checkpoint"]
    validation_cfg = config["validation"]
    log_freq = config["logging"]["log_freq_steps"]
    run_start = time.perf_counter()
    recent_loss, recent_count = 0.0, 0
    previous_step_end = run_start
    try:
        policy.train()
        with tqdm(
            total=target_step - start_step, desc="RL100 IL", unit="step",
            dynamic_ncols=True,
            disable=not (config["logging"]["progress_bar"] and sys.stderr.isatty()),
        ) as progress:
            for step, sample in enumerate(train_loader, start=start_step + 1):
                read_time = time.perf_counter() - previous_step_end
                ready = prepare_batch(sample, device)
                update_start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                loss, _ = policy.compute_loss(ready)
                if not torch.isfinite(loss).all():
                    raise FloatingPointError(f"Nonfinite train loss at step {step}.")
                loss.backward()
                optimizer.step()
                scheduler.step()
                ema.step(policy)
                update_time = time.perf_counter() - update_start
                progress.update(1)
                recent_loss += float(loss.item())
                recent_count += 1
                epoch, offset = divmod(step, steps_per_epoch)
                epoch_fraction = step / steps_per_epoch
                if step % log_freq == 0 or step == total_steps:
                    elapsed = time.perf_counter() - run_start
                    step_speed = (step - start_step) / elapsed if elapsed > 0 else 0
                    eta = (total_steps - step) / step_speed if step_speed > 0 else 0
                    metrics = {
                        "loss": recent_loss / recent_count,
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "epoch_equivalent": epoch_fraction,
                        "completed_epochs": epoch,
                        "epoch_batch_offset": offset,
                        "elapsed_s": elapsed,
                        "eta_s": eta,
                        "steps_per_s": step_speed,
                        "data_s": read_time,
                        "update_s": update_time,
                    }
                    logger.log("train", step, **metrics)
                    progress.set_postfix(
                        loss=f"{metrics['loss']:.4f}",
                        lr=f"{metrics['lr']:.1e}", epoch=f"{epoch_fraction:.2f}",
                    )
                    tqdm.write(
                        f"[TRAIN] step={step}/{total_steps} epoch={epoch_fraction:.2f} "
                        f"loss={metrics['loss']:.6f} lr={metrics['lr']:.2e} "
                        f"speed={step_speed:.2f}step/s "
                        f"ETA={timedelta(seconds=int(eta))}"
                    )
                    recent_loss, recent_count = 0.0, 0

                val_due = (
                    val_loader is not None and
                    (is_due(validation_cfg, step, steps_per_epoch) or
                     (step == total_steps and validation_cfg.get("on_finish", True)))
                )
                if val_due:
                    val_start = time.perf_counter()
                    val_loss = _validate(
                        policy, val_loader, device, int(validation_cfg["seed"])
                    )
                    logger.log("val", step, loss=val_loss,
                               elapsed_s=time.perf_counter() - val_start)
                    tqdm.write(f"[VAL] step={step} loss={val_loss:.6f}")
                    if best_val_loss is None or val_loss < best_val_loss:
                        best_val_loss, best_step = val_loss, step
                        if checkpoint_cfg["enabled"]:
                            _save_best(
                                run_dir / "checkpoints" / "best.ckpt",
                                policy, step, val_loss, contract,
                            )
                    # The training loop always returns to its pre-validation mode.
                save_due = (checkpoint_cfg["enabled"] and
                            (is_due(checkpoint_cfg, step, steps_per_epoch) or
                             (step == total_steps and checkpoint_cfg["save_on_finish"]) or
                             (stop_after is not None and step == target_step)))
                if save_due:
                    _checkpoint_periodic(
                        run_dir, config, step, policy, optimizer, scheduler, ema,
                        contract, best_val_loss, best_step,
                    )
                    tqdm.write(f"[SAVE] step={step} -> checkpoints/last.ckpt")
                previous_step_end = time.perf_counter()
        if start_step >= total_steps:
            print(f"[IL] Training already finished at step={start_step}.", flush=True)
    finally:
        logger.close()
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Native RL-100 DP-2D IL training")
    parser.add_argument("--config", type=Path,
                        help="YAML profile for a new run; optional on Resume.")
    parser.add_argument("--resume", type=Path,
                        help="Existing run directory (loads checkpoints/last.ckpt).")
    args = parser.parse_args()
    if args.config is None and args.resume is None:
        parser.error("Either --config or --resume is required.")
    if args.config is not None:
        config_path = args.config
    else:
        # Read original configuration before any dataset/model setup.
        run_dir = Path(args.resume).expanduser().resolve()
        if run_dir.is_file():
            if run_dir.name != "last.ckpt" or run_dir.parent.name != "checkpoints":
                parser.error("--resume file must be checkpoints/last.ckpt")
            run_dir = run_dir.parent.parent
        config_path = run_dir / "config_resolved.yaml"
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    run(config, resume=args.resume)


if __name__ == "__main__":
    main()
