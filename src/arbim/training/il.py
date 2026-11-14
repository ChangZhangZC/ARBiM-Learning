"""Minimal native RL-100 DP-2D imitation-learning entry for LeRobot v3."""

from __future__ import annotations

import argparse
import copy
import random
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


def _contract(cfg: dict[str, Any], dataset: Any) -> dict[str, Any]:
    keys = ("dataset", "policy", "vision", "schedulers", "optimizer", "lr_scheduler", "ema", "dataloader")
    return {
        "config": {key: copy.deepcopy(cfg[key]) for key in keys},
        "dataset_spec": copy.deepcopy(dataset.spec),
        "train_episodes": tuple(dataset._train_episodes),
        "val_episodes": tuple(dataset._val_episodes),
        "num_epochs": cfg["training"]["num_epochs"],
        "max_train_batches": cfg["training"].get("max_train_batches"),
    }


def save_checkpoint(
    path: str | Path, policy: torch.nn.Module, optimizer: torch.optim.Optimizer,
    scheduler: Any, ema: Any, *, next_epoch: int, global_step: int,
    contract: dict[str, Any],
) -> None:
    """Save an epoch-boundary ARBiM checkpoint with unmodified RL1002D keys."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 1,
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
        "ema_policy": ema.averaged_model.state_dict(),
        "ema_optimization_step": ema.optimization_step,
        "ema_decay": ema.decay,
        "next_epoch": next_epoch,
        "global_step": global_step,
        "contract": contract,
        "rng": _rng_state(),
    }, path)


def load_checkpoint(
    path: str | Path, policy: torch.nn.Module, optimizer: torch.optim.Optimizer,
    scheduler: Any, ema: Any, *, contract: dict[str, Any], device: torch.device,
) -> tuple[int, int]:
    """Strict epoch-boundary restore; never silently accept incompatible models."""
    from rl_100.common.pytorch_util import optimizer_to

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or payload.get("contract") != contract:
        raise ValueError("Checkpoint version or data/model/training contract mismatch.")
    policy.load_state_dict(payload["policy"], strict=True)
    ema.averaged_model.load_state_dict(payload["ema_policy"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    optimizer_to(optimizer, device)
    scheduler.load_state_dict(payload["lr_scheduler"])
    ema.optimization_step = int(payload["ema_optimization_step"])
    ema.decay = float(payload["ema_decay"])
    _restore_rng(payload["rng"])
    return int(payload["next_epoch"]), int(payload["global_step"])


def _loader(dataset: Any, config: dict[str, Any], shuffle: bool) -> DataLoader:
    options = dict(config)
    if options.get("num_workers", 0) == 0:
        options.pop("prefetch_factor", None)
        options["persistent_workers"] = False
    return DataLoader(dataset, shuffle=shuffle, **options)


def run(config: dict[str, Any], *, resume: str | Path | None = None) -> None:
    """Train with native RL1002D.compute_loss; no RL workspace or custom loss."""
    from rl_100.model.common.lr_scheduler import get_scheduler
    from rl_100.model.diffusion.ema_model import EMAModel
    from arbim.data.lerobot_adapter import LeRobotDP2DDataset, build_shape_meta

    train_cfg = config["training"]
    _seed_everything(int(train_cfg["seed"]))
    device = torch.device(train_cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing silent CPU fallback.")
    train_dataset = LeRobotDP2DDataset(**config["dataset"], split="train")
    val_dataset = train_dataset.get_validation_dataset() if config["dataset"]["val_ratio"] > 0 else None
    train_loader = _loader(train_dataset, config["dataloader"], shuffle=True)
    val_loader = _loader(val_dataset, config["dataloader"], shuffle=False) if val_dataset is not None else None
    max_train = train_cfg.get("max_train_batches")
    batches_per_epoch = min(len(train_loader), max_train) if max_train is not None else len(train_loader)
    if batches_per_epoch < 1:
        raise ValueError("Empty training epoch.")
    epochs = int(train_cfg["num_epochs"])
    if epochs < 1:
        raise ValueError("num_epochs must be positive.")
    if config["policy"]["n_action_steps"] != train_dataset.prediction_horizon:
        raise ValueError("n_action_steps must match horizon-n_obs_steps+1 for V1.")

    policy = build_rl100_dp(build_shape_meta(train_dataset.spec), {
        **config, "policy": {**config["policy"], "horizon": train_dataset.horizon,
                             "n_obs_steps": train_dataset.n_obs_steps},
    })
    policy.set_normalizer(train_dataset.get_normalizer(mode="limits"))
    policy.to(device)
    ema_policy = copy.deepcopy(policy)
    ema = EMAModel(model=ema_policy, **config["ema"])
    ema.averaged_model.to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), **config["optimizer"])
    lr_cfg = config["lr_scheduler"]
    scheduler = get_scheduler(
        lr_cfg["name"], optimizer=optimizer,
        num_warmup_steps=lr_cfg["warmup_steps"],
        num_training_steps=epochs * batches_per_epoch,
    )
    contract = _contract(config, train_dataset)
    start_epoch, global_step = 0, 0
    if resume is not None:
        start_epoch, global_step = load_checkpoint(
            resume, policy, optimizer, scheduler, ema, contract=contract, device=device)
    output_dir = Path(train_cfg["output_dir"])
    for epoch in range(start_epoch, epochs):
        policy.train()
        train_losses = []
        for index, batch in enumerate(train_loader):
            if index >= batches_per_epoch:
                break
            batch = prepare_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = policy.compute_loss(batch)
            if not torch.isfinite(loss).all():
                raise FloatingPointError(f"Nonfinite training loss at epoch={epoch}, batch={index}.")
            loss.backward()
            optimizer.step()
            scheduler.step()
            ema.step(policy)
            global_step += 1
            train_losses.append(loss.item())
        metrics = {"epoch": epoch + 1, "global_step": global_step, "train_loss": float(np.mean(train_losses))}
        validate_every = int(train_cfg["validate_every"])
        if val_loader is not None and validate_every > 0 and (epoch + 1) % validate_every == 0:
            policy.eval()
            val_losses = []
            max_val = train_cfg.get("max_val_batches")
            with torch.no_grad():
                for index, batch in enumerate(val_loader):
                    if max_val is not None and index >= max_val:
                        break
                    loss, _ = policy.compute_loss(prepare_batch(batch, device))
                    if not torch.isfinite(loss).all():
                        raise FloatingPointError("Nonfinite validation loss.")
                    val_losses.append(loss.item())
            if val_losses:
                metrics["val_loss"] = float(np.mean(val_losses))
        print(metrics, flush=True)
        checkpoint_every = int(train_cfg["checkpoint_every"])
        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            save_checkpoint(
                output_dir / "latest.ckpt", policy, optimizer, scheduler, ema,
                next_epoch=epoch + 1, global_step=global_step, contract=contract,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Native RL100 DP-2D IL training")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    run(config, resume=args.resume)


if __name__ == "__main__":
    main()
