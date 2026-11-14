"""Training-loop orchestration only; leaves the native RL-100 policy untouched."""

from __future__ import annotations

import copy
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Sampler


def _positive_or_none(value: Any, key: str) -> int | None:
    if value is not None and (type(value) is not int or value <= 0):
        raise ValueError(f"{key} must be a positive integer or null.")
    return value



_DATALOADER_FIELDS = ("batch_size", "num_workers", "pin_memory",
                      "persistent_workers", "prefetch_factor")


def loader_options(cfg: dict[str, Any]) -> dict[str, Any]:
    """Read DataLoader arguments from the canonical training settings."""
    return {key: cfg["training"][key] for key in _DATALOADER_FIELDS
            if key in cfg["training"]}


def normalize_config(cfg: dict[str, Any], *, resume: bool) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Support strict Resume of pre-restructure v2 runs, not legacy fresh runs.

    Original YAML/contract are preserved for existing run directories; the new
    schema alone is used when creating a fresh experiment.
    """
    if not isinstance(cfg, dict):
        raise ValueError("Training configuration must be a YAML mapping.")
    legacy = ("experiment" in cfg or "dataloader" in cfg
              or "horizon" in cfg.get("dataset", {})
              or "n_obs_steps" in cfg.get("dataset", {}))
    if not legacy:
        return cfg, None
    if not resume:
        raise ValueError("Legacy configuration: move experiment/dataloader to training "
                         "and dataset.horizon/n_obs_steps to policy.")
    original = copy.deepcopy(cfg)
    if not (isinstance(original.get("experiment"), dict)
            and isinstance(original.get("dataloader"), dict)
            and "horizon" in original["dataset"]
            and "n_obs_steps" in original["dataset"]):
        raise ValueError("Incomplete legacy run configuration; cannot safely resume.")
    canonical = copy.deepcopy(original)
    canonical["policy"]["horizon"] = canonical["dataset"].pop("horizon")
    canonical["policy"]["n_obs_steps"] = canonical["dataset"].pop("n_obs_steps")
    canonical["training"].update(canonical.pop("experiment"))
    canonical["training"].update(canonical.pop("dataloader"))
    return canonical, original


def check_config(cfg: dict[str, Any]) -> None:
    """Validate the cleaned schema before any dataset/model construction."""
    for key in ("dataset", "policy", "vision", "schedulers", "optimizer",
                "lr_scheduler", "ema", "training", "output", "checkpoint",
                "validation", "logging"):
        if key not in cfg or not isinstance(cfg[key], dict):
            raise ValueError(f"Missing configuration mapping: {key}")
    if "experiment" in cfg or "dataloader" in cfg:
        raise ValueError("Use training.task/batch_size instead of experiment/dataloader.")
    tr, policy = cfg["training"], cfg["policy"]
    for old in ("num_epochs", "max_train_batches", "max_val_batches",
                "validate_every", "checkpoint_every"):
        if old in tr:
            raise ValueError(f"Legacy training.{old} is unsupported.")
    steps = _positive_or_none(tr.get("max_steps"), "training.max_steps")
    epochs = _positive_or_none(tr.get("max_epochs"), "training.max_epochs")
    if steps is None and epochs is None:
        raise ValueError("Set training.max_steps and/or training.max_epochs.")
    if _positive_or_none(tr.get("batch_size"), "training.batch_size") is None:
        raise ValueError("training.batch_size is required.")
    if "drop_last" in tr:
        raise ValueError("Do not use training.drop_last; a full epoch includes all eligible windows.")
    workers = tr.get("num_workers", 0)
    if type(workers) is not int or workers < 0:
        raise ValueError("training.num_workers must be a nonnegative integer.")
    if workers == 0 and tr.get("persistent_workers"):
        raise ValueError("persistent_workers requires num_workers > 0.")
    if workers == 0 and tr.get("prefetch_factor") is not None:
        raise ValueError("prefetch_factor requires num_workers > 0.")
    for name in ("every_steps", "every_epochs"):
        for section in ("checkpoint", "validation"):
            _positive_or_none(cfg[section].get(name), f"{section}.{name}")
    if _positive_or_none(cfg["logging"].get("log_freq_steps"), "logging.log_freq_steps") is None:
        raise ValueError("logging.log_freq_steps is required.")
    if _positive_or_none(cfg["checkpoint"].get("keep_last"), "checkpoint.keep_last") is None:
        raise ValueError("checkpoint.keep_last is required.")
    for section, field in (("logging", "progress_bar"), ("logging", "jsonl"),
                           ("validation", "on_finish"), ("checkpoint", "enabled"),
                           ("checkpoint", "save_on_finish")):
        if not isinstance(cfg[section].get(field), bool):
            raise ValueError(f"{section}.{field} must be boolean.")
    if not isinstance(cfg["validation"].get("seed"), int):
        raise ValueError("validation.seed must be integer.")
    if (cfg["checkpoint"]["enabled"] and not cfg["checkpoint"]["save_on_finish"]
        and cfg["checkpoint"].get("every_steps") is None
        and cfg["checkpoint"].get("every_epochs") is None):
        raise ValueError("Checkpoint enabled but all save triggers are disabled.")
    for section in ("tensorboard", "wandb"):
        item = cfg["logging"].get(section)
        if not isinstance(item, dict) or not isinstance(item.get("enabled"), bool):
            raise ValueError(f"logging.{section}.enabled must be boolean.")
    if cfg["logging"]["wandb"]["enabled"]:
        if not cfg["logging"]["wandb"].get("project"):
            raise ValueError("Enabled WandB requires logging.wandb.project.")
        if cfg["logging"]["wandb"].get("mode", "online") not in ("online", "offline", "disabled"):
            raise ValueError("logging.wandb.mode must be online, offline or disabled.")
    for key in ("task", "method"):
        value = tr.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", value):
            raise ValueError(f"training.{key} must be a safe path component.")
    run_name = tr.get("run_name")
    if run_name is not None and (
        not isinstance(run_name, str)
        or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", run_name)
    ):
        raise ValueError("training.run_name must be null or a safe path component.")
    if not isinstance(policy.get("horizon"), int) or not isinstance(policy.get("n_obs_steps"), int):
        raise ValueError("policy.horizon and policy.n_obs_steps must be positive integers.")
    if policy["n_obs_steps"] < 1 or policy["horizon"] < policy["n_obs_steps"]:
        raise ValueError("policy.horizon must be >= policy.n_obs_steps >= 1.")
    if policy["n_action_steps"] != policy["horizon"] - policy["n_obs_steps"] + 1:
        raise ValueError("policy.n_action_steps must equal horizon-n_obs_steps+1 for current V1.")
    seed = tr.get("seed")
    if type(seed) is not int or seed < 0:
        raise ValueError("training.seed must be nonnegative integer.")
    if cfg["lr_scheduler"]["warmup_steps"] < 0:
        raise ValueError("lr_scheduler.warmup_steps must be nonnegative.")
    if cfg["output"].get("output_dir") is not None and not str(cfg["output"]["output_dir"]).strip():
        raise ValueError("output.output_dir must be null or a nonempty path.")
    if cfg["output"].get("output_dir") is None and not cfg["output"].get("root"):
        raise ValueError("output.root is required when output.output_dir is null.")


def resolve_training_budget(cfg: dict[str, Any], train_windows: int) -> tuple[int, int]:
    """Return (effective optimizer steps, steps per full epoch)."""
    batch_size = cfg["training"]["batch_size"]
    if train_windows < 1:
        raise ValueError("No eligible training windows.")
    steps_per_epoch = math.ceil(train_windows / batch_size)
    tr = cfg["training"]
    caps = [n for n in (
        tr["max_steps"],
        tr["max_epochs"] * steps_per_epoch if tr["max_epochs"] is not None else None,
    ) if n is not None]
    return min(caps), steps_per_epoch


class StepBatchSampler(Sampler[list[int]]):
    """Address each batch by global optimizer step, independently of prefetch.

    Epoch i uses a private CPU permutation seeded with seed+i. Workers cannot
    perturb sample indices. Stochastic worker-side transforms are not bit-exact
    on resume; for strict equality use deterministic dataset preprocessing.
    """

    def __init__(self, n_samples: int, batch_size: int, seed: int,
                 start_step: int, stop_step: int):
        if n_samples < 1 or batch_size < 1 or start_step < 0 or stop_step < start_step:
            raise ValueError("Invalid StepBatchSampler range.")
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.seed = seed
        self.start_step = start_step
        self.stop_step = stop_step
        self.steps_per_epoch = math.ceil(n_samples / batch_size)

    def __iter__(self):
        last_epoch, order = -1, None
        for zero_step in range(self.start_step, self.stop_step):
            epoch, batch_index = divmod(zero_step, self.steps_per_epoch)
            if epoch != last_epoch:
                generator = torch.Generator().manual_seed(self.seed + epoch)
                order = torch.randperm(self.n_samples, generator=generator)
                last_epoch = epoch
            offset = batch_index * self.batch_size
            yield order[offset:offset + self.batch_size].tolist()

    def __len__(self) -> int:
        return self.stop_step - self.start_step


def resolve_run_directory(cfg: dict[str, Any], resume: str | Path | None) -> Path:
    if resume is not None:
        resume_path = Path(resume).expanduser().resolve()
        if resume_path.is_file() and (resume_path.name != "last.ckpt" or
                                      resume_path.parent.name != "checkpoints"):
            raise ValueError("Resume file must be checkpoints/last.ckpt; snapshots are not auto-selected.")
        run_dir = resume_path.parent.parent if resume_path.is_file() else resume_path
        if not (run_dir / "checkpoints" / "last.ckpt").is_file():
            raise FileNotFoundError(f"No resumable checkpoints/last.ckpt under {run_dir}")
        return run_dir
    output = cfg["output"]
    if output.get("output_dir") is not None:
        run_dir = Path(output["output_dir"])
    else:
        prefix = Path(output["root"]) / cfg["training"]["task"] / cfg["training"]["method"]
        run_name = cfg["training"].get("run_name") or datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = prefix / run_name
    run_dir = run_dir.expanduser().resolve()
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists; choose a new name or use --resume: {run_dir}")
    return run_dir


def is_due(config: dict[str, Any], step: int, steps_per_epoch: int) -> bool:
    """True when a configured step or completed-epoch interval is reached."""
    by_steps = config.get("every_steps")
    by_epochs = config.get("every_epochs")
    return bool((by_steps and step % by_steps == 0) or
                (by_epochs and step % steps_per_epoch == 0 and
                 (step // steps_per_epoch) % by_epochs == 0))
