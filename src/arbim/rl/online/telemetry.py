"""Durable Online PPO events and local TensorBoard using the IL scalar backend.

Online JSONL intentionally keeps its existing event/update/env_steps format.
TensorBoard alone uses the generic IL MetricLogger, with its JSONL sink off.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

from arbim.training.telemetry import MetricLogger


def reconcile_online_metrics(path: Path, *, updates: int, env_steps: int) -> None:
    """Move records newer than the durable PPO boundary out of active metrics.

    An interrupted rollout may have appended an event without a checkpoint.
    Keep these records separately for forensic analysis instead of discarding them.
    """
    if not path.is_file():
        return
    kept: list[str] = []
    uncommitted: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        # Old evaluation entries may not carry env_steps. Use update counter.
        newer = (
            int(record.get("env_steps", -1)) > env_steps
            or int(record.get("updates", -1)) > updates
        )
        (uncommitted if newer else kept).append(line)
    if not uncommitted:
        return
    archive = path.with_name("metrics.uncommitted.jsonl")
    with archive.open("a", encoding="utf-8") as out:
        out.write("\n".join(uncommitted) + "\n")
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")



def reconcile_native_ratio_csv(
    path: Path, *, updates: int, env_steps: int,
) -> None:
    """Archive ratio statistics belonging to updates newer than the checkpoint."""
    if not path.is_file():
        return
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        if not fieldnames:
            return
        rows = list(reader)
    kept = [
        row for row in rows
        if int(row["update"]) <= updates and int(row["env_steps"]) <= env_steps
    ]
    discarded = [
        row for row in rows
        if int(row["update"]) > updates or int(row["env_steps"]) > env_steps
    ]
    if not discarded:
        return
    archive = path.with_name("ratio.uncommitted.csv")
    with archive.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerows(discarded)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)


class OnlineTelemetry:
    def __init__(
        self, cfg: Mapping[str, Any], run_dir: Path, *,
        resumed: bool, resume_updates: int, resume_env_steps: int,
        migrate_legacy_tb: bool = False,
    ) -> None:
        self.run_dir = run_dir
        self.metrics = run_dir / "metrics.jsonl"
        self.console = run_dir / "train.log"
        self.jsonl_enabled = bool(cfg.get("jsonl", True))
        if resumed:
            reconcile_online_metrics(
                self.metrics, updates=resume_updates, env_steps=resume_env_steps
            )
            reconcile_native_ratio_csv(
                run_dir / "diagnostics" / "ratio_stats.csv",
                updates=resume_updates, env_steps=resume_env_steps,
            )
        # Legacy Task 7 events use env_steps as their TensorBoard x-axis.
        # Never silently mix them with PPO-update indexed events.
        tb_dir = (
            run_dir / "tensorboard_ppo_updates"
            if migrate_legacy_tb or (run_dir / "tensorboard_ppo_updates").exists()
            else run_dir / "tensorboard"
        )
        tb_cfg = dict(cfg["tensorboard"])
        tb_cfg["log_dir"] = str(tb_dir)
        logging_cfg = {
            "logging": {
                "jsonl": False,
                "tensorboard": tb_cfg,
                "wandb": {"enabled": False},
            }
        }
        self.sink = MetricLogger(
            logging_cfg, run_dir, resumed=resumed, resume_step=resume_updates
        )
        if resumed:
            self.message(f"[RESUME] update={resume_updates} env_steps={resume_env_steps}")

    def message(self, message: str) -> None:
        print(message, flush=True)
        with self.console.open("a", encoding="utf-8") as out:
            out.write(message + "\n")

    def event(
        self, phase: str, record: Mapping[str, Any], *,
        summary: str, groups: Mapping[str, Mapping[str, float | int]] | None = None,
    ) -> None:
        if self.jsonl_enabled:
            with self.metrics.open("a", encoding="utf-8") as out:
                out.write(json.dumps(dict(record), ensure_ascii=False, allow_nan=False) + "\n")
        if groups:
            step = int(record["updates"])  # primary Online training axis is PPO updates
            for group, values in groups.items():
                scalars = {
                    name: value for name, value in values.items()
                    if isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                }
                if scalars:
                    self.sink.log(group, step, **scalars)
        self.message(summary)

    def close(self) -> None:
        self.sink.close()


def native_ratio_summary(ppo: Any) -> dict[str, float]:
    """Read already-collected upstream diagnostics; never recompute PPO ratios."""
    records = getattr(ppo, "_ratio_records", ())
    if not records:
        return {}
    keys = ("ratio_mean", "ratio_q05", "ratio_q50", "ratio_q95", "delta_mean")
    return {
        key: sum(float(row[key]) for row in records if key in row)
        / sum(key in row for row in records)
        for key in keys if any(key in row for row in records)
    }

def append_native_ratio_records(
    ppo: Any, run_dir: Path, *, updates: int, env_steps: int,
) -> None:
    """Persist upstream diagnostic records without its CSV-overwrite behavior.

    The pinned native flush_ratio_logs() opens CSV with mode='w' after each
    internal prune; use an append-only external sink while retaining native
    ratio calculations exactly as recorded.
    """
    rows = getattr(ppo, "_ratio_records", None)
    if not rows:
        return
    folder = run_dir / "diagnostics"
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / "ratio_stats.csv"
    keys = ["update", "env_steps", *rows[0].keys()]
    with csv_path.open("a", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=keys)
        if out.tell() == 0:
            writer.writeheader()
        writer.writerows(
            {"update": updates, "env_steps": env_steps, **row} for row in rows
        )
    rows.clear()
    delta_cache = getattr(ppo, "_delta_cache", None)
    if isinstance(delta_cache, list):
        delta_cache.clear()
