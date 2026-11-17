"""Optional metric sinks. Neither model nor training RNG depends on logging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MetricLogger:
    def __init__(self, cfg: dict[str, Any], run_dir: Path, *, resumed: bool,
                 resume_step: int):
        self.cfg = cfg
        self.run_dir = run_dir
        self.jsonl = run_dir / "metrics.jsonl"
        self.tb = None
        self.wb = None
        if resumed and cfg["logging"].get("jsonl", True):
            trim_metrics(self.jsonl, resume_step)
        tb_cfg = cfg["logging"]["tensorboard"]
        wb_cfg = cfg["logging"]["wandb"]
        try:
            if tb_cfg["enabled"]:
                from torch.utils.tensorboard import SummaryWriter
                options = {"purge_step": resume_step + 1} if resumed else {}
                self.tb = SummaryWriter(log_dir=str(run_dir / "tensorboard"), **options)
            if wb_cfg["enabled"]:
                import wandb
                id_path = run_dir / "wandb_run_id.txt"
                run_id = id_path.read_text(encoding="utf-8").strip() if resumed and id_path.is_file() else None
                if resumed and id_path.is_file() and not run_id:
                    raise ValueError("Empty WandB run ID in existing run.")
                self.wb = wandb.init(
                    project=wb_cfg["project"], entity=wb_cfg.get("entity"),
                    mode=wb_cfg.get("mode", "online"), dir=str(run_dir),
                    name=run_dir.name, id=run_id,
                    resume="allow" if run_id else None,
                    config=cfg,
                )
                if self.wb is None or not getattr(self.wb, "id", None):
                    raise RuntimeError("WandB initialization failed.")
                if not run_id:
                    id_path.write_text(self.wb.id + "\n", encoding="utf-8")
                self.wb.define_metric("global_step")
                self.wb.define_metric("train/*", step_metric="global_step")
                self.wb.define_metric("val/*", step_metric="global_step")
        except BaseException:
            self.close()
            raise

    def log(self, phase: str, step: int, **metrics: float | int | str) -> None:
        entry: dict[str, Any] = {"phase": phase, "step": step, **metrics}
        if self.cfg["logging"].get("jsonl", True):
            with self.jsonl.open("a", encoding="utf-8") as out:
                out.write(json.dumps(entry, allow_nan=False) + "\n")
        scalar = {f"{phase}/{key}": value for key, value in metrics.items()
                  if isinstance(value, (int, float)) and not isinstance(value, bool)}
        if self.tb is not None:
            for key, value in scalar.items():
                self.tb.add_scalar(key, value, global_step=step)
            self.tb.flush()
        if self.wb is not None:
            # Use global_step as the chart x-axis; train and val may share a step.
            self.wb.log({"global_step": step, **scalar})

    def close(self) -> None:
        if self.tb is not None:
            self.tb.close()
            self.tb = None
        if self.wb is not None:
            self.wb.finish()
            self.wb = None


def trim_metrics(path: Path, resume_step: int) -> None:
    """Drop metrics logged after last durable model checkpoint (or at same step twice)."""
    if not path.is_file():
        return
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if int(entry["step"]) <= resume_step:
            kept.append(line)
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
