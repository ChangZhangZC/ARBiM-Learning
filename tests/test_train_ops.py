"""CPU regression checks for native-RL100 training orchestration (no GPU download)."""

from __future__ import annotations

import copy
import json
import sys
import types

import pytest
import torch
import yaml

from arbim.training.run_utils import (
    StepBatchSampler, check_config, is_due, resolve_run_directory,
    resolve_training_budget,
)


def profile():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "configs/il/duobench_transfer_cube.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_config_rejects_invalid_budgets_and_legacy_smoke_keys():
    cfg = profile()
    assert "max_train_batches" not in cfg["training"]
    assert "max_val_batches" not in cfg["training"]
    check_config(cfg)
    cfg["training"]["max_epochs"] = None
    cfg["training"]["max_steps"] = None
    with pytest.raises(ValueError, match="max_steps"):
        check_config(cfg)
    cfg["training"]["max_steps"] = True
    with pytest.raises(ValueError, match="positive integer"):
        check_config(cfg)
    cfg["training"]["max_steps"] = 9
    cfg["checkpoint"]["every_steps"] = 0
    with pytest.raises(ValueError, match="checkpoint.every_steps"):
        check_config(cfg)
    cfg["checkpoint"]["every_steps"] = None
    cfg["experiment"]["task"] = "../escape"
    with pytest.raises(ValueError, match="experiment.task"):
        check_config(cfg)


def test_steps_and_epochs_whichever_first():
    cfg = profile()
    cfg["dataloader"]["batch_size"] = 8
    cfg["training"]["max_steps"] = None
    cfg["training"]["max_epochs"] = 2
    assert resolve_training_budget(cfg, 15) == (4, 2)
    cfg["training"]["max_steps"] = 3
    assert resolve_training_budget(cfg, 15) == (3, 2)
    cfg["training"]["max_epochs"] = None
    assert resolve_training_budget(cfg, 15) == (3, 2)


def test_resume_sampler_rebuilds_exact_next_batch_and_epoch_boundaries():
    batches = list(StepBatchSampler(13, 5, 12, 0, 8))
    assert len(batches) == 8
    assert batches[3:8] == list(StepBatchSampler(13, 5, 12, 3, 8))
    assert [len(b) for b in batches] == [5, 5, 3, 5, 5, 3, 5, 5]
    for offset in (0, 3, 6):
        assert sorted(sum(batches[offset:offset + 3], [])) == list(range(13))


def test_step_and_epoch_checkpoint_triggers():
    cfg = {"every_steps": 5, "every_epochs": 2}
    assert [i for i in range(1, 15) if is_due(cfg, i, 3)] == [5, 6, 10, 12]


def test_run_path_isolated_and_resume_keeps_original(tmp_path):
    cfg = profile()
    cfg["output"]["root"] = str(tmp_path)
    cfg["experiment"]["run_name"] = "test1"
    path = resolve_run_directory(cfg, None)
    assert path == tmp_path / "duobench_transfer_cube" / "rl100_dp2d" / "test1"
    (path / "checkpoints").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        resolve_run_directory(cfg, None)
    (path / "checkpoints" / "last.ckpt").write_bytes(b"placeholder")
    assert resolve_run_directory(cfg, path) == path


@pytest.fixture
def fake_native(monkeypatch):
    """Mirror Policy/EMA/LR/data interfaces with tiny deterministic CPU modules."""
    from arbim.training import il

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, *args, split="train", **kwargs):
            self.split = split
            self.horizon = 1
            self.n_obs_steps = 1
            self.prediction_horizon = 1
            self.spec = {
                "state_dim": 2, "action_dim": 2,
                "state_key": "observation.state", "action_key": "action",
                "camera_keys": ("observation.images.head",),
                "camera_shapes": {"observation.images.head": (3, 4, 4)},
                "camera_obs_keys": {"observation.images.head": "rgb_0"},
            }
            self._train_episodes = (0, 1, 2)
            self._val_episodes = (3,)

        def __len__(self):
            return 7 if self.split == "train" else 3

        def __getitem__(self, i):
            x = torch.tensor([float(i), float(i % 3)], dtype=torch.float32)
            return {
                "obs": {
                    "agent_pos": x[None],
                    "rgb_0": torch.zeros(1, 3, 4, 4, dtype=torch.uint8),
                },
                "action": (x * 0.5)[None],
                "action_is_pad": torch.zeros(1, dtype=torch.bool),
            }

        def get_normalizer(self, mode):
            return torch.nn.Identity()

        def get_validation_dataset(self):
            return Dataset(split="val")

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(2, 2)
            self.use_aug = False

        def set_normalizer(self, normalizer):
            self.normalizer = normalizer

        def compute_loss(self, batch):
            state = batch["obs"]["agent_pos"][:, -1]
            pred = self.layer(state)
            noise = torch.randn_like(pred)
            return ((pred + 0.1 * noise - batch["action"][:, 0]) ** 2).mean(), {}

    class EMA:
        def __init__(self, model, **kwargs):
            self.averaged_model = model
            self.optimization_step = 0
            self.decay = 0.75

        def step(self, model):
            self.optimization_step += 1
            with torch.no_grad():
                for dst, src in zip(self.averaged_model.parameters(), model.parameters()):
                    dst.lerp_(src, 0.25)

    modules = {}
    for name in (
        "rl_100", "rl_100.common", "rl_100.common.pytorch_util",
        "rl_100.model", "rl_100.model.common",
        "rl_100.model.common.lr_scheduler",
        "rl_100.model.diffusion", "rl_100.model.diffusion.ema_model",
    ):
        module = types.ModuleType(name)
        module.__path__ = []
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
    modules["rl_100.common.pytorch_util"].optimizer_to = lambda *args: None
    modules["rl_100.model.common.lr_scheduler"].get_scheduler = (
        lambda name, optimizer, num_warmup_steps, num_training_steps:
        torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: max(0.1, 1 - step / num_training_steps)
        )
    )
    modules["rl_100.model.diffusion.ema_model"].EMAModel = EMA
    adapter = types.ModuleType("arbim.data.lerobot_adapter")
    adapter.LeRobotDP2DDataset = Dataset
    adapter.build_shape_meta = lambda spec: {"obs": {}, "action": {}}
    monkeypatch.setitem(sys.modules, "arbim.data.lerobot_adapter", adapter)
    monkeypatch.setattr(il, "build_rl100_dp", lambda *args, **kwargs: Policy())
    return il


@pytest.mark.parametrize("interruption", [4, 5])
def test_exact_resume_matches_uninterrupted_policy_optimizer_ema_rng(
    fake_native, tmp_path, interruption,
):
    def config(folder):
        cfg = profile()
        cfg["output"]["output_dir"] = str(folder)
        cfg["dataloader"].update(
            batch_size=2, num_workers=0,
            prefetch_factor=None, persistent_workers=False,
        )
        cfg["dataset"]["horizon"] = 1
        cfg["dataset"]["n_obs_steps"] = 1
        cfg["policy"]["n_action_steps"] = 1
        cfg["training"].update(device="cpu", max_steps=9, max_epochs=4)
        cfg["lr_scheduler"]["warmup_steps"] = 0
        cfg["checkpoint"].update(every_steps=3, every_epochs=1, keep_last=2)
        cfg["validation"].update(every_steps=2, every_epochs=None, on_finish=True)
        cfg["logging"]["tensorboard"]["enabled"] = False
        cfg["logging"]["wandb"]["enabled"] = False
        cfg["logging"]["log_freq_steps"] = 2
        return cfg

    full_cfg = config(tmp_path / "full")
    resumed_cfg = config(tmp_path / "resumed")
    fake_native.run(full_cfg)
    fake_native.run(resumed_cfg, stop_after=interruption)
    fake_native.run(resumed_cfg, resume=tmp_path / "resumed")
    full = torch.load(tmp_path / "full/checkpoints/last.ckpt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed/checkpoints/last.ckpt", weights_only=True)
    assert full["global_step"] == resumed["global_step"] == 9
    for key in ("policy", "ema_policy"):
        assert full[key].keys() == resumed[key].keys()
        for tensor in full[key]:
            torch.testing.assert_close(full[key][tensor], resumed[key][tensor], rtol=0, atol=0)
    assert full["optimizer"]["state"].keys() == resumed["optimizer"]["state"].keys()
    for idx, entry in full["optimizer"]["state"].items():
        for name, value in entry.items():
            other = resumed["optimizer"]["state"][idx][name]
            if torch.is_tensor(value):
                torch.testing.assert_close(value, other, rtol=0, atol=0)
            else:
                assert value == other
    assert full["lr_scheduler"] == resumed["lr_scheduler"]
    assert full["ema_optimization_step"] == resumed["ema_optimization_step"]
    torch.testing.assert_close(full["rng"]["torch"], resumed["rng"]["torch"])
    assert len(list((tmp_path / "resumed/checkpoints").glob("step_*.ckpt"))) <= 2
    lines = [json.loads(s) for s in (tmp_path / "resumed/metrics.jsonl").read_text().splitlines()]
    assert all(item["step"] <= 9 for item in lines)


def test_resume_rejects_changed_budget_before_mutating_checkpoint(fake_native, tmp_path):
    cfg = profile()
    cfg["output"]["output_dir"] = str(tmp_path / "run")
    cfg["dataloader"].update(
        batch_size=2, num_workers=0, prefetch_factor=None, persistent_workers=False,
    )
    cfg["dataset"].update(horizon=1, n_obs_steps=1)
    cfg["policy"]["n_action_steps"] = 1
    cfg["training"].update(device="cpu", max_steps=6, max_epochs=None)
    cfg["logging"]["tensorboard"]["enabled"] = False
    cfg["checkpoint"]["every_steps"] = 1
    fake_native.run(cfg, stop_after=2)
    changed = copy.deepcopy(cfg)
    changed["training"]["max_steps"] = 8
    with pytest.raises(ValueError, match="contract"):
        fake_native.run(changed, resume=tmp_path / "run")
