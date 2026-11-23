"""CPU tests for native batch semantics and checkpoint recovery."""

import copy
import sys
import types

import pytest
import torch

from arbim.training.train_rl100_dp2d import (
    _checkpoint_periodic, _validation_without_random_shifts, load_checkpoint,
    read_checkpoint, prepare_batch, save_checkpoint,
)


def test_prepare_batch_preserves_native_loss_contract():
    batch = {
        "obs": {
            "agent_pos": torch.ones(2, 2, 16),
            "head": torch.full((2, 2, 3, 8, 8), 255, dtype=torch.uint8),
            "left": torch.full((2, 2, 3, 8, 8), 128, dtype=torch.uint8),
        },
        "action": torch.ones(2, 5, 16),
        "action_is_pad": torch.tensor([[True, False, False, False, False]] * 2),
    }
    ready = prepare_batch(batch, torch.device("cpu"))
    assert ready["obs"]["head"].dtype == torch.float32
    assert ready["obs"]["head"].min() == 1
    assert ready["obs"]["left"][0, 0, 0, 0, 0].item() == pytest.approx(128 / 255)
    assert ready["action"].shape == (2, 5, 16)
    assert ready["obs"]["agent_pos"].shape == (2, 2, 16)
    batch["action_is_pad"][0, -1] = True
    with pytest.raises(ValueError, match="padded"):
        prepare_batch(batch, torch.device("cpu"))


@pytest.mark.parametrize("use_aug", [False, True])
def test_validation_temporarily_disables_random_shifts(use_aug):
    policy = types.SimpleNamespace(use_aug=use_aug)
    with _validation_without_random_shifts(policy):
        assert policy.use_aug is False
    assert policy.use_aug is use_aug

    with pytest.raises(RuntimeError, match="interrupted"):
        with _validation_without_random_shifts(policy):
            assert policy.use_aug is False
            raise RuntimeError("interrupted")
    assert policy.use_aug is use_aug



def test_checkpoint_strict_roundtrip(tmp_path, monkeypatch):
    pkg = types.ModuleType("rl_100")
    pkg.__path__ = []
    common = types.ModuleType("rl_100.common")
    common.__path__ = []
    util = types.ModuleType("rl_100.common.pytorch_util")
    util.optimizer_to = lambda optimizer, device: None
    for name, obj in (("rl_100", pkg), ("rl_100.common", common),
                      ("rl_100.common.pytorch_util", util)):
        monkeypatch.setitem(sys.modules, name, obj)

    def components(policy=None):
        policy = policy if policy is not None else torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 0.5)
        ema = types.SimpleNamespace(averaged_model=copy.deepcopy(policy), optimization_step=0, decay=0.0)
        return policy, optimizer, scheduler, ema

    policy, optimizer, scheduler, ema = components()
    optimizer.zero_grad()
    policy(torch.ones(2, 3)).square().mean().backward()
    optimizer.step()
    scheduler.step()
    ema.averaged_model.load_state_dict(policy.state_dict())
    ema.optimization_step, ema.decay = 7, 0.9
    path = tmp_path / "last.ckpt"
    contract = {"dataset_spec": {"state_dim": 3}, "total_steps": 20, "steps_per_epoch": 5}
    save_checkpoint(path, policy, optimizer, scheduler, ema,
                    global_step=7, contract=contract, best_val_loss=0.5, best_step=5)

    restored = torch.nn.Linear(3, 2)
    payload = read_checkpoint(path, contract=contract)
    restored.load_state_dict(payload["policy"], strict=True)
    restored.to(torch.device("cpu"))
    restored, restored_optim, restored_sched, restored_ema = components(restored)
    step, best_loss, best_step = load_checkpoint(
        payload, restored, restored_optim, restored_sched, restored_ema,
        device=torch.device("cpu"))
    assert (step, best_loss, best_step) == (7, 0.5, 5)
    assert restored_ema.optimization_step == 7
    assert restored_ema.decay == 0.9
    assert restored_sched.state_dict()["last_epoch"] == scheduler.state_dict()["last_epoch"]
    assert restored_optim.state_dict()["state"]
    assert torch.load(path, weights_only=True)["batch_offset"] == 2
    for before, after in zip(policy.parameters(), restored.parameters()):
        torch.testing.assert_close(before, after)
    with pytest.raises(ValueError, match="contract"):
        read_checkpoint(path, contract={**contract, "total_steps": 21})


def test_resume_rebuilds_optimizer_after_dynamic_native_normalizer_load(tmp_path, monkeypatch):
    """Mirror RL-100 DictOfTensorMixin's replacement of ParameterDict entries."""
    pkg = types.ModuleType("rl_100")
    pkg.__path__ = []
    common = types.ModuleType("rl_100.common")
    common.__path__ = []
    util = types.ModuleType("rl_100.common.pytorch_util")
    util.optimizer_to = lambda optimizer, device: None
    for name, mod in (("rl_100", pkg), ("rl_100.common", common),
                      ("rl_100.common.pytorch_util", util)):
        monkeypatch.setitem(sys.modules, name, mod)

    class DynamicNormalizer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.params_dict = torch.nn.ParameterDict({
                "scale": torch.nn.Parameter(torch.ones(2), requires_grad=False),
            })

        def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                                  strict, missing_keys, unexpected_keys, error_msgs):
            # Native DictOfTensorMixin performs this replacement with CPU tensors.
            self.params_dict = torch.nn.ParameterDict({
                "scale": torch.nn.Parameter(
                    state_dict[prefix + "params_dict.scale"].clone(),
                    requires_grad=False,
                )
            })

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(2, 2)
            self.normalizer = DynamicNormalizer()

        def forward(self, value):
            return self.layer(value * self.normalizer.params_dict["scale"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy = Policy().to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1)
    ema = types.SimpleNamespace(averaged_model=copy.deepcopy(policy),
                                optimization_step=1, decay=0.9)
    optimizer.zero_grad()
    policy(torch.ones(2, 2, device=device)).square().mean().backward()
    optimizer.step()
    scheduler.step()
    checkpoint = tmp_path / "last.ckpt"
    contract = {"total_steps": 10, "steps_per_epoch": 4}
    save_checkpoint(checkpoint, policy, optimizer, scheduler, ema,
                    global_step=1, contract=contract, best_val_loss=None, best_step=None)

    payload = read_checkpoint(checkpoint, contract=contract)
    resumed = Policy()
    resumed.load_state_dict(payload["policy"], strict=True)
    resumed.to(device)
    resumed_ema = types.SimpleNamespace(averaged_model=copy.deepcopy(resumed),
                                        optimization_step=0, decay=0.0)
    resumed_opt = torch.optim.AdamW(resumed.parameters(), lr=0.01)
    resumed_sch = torch.optim.lr_scheduler.LambdaLR(resumed_opt, lr_lambda=lambda step: 1)
    assert load_checkpoint(payload, resumed, resumed_opt, resumed_sch, resumed_ema,
                           device=device)[0] == 1
    optimizer_ids = {id(p) for group in resumed_opt.param_groups for p in group["params"]}
    assert id(resumed.normalizer.params_dict["scale"]) in optimizer_ids
    assert all(p.device == device for p in resumed.parameters())
    assert all(p.device == device for p in resumed_ema.averaged_model.parameters())
    assert resumed(torch.ones(2, 2, device=device)).device == device

    # Old order: on-device model and optimizer exist before loading native
    # Normalizer. The optimizer then references an orphaned Parameter.
    stale = Policy().to(device)
    stale_opt = torch.optim.AdamW(stale.parameters(), lr=0.01)
    stale_sch = torch.optim.lr_scheduler.LambdaLR(stale_opt, lr_lambda=lambda step: 1)
    stale_ema = types.SimpleNamespace(averaged_model=copy.deepcopy(stale),
                                      optimization_step=0, decay=0.0)
    stale.load_state_dict(payload["policy"], strict=True)
    with pytest.raises(ValueError, match="stale Policy parameters"):
        load_checkpoint(payload, stale, stale_opt, stale_sch, stale_ema, device=device)


def test_periodic_snapshots_are_raw_only_and_last_remains_resumable(tmp_path):
    policy = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1)
    ema = types.SimpleNamespace(averaged_model=copy.deepcopy(policy),
                                optimization_step=1, decay=0.9)
    optimizer.zero_grad()
    policy(torch.ones(2, 3)).square().mean().backward()
    optimizer.step()
    scheduler.step()
    contract = {"total_steps": 10, "steps_per_epoch": 4}
    cfg = {"checkpoint": {"keep_last": 2}}
    checkpoint_dir = tmp_path / "checkpoints"

    for step in (1, 2, 3):
        _checkpoint_periodic(
            tmp_path, cfg, step, policy, optimizer, scheduler, ema,
            contract, best_loss=0.5, best_step=1,
        )
        last = read_checkpoint(checkpoint_dir / "last.ckpt", contract=contract)
        snapshot_path = checkpoint_dir / f"step_{step:08d}.ckpt"
        snapshot = torch.load(snapshot_path, weights_only=True)
        assert last["checkpoint_role"] == "training"
        assert {"optimizer", "lr_scheduler", "ema_policy", "rng"} <= set(last)
        assert last["global_step"] == snapshot["global_step"] == step
        assert snapshot["checkpoint_role"] == "policy_snapshot"
        assert snapshot["selected_weights"] == "raw"
        assert set(snapshot) == {
            "format_version", "checkpoint_role", "policy",
            "global_step", "contract", "selected_weights",
        }
        assert snapshot["contract"] == contract
        for name in last["policy"]:
            torch.testing.assert_close(last["policy"][name], snapshot["policy"][name])
        assert snapshot_path.stat().st_ino != (checkpoint_dir / "last.ckpt").stat().st_ino
        with pytest.raises(ValueError, match="resumable"):
            read_checkpoint(snapshot_path, contract=contract)

    assert not (checkpoint_dir / "step_00000001.ckpt").exists()
    assert [p.name for p in sorted(checkpoint_dir.glob("step_*.ckpt"))] == [
        "step_00000002.ckpt", "step_00000003.ckpt",
    ]
    previous_last = (checkpoint_dir / "last.ckpt").read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _checkpoint_periodic(
            tmp_path, cfg, 3, policy, optimizer, scheduler, ema,
            contract, best_loss=0.5, best_step=1,
        )
    assert (checkpoint_dir / "last.ckpt").read_bytes() == previous_last


def test_snapshot_write_failure_keeps_last_resumable_and_no_partial_file(
        tmp_path, monkeypatch):
    policy = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1)
    ema = types.SimpleNamespace(averaged_model=copy.deepcopy(policy),
                                optimization_step=1, decay=0.9)
    contract = {"total_steps": 10, "steps_per_epoch": 4}
    cfg = {"checkpoint": {"keep_last": 2}}
    original_save = torch.save

    def failing_save(value, destination, *args, **kwargs):
        if str(destination).endswith("step_00000001.ckpt.tmp"):
            raise OSError("simulated snapshot write failure")
        return original_save(value, destination, *args, **kwargs)

    monkeypatch.setattr(torch, "save", failing_save)
    with pytest.raises(OSError, match="simulated snapshot"):
        _checkpoint_periodic(
            tmp_path, cfg, 1, policy, optimizer, scheduler, ema,
            contract, best_loss=None, best_step=None,
        )
    checkpoint_dir = tmp_path / "checkpoints"
    assert read_checkpoint(checkpoint_dir / "last.ckpt", contract=contract)["global_step"] == 1
    assert not (checkpoint_dir / "step_00000001.ckpt").exists()
    assert not (checkpoint_dir / "step_00000001.ckpt.tmp").exists()
