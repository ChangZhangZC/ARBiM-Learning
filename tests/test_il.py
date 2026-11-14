"""CPU tests for native batch semantics and checkpoint recovery."""

import copy
import sys
import types

import pytest
import torch

from arbim.training.il import load_checkpoint, prepare_batch, save_checkpoint


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

    def components():
        policy = torch.nn.Linear(3, 2)
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
    path = tmp_path / "latest.ckpt"
    contract = {"dataset_spec": {"state_dim": 3}, "num_epochs": 8}
    save_checkpoint(path, policy, optimizer, scheduler, ema,
                    next_epoch=3, global_step=7, contract=contract)

    restored, restored_optim, restored_sched, restored_ema = components()
    start_epoch, step = load_checkpoint(
        path, restored, restored_optim, restored_sched, restored_ema,
        contract=contract, device=torch.device("cpu"))
    assert (start_epoch, step) == (3, 7)
    assert restored_ema.optimization_step == 7
    assert restored_ema.decay == 0.9
    assert restored_sched.state_dict()["last_epoch"] == scheduler.state_dict()["last_epoch"]
    assert restored_optim.state_dict()["state"]
    for before, after in zip(policy.parameters(), restored.parameters()):
        torch.testing.assert_close(before, after)
    with pytest.raises(ValueError, match="contract"):
        load_checkpoint(path, restored, restored_optim, restored_sched,
                        restored_ema, contract={"dataset_spec": {}}, device=torch.device("cpu"))
