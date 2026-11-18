"""Check the native PPO bridge and the fail-fast Online scope gates."""
from types import SimpleNamespace

import pytest
import torch

from arbim.rl.online.policy_adapter import validate_online_policy
from arbim.rl.online.ppo_bridge import update_online_ppo
from arbim.training.train_online_rl100_dp2d import validate_config


class FakeOptimizerPPO:
    def __init__(self):
        self.args = SimpleNamespace(batch_size=32)
        self.called = False

    def dp_align_update_no_share(self, buffer, total_steps):
        assert self.args.batch_size == buffer.count
        assert total_steps == 24
        self.called = True
        return 0.2, 0.1, 0.0, 0.0


def test_online_bridge_calls_native_update_not_custom_math():
    ppo = FakeOptimizerPPO()
    buffer = SimpleNamespace(count=7)
    result = update_online_ppo(ppo, buffer, total_steps=24)
    assert ppo.called and ppo.args.batch_size == 7
    assert result == {"actor_loss": 0.2, "critic_loss": 0.1,
                      "bc_loss": 0.0, "distill_loss": 0.0}


def test_ratio_rejects_unsafe_h_gt_1():
    policy = SimpleNamespace(
        is_flow=False, w_pc=False, no_pre_action=True, use_aug=False,
        n_action_steps=4, n_obs_steps=2, horizon=5, num_inference_steps=10,
    )
    with pytest.raises(ValueError, match="H>1"):
        validate_online_policy(policy, "per_action_step")
    validate_online_policy(policy, "chunk_joint")


def test_default_config_is_strict(tmp_path):
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] /
        "configs/online/rl100_dp2d_hinge_chest.yaml").read_text())
    cfg["policy"]["checkpoint_path"] = "/verified/hinge_chest/best.ckpt"
    validate_config(cfg)
    for fix in (False, True):
        for share in (False, True):
            cfg["ppo"]["fix_encoder"] = fix
            cfg["ppo"]["share_encoder"] = share
            validate_config(cfg)
    cfg["ppo"]["fix_encoder"] = "false"
    with pytest.raises(ValueError, match="ppo.fix_encoder"):
        validate_config(cfg)
    cfg["ppo"]["fix_encoder"] = False
    cfg["ppo"]["share_encoder"] = None
    with pytest.raises(ValueError, match="ppo.share_encoder"):
        validate_config(cfg)
