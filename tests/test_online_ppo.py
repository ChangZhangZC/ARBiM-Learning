"""Check the native PPO bridge and the fail-fast Online scope gates."""
from types import SimpleNamespace

import pytest
import torch

from arbim.rl.online.policy_adapter import validate_online_policy
from arbim.rl.online.ppo_bridge import update_online_ppo, count_ppo_steps, _variable_span_advantage
from arbim.training.train_online_rl100_dp2d import validate_config


class FakeOptimizerPPO:
    def __init__(self):
        self.args = SimpleNamespace(batch_size=32)
        self.called = False

    def dp_align_update_no_share(self, buffer, total_steps, precomputed=None):
        assert precomputed is None
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


def test_ppo_minibatch_budget_is_not_rollout_updates():
    assert count_ppo_steps(1024, 128, 5) == 40
    assert count_ppo_steps(1054, 128, 5) == 45


def test_variable_span_gae_uses_real_discount_and_bootstrap():
    class FakeValue:
        args = SimpleNamespace(gamma=0.5, lamda=0.95, share_encoder=False)

        def _compute_critic_values_in_chunks(self, s, next_s, use_obs2latent=False):
            assert not use_obs2latent
            return torch.tensor([[2.0], [3.0]]), torch.tensor([[4.0], [5.0]])

    class FakeTransitions:
        count = 2

        def executed_spans(self):
            return [4, 1]

        def numpy_to_tensor(self):
            obs = {"agent_pos": torch.zeros(2, 1, 1)}
            return (obs, None, None, torch.tensor([[1.0], [2.0]]), obs,
                    torch.tensor([[0.0], [0.0]]), torch.tensor([[0.0], [1.0]]))

    result = _variable_span_advantage(FakeValue(), FakeTransitions())
    # Last transition is time-limit truncation, not a true terminal: bootstrap V(s_next).
    last_delta = 2 + 0.5 * 5 - 3
    first_delta = 1 + 0.5**4 * 4 - 2
    expected_first = first_delta + 0.5**4 * 0.95 * last_delta
    torch.testing.assert_close(result["adv"], torch.tensor([[expected_first], [last_delta]]))
    torch.testing.assert_close(result["v_target"], torch.tensor([[expected_first + 2], [last_delta + 3]]))


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


@pytest.mark.parametrize("recon,value_recon,per_step_recon", [
    (False, False, False), (True, False, False), (True, True, True),
    (False, True, False),
])
def test_online_reconvib_configuration_is_valid(recon, value_recon, per_step_recon):
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] /
        "configs/online/rl100_dp2d_hinge_chest.yaml").read_text())
    cfg["policy"]["checkpoint_path"] = "/verified/hinge_chest/best.ckpt"
    cfg["ppo"].update(recon=recon, value_recon=value_recon,
                      per_step_recon=per_step_recon,
                      fix_encoder=False, share_encoder=False)
    validate_config(cfg)


@pytest.mark.parametrize("settings,expected_error", [
    ({"recon": True, "fix_encoder": True}, "fix_encoder"),
    ({"value_recon": True, "share_encoder": True}, "share_encoder"),
    ({"per_step_recon": True, "recon": False}, "per_step_recon"),
    ({"recon": "true"}, "ppo.recon"),
])
def test_online_reconvib_rejects_unusable_combinations(settings, expected_error):
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] /
        "configs/online/rl100_dp2d_hinge_chest.yaml").read_text())
    cfg["policy"]["checkpoint_path"] = "/verified/hinge_chest/best.ckpt"
    cfg["ppo"].update(settings)
    with pytest.raises(ValueError, match=expected_error):
        validate_config(cfg)
