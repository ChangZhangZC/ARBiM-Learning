"""Native action trajectory / probability contract (no MuJoCo required)."""
import pytest
import torch

from arbim.rl.online.policy_adapter import sample_online, validate_online_policy


class FakePolicy(torch.nn.Module):
    def __init__(self, steps=4, k=3):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(()))
        self.is_flow = False
        self.w_pc = False
        self.no_pre_action = True
        self.use_aug = False
        self.n_action_steps = steps
        self.n_obs_steps = 2
        self.horizon = 5
        self.action_dim = 16
        self.num_inference_steps = k

    def all_step_action_logprob(self, obs, fix_encoder=True, use_cm=False):
        assert not use_cm
        self.last_fix_encoder = fix_encoder
        b = obs["agent_pos"].shape[0]
        x = torch.arange((self.num_inference_steps + 1) * b * 4 * 16).float().reshape(
            self.num_inference_steps + 1, b, 4, 16
        )
        lp = torch.full((self.num_inference_steps, b, 4, 16), -0.5)
        return x[-1, :, :self.n_action_steps], x, lp


def test_chunk_sampling_preserves_one_ddim_chain():
    policy = FakePolicy()
    validate_online_policy(policy, "chunk_joint")
    result = sample_online(policy, {"agent_pos": torch.zeros(2, 2, 16)})
    assert result.action.shape == (2, 4, 16)
    assert result.all_x.shape == (4, 2, 4, 16)
    assert result.a_logprob.shape == (3, 2, 4, 16)
    assert torch.equal(result.action, result.all_x[-1])


def test_single_step_supported_with_original_per_action_ratio():
    policy = FakePolicy(steps=1)
    validate_online_policy(policy, "per_action_step")
    result = sample_online(policy, {"agent_pos": torch.zeros(1, 2, 16)})
    assert result.action.shape == (1, 1, 16)


def test_upstream_multistep_ratio_bug_is_not_silently_patched():
    with pytest.raises(ValueError, match="H>1"):
        validate_online_policy(FakePolicy(steps=4), "per_action_step")


@pytest.mark.parametrize("fix_encoder", [False, True])
def test_rollout_forwards_actor_encoder_mode_to_native_sampler(fix_encoder):
    policy = FakePolicy()
    result = sample_online(
        policy, {"agent_pos": torch.zeros(1, 2, 16)}, fix_encoder=fix_encoder
    )
    assert result.action.shape == (1, 4, 16)
    assert policy.last_fix_encoder is fix_encoder
