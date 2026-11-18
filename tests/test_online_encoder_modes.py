"""Four native RL-100 Online PPO Encoder paths (CPU, without DuoBench).

The native PPO bridge and ValueMLP are instantiated, not substituted by a
surrogate PPO loss. A cheap RGB-like encoder lets gradient/optimizer ownership,
checkpoint reconstruction and shared-versus-independent inputs be checked.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from arbim.rl.online.checkpoint import (
    read_online_checkpoint, restore_online_checkpoint, save_online_checkpoint,
)
from arbim.rl.online.ppo_bridge import build_online_ppo


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
        with torch.no_grad():
            self.linear.weight.fill_(0.2)
            self.linear.bias.fill_(0.1)

    def forward(self, observation):
        return self.linear(observation["agent_pos"])


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.obs_encoder = TinyEncoder()
        self.model = torch.nn.Linear(8, 1)
        self.is_flow = False
        self.w_pc = False
        self.no_pre_action = True
        self.use_aug = False
        self.n_action_steps = 4
        self.n_obs_steps = 2
        self.horizon = 5
        self.obs_feature_dim = 4
        self.action_dim = 1
        self.num_inference_steps = 3


def config():
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs/online/rl100_dp2d_hinge_chest.yaml").read_text())
    cfg["policy"]["checkpoint_path"] = "/tmp/test/il.ckpt"
    return cfg


def create_bridge(cfg):
    settings = dict(cfg["ppo"])
    settings["max_train_steps"] = 1000
    return build_online_ppo(TinyPolicy(), settings)


def param_ids(parameters):
    return {id(p) for p in parameters}


@pytest.mark.parametrize("fix_encoder", [False, True])
@pytest.mark.parametrize("share_encoder", [False, True])
def test_four_modes_native_optimizers_and_critic_contract(fix_encoder, share_encoder):
    cfg = config()
    cfg["ppo"]["fix_encoder"] = fix_encoder
    cfg["ppo"]["share_encoder"] = share_encoder
    ppo = create_bridge(cfg)

    actor_encoder = ppo._policy.obs_encoder
    actor_ids = param_ids(actor_encoder.parameters())
    actor_optimizer_ids = param_ids(
        p for group in ppo.optimizer_actor.param_groups for p in group["params"]
    )
    critic_optimizer_ids = param_ids(
        p for group in ppo.optimizer_critic.param_groups for p in group["params"]
    )

    assert ppo.args.fix_encoder is fix_encoder
    assert ppo.args.share_encoder is share_encoder
    assert all(p.requires_grad is (not fix_encoder) for p in actor_encoder.parameters())
    assert bool(actor_ids & actor_optimizer_ids) is (not fix_encoder)
    assert not actor_optimizer_ids & critic_optimizer_ids
    assert all(not p.requires_grad for p in ppo._old_policy.parameters())

    raw = {"agent_pos": torch.ones(4, 3)}
    if share_encoder:
        assert ppo.critic._obs_encoder is None
        with torch.no_grad():
            features = actor_encoder(raw).reshape(2, -1)
        value = ppo.critic(features)
    else:
        separate = ppo.critic._obs_encoder
        assert separate is not actor_encoder
        assert param_ids(separate.parameters()).isdisjoint(actor_ids)
        assert all(p.requires_grad for p in separate.parameters())
        assert param_ids(separate.parameters()) <= critic_optimizer_ids
        assert torch.equal(separate.linear.weight, actor_encoder.linear.weight)
        assert separate.linear.weight.data_ptr() != actor_encoder.linear.weight.data_ptr()
        value = ppo.critic(raw)
    assert value.shape == (2, 1)
    assert torch.isfinite(value).all()


@pytest.mark.parametrize("fix_encoder", [False, True])
@pytest.mark.parametrize("share_encoder", [False, True])
def test_four_modes_actor_and_critic_gradient_isolation(fix_encoder, share_encoder):
    cfg = config()
    cfg["ppo"]["fix_encoder"] = fix_encoder
    cfg["ppo"]["share_encoder"] = share_encoder
    ppo = create_bridge(cfg)
    actor_encoder = ppo._policy.obs_encoder
    before = actor_encoder.linear.weight.detach().clone()
    raw = {"agent_pos": torch.ones(2, 3)}

    ppo.optimizer_actor.zero_grad()
    feature = actor_encoder(raw).reshape(1, -1)
    loss = ppo._policy.model(feature).sum()
    loss.backward()
    actor_gradient = actor_encoder.linear.weight.grad
    assert (actor_gradient is not None) is (not fix_encoder)
    ppo.optimizer_actor.step()
    assert (not torch.equal(before, actor_encoder.linear.weight)) is (not fix_encoder)

    ppo.optimizer_actor.zero_grad()
    ppo.optimizer_critic.zero_grad()
    if share_encoder:
        with torch.no_grad():
            value_input = actor_encoder(raw).reshape(1, -1)
    else:
        value_input = raw
    ppo.critic(value_input).sum().backward()
    assert actor_encoder.linear.weight.grad is None
    if not share_encoder:
        # An independent Value Encoder must receive its own gradients,
        # including when the *Actor* Encoder is frozen.
        encoder = ppo.critic._obs_encoder
        # Direct encoder proxy avoids fragile assumptions about ReLU signs.
        ppo.optimizer_critic.zero_grad()
        encoder(raw).sum().backward()
        assert encoder.linear.weight.grad is not None


@pytest.mark.parametrize("fix_encoder,share_encoder", [
    (False, False), (False, True), (True, False), (True, True),
])
def test_four_modes_full_checkpoint_restores_encoder_state(tmp_path, fix_encoder, share_encoder):
    cfg = config()
    cfg["ppo"]["fix_encoder"] = fix_encoder
    cfg["ppo"]["share_encoder"] = share_encoder
    original = create_bridge(cfg)
    target_path = tmp_path / "last.ckpt"
    with torch.no_grad():
        original._policy.obs_encoder.linear.weight.fill_(0.43)
        if not share_encoder:
            original.critic._obs_encoder.linear.weight.fill_(0.57)
    save_online_checkpoint(
        target_path, original, config=cfg, updates=1, episode_count=2,
        env_steps=32, decision_steps=8, original_checkpoint="/tmp/test/il.ckpt",
    )
    payload = read_online_checkpoint(
        target_path, config=cfg, original_checkpoint="/tmp/test/il.ckpt",
    )
    restored = create_bridge(cfg)
    assert restore_online_checkpoint(restored, payload) == (1, 2, 32)
    assert torch.equal(
        restored._policy.obs_encoder.linear.weight,
        original._policy.obs_encoder.linear.weight,
    )
    if not share_encoder:
        assert torch.equal(
            restored.critic._obs_encoder.linear.weight,
            original.critic._obs_encoder.linear.weight,
        )
    assert all(not p.requires_grad for p in restored._old_policy.parameters())
    assert all(
        p.requires_grad is (not fix_encoder)
        for p in restored._policy.obs_encoder.parameters()
    )
