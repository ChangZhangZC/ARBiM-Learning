"""Native constructor wiring tests: no upstream weight downloads."""

import sys
import types

import pytest

from arbim.policy.rl100_dp import build_rl100_dp


@pytest.mark.parametrize("weights", ["r3m", "IMAGENET1K_V1", None])
def test_factory_uses_unmodified_native_classes(monkeypatch, weights):
    def module(name):
        m = types.ModuleType(name)
        m.__path__ = []
        monkeypatch.setitem(sys.modules, name, m)
        return m

    module("hydra").utils = module("hydra.utils")
    sys.modules["hydra.utils"].instantiate = lambda cfg: ("native scheduler", cfg["_target_"])
    for name in ("rl_100", "rl_100.model", "rl_100.model.vision", "rl_100.policy"):
        module(name)
    getter = module("rl_100.model.vision.model_getter")
    getter.get_resnet = lambda **kwargs: ("native resnet", kwargs)
    encoder_module = module("rl_100.model.vision.multi_image_obs_encoder")

    class Encoder:
        def __init__(self, **kwargs):
            self.args = kwargs

    encoder_module.MultiImageObsEncoder = Encoder
    actor_module = module("rl_100.policy.rl100_2d")

    class Actor:
        def __init__(self, **kwargs):
            self.args = kwargs

    actor_module.RL1002D = Actor
    config = {
        "vision": {
            "rgb_model": {"name": "resnet18", "weights": weights},
            "share_rgb_model": False, "crop_shape": None,
            "use_recon": False, "use_vib": False,
        },
        "policy": {
            "model": "dp3", "scheduler_type": "ddim", "use_aug": False,
            "use_recon": False, "horizon": 5, "n_obs_steps": 2, "n_action_steps": 4,
        },
        "schedulers": {"cm": {"_target_": "cm"}, "ddim": {"_target_": "ddim"}},
    }
    meta = {"obs": {"agent_pos": {"shape": [16], "type": "low_dim"}}}
    actor = build_rl100_dp(meta, config)
    assert isinstance(actor, Actor)
    assert actor.args["shape_meta"] is meta
    assert actor.args["obs_encoder"].args["rgb_model"][1] == {"name": "resnet18", "weights": weights}
    assert actor.args["ddim_noise_scheduler"] == ("native scheduler", "ddim")
    assert actor.args["n_action_steps"] == 4

    def unavailable_weights(**kwargs):
        raise RuntimeError("pretrained weights unavailable")

    getter.get_resnet = unavailable_weights
    with pytest.raises(RuntimeError, match="pretrained weights unavailable"):
        build_rl100_dp(meta, config)
