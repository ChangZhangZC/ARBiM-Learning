"""Native constructor wiring tests: no upstream weight downloads."""

import sys
import types

import pytest

from arbim.policy.rl100_dp import build_rl100_dp


@pytest.mark.parametrize("weights", ["r3m", "IMAGENET1K_V1", None])
@pytest.mark.parametrize(
    "share_rgb_model,resize_shape,crop_shape,random_crop,use_aug",
    [
        (False, [224, 224], None, True, False), # unchanged baseline
        (True, [224, 224], None, True, False), # shared backbone
        (False, [224, 224], [200, 200], True, True), # random crop + shifts
        (False, [256, 256], [200, 200], False, False), # alternative resize + center crop
        (True, [224, 224], [200, 200], True, True), # combined native options
        (False, None, None, True, False), # keep original image resolution
    ],
)
def test_factory_uses_unmodified_native_classes(
    monkeypatch, weights, share_rgb_model, resize_shape, crop_shape, random_crop, use_aug
):
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
            "share_rgb_model": share_rgb_model, "resize_shape": resize_shape, "crop_shape": crop_shape,
            "random_crop": random_crop, "use_recon": False, "use_vib": False,
        },
        "policy": {
            "model": "dp3", "scheduler_type": "ddim", "use_aug": use_aug,
            "use_recon": False, "horizon": 5, "n_obs_steps": 2, "n_action_steps": 4,
        },
        "schedulers": {"cm": {"_target_": "cm"}, "ddim": {"_target_": "ddim"}},
    }
    meta = {"obs": {"agent_pos": {"shape": [16], "type": "low_dim"}}}
    actor = build_rl100_dp(meta, config)
    assert isinstance(actor, Actor)
    assert actor.args["shape_meta"] is meta
    assert actor.args["obs_encoder"].args["rgb_model"][1] == {"name": "resnet18", "weights": weights}
    assert actor.args["obs_encoder"].args["share_rgb_model"] is share_rgb_model
    assert actor.args["obs_encoder"].args["resize_shape"] == resize_shape
    assert actor.args["obs_encoder"].args["crop_shape"] == crop_shape
    assert actor.args["obs_encoder"].args["random_crop"] is random_crop
    assert actor.args["use_aug"] is use_aug
    assert actor.args["ddim_noise_scheduler"] == ("native scheduler", "ddim")
    assert actor.args["n_action_steps"] == 4

    def unavailable_weights(**kwargs):
        raise RuntimeError("pretrained weights unavailable")

    getter.get_resnet = unavailable_weights
    with pytest.raises(RuntimeError, match="pretrained weights unavailable"):
        build_rl100_dp(meta, config)


@pytest.mark.parametrize("use_recon,use_vib", [(False, False), (False, True), (True, False), (True, True)])
def test_factory_passes_native_visual_auxiliary_options(monkeypatch, use_recon, use_vib):
    """Native VIB and reconstruction heads are constructed by upstream, not ARBiM."""
    import sys
    import types

    def module(name):
        m = types.ModuleType(name)
        m.__path__ = []
        monkeypatch.setitem(sys.modules, name, m)
        return m

    module("hydra").utils = module("hydra.utils")
    sys.modules["hydra.utils"].instantiate = lambda cfg: cfg
    for name in ("rl_100", "rl_100.model", "rl_100.model.vision", "rl_100.policy"):
        module(name)
    module("rl_100.model.vision.model_getter").get_resnet = lambda **kwargs: kwargs

    class Encoder:
        def __init__(self, **kwargs):
            self.options = kwargs

    class Policy:
        def __init__(self, **kwargs):
            self.options = kwargs

    module("rl_100.model.vision.multi_image_obs_encoder").MultiImageObsEncoder = Encoder
    module("rl_100.policy.rl100_2d").RL1002D = Policy
    config = {
        "vision": {
            "rgb_model": {"name": "resnet18", "weights": None},
            "use_recon": use_recon, "use_vib": use_vib,
            "kl_beta": 2.5e-4, "recon_loss_weight": 0.05,
        },
        "policy": {
            "model": "dp3", "scheduler_type": "ddim", "use_recon": use_recon,
        },
        "schedulers": {"ddim": {"_target_": "ddim"}, "cm": {"_target_": "cm"}},
    }
    policy = build_rl100_dp({}, config)
    assert policy.options["obs_encoder"].options["use_recon"] is use_recon
    assert policy.options["obs_encoder"].options["use_vib"] is use_vib
    assert policy.options["obs_encoder"].options["kl_beta"] == 2.5e-4
    assert policy.options["use_recon"] is use_recon
    config["policy"]["use_recon"] = not use_recon
    with pytest.raises(ValueError, match="must match"):
        build_rl100_dp({}, config)
