"""Construct the pinned, unmodified RL-100 2D diffusion policy for IL."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


def build_rl100_dp(shape_meta: Mapping[str, Any], config: Mapping[str, Any]):
    """Build native RL1002D with the original encoder, UNet and schedulers.

    Keep shape_meta's UserDict feature attributes intact: RL-100 dict_apply
    recursively descends into plain dicts and otherwise loses feature shapes.
    """
    from hydra.utils import instantiate
    from rl_100.model.vision.model_getter import get_resnet
    from rl_100.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
    from rl_100.policy.rl100_2d import RL1002D

    policy_cfg = dict(config["policy"])
    vision_cfg = dict(config["vision"])
    rgb_cfg = dict(vision_cfg.pop("rgb_model"))
    if policy_cfg.get("model") != "dp3" or policy_cfg.get("scheduler_type") != "ddim":
        raise ValueError("Task 3 V1 requires native model='dp3' and scheduler_type='ddim'.")
    if rgb_cfg != {"name": "resnet18", "weights": "r3m"}:
        raise ValueError("Task 3 V1 requires ResNet18 with R3M weights (no fallback).")
    if vision_cfg.get("share_rgb_model") is not False:
        raise ValueError("Task 3 V1 requires separate native camera encoders.")
    if vision_cfg.get("crop_shape") is not None or policy_cfg.get("use_aug") is not False:
        raise ValueError("Task 3 V1 disables random crop and additional RGB shifts.")
    if policy_cfg.get("use_recon") or vision_cfg.get("use_recon") or vision_cfg.get("use_vib"):
        raise ValueError("Task 3 V1 disables reconstruction and VIB objectives.")

    rgb_model = get_resnet(**rgb_cfg)
    encoder = MultiImageObsEncoder(shape_meta=shape_meta, rgb_model=rgb_model, **vision_cfg)
    schedulers = config["schedulers"]
    ddim_scheduler = instantiate(schedulers["ddim"])
    cm_scheduler = instantiate(schedulers["cm"])
    return RL1002D(
        shape_meta=shape_meta,
        obs_encoder=encoder,
        ddim_noise_scheduler=ddim_scheduler,
        cm_noise_scheduler=cm_scheduler,
        **policy_cfg,
    )


def load_native_policy_weights(policy: torch.nn.Module, path: str | Path) -> None:
    """Strictly load a native model state_dict or an ARBiM training checkpoint.

    The caller must initialize the native LinearNormalizer before loading,
    since its feature parameters are dynamically registered when fitted.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("Expected a model state_dict or ARBiM checkpoint dictionary.")
    state = payload["policy"] if "policy" in payload else payload
    policy.load_state_dict(state, strict=True)
