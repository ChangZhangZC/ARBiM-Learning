"""LeRobot v3 dataset adapter for RL-100 2D imitation learning.

This module changes the *data interface*, not RL-100's policy, loss, diffusion,
image encoder or normalizer. It requires the pinned LeRobot and RL-100 sources.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rl_100.dataset.base_dataset import BaseDataset
from rl_100.model.common.normalizer import LinearNormalizer


def infer_dataset_spec(
    metadata: Any,
    *,
    state_key: str | None = None,
    action_key: str = "action",
    camera_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Infer RL-100 dimensions and RGB keys from LeRobot v3 metadata."""
    features: Mapping[str, dict] = metadata.features
    if state_key is None:
        state_key = next((k for k in ("observation.state", "state") if k in features), None)
        if state_key is None:
            raise ValueError("Cannot infer robot state; pass state_key explicitly.")
    for key, label in ((state_key, "state"), (action_key, "action")):
        if key not in features:
            raise ValueError(f"Missing {label} feature {key!r} in LeRobot metadata.")
        if features[key]["dtype"] in ("image", "video"):
            raise ValueError(f"{label} feature {key!r} must be numeric, not visual.")
    if state_key == action_key:
        raise ValueError("State and action must reference different features.")

    state_shape = tuple(int(x) for x in features[state_key]["shape"])
    action_shape = tuple(int(x) for x in features[action_key]["shape"])
    if not state_shape or any(x < 1 for x in state_shape):
        raise ValueError(f"Invalid state shape: {state_shape}")
    if not action_shape or any(x < 1 for x in action_shape):
        raise ValueError(f"Invalid action shape: {action_shape}")
    if features[state_key]["dtype"] not in ("float32", "float64"):
        raise ValueError(f"State must be floating-point: {state_key!r}")
    if features[action_key]["dtype"] not in ("float32", "float64"):
        raise ValueError(f"Action must be floating-point: {action_key!r}")

    if camera_keys is None:
        candidates = [k for k, ft in features.items() if ft["dtype"] in ("image", "video")]
        selected = [
            k for k in candidates
            if not features[k].get("video.is_depth_map", False)
            and len(features[k]["shape"]) == 3
            and int(features[k]["shape"][-1]) == 3
        ]
    else:
        selected = list(camera_keys)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("At least one unique RGB camera key is required for DP-2D.")

    image_shapes: dict[str, tuple[int, int, int]] = {}
    for key in selected:
        if key not in features:
            raise ValueError(f"RGB camera feature {key!r} not found in metadata.")
        feature = features[key]
        shape = tuple(int(x) for x in feature["shape"])
        if feature["dtype"] not in ("image", "video") or len(shape) != 3 or shape[-1] != 3:
            raise ValueError(f"Feature {key!r} is not HWC RGB: {shape}.")
        if feature.get("video.is_depth_map", False):
            raise ValueError(f"Feature {key!r} is marked as a depth map, not RGB.")
        if any(x < 1 for x in shape):
            raise ValueError(f"Invalid RGB shape for {key!r}: {shape}.")
        image_shapes[key] = (3, shape[0], shape[1])

    fps = int(metadata.fps)
    if fps < 1:
        raise ValueError(f"Invalid dataset fps: {fps}")
    return {
        "state_key": state_key,
        "action_key": action_key,
        "camera_keys": tuple(selected),
        "state_shape": state_shape,
        "action_shape": action_shape,
        "state_dim": int(np.prod(state_shape)),
        "action_dim": int(np.prod(action_shape)),
        "camera_shapes": image_shapes,
        "fps": fps,
    }


def build_shape_meta(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Build shape_meta accepted by RL1002D and MultiImageObsEncoder."""
    obs = {"agent_pos": {"shape": [spec["state_dim"]], "type": "low_dim"}}
    for key in spec["camera_keys"]:
        obs[key] = {"shape": list(spec["camera_shapes"][key]), "type": "rgb"}
    return {"obs": obs, "action": {"shape": [spec["action_dim"]]}}


def _float_rgb(image: Any, key: str, n_steps: int, shape: tuple[int, ...]) -> torch.Tensor:
    image = torch.as_tensor(image)
    if image.ndim == 3 and n_steps == 1:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[0] != n_steps or tuple(image.shape[1:]) != shape:
        raise ValueError(f"Camera {key!r}: expected [{n_steps}, {', '.join(map(str, shape))}], got {tuple(image.shape)}")
    if image.dtype == torch.uint8:
        image = image.to(torch.float32) / 255.0
    elif image.is_floating_point():
        image = image.to(torch.float32)
    else:
        raise ValueError(f"Unsupported RGB dtype for {key!r}: {image.dtype}")
    if not torch.isfinite(image).all() or image.min() < -1e-6 or image.max() > 1 + 1e-6:
        raise ValueError(f"Camera {key!r}: RGB must be finite in [0,1] after decoding.")
    return image


def _flat_window(value: Any, key: str, steps: int, dim: int) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 1 and steps == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.numel() != steps * dim or tensor.shape[0] != steps:
        raise ValueError(f"Feature {key!r}: expected [{steps}, ..., D={dim}], got {tuple(tensor.shape)}")
    tensor = tensor.reshape(steps, dim)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"Feature {key!r} contains NaN or Inf.")
    return tensor


class LeRobotDP2DDataset(BaseDataset):
    """LeRobot v3 time windows with the existing RL-100 2D Dataset contract.

    One sample is anchored to one recorded frame. Observations end at that
    frame; actions start at t-(n_obs_steps-1). This matches RL1002D's fixed
    no_pre_action=True loss slicing. LeRobot performs episode-edge padding.

    Normalization is not applied here; RL1002D.compute_loss owns that step.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        *,
        horizon: int = 5,
        n_obs_steps: int = 2,
        val_ratio: float = 0.02,
        seed: int = 42,
        max_train_episodes: int | None = None,
        state_key: str | None = None,
        action_key: str = "action",
        camera_keys: Sequence[str] | None = None,
        split: str = "train",
        video_backend: str | None = None,
    ):
        super().__init__()
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'.")
        if n_obs_steps < 1 or horizon < n_obs_steps:
            raise ValueError("Require horizon >= n_obs_steps >= 1 for RL1002D no_pre_action=True.")
        if not 0 <= val_ratio < 1:
            raise ValueError("val_ratio must be in [0,1).")
        if max_train_episodes is not None and max_train_episodes < 1:
            raise ValueError("max_train_episodes must be positive.")

        from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

        self.repo_id = repo_id
        self.root = root
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.val_ratio = val_ratio
        self.seed = seed
        self.max_train_episodes = max_train_episodes
        self.split = split
        self.video_backend = video_backend
        self._state_key = state_key
        self._action_key = action_key
        self._camera_keys = tuple(camera_keys) if camera_keys is not None else None

        metadata = LeRobotDatasetMetadata(repo_id, root=root)
        self.spec = infer_dataset_spec(
            metadata, state_key=state_key, action_key=action_key, camera_keys=self._camera_keys
        )
        n_episodes = int(metadata.total_episodes)
        if n_episodes == 0:
            raise ValueError("LeRobot dataset contains no episodes.")
        rng = np.random.default_rng(seed)
        n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1) if val_ratio > 0 else 0
        val_ids = sorted(int(i) for i in rng.choice(n_episodes, n_val, replace=False))
        train_ids = sorted(set(range(n_episodes)) - set(val_ids))
        if max_train_episodes is not None and len(train_ids) > max_train_episodes:
            train_ids = sorted(int(i) for i in rng.choice(train_ids, max_train_episodes, replace=False))
        self._train_episodes = tuple(train_ids)
        self._val_episodes = tuple(val_ids)
        selected = train_ids if split == "train" else val_ids

        self._dataset = None
        if selected:
            fps = self.spec["fps"]
            obs_steps = [i / fps for i in range(1 - n_obs_steps, 1)]
            action_steps = [i / fps for i in range(1 - n_obs_steps, horizon - n_obs_steps + 1)]
            deltas = {self.spec["state_key"]: obs_steps, self.spec["action_key"]: action_steps}
            for key in self.spec["camera_keys"]:
                deltas[key] = obs_steps
            self._dataset = LeRobotDataset(
                repo_id, root=root, episodes=list(selected), delta_timestamps=deltas,
                video_backend=video_backend,
            )

    def __len__(self) -> int:
        return len(self._dataset) if self._dataset is not None else 0

    def get_length(self) -> int:
        return len(self)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._dataset is None:
            raise IndexError("Empty dataset split.")
        raw = self._dataset[idx]
        spec = self.spec
        obs = {
            "agent_pos": _flat_window(
                raw[spec["state_key"]], spec["state_key"], self.n_obs_steps, spec["state_dim"]
            )
        }
        for key in spec["camera_keys"]:
            obs[key] = _float_rgb(raw[key], key, self.n_obs_steps, spec["camera_shapes"][key])
        action = _flat_window(raw[spec["action_key"]], spec["action_key"], self.horizon, spec["action_dim"])
        return {"obs": obs, "action": action}

    def get_validation_dataset(self) -> LeRobotDP2DDataset:
        return type(self)(
            repo_id=self.repo_id,
            root=self.root,
            horizon=self.horizon,
            n_obs_steps=self.n_obs_steps,
            val_ratio=self.val_ratio,
            seed=self.seed,
            max_train_episodes=self.max_train_episodes,
            state_key=self.spec["state_key"],
            action_key=self.spec["action_key"],
            camera_keys=self.spec["camera_keys"],
            split="val",
            video_backend=self.video_backend,
        )

    def get_shape_info(self, horizon: int | None = None, n_obs_steps: int | None = None) -> dict[str, Any]:
        if n_obs_steps is not None and n_obs_steps != self.n_obs_steps:
            raise ValueError(f"n_obs_steps mismatch: {n_obs_steps} != {self.n_obs_steps}")
        if horizon is not None and horizon not in (self.horizon, self.horizon - self.n_obs_steps + 1):
            raise ValueError("horizon does not match the configured RL1002D action window.")
        shape_meta = build_shape_meta(self.spec)
        return {
            "obs": {key: tuple(value["shape"]) for key, value in shape_meta["obs"].items()},
            "action": tuple(shape_meta["action"]["shape"]),
        }

    def _get_numeric_columns(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Read training state/action columns only; do not decode RGB."""
        if self.split != "train" or self._dataset is None:
            raise RuntimeError("Normalizer statistics must be fitted on the training split.")
        keys = (self.spec["state_key"], self.spec["action_key"])
        data = self._dataset.hf_dataset.select_columns(list(keys))
        chunks: dict[str, list[torch.Tensor]] = {k: [] for k in keys}
        for start in range(0, len(data), 8192):
            rows = data[start : start + 8192]
            for key in keys:
                tensors = [torch.as_tensor(x, dtype=torch.float32).reshape(-1) for x in rows[key]]
                chunks[key].append(torch.stack(tensors))
        state = torch.cat(chunks[keys[0]], dim=0)
        action = torch.cat(chunks[keys[1]], dim=0)
        if not torch.isfinite(state).all() or not torch.isfinite(action).all():
            raise ValueError("Training state/action contains NaN or Inf.")
        if state.shape[1] != self.spec["state_dim"] or action.shape[1] != self.spec["action_dim"]:
            raise ValueError("Numeric feature dimensions do not match metadata.")
        return state, action

    def get_normalizer(self, mode: str = "limits", **kwargs: Any) -> LinearNormalizer:
        state, action = self._get_numeric_columns()
        normalizer = LinearNormalizer()
        normalizer.fit(
            data={"agent_pos": state, "action": action},
            last_n_dims=1, mode=mode, **kwargs,
        )
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        _, actions = self._get_numeric_columns()
        return actions
