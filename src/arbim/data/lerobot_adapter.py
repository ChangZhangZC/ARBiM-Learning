"""LeRobot v3 dataset adapter for RL-100 2D imitation learning.

This module changes the *data interface*, not RL-100's policy, loss, diffusion,
image encoder or normalizer. It requires the pinned LeRobot and RL-100 sources.
"""

from __future__ import annotations

from collections import UserDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rl_100.dataset.base_dataset import BaseDataset
from rl_100.model.common.normalizer import LinearNormalizer


def _is_depth_feature(key: str, feature: Mapping[str, Any]) -> bool:
    """Exclude known depth fields, including three-channel encoded depth."""
    info = feature.get("info") or {}
    video_info = feature.get("video_info") or {}
    flags = (
        feature.get("video.is_depth_map"), feature.get("is_depth_map"),
        info.get("is_depth_map"), info.get("video.is_depth_map"),
        video_info.get("is_depth_map"), video_info.get("video.is_depth_map"),
    )
    tokens = key.lower().replace("-", "_").replace(".", "_").replace("/", "_").split("_")
    return any(bool(flag) for flag in flags) or "depth" in tokens


def _rgb_shape(feature: Mapping[str, Any], key: str) -> tuple[tuple[int, int, int], str]:
    """Convert HWC or CHW metadata to the CHW shape required by RL1002D."""
    shape = tuple(int(x) for x in feature["shape"])
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError(f"RGB camera {key!r} must have 3 positive dimensions, got {shape}.")
    first, last = shape[0] == 3, shape[-1] == 3
    if first and last:
        names = [str(x).lower() for x in feature.get("names", ())]
        if len(names) == 3 and names[0] in ("channel", "channels"):
            return shape, "CHW"
        if len(names) == 3 and names[-1] in ("channel", "channels"):
            return (3, shape[0], shape[1]), "HWC"
        raise ValueError(f"Ambiguous RGB layout for {key!r}: {shape}; supply channel dimension names.")
    if last:
        return (3, shape[0], shape[1]), "HWC"
    if first:
        return shape, "CHW"
    raise ValueError(f"RGB camera {key!r} must have 3 channels, got {shape}.")


def _feature_names(feature: Mapping[str, Any], label: str, dim: int) -> tuple[str, ...]:
    names = feature.get("names")
    if isinstance(names, Mapping):
        names = names.get(f"{label}_names")
    if names is None:
        return ()
    if not isinstance(names, (list, tuple)) or len(names) != dim:
        raise ValueError(f"{label} names must match flattened dimension {dim}.")
    return tuple(str(name) for name in names)


def infer_dataset_spec(
    metadata: Any,
    *,
    state_key: str | None = None,
    action_key: str = "action",
    camera_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Infer data dimensions; physical joint/command semantics remain embodiment-specific."""
    features: Mapping[str, dict] = metadata.features
    if state_key is None:
        state_key = next((k for k in ("observation.state", "state") if k in features), None)
        if state_key is None:
            raise ValueError("Cannot infer robot state; pass state_key explicitly.")
    if state_key == action_key:
        raise ValueError("State and action must reference different features.")
    for key, label in ((state_key, "state"), (action_key, "action")):
        if key not in features:
            raise ValueError(f"Missing {label} feature {key!r}.")
        if features[key].get("dtype") not in ("float32", "float64"):
            raise ValueError(f"{label} feature {key!r} must be floating-point.")

    state_shape = tuple(int(x) for x in features[state_key]["shape"])
    action_shape = tuple(int(x) for x in features[action_key]["shape"])
    if not state_shape or not action_shape or min((*state_shape, *action_shape)) < 1:
        raise ValueError("State/action shapes must contain positive dimensions.")
    state_dim, action_dim = int(np.prod(state_shape)), int(np.prod(action_shape))
    if camera_keys is None:
        selected = sorted(
            key for key, ft in features.items()
            if ft.get("dtype") in ("image", "video")
            and not _is_depth_feature(key, ft)
            and len(ft.get("shape", ())) == 3
            and (ft["shape"][0] == 3 or ft["shape"][-1] == 3)
        )
    else:
        selected = list(camera_keys)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("At least one unique RGB camera is required for this DP-2D adapter.")
    shapes, layouts = {}, {}
    for key in selected:
        if key not in features or features[key].get("dtype") not in ("image", "video"):
            raise ValueError(f"Camera {key!r} must be an image/video feature.")
        if _is_depth_feature(key, features[key]):
            raise ValueError(f"Camera {key!r} is a depth feature, not RGB.")
        shapes[key], layouts[key] = _rgb_shape(features[key], key)

    fps = float(metadata.fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid FPS: {metadata.fps!r}.")
    return {
        "state_key": state_key, "action_key": action_key,
        "state_shape": state_shape, "action_shape": action_shape,
        "state_dim": state_dim, "action_dim": action_dim,
        "state_names": _feature_names(features[state_key], "state", state_dim),
        "action_names": _feature_names(features[action_key], "action", action_dim),
        "camera_keys": tuple(selected), "camera_shapes": shapes,
        "camera_layouts": layouts, "fps": fps,
    }


def build_shape_meta(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Build native RL1002D shape_meta.

    The upstream dict_apply recurses into plain dicts, so its inner feature
    mappings must be UserDict. Convert them with dict(value) when serializing.
    """
    obs = {"agent_pos": UserDict({"shape": [spec["state_dim"]], "type": "low_dim"})}
    for key in spec["camera_keys"]:
        obs[key] = UserDict({"shape": list(spec["camera_shapes"][key]), "type": "rgb"})
    return {"obs": obs, "action": {"shape": [spec["action_dim"]]}}


def _rgb_window(image: Any, key: str, steps: int, shape: tuple[int, ...]) -> torch.Tensor:
    """Accept CHW/HWC windows, keeping uint8 compact until trainer transfer."""
    image = torch.as_tensor(image)
    if image.ndim == 3 and steps == 1:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[0] != steps:
        raise ValueError(f"Camera {key!r} expected {steps} frames, got {tuple(image.shape)}.")
    if tuple(image.shape[1:]) == shape:
        pass
    elif tuple(image.shape[1:]) == (shape[1], shape[2], 3):
        image = image.permute(0, 3, 1, 2).contiguous()
    else:
        raise ValueError(f"Camera {key!r} expected CHW {shape}, got {tuple(image.shape[1:])}.")
    if image.dtype == torch.uint8:
        return image
    if not image.is_floating_point():
        raise ValueError(f"Camera {key!r} has unsupported dtype {image.dtype}.")
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError(f"Camera {key!r} floating RGB must be finite and in [0,1].")
    return image.to(torch.float32)


def _flat_window(value: Any, key: str, steps: int, shape: tuple[int, ...]) -> torch.Tensor:
    """Validate original feature shape before flattening robot dimensions."""
    tensor = torch.as_tensor(value, dtype=torch.float32)
    expected = (steps, *shape)
    if steps == 1 and tuple(tensor.shape) == shape:
        tensor = tensor.unsqueeze(0)
    if tuple(tensor.shape) != expected:
        raise ValueError(f"Feature {key!r}: expected {expected}, got {tuple(tensor.shape)}.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"Feature {key!r} contains NaN or Inf.")
    return tensor.reshape(steps, -1)


class LeRobotDP2DDataset(BaseDataset):
    """LeRobot v3 windows for RL1002D IL with an episode-level holdout.

    Obs are [t-To+1,...,t], actions are [t-To+1,...,t+H-To].
    Native RL1002D(no_pre_action=True) supervises actions starting at t.
    By default, exclude anchors with padded future supervised actions.
    RGB may be uint8: the IL trainer must convert RGB to float32 [0,1]
    before calling RL1002D.compute_loss(). Do not normalize state/action here.
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
        drop_padded_action_targets: bool = True,
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

        self.repo_id, self.root = repo_id, root
        self.horizon, self.n_obs_steps = horizon, n_obs_steps
        self.prediction_horizon = horizon - n_obs_steps + 1
        self.val_ratio, self.seed = val_ratio, seed
        self.max_train_episodes, self.split = max_train_episodes, split
        self.video_backend = video_backend
        self.drop_padded_action_targets = drop_padded_action_targets
        self._state_key, self._action_key = state_key, action_key
        self._camera_keys = tuple(camera_keys) if camera_keys is not None else None

        metadata = LeRobotDatasetMetadata(repo_id, root=root)
        self.spec = infer_dataset_spec(
            metadata, state_key=state_key, action_key=action_key, camera_keys=self._camera_keys
        )
        n_episodes = int(metadata.total_episodes)
        if n_episodes < 1:
            raise ValueError("LeRobot dataset contains no episodes.")
        if val_ratio and n_episodes < 2:
            raise ValueError("At least two episodes required for held-out validation; use val_ratio=0 to disable.")
        rng = np.random.default_rng(seed)
        n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1) if val_ratio else 0
        val_ids = sorted(int(i) for i in rng.choice(n_episodes, n_val, replace=False))
        train_ids = sorted(set(range(n_episodes)) - set(val_ids))
        if max_train_episodes is not None and len(train_ids) > max_train_episodes:
            train_ids = sorted(int(i) for i in rng.choice(train_ids, max_train_episodes, replace=False))
        self._train_episodes, self._val_episodes = tuple(train_ids), tuple(val_ids)
        selected = train_ids if split == "train" else val_ids

        self._dataset = None
        self._valid_indices: np.ndarray | None = None
        if selected:
            fps = self.spec["fps"]
            obs_steps = [i / fps for i in range(1 - n_obs_steps, 1)]
            action_steps = [i / fps for i in range(1 - n_obs_steps, horizon - n_obs_steps + 1)]
            deltas = {self.spec["state_key"]: obs_steps, self.spec["action_key"]: action_steps}
            for key in self.spec["camera_keys"]:
                deltas[key] = obs_steps
            self._dataset = LeRobotDataset(
                repo_id, root=root, episodes=list(selected), delta_timestamps=deltas,
                video_backend=video_backend, return_uint8=True,
            )
            if drop_padded_action_targets:
                # These columns are metadata only. Absolute frame IDs remain
                # global even when LeRobotDataset filters the selected episodes.
                table = self._dataset.hf_dataset.data
                frame_idx = table.column("index").to_numpy(zero_copy_only=False)
                episode_idx = table.column("episode_index").to_numpy(zero_copy_only=False)
                ep_ends = np.asarray(
                    [metadata.episodes[int(e)]["dataset_to_index"] for e in range(n_episodes)],
                    dtype=np.int64,
                )
                self._valid_indices = np.flatnonzero(
                    frame_idx + (horizon - n_obs_steps) < ep_ends[episode_idx]
                )
                if len(self._valid_indices) == 0:
                    raise ValueError(f"No unpadded action windows in {split} episodes; reduce horizon.")

    def __len__(self) -> int:
        if self._dataset is None:
            return 0
        return len(self._valid_indices) if self._valid_indices is not None else len(self._dataset)

    def get_length(self) -> int:
        return len(self)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._dataset is None:
            raise IndexError("Empty dataset split.")
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        raw_idx = int(self._valid_indices[idx]) if self._valid_indices is not None else idx
        raw = self._dataset[raw_idx]
        spec = self.spec
        obs = {
            "agent_pos": _flat_window(
                raw[spec["state_key"]], spec["state_key"], self.n_obs_steps, spec["state_shape"]
            )
        }
        for key in spec["camera_keys"]:
            obs[key] = _rgb_window(raw[key], key, self.n_obs_steps, spec["camera_shapes"][key])
        action = _flat_window(
            raw[spec["action_key"]], spec["action_key"], self.horizon, spec["action_shape"]
        )
        pad_key = f"{spec['action_key']}_is_pad"
        if pad_key not in raw:
            raise KeyError(f"LeRobot did not return {pad_key!r} for action window.")
        action_is_pad = torch.as_tensor(raw[pad_key], dtype=torch.bool)
        if tuple(action_is_pad.shape) != (self.horizon,):
            raise ValueError(f"Expected {pad_key} shape [{self.horizon}], got {tuple(action_is_pad.shape)}.")
        if self.drop_padded_action_targets and action_is_pad[self.n_obs_steps - 1:].any():
            raise ValueError("Supervised action is padded; check LeRobot episode sampling.")
        return {"obs": obs, "action": action, "action_is_pad": action_is_pad}

    def get_validation_dataset(self) -> LeRobotDP2DDataset:
        if not self._val_episodes:
            raise ValueError("No validation episodes; enable val_ratio or skip validation in trainer.")
        return type(self)(
            repo_id=self.repo_id, root=self.root, horizon=self.horizon,
            n_obs_steps=self.n_obs_steps, val_ratio=self.val_ratio, seed=self.seed,
            max_train_episodes=self.max_train_episodes, state_key=self.spec["state_key"],
            action_key=self.spec["action_key"], camera_keys=self.spec["camera_keys"],
            split="val", video_backend=self.video_backend,
            drop_padded_action_targets=self.drop_padded_action_targets,
        )

    def get_shape_info(
        self, horizon: int | None = None, n_obs_steps: int | None = None
    ) -> dict[str, Any]:
        """Return feature shapes; accept sampled H or modeled H-To+1.

        The native RL-100 training workspace sometimes passes modeled horizon
        rather than the complete action-window horizon to this method.
        """
        if n_obs_steps is not None and n_obs_steps != self.n_obs_steps:
            raise ValueError(f"n_obs_steps mismatch: {n_obs_steps} != {self.n_obs_steps}")
        if horizon is not None and horizon not in (self.horizon, self.prediction_horizon):
            raise ValueError("Expected sample horizon or modeled prediction_horizon.")
        shape_meta = build_shape_meta(self.spec)
        return {
            "obs": {key: tuple(value["shape"]) for key, value in shape_meta["obs"].items()},
            "action": tuple(shape_meta["action"]["shape"]),
        }

    def _iter_numeric_chunks(self, keys: Sequence[str], chunk_size: int = 8192):
        """Read train-only numeric Parquet columns; do not decode RGB."""
        if self.split != "train" or self._dataset is None:
            raise RuntimeError("Normalizer/actions must use the training split.")
        data = self._dataset.hf_dataset.select_columns(list(keys))
        dims = {
            self.spec["state_key"]: self.spec["state_dim"],
            self.spec["action_key"]: self.spec["action_dim"],
        }
        for start in range(0, len(data), chunk_size):
            rows = data[start : start + chunk_size]
            result = {}
            for key in keys:
                values = [torch.as_tensor(row, dtype=torch.float32).reshape(-1) for row in rows[key]]
                tensor = torch.stack(values)
                if tensor.shape[1] != dims[key] or not torch.isfinite(tensor).all():
                    raise ValueError(f"Invalid shape or non-finite values in {key!r} train data.")
                result[key] = tensor
            yield result

    def get_normalizer(self, mode: str = "limits", **kwargs: Any) -> LinearNormalizer:
        """Fit native RL100 limits from streaming extrema over train episodes.

        Fitting two endpoint rows preserves scale/offset but not true mean/std;
        input_stats moments must not be interpreted as full-data moments.
        """
        if mode != "limits":
            raise ValueError("Streaming normalizer currently supports only native mode='limits'.")
        keys = (self.spec["state_key"], self.spec["action_key"])
        bounds = {
            key: [torch.full((dim,), float("inf")), torch.full((dim,), -float("inf"))]
            for key, dim in (
                (keys[0], self.spec["state_dim"]), (keys[1], self.spec["action_dim"])
            )
        }
        for chunk in self._iter_numeric_chunks(keys):
            for key in keys:
                bounds[key][0] = torch.minimum(bounds[key][0], chunk[key].amin(dim=0))
                bounds[key][1] = torch.maximum(bounds[key][1], chunk[key].amax(dim=0))
        normalizer = LinearNormalizer()
        normalizer.fit(
            data={
                "agent_pos": torch.stack(bounds[keys[0]]),
                "action": torch.stack(bounds[keys[1]]),
            },
            last_n_dims=1, mode=mode, **kwargs,
        )
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        """RL-100 compatibility API; materializes all actions when called."""
        key = self.spec["action_key"]
        return torch.cat([item[key] for item in self._iter_numeric_chunks((key,))], dim=0)
