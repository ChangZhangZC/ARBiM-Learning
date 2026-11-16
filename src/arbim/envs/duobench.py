"""DuoBench Hinge-Chest environment and data-contract adapter.

Only the environment boundary lives here. Model loading, observation-history
stacking, Action Chunk scheduling, and evaluation belong to other modules.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

ARMS = ("left", "right")
CAMERAS = ("head", "left_wrist", "right_wrist")
JOINTS_PER_ARM = 7
DIM = 16
RCS_NAMES = tuple(
    name
    for arm in ARMS
    for name in (*(f"{arm}_joint_{i}" for i in range(JOINTS_PER_ARM)), f"{arm}_gripper")
)


def validate_dataset_spec(spec: Mapping[str, Any], camera_map: Mapping[str, str]) -> None:
    """Reject incompatible checkpoint semantics, not just mismatched dimensions.

    RCS LeRobot's joint converter writes left joints/gripper followed by right
    joints/gripper. Unnamed 16D datasets cannot establish this ordering.
    """
    if spec.get("state_dim") != DIM or spec.get("action_dim") != DIM:
        raise ValueError("DuoBench joint adapter requires 16D state and action.")
    if tuple(spec.get("state_shape", ())) != (DIM,) or tuple(spec.get("action_shape", ())) != (DIM,):
        raise ValueError("Expected flat [16] state/action features from RCS LeRobot conversion.")
    for key in ("state_names", "action_names"):
        if tuple(spec.get(key) or ()) != RCS_NAMES:
            raise ValueError(
                f"{key} must identify left joints/gripper then right joints/gripper; "
                "unnamed or differently ordered features are not supported."
            )
    keys = tuple(spec.get("camera_keys", ()))
    if len(keys) != 3 or len(set(keys)) != 3:
        raise ValueError("Hinge-Chest V1 requires exactly three RGB camera keys.")
    if not isinstance(camera_map, Mapping) or set(camera_map) != set(keys):
        raise ValueError("env.camera_map must explicitly map every checkpoint camera key.")
    if set(camera_map.values()) != set(CAMERAS) or len(camera_map) != 3:
        raise ValueError("env.camera_map must map one-to-one to head, left_wrist, right_wrist.")
    shapes = spec.get("camera_shapes")
    if not isinstance(shapes, Mapping):
        raise ValueError("Checkpoint is missing camera_shapes.")
    for key in keys:
        shape = tuple(shapes.get(key, ()))
        if len(shape) != 3 or shape[0] != 3 or any(type(d) is not int or d < 1 for d in shape):
            raise ValueError(f"Invalid RGB CHW camera shape for {key!r}: {shape!r}.")


def create_duobench_env(settings: Mapping[str, Any]):
    """Build the native gymnasium environment without rewriting DuoBench."""
    if settings.get("backend", "duobench") != "duobench":
        raise ValueError("This adapter supports only the 'duobench' backend.")
    if settings.get("task") != "hinge_chest":
        raise ValueError("Task 5.2 currently supports only 'hinge_chest'.")
    if settings.get("control_mode", "joints") != "joints":
        raise ValueError("The RCS LeRobot action contract requires control_mode='joints'.")
    if settings.get("relative_to", "none") != "none":
        raise ValueError("The RCS LeRobot action contract requires relative_to='none'.")
    if settings.get("binary_gripper", True) is not True:
        raise ValueError("Task 5.2 requires RCS binary_gripper=True.")
    frequency = settings.get("frequency", 30)
    if type(frequency) not in (int, float) or not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("env.frequency must be a positive finite number.")
    for flag in ("headless", "realtime", "async_control"):
        if flag in settings and type(settings[flag]) is not bool:
            raise ValueError(f"env.{flag} must be boolean.")

    import gymnasium as gym
    from duobench.tasks.hinge_chest import HingeChestEnvConfig
    from rcs.envs.base import ControlMode, RelativeTo

    cfg = HingeChestEnvConfig().config()
    cfg.control_mode = ControlMode.JOINTS
    cfg.relative_to = RelativeTo.NONE
    cfg.max_relative_movement = None
    cfg.wrapper_cfg.binary_gripper = True
    cfg.wrapper_cfg.include_depth = False
    cfg.headless = settings.get("headless", True)
    # Mutate, rather than replace, DuoBench's SimConfig: it sets extra
    # MuJoCo memory for contact-rich scenes.
    cfg.sim_cfg.frequency = frequency
    cfg.sim_cfg.realtime = settings.get("realtime", False)
    cfg.sim_cfg.async_control = settings.get("async_control", True)
    return gym.make("duobench/hinge_chest", cfg=cfg)


def to_env_action(action: Any) -> dict[str, dict[str, np.ndarray]]:
    """Translate a single *unnormalized* RCS 16D action to native joint commands."""
    vector = np.asarray(action, dtype=np.float32)
    if vector.shape != (DIM,) or not np.isfinite(vector).all():
        raise ValueError(f"Expected one finite 16D action, got {vector.shape}.")
    return {
        arm: {
            "joints": vector[offset:offset + JOINTS_PER_ARM].copy(),
            # Match the native RCS gripper wrapper's [0,1] clipping.
            "gripper": np.clip(vector[offset + JOINTS_PER_ARM:offset + 8], 0.0, 1.0).copy(),
        }
        for arm, offset in (("left", 0), ("right", 8))
    }


def _rgb_frame(frame: Any, shape: tuple[int, int, int], key: str) -> np.ndarray:
    """Match the dataset's HWC RGB resolution using the converter's resize."""
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError(f"Camera {key!r} must be HWC uint8 RGB, got {image.shape} {image.dtype}.")
    height, width = shape[1:]
    if image.shape[:2] == (height, width):
        return image
    # RCS's LeRobot converter uses torchvision v2.Resize on CHW uint8 tensors.
    import torch
    from torchvision.transforms import v2

    chw = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
    resized = v2.Resize((height, width))(chw)
    return resized.permute(1, 2, 0).contiguous().numpy()


def to_dataset_observation(
    raw: Mapping[str, Any],
    spec: Mapping[str, Any],
    camera_map: Mapping[str, str],
    *,
    binarize_state_gripper: bool = True,
    gripper_threshold: float = 0.9,
) -> dict[str, np.ndarray]:
    """Convert one raw DuoBench frame into original LeRobot feature keys.

    No observation window, batching, state normalization, or RL100 diffusion
    is applied here. The caller handles temporal stacking before inference.
    """
    if not isinstance(binarize_state_gripper, bool):
        raise ValueError("binarize_state_gripper must be boolean.")
    if not np.isfinite(gripper_threshold) or not 0.0 <= gripper_threshold <= 1.0:
        raise ValueError("gripper_threshold must lie in [0, 1].")
    validate_dataset_spec(spec, camera_map)
    states = []
    for arm in ARMS:
        item = raw.get(arm)
        if not isinstance(item, Mapping):
            raise KeyError(f"Missing DuoBench {arm!r} observation.")
        joints = np.asarray(item["joints"], dtype=np.float32)
        gripper = np.asarray(item["gripper"], dtype=np.float32)
        if joints.shape != (JOINTS_PER_ARM,) or gripper.shape != (1,):
            raise ValueError(f"{arm} requires joints [7] and gripper [1].")
        if not np.isfinite(joints).all() or not np.isfinite(gripper).all():
            raise ValueError(f"{arm} observation contains NaN or Inf.")
        if binarize_state_gripper:
            gripper = (gripper > gripper_threshold).astype(np.float32)
        states.append(np.concatenate((joints, gripper)))
    result = {spec["state_key"]: np.concatenate(states).astype(np.float32)}
    raw_frames = raw.get("frames")
    if not isinstance(raw_frames, Mapping):
        raise ValueError("DuoBench RGB frames unavailable; check camera_available.")
    for key in spec["camera_keys"]:
        camera = camera_map[key]
        try:
            frame = raw_frames[camera]["rgb"]["data"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Missing DuoBench RGB frame {camera!r} for {key!r}.") from exc
        result[key] = _rgb_frame(frame, tuple(spec["camera_shapes"][key]), key)
    return result


class DuoBenchAdapter:
    """Thin Gymnasium wrapper; all temporal policy state stays in rollout code."""

    def __init__(
        self,
        env: Any,
        spec: Mapping[str, Any],
        camera_map: Mapping[str, str],
        *,
        binarize_state_gripper: bool = True,
        gripper_threshold: float = 0.9,
    ):
        validate_dataset_spec(spec, camera_map)
        self.env, self.spec, self.camera_map = env, spec, dict(camera_map)
        self.binarize_state_gripper = binarize_state_gripper
        self.gripper_threshold = gripper_threshold
        self.last_raw_observation: Mapping[str, Any] | None = None
        # Check nested action shape when the environment provides Gym spaces.
        if hasattr(env, "action_space"):
            for arm in ARMS:
                try:
                    left_or_right = env.action_space.spaces[arm].spaces
                    joint_shape = left_or_right["joints"].shape
                    gripper_shape = left_or_right["gripper"].shape
                except (AttributeError, KeyError, TypeError) as exc:
                    raise ValueError(f"DuoBench action space has no {arm} joints/gripper.") from exc
                if tuple(joint_shape) != (7,) or tuple(gripper_shape) != (1,):
                    raise ValueError(f"DuoBench {arm} action space must be joints [7], gripper [1].")

    def observation(self, raw: Mapping[str, Any]) -> dict[str, np.ndarray]:
        return to_dataset_observation(
            raw, self.spec, self.camera_map,
            binarize_state_gripper=self.binarize_state_gripper,
            gripper_threshold=self.gripper_threshold,
        )

    def reset(self, *, seed: int | None = None):
        raw, info = self.env.reset(seed=seed)
        self.last_raw_observation = raw
        return self.observation(raw), info

    def step(self, action: Any):
        raw, reward, terminated, truncated, info = self.env.step(to_env_action(action))
        self.last_raw_observation = raw
        return self.observation(raw), reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()
