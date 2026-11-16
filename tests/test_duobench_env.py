"""Task 5.2: pure adapter tests plus opt-in real DuoBench smoke test."""
from __future__ import annotations

import copy
import os
from types import SimpleNamespace

import numpy as np
import pytest

from arbim.envs.duobench import (
    CAMERAS, RCS_NAMES, DuoBenchAdapter, create_duobench_env,
    to_dataset_observation, to_env_action, validate_dataset_spec,
)


@pytest.fixture
def spec():
    keys = (
        "observation.images.head",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    )
    return {
        "state_key": "observation.state",
        "action_key": "action",
        "state_dim": 16,
        "action_dim": 16,
        "state_shape": (16,),
        "action_shape": (16,),
        "state_names": RCS_NAMES,
        "action_names": RCS_NAMES,
        "camera_keys": keys,
        "camera_shapes": {key: (3, 2, 3) for key in keys},
    }


@pytest.fixture
def camera_map(spec):
    return dict(zip(spec["camera_keys"], CAMERAS, strict=True))


def raw_observation(gripper=0.95):
    return {
        "left": {"joints": np.arange(7, dtype=np.float32), "gripper": [gripper]},
        "right": {"joints": np.arange(7, dtype=np.float32) + 10, "gripper": [0.5]},
        "frames": {
            camera: {"rgb": {"data": np.full((2, 3, 3), i, dtype=np.uint8)}}
            for i, camera in enumerate(CAMERAS)
        },
    }


def test_observation_matches_rcs_converter(spec, camera_map):
    obs = to_dataset_observation(raw_observation(), spec, camera_map)
    assert set(obs) == {"observation.state", *spec["camera_keys"]}
    np.testing.assert_array_equal(
        obs["observation.state"],
        [0, 1, 2, 3, 4, 5, 6, 1, 10, 11, 12, 13, 14, 15, 16, 0],
    )
    for i, key in enumerate(spec["camera_keys"]):
        assert obs[key].shape == (2, 3, 3)
        assert obs[key].dtype == np.uint8
        assert np.all(obs[key] == i)


def test_unbinarized_gripper_state(spec, camera_map):
    obs = to_dataset_observation(raw_observation(), spec, camera_map, binarize_state_gripper=False)
    assert obs["observation.state"][7] == pytest.approx(0.95)
    assert obs["observation.state"][15] == pytest.approx(0.5)


def test_action_left_right_order_and_clipping():
    action = np.arange(16, dtype=np.float32)
    action[7], action[15] = -0.2, 1.4
    decoded = to_env_action(action)
    np.testing.assert_array_equal(decoded["left"]["joints"], np.arange(7))
    np.testing.assert_array_equal(decoded["right"]["joints"], np.arange(8, 15))
    np.testing.assert_array_equal(decoded["left"]["gripper"], [0])
    np.testing.assert_array_equal(decoded["right"]["gripper"], [1])
    assert set(decoded) == {"left", "right"}


@pytest.mark.parametrize("action", [
    np.zeros(15), np.zeros((1, 16)), [float("nan")] * 16,
])
def test_invalid_action_rejected(action):
    with pytest.raises(ValueError):
        to_env_action(action)


def test_mismatched_feature_names_fail_closed(spec, camera_map):
    bad = copy.deepcopy(spec)
    bad["action_names"] = ()
    with pytest.raises(ValueError, match="action_names"):
        validate_dataset_spec(bad, camera_map)
    bad = copy.deepcopy(spec)
    bad["state_names"] = tuple(reversed(RCS_NAMES))
    with pytest.raises(ValueError, match="state_names"):
        validate_dataset_spec(bad, camera_map)


def test_camera_map_and_shape_validation(spec, camera_map):
    with pytest.raises(ValueError, match="camera_map"):
        validate_dataset_spec(spec, {spec["camera_keys"][0]: "head"})
    bad = copy.deepcopy(spec)
    bad["camera_shapes"][spec["camera_keys"][0]] = (1, 2, 3)
    with pytest.raises(ValueError, match="RGB CHW"):
        validate_dataset_spec(bad, camera_map)


def test_missing_camera_and_bad_rgb_are_errors(spec, camera_map):
    raw = raw_observation()
    del raw["frames"]["head"]
    with pytest.raises(ValueError, match="Missing DuoBench RGB"):
        to_dataset_observation(raw, spec, camera_map)
    raw = raw_observation()
    raw["frames"]["head"]["rgb"]["data"] = np.zeros((2, 3, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="HWC uint8 RGB"):
        to_dataset_observation(raw, spec, camera_map)


class FakeEnv:
    def __init__(self):
        self.actions = []
        self.closed = False

    def reset(self, *, seed=None):
        self.seed = seed
        return raw_observation(), {"success": False, "stage": 0}

    def step(self, action):
        self.actions.append(action)
        return raw_observation(), 1.0, True, False, {"success": True, "stage": 3}

    def close(self):
        self.closed = True


def test_adapter_reset_step_close(spec, camera_map):
    native = FakeEnv()
    adapter = DuoBenchAdapter(native, spec, camera_map)
    obs, info = adapter.reset(seed=123)
    assert native.seed == 123
    assert obs["observation.state"].shape == (16,)
    assert info["stage"] == 0
    action = np.zeros(16, dtype=np.float32)
    obs, reward, terminated, truncated, info = adapter.step(action)
    assert len(native.actions) == 1 and native.actions[0]["left"]["joints"].shape == (7,)
    assert (reward, terminated, truncated) == (1.0, True, False)
    assert info["success"] is True and info["stage"] == 3
    assert adapter.last_raw_observation is not None
    adapter.close()
    assert native.closed


def test_build_env_rejects_unsupported_modes_without_importing_sim():
    with pytest.raises(ValueError, match="hinge_chest"):
        create_duobench_env({"task": "transfer_cube"})
    with pytest.raises(ValueError, match="joints"):
        create_duobench_env({"task": "hinge_chest", "control_mode": "cartesian"})
    with pytest.raises(ValueError, match="frequency"):
        create_duobench_env({"task": "hinge_chest", "frequency": 0})


@pytest.mark.skipif(os.environ.get("ARBIM_DUOBENCH_SMOKE") != "1",
                    reason="Set ARBIM_DUOBENCH_SMOKE=1 on a workstation with DuoBench/RCS/MuJoCo.")
def test_real_hinge_chest_reset_step(spec, camera_map):
    # Optional integration test: no Policy/Checkpoint is involved.
    real_spec = dict(spec)
    real_spec["camera_shapes"] = {key: (3, 224, 224) for key in spec["camera_keys"]}
    native = create_duobench_env({"task": "hinge_chest", "headless": True})
    try:
        adapter = DuoBenchAdapter(native, real_spec, camera_map)
        obs, info = adapter.reset(seed=7)
        assert obs["observation.state"].shape == (16,)
        assert set(real_spec["camera_keys"]).issubset(obs)
        _, reward, terminated, truncated, info = adapter.step(obs["observation.state"])
        assert "success" in info and "stage" in info
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
    finally:
        native.close()
