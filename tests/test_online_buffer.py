"""Episode boundaries and native PPO tensor layouts."""
import numpy as np
import pytest
import torch

from arbim.rl.online.buffer import OnlinePPOBuffer


@pytest.fixture
def spec():
    return {
        "state_key": "observation.state",
        "camera_keys": ("head", "left", "right"),
        "state_dim": 16,
        "camera_shapes": {k: (3, 2, 3) for k in ("head", "left", "right")},
        "state_shape": (16,),
        "camera_obs_keys": {"head": "rgb_0", "left": "rgb_1", "right": "rgb_2"},
        "action_dim": 16,
        "action_shape": (16,),
        "action_key": "action",
    }


def obs():
    return {
        "observation.state": np.zeros((2, 16), dtype=np.float32),
        **{key: np.zeros((2, 2, 3, 3), dtype=np.uint8) for key in ("head", "left", "right")},
    }


def test_buffer_returns_original_native_ppo_order_and_compact_rgb(spec, monkeypatch):
    from arbim.rl.online import buffer as module

    def fake_prepare(batch, spec, n_obs_steps, *, device):
        return {
            "agent_pos": torch.as_tensor(batch["observation.state"], device=device),
            **{f"rgb_{i}": torch.as_tensor(batch[key], device=device)
               for i, key in enumerate(spec["camera_keys"])},
        }

    monkeypatch.setattr(module, "prepare_observation", fake_prepare)
    storage = OnlinePPOBuffer(
        spec=spec, n_obs_steps=2, prediction_steps=4, num_inference_steps=3,
        action_dim=16, device="cpu",
    )
    frame = obs()
    storage.store(frame, np.ones((4, 4, 16)), np.zeros((3, 4, 16)),
                  0.5, frame, True, True)
    assert storage.records[0]["obs"]["head"].dtype == np.uint8
    frame["head"][0, 0, 0, 0] = 255
    assert storage.records[0]["obs"]["head"][0, 0, 0, 0] == 0
    s, a, lp, r, next_s, dw, done = storage.numpy_to_tensor()
    assert s["agent_pos"].shape == (1, 2, 16)
    assert a.shape == (1, 4, 4, 16)
    assert lp.shape == (1, 3, 4, 16)
    assert (r.item(), dw.item(), done.item()) == (0.5, 1.0, 1.0)
    assert next_s["rgb_2"].shape == (1, 2, 2, 3, 3)
    storage.clear()
    assert storage.count == 0


def test_buffer_rejects_unsafe_shapes_and_termination(spec):
    storage = OnlinePPOBuffer(spec=spec, n_obs_steps=2, prediction_steps=4,
                              num_inference_steps=3, action_dim=16, device="cpu")
    frame = obs()
    with pytest.raises(ValueError, match="shape"):
        storage.store(frame, np.zeros((4, 16)), np.zeros((3, 4, 16)),
                      0, frame, False, False)
    with pytest.raises(ValueError, match="dw=True"):
        storage.store(frame, np.zeros((4, 4, 16)), np.zeros((3, 4, 16)),
                      0, frame, False, True)
    assert storage.count == 0


def test_executed_spans_are_separate_from_native_ddim_tensor_shapes(spec):
    storage = OnlinePPOBuffer(spec=spec, n_obs_steps=2, prediction_steps=4,
                              num_inference_steps=3, action_dim=16, device="cpu", action_span=4)
    frame = obs()
    trajectory = np.zeros((4, 4, 16), dtype=np.float32)
    logprob = np.zeros((3, 4, 16), dtype=np.float32)
    storage.store(frame, trajectory, logprob, 1.0, frame, False, False, executed_steps=4)
    storage.store(frame, trajectory, logprob, 2.0, frame, True, False, executed_steps=1)
    assert storage.has_partial_spans()
    assert storage.executed_spans() == [4, 1]
    assert storage.numpy_to_tensor()[1].shape == (2, 4, 4, 16)
    with pytest.raises(ValueError, match="executed_steps"):
        storage.store(frame, trajectory, logprob, 0.0, frame, True, False, executed_steps=5)
