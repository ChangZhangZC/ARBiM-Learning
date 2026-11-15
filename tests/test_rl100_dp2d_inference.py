"""CPU-only Task 4 tests, with a tiny native-shaped Policy test double.

No R3M downloads, LeRobot datasets, GPU, or upstream RL100 modifications.
"""

from __future__ import annotations

import copy
import sys
import types

import pytest
import torch

from arbim.policy.inference_rl100_dp2d import (
    _model_config, _read_inference_config, load_inference_policy,
    prepare_observation, predict, seed_inference,
)


@pytest.fixture
def contract():
    spec = {
        "state_key": "observation.state", "action_key": "action",
        "state_shape": (2, 3), "action_shape": (2, 3),
        "state_dim": 6, "action_dim": 6,
        "state_names": (), "action_names": (),
        "camera_keys": ("observation.images.front",),
        "camera_shapes": {"observation.images.front": (3, 4, 5)},
        "camera_layouts": {"observation.images.front": "HWC"},
        "camera_obs_keys": {"observation.images.front": "rgb_0"},
        "fps": 30.0,
    }
    return {
        "dataset_spec": spec,
        "dataset": {"repo_id": "fake/dataset", "root": "/unused", "val_ratio": 0.1, "seed": 1},
        "policy": {"model": "dp3", "scheduler_type": "ddim",
                   "horizon": 5, "n_obs_steps": 2, "n_action_steps": 4},
        "vision": {"rgb_model": {"name": "resnet18", "weights": "r3m"}},
        "schedulers": {"ddim": {"_target_": "fake.ddim"}, "cm": {"_target_": "fake.cm"}},
    }


class TinyNormalizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.params_dict = torch.nn.ModuleDict({
            key: torch.nn.ParameterDict({
                "scale": torch.nn.Parameter(torch.ones(6), requires_grad=False),
                "offset": torch.nn.Parameter(torch.zeros(6), requires_grad=False),
            })
            for key in ("agent_pos", "action")
        })


class TinyNativePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6, 6)
        self.normalizer = TinyNormalizer()
        self.horizon, self.n_obs_steps, self.n_action_steps = 5, 2, 4
        self.action_dim, self.no_pre_action = 6, True
        self.last_deterministic = None

    def predict_action(self, observation, *, deterministic=False):
        self.last_deterministic = deterministic
        assert observation["agent_pos"].dtype == torch.float32
        assert observation["rgb_0"].shape[1:] == (2, 3, 4, 5)
        out = self.linear(observation["agent_pos"][:, -1])
        full = out[:, None, :].expand(-1, 4, -1)
        return {"action": full[:, :self.n_action_steps], "action_pred": full}


@pytest.fixture
def native_stubs(monkeypatch):
    from arbim.data import lerobot_adapter as real_adapter

    adapter = types.ModuleType("arbim.data.lerobot_adapter")
    adapter.build_shape_meta = lambda spec: {"obs": {"agent_pos": {"shape": [spec["state_dim"]]}},
                                             "action": {"shape": [spec["action_dim"]]}}
    adapter._flat_window = real_adapter._flat_window
    adapter._rgb_window = real_adapter._rgb_window
    monkeypatch.setitem(sys.modules, "arbim.data.lerobot_adapter", adapter)
    from arbim.policy import rl100_dp as real_policy

    factory = types.ModuleType("arbim.policy.rl100_dp")
    factory.build_rl100_dp = lambda shape_meta, config: TinyNativePolicy()
    monkeypatch.setitem(sys.modules, "arbim.policy.rl100_dp", factory)
    return adapter


def _checkpoint(tmp_path, contract, *, role="best_policy"):
    raw, ema = TinyNativePolicy(), TinyNativePolicy()
    with torch.no_grad():
        raw.linear.bias.fill_(0.2)
        ema.linear.bias.fill_(0.7)
    payload = {
        "format_version": 2,
        "checkpoint_role": role,
        "policy": raw.state_dict(),
        "contract": contract,
        "global_step": 12,
    }
    if role == "best_policy":
        payload.update(selected_weights="raw", val_loss=0.3)
    else:
        payload["ema_policy"] = ema.state_dict()
    path = tmp_path / ("best.ckpt" if role == "best_policy" else "last.ckpt")
    torch.save(payload, path)
    return path, raw, ema


def test_best_checkpoint_strict_raw_load(tmp_path, contract, native_stubs):
    path, raw, _ = _checkpoint(tmp_path, contract)
    policy, spec, cfg = load_inference_policy(path)
    assert cfg["policy"]["horizon"] == 5
    assert spec == contract["dataset_spec"]
    assert not policy.training
    assert policy.linear.weight.device.type == "cpu"
    for key, value in raw.state_dict().items():
        torch.testing.assert_close(policy.state_dict()[key], value)


def test_training_checkpoint_ema_and_raw_selection(tmp_path, contract, native_stubs):
    path, raw, ema = _checkpoint(tmp_path, contract, role="training")
    for choice, expected in (("raw", raw), ("ema", ema)):
        loaded, _, _ = load_inference_policy(path, weights=choice)
        torch.testing.assert_close(loaded.linear.bias, expected.linear.bias)
        torch.testing.assert_close(loaded.normalizer.params_dict["action"]["scale"],
                                   expected.normalizer.params_dict["action"]["scale"])


def test_best_rejects_ema(tmp_path, contract, native_stubs):
    path, _, _ = _checkpoint(tmp_path, contract)
    with pytest.raises(ValueError, match="EMA requires"):
        load_inference_policy(path, weights="ema")


@pytest.mark.parametrize("corruption,error", [
    ({"format_version": 1}, "version-2"),
    ({"checkpoint_role": "unknown"}, "checkpoint_role"),
    ({"contract": {}}, "contract"),
    ({"selected_weights": "ema"}, "selected_weights"),
    ({"policy": {}}, "state_dict"),
])
def test_checkpoint_metadata_guard(tmp_path, contract, native_stubs, corruption, error):
    path, _, _ = _checkpoint(tmp_path, contract)
    payload = torch.load(path, weights_only=True)
    payload.update(corruption)
    torch.save(payload, path)
    with pytest.raises(ValueError, match=error):
        load_inference_policy(path)


def test_checkpoint_strict_key_mismatch(tmp_path, contract, native_stubs):
    path, _, _ = _checkpoint(tmp_path, contract)
    payload = torch.load(path, weights_only=True)
    del payload["policy"]["linear.bias"]
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_inference_policy(path)


def test_checkpoint_rejects_missing_action_normalizer(tmp_path, contract, native_stubs):
    path, _, _ = _checkpoint(tmp_path, contract)
    # Build a Policy with a perfectly strict but incomplete native normalizer.
    class WrongPolicy(TinyNativePolicy):
        def __init__(self):
            super().__init__()
            del self.normalizer.params_dict["action"]

    native_stubs_policy = sys.modules["arbim.policy.rl100_dp"]
    native_stubs_policy.build_rl100_dp = lambda *_: WrongPolicy()
    payload = torch.load(path, weights_only=True)
    for key in list(payload["policy"]):
        if key.startswith("normalizer.params_dict.action."):
            del payload["policy"][key]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="Normalizer is missing action"):
        load_inference_policy(path)


def test_observation_adapter_accepts_original_and_native_windows(contract, native_stubs):
    spec = contract["dataset_spec"]
    state = torch.arange(12, dtype=torch.float32).reshape(1, 2, 2, 3)
    rgb_hwc = torch.full((1, 2, 4, 5, 3), 128, dtype=torch.uint8)
    raw = {"observation.state": state, "observation.images.front": rgb_hwc}
    obs = prepare_observation(raw, spec, 2)
    assert obs["agent_pos"].shape == (1, 2, 6)
    assert obs["rgb_0"].shape == (1, 2, 3, 4, 5)
    assert obs["rgb_0"].dtype == torch.float32
    torch.testing.assert_close(obs["rgb_0"], torch.full_like(obs["rgb_0"], 128 / 255))
    native = {"obs": {"agent_pos": obs["agent_pos"], "rgb_0": obs["rgb_0"]}}
    again = prepare_observation(native, spec, 2)
    for key in obs:
        torch.testing.assert_close(obs[key], again[key])


def test_observation_adapter_rejects_malformed_input(contract, native_stubs):
    spec = contract["dataset_spec"]
    good = {"agent_pos": torch.ones(1, 2, 6),
            "rgb_0": torch.zeros(1, 2, 3, 4, 5)}
    with pytest.raises(KeyError, match="Missing observation"):
        prepare_observation({"agent_pos": good["agent_pos"]}, spec, 2)
    with pytest.raises(ValueError, match="Ambiguous"):
        prepare_observation({**good, "observation.images.front": good["rgb_0"]}, spec, 2)
    with pytest.raises(ValueError, match="expected"):
        prepare_observation({**good, "rgb_0": torch.zeros(1, 2, 3, 5, 4)}, spec, 2)
    with pytest.raises(ValueError, match="must be batched"):
        prepare_observation({**good, "agent_pos": torch.ones(2, 6)}, spec, 2)
    with pytest.raises(ValueError, match="floating RGB"):
        prepare_observation({**good, "rgb_0": torch.full((1, 2, 3, 4, 5), 255.0)}, spec, 2)


def test_predict_modes_shapes_and_values(contract, native_stubs):
    policy = TinyNativePolicy().eval()
    obs = prepare_observation(
        {"agent_pos": torch.randn(1, 2, 6),
         "rgb_0": torch.zeros(1, 2, 3, 4, 5)}, contract["dataset_spec"], 2,
    )
    for deterministic in (True, False):
        result = predict(policy, obs, deterministic=deterministic)
        assert policy.last_deterministic is deterministic
        assert result["action"].shape == (1, 4, 6)
        assert result["action_pred"].shape == (1, 4, 6)
        assert torch.isfinite(result["action"]).all()


def test_predict_rejects_nonfinite_output(contract, native_stubs):
    class BadPolicy(TinyNativePolicy):
        def predict_action(self, observation, *, deterministic=False):
            result = super().predict_action(observation, deterministic=deterministic)
            return {key: value * float("nan") for key, value in result.items()}

    obs = prepare_observation(
        {"agent_pos": torch.ones(1, 2, 6), "rgb_0": torch.zeros(1, 2, 3, 4, 5)},
        contract["dataset_spec"], 2,
    )
    with pytest.raises(FloatingPointError, match="NaN"):
        predict(BadPolicy().eval(), obs)


def test_legacy_checkpoint_horizon_recovery(contract):
    old = copy.deepcopy(contract)
    old["dataset"]["horizon"] = old["policy"].pop("horizon")
    old["dataset"]["n_obs_steps"] = old["policy"].pop("n_obs_steps")
    config = _model_config(old)
    assert config["policy"]["horizon"] == 5
    assert config["policy"]["n_obs_steps"] == 2


def test_inference_yaml_reads_runtime_only(tmp_path):
    path = tmp_path / "inference.yaml"
    path.write_text(
        "inference:\n"
        "  checkpoint_path: /tmp/model/best.ckpt\n"
        "  device: cuda:1\n"
        "  weights: raw\n"
        "  deterministic: true\n"
        "  n_action_steps: 2\n"
        "  num_inference_steps: 8\n"
        "  seed: 123\n",
        encoding="utf-8",
    )
    options = _read_inference_config(path)
    assert options["checkpoint_path"] == "/tmp/model/best.ckpt"
    assert options["device"] == "cuda:1"
    assert options["weights"] == "raw"
    assert options["deterministic"] is True
    assert options["n_action_steps"] == 2
    assert options["num_inference_steps"] == 8
    assert options["seed"] == 123


@pytest.mark.parametrize(("text", "error"), [
    ("policy:\n  horizon: 5\n", "inference"),
    ("inference: {}\npolicy:\n  horizon: 5\n", "top-level"),
    ("inference:\n  horizon: 5\n", "Unsupported"),
    ("inference:\n  weights: invalid\n", "weights"),
    ("inference:\n  deterministic: 'false'\n", "boolean"),
    ("inference:\n  samples: 1\n", "Unsupported"),
    ("inference:\n  dataset_root: /tmp/data\n", "Unsupported"),
    ("inference:\n  n_action_steps: 0\n", "positive integer"),
    ("inference:\n  n_action_steps: 1.5\n", "positive integer"),
    ("inference:\n  num_inference_steps: 0\n", "num_inference_steps"),
    ("inference:\n  num_inference_steps: true\n", "num_inference_steps"),
    ("inference:\n  seed: -1\n", "seed"),
    ("inference:\n  seed: true\n", "seed"),
    ("inference:\n  seed: 1.5\n", "seed"),
])
def test_inference_yaml_rejects_invalid_settings(tmp_path, text, error):
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        _read_inference_config(path)


@pytest.mark.parametrize("steps", [1, 2, 4])
def test_inference_action_steps_override(tmp_path, contract, native_stubs, steps):
    path, _, _ = _checkpoint(tmp_path, contract)
    policy, spec, saved_config = load_inference_policy(path, n_action_steps=steps)
    assert saved_config["policy"]["n_action_steps"] == 4  # Checkpoint remains untouched.
    assert policy.n_action_steps == steps
    obs = prepare_observation(
        {"agent_pos": torch.ones(1, 2, 6), "rgb_0": torch.zeros(1, 2, 3, 4, 5)},
        spec, 2,
    )
    result = predict(policy, obs)
    assert result["action_pred"].shape == (1, 4, 6)
    assert result["action"].shape == (1, steps, 6)


@pytest.mark.parametrize("steps", [0, 5, -1, True, 2.5])
def test_inference_action_steps_rejects_invalid(tmp_path, contract, native_stubs, steps):
    path, _, _ = _checkpoint(tmp_path, contract)
    with pytest.raises(ValueError, match="n_action_steps"):
        load_inference_policy(path, n_action_steps=steps)


def test_ddim_steps_override_rebuilds_native_constructor(tmp_path, contract, native_stubs):
    """Factory must see the override *before* Policy and DDIMSolver construction."""
    contract["schedulers"]["ddim"]["num_train_timesteps"] = 100
    contract["policy"]["num_inference_steps"] = 10
    contract["policy"]["ddim_inference_steps"] = 9  # Native kwargs take priority.
    path, _, _ = _checkpoint(tmp_path, contract)
    original = torch.load(path, weights_only=True)["contract"]
    seen = []

    def factory(shape_meta, config):
        seen.append(copy.deepcopy(config))
        model = TinyNativePolicy()
        model.ddim_inference_steps = config["policy"].get(
            "ddim_inference_steps", config["policy"]["num_inference_steps"]
        )
        return model

    native_stubs_policy = sys.modules["arbim.policy.rl100_dp"]
    native_stubs_policy.build_rl100_dp = factory
    policy, spec, cfg = load_inference_policy(path, num_inference_steps=6)
    assert policy.ddim_inference_steps == 6
    assert seen[0]["policy"]["num_inference_steps"] == 6
    assert seen[0]["policy"]["ddim_inference_steps"] == 6
    assert cfg["policy"]["num_inference_steps"] == 6
    assert torch.load(path, weights_only=True)["contract"] == original
    # No override retains all original constructor settings.
    untouched, _, original_cfg = load_inference_policy(path)
    assert untouched.ddim_inference_steps == 9
    assert original_cfg["policy"]["num_inference_steps"] == 10


@pytest.mark.parametrize("steps", [0, 101, -5, True, 1.5])
def test_ddim_steps_override_rejects_out_of_range(tmp_path, contract, native_stubs, steps):
    contract["schedulers"]["ddim"]["num_train_timesteps"] = 100
    path, _, _ = _checkpoint(tmp_path, contract)
    with pytest.raises(ValueError, match="num_inference_steps"):
        load_inference_policy(path, num_inference_steps=steps)


def test_ddim_steps_requires_saved_train_schedule(tmp_path, contract, native_stubs):
    path, _, _ = _checkpoint(tmp_path, contract)
    with pytest.raises(ValueError, match="num_train_timesteps"):
        load_inference_policy(path, num_inference_steps=6)


def test_seed_inference_once_and_null_preserves_rng():
    previous_state = torch.random.get_rng_state().clone()
    try:
        seed_inference(1234)
        first = torch.randn(3)
        second = torch.randn(3)
        seed_inference(1234)
        torch.testing.assert_close(first, torch.randn(3))
        torch.testing.assert_close(second, torch.randn(3))
        before = torch.random.get_rng_state().clone()
        seed_inference(None)
        assert torch.equal(before, torch.random.get_rng_state())
    finally:
        torch.random.set_rng_state(previous_state)


@pytest.mark.parametrize("seed", [-1, True, 1.5, 2**63])
def test_seed_inference_invalid(seed):
    with pytest.raises(ValueError, match="seed"):
        seed_inference(seed)
