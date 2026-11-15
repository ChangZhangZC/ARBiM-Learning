"""Checkpoint-backed, embodiment-independent inference for native RL100 DP-2D.

No environment stepping, robot control, extra normalization or diffusion changes.
Run the smoke CLI with: python -m arbim.policy.inference_rl100_dp2d --help
"""

from __future__ import annotations

import argparse
import copy
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


def _model_config(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Use the checkpoint as the source of truth, including older v2 contracts."""
    for key in ("dataset_spec", "dataset", "policy", "vision", "schedulers"):
        if not isinstance(contract.get(key), Mapping):
            raise ValueError(f"Checkpoint contract missing mapping {key!r}.")
    config = {key: copy.deepcopy(contract[key]) for key in ("dataset", "policy", "vision", "schedulers")}
    # Old v2 runs kept the two time dimensions under dataset, not policy.
    for key in ("horizon", "n_obs_steps"):
        if key not in config["policy"] and key in config["dataset"]:
            config["policy"][key] = config["dataset"][key]
    policy = config["policy"]
    if any(type(policy.get(k)) is not int or policy[k] < 1
           for k in ("horizon", "n_obs_steps", "n_action_steps")):
        raise ValueError("Checkpoint lacks valid horizon/n_obs_steps/n_action_steps.")
    if not policy["n_obs_steps"] <= policy["horizon"]:
        raise ValueError("Checkpoint has n_obs_steps > horizon.")
    if policy["n_action_steps"] > policy["horizon"] - policy["n_obs_steps"] + 1:
        raise ValueError("Checkpoint n_action_steps exceeds native prediction horizon.")
    return config


def _validate_spec(spec: Mapping[str, Any]) -> None:
    required = ("state_key", "action_key", "state_shape", "action_shape",
                "state_dim", "action_dim", "camera_keys", "camera_shapes", "camera_obs_keys")
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(f"Checkpoint dataset_spec missing fields: {missing}")
    from math import prod
    if any(type(spec[key]) is not int or spec[key] < 1 for key in ("state_dim", "action_dim")):
        raise ValueError("Invalid state/action dimensions in checkpoint.")
    for key, dim in (("state_shape", spec["state_dim"]), ("action_shape", spec["action_dim"])):
        shape = spec[key]
        if not shape or prod(shape) != dim:
            raise ValueError(f"Checkpoint {key} does not match its flattened dimension.")
    cameras = tuple(spec["camera_keys"])
    aliases = spec["camera_obs_keys"]
    if not cameras or len(cameras) != len(set(cameras)):
        raise ValueError("Checkpoint must list unique RGB cameras.")
    if set(aliases) != set(cameras) or len(set(aliases.values())) != len(cameras):
        raise ValueError("Checkpoint camera alias mapping is incomplete or ambiguous.")
    if "agent_pos" in aliases.values():
        raise ValueError("Camera aliases cannot overwrite agent_pos.")
    for key in cameras:
        shape = spec["camera_shapes"].get(key)
        if shape is None or len(shape) != 3 or shape[0] != 3 or any(d < 1 for d in shape):
            raise ValueError(f"Invalid RGB camera shape for {key!r}.")


def _check_normalizer(policy: torch.nn.Module, spec: Mapping[str, Any], device: torch.device) -> None:
    params = policy.normalizer.params_dict
    for name, dim in (("agent_pos", spec["state_dim"]), ("action", spec["action_dim"])):
        if name not in params or "scale" not in params[name] or "offset" not in params[name]:
            raise ValueError(f"Native Normalizer is missing {name} scale/offset.")
        scale, offset = params[name]["scale"], params[name]["offset"]
        if scale.shape != (dim,) or offset.shape != (dim,):
            raise ValueError(f"Normalizer {name} dimension does not match dataset_spec ({dim}).")
        if (scale.device != device or offset.device != device
                or not torch.isfinite(scale).all() or not torch.isfinite(offset).all()
                or torch.any(scale == 0)):
            raise ValueError(f"Normalizer {name} contains invalid or off-device parameters.")


def load_inference_policy(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    weights: str = "raw",
    n_action_steps: int | None = None,
    num_inference_steps: int | None = None,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    """Return (native policy, saved dataset_spec, saved model config).

    Supports Task 3 best.ckpt (raw) and last.ckpt (raw or EMA). The native
    Normalizer dynamically re-registers parameters on CPU during strict load.
    n_action_steps optionally shortens the native inference output slice only.
    num_inference_steps is applied to a copy of saved constructor config, so
    native DDIM scheduler/solver are rebuilt consistently without touching ckpt.
    """
    if weights not in ("raw", "ema"):
        raise ValueError("weights must be 'raw' or 'ema'.")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 2:
        raise ValueError("Expected an ARBiM Task 3 version-2 checkpoint.")
    role = payload.get("checkpoint_role")
    if role not in ("best_policy", "training"):
        raise ValueError(f"Unsupported checkpoint_role: {role!r}")
    if role == "best_policy" and payload.get("selected_weights") != "raw":
        raise ValueError("best_policy checkpoint must declare selected_weights='raw'.")
    if role == "best_policy" and weights == "ema":
        raise ValueError("best.ckpt contains raw Policy only; EMA requires last.ckpt.")
    if not isinstance(payload.get("contract"), Mapping):
        raise ValueError("Checkpoint is missing its saved contract.")
    contract = payload["contract"]
    config = _model_config(contract)
    if num_inference_steps is not None:
        train_steps = config["schedulers"]["ddim"].get("num_train_timesteps")
        if type(train_steps) is not int or train_steps < 1:
            raise ValueError("Checkpoint has no valid DDIM num_train_timesteps for inference override.")
        if (type(num_inference_steps) is not int
                or not 1 <= num_inference_steps <= train_steps):
            raise ValueError(f"num_inference_steps must be an integer in [1, {train_steps}].")
        config["policy"]["num_inference_steps"] = num_inference_steps
        if "ddim_inference_steps" in config["policy"]:
            config["policy"]["ddim_inference_steps"] = num_inference_steps
    spec = copy.deepcopy(dict(contract["dataset_spec"]))
    _validate_spec(spec)
    state = payload.get("ema_policy" if weights == "ema" else "policy")
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"Checkpoint is missing the {weights} Policy state_dict.")

    from arbim.data.lerobot_adapter import build_shape_meta
    from arbim.policy.rl100_dp import build_rl100_dp

    policy = build_rl100_dp(build_shape_meta(spec), config)  # CPU construction
    if (num_inference_steps is not None
            and getattr(policy, "ddim_inference_steps", None) != num_inference_steps):
        raise ValueError("Native RL1002D did not apply the requested DDIM inference steps.")
    policy.load_state_dict(state, strict=True)
    policy.to(device)
    policy.eval()
    if any(param.device != device for param in policy.parameters()):
        raise ValueError("Native Policy contains off-device parameters.")
    _check_normalizer(policy, spec, device)
    if n_action_steps is not None:
        max_steps = policy.horizon - policy.n_obs_steps + 1
        if type(n_action_steps) is not int or not 1 <= n_action_steps <= max_steps:
            raise ValueError(f"n_action_steps must be an integer in [1, {max_steps}].")
        policy.n_action_steps = n_action_steps
    return policy, spec, config


def prepare_observation(
    observation: Mapping[str, Any],
    spec: Mapping[str, Any],
    n_obs_steps: int,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Convert *batched* LeRobot or Task-3-native windows to native observations.

    Each input has a leading [B, To] window, with To=n_obs_steps. Accept
    LeRobot feature keys or the saved native aliases; never normalize state
    or call LeRobot pre/postprocessors. RGB is CHW float32 in [0,1].
    """
    from arbim.data.lerobot_adapter import _flat_window, _rgb_window

    _validate_spec(spec)
    if type(n_obs_steps) is not int or n_obs_steps < 1:
        raise ValueError("n_obs_steps must be a positive integer.")
    source = observation.get("obs", observation)
    if not isinstance(source, Mapping):
        raise TypeError("Expected an observation mapping or {'obs': mapping}.")

    def pick(native: str, original: str) -> Any:
        if native in source and original in source and native != original:
            raise ValueError(f"Ambiguous observation: both {native!r} and {original!r} provided.")
        key = native if native in source else original
        if key not in source:
            raise KeyError(f"Missing observation {original!r} (native alias {native!r}).")
        return source[key]

    def batch_windows(value: Any, key: str, shape: tuple[int, ...], rgb: bool) -> torch.Tensor:
        tensor = torch.as_tensor(value)
        if tensor.ndim < 3 or tensor.shape[0] < 1 or tensor.shape[1] != n_obs_steps:
            raise ValueError(f"{key!r} must be batched [B,{n_obs_steps},...], got {tuple(tensor.shape)}.")
        if not rgb and tuple(tensor.shape[2:]) == (spec["state_dim"],):
            shape = (spec["state_dim"],)  # Already flattened native agent_pos.
        convert = _rgb_window if rgb else _flat_window
        return torch.stack([convert(frame, key, n_obs_steps, shape) for frame in tensor])

    obs = {"agent_pos": batch_windows(
        pick("agent_pos", spec["state_key"]), spec["state_key"],
        tuple(spec["state_shape"]), rgb=False,
    )}
    batch_size = obs["agent_pos"].shape[0]
    for key in spec["camera_keys"]:
        alias = spec["camera_obs_keys"][key]
        rgb = batch_windows(pick(alias, key), key, tuple(spec["camera_shapes"][key]), rgb=True)
        if rgb.shape[0] != batch_size:
            raise ValueError(f"Camera {key!r} batch size differs from state.")
        if rgb.dtype == torch.uint8:
            rgb = rgb.float().div_(255.0)
        obs[alias] = rgb

    target = torch.device(device)
    return {key: value.to(target, dtype=torch.float32) for key, value in obs.items()}


def predict(
    policy: torch.nn.Module,
    observation: Mapping[str, torch.Tensor],
    *,
    deterministic: bool = True,
) -> dict[str, torch.Tensor]:
    """Call *only* native predict_action, checking its original output contract."""
    with torch.inference_mode():
        result = policy.predict_action(dict(observation), deterministic=deterministic)
    if not isinstance(result, Mapping) or not {"action", "action_pred"} <= result.keys():
        raise ValueError("Native predict_action must return action and action_pred.")
    action, full = result["action"], result["action_pred"]
    if not torch.is_tensor(action) or not torch.is_tensor(full):
        raise TypeError("Native action outputs must be tensors.")
    b = observation["agent_pos"].shape[0]
    t = policy.horizon - (policy.n_obs_steps - 1) if policy.no_pre_action else policy.horizon
    if tuple(full.shape) != (b, t, policy.action_dim):
        raise ValueError(f"Unexpected action_pred shape: {tuple(full.shape)}")
    if tuple(action.shape) != (b, policy.n_action_steps, policy.action_dim):
        raise ValueError(f"Unexpected action shape: {tuple(action.shape)}")
    start = 0 if policy.no_pre_action else policy.n_obs_steps - 1
    if not torch.isfinite(action).all() or not torch.isfinite(full).all():
        raise FloatingPointError("Native Policy returned NaN or Inf actions.")
    if not torch.equal(action, full[:, start:start + policy.n_action_steps]):
        raise ValueError("Native action does not match the action_pred slice.")
    target = next(policy.parameters()).device
    if action.device != target or full.device != target:
        raise ValueError("Native Policy returned actions on the wrong device.")
    return dict(result)



def seed_inference(seed: int | None) -> None:
    """Seed PyTorch once per run, not on each predict(); null preserves RNG state.

    Native deterministic DDIM uses its own fixed seed=0 for initial noise.
    """
    if seed is None:
        return
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative integer below 2**63 or null.")
    torch.manual_seed(seed)


def _read_inference_config(path: str | Path) -> dict[str, Any]:
    """Read only runtime settings; model structure always comes from checkpoint."""
    import yaml

    with Path(path).expanduser().open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict) or not isinstance(document.get("inference"), dict):
        raise ValueError("Inference YAML must contain an 'inference' mapping.")
    if set(document) != {"inference"}:
        raise ValueError("Task 4 YAML only supports the top-level 'inference' mapping.")
    settings = document["inference"]
    allowed = {"checkpoint_path", "device", "weights", "deterministic",
               "n_action_steps", "num_inference_steps", "seed"}
    unexpected = set(settings) - allowed
    if unexpected:
        raise ValueError(f"Unsupported Task 4 inference settings: {sorted(unexpected)}")
    for key in ("checkpoint_path", "device"):
        value = settings.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"inference.{key} must be a nonempty string or null.")
    if settings.get("weights", "raw") not in ("raw", "ema"):
        raise ValueError("inference.weights must be 'raw' or 'ema'.")
    if type(settings.get("deterministic", True)) is not bool:
        raise ValueError("inference.deterministic must be boolean.")
    steps = settings.get("n_action_steps")
    if steps is not None and (type(steps) is not int or steps < 1):
        raise ValueError("inference.n_action_steps must be a positive integer or null.")
    steps = settings.get("num_inference_steps")
    if steps is not None and (type(steps) is not int or steps < 1):
        raise ValueError("inference.num_inference_steps must be a positive integer or null.")
    seed = settings.get("seed")
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**63):
        raise ValueError("inference.seed must be a nonnegative integer below 2**63 or null.")
    return settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Native RL100 DP-2D checkpoint GPU/CPU inference smoke test")
    parser.add_argument("--config", type=Path, help="Optional configs/inference/rl100_dp2d.yaml")
    parser.add_argument("--checkpoint", type=Path, help="Override inference.checkpoint_path")
    parser.add_argument("--device", help="Override inference.device (default: cuda:0)")
    parser.add_argument("--weights", choices=("raw", "ema"), help="Override inference.weights (default: raw)")
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--stochastic", action="store_false", dest="deterministic",
                          help="Use native stochastic DDIM sampling")
    sampling.add_argument("--deterministic", action="store_true", dest="deterministic",
                          help="Use native deterministic DDIM sampling (default)")
    parser.set_defaults(deterministic=None)
    parser.add_argument("--n-action-steps", type=int, help="Override inference.n_action_steps (null uses checkpoint)")
    parser.add_argument("--num-inference-steps", type=int,
                        help="Override native DDIM denoising steps (null uses checkpoint)")
    parser.add_argument("--seed", type=int,
                        help="Set global PyTorch RNG once per inference run (null leaves RNG unchanged)")
    parser.add_argument("--dataset-root", type=Path, help="Smoke test only: relocated LeRobot dataset root")
    parser.add_argument("--samples", type=int, default=1, help="Smoke test only: number of validation windows")
    args = parser.parse_args()

    options = _read_inference_config(args.config) if args.config is not None else {}
    checkpoint_path = args.checkpoint or options.get("checkpoint_path")
    if checkpoint_path is None:
        parser.error("Provide --checkpoint or --config with inference.checkpoint_path.")
    device = torch.device(args.device if args.device is not None else options.get("device") or "cuda:0")
    weights = args.weights if args.weights is not None else options.get("weights", "raw")
    deterministic = (args.deterministic if args.deterministic is not None
                     else options.get("deterministic", True))
    n_action_steps = (args.n_action_steps if args.n_action_steps is not None
                      else options.get("n_action_steps"))
    num_inference_steps = (args.num_inference_steps if args.num_inference_steps is not None
                           else options.get("num_inference_steps"))
    seed = args.seed if args.seed is not None else options.get("seed")
    dataset_root = args.dataset_root
    samples = args.samples
    if samples < 1:
        parser.error("--samples must be positive.")

    policy, spec, config = load_inference_policy(
        checkpoint_path, device=device, weights=weights,
        n_action_steps=n_action_steps, num_inference_steps=num_inference_steps,
    )
    seed_inference(seed)
    from arbim.data.lerobot_adapter import LeRobotDP2DDataset

    dataset_cfg = dict(config["dataset"])
    dataset_cfg.pop("horizon", None)
    dataset_cfg.pop("n_obs_steps", None)
    if dataset_root is not None:
        dataset_cfg["root"] = Path(dataset_root)
    dataset_cfg.update(
        horizon=config["policy"]["horizon"], n_obs_steps=config["policy"]["n_obs_steps"],
        state_key=spec["state_key"], action_key=spec["action_key"],
        camera_keys=spec["camera_keys"], split="val",
    )
    dataset = LeRobotDP2DDataset(**dataset_cfg)
    if dataset.spec != spec:
        raise ValueError("Validation dataset metadata does not match checkpoint dataset_spec.")
    if len(dataset) < samples:
        raise ValueError(f"Requested {samples} validation windows, only {len(dataset)} available.")

    print(f"checkpoint={checkpoint_path} weights={weights} device={device} "
          f"deterministic={deterministic} n_action_steps={policy.n_action_steps} "
          f"ddim_inference_steps={policy.ddim_inference_steps} seed={seed}")
    for idx in range(samples):
        # __getitem__ supplies [To,...]; batching produces [1,To,...].
        raw = {key: value.unsqueeze(0) for key, value in dataset[idx]["obs"].items()}
        obs = prepare_observation(raw, spec, config["policy"]["n_obs_steps"], device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        result = predict(policy, obs, deterministic=deterministic)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        action = result["action"]
        print(f"sample={idx} obs={[(k, tuple(v.shape)) for k, v in obs.items()]} "
              f"action={tuple(action.shape)} action_pred={tuple(result['action_pred'].shape)} "
              f"dtype={action.dtype} device={action.device} finite=True "
              f"min={action.min().item():.6g} max={action.max().item():.6g} "
              f"inference_ms={elapsed_ms:.2f}")


if __name__ == "__main__":
    main()
