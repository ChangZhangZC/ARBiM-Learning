"""Small on-policy buffer compatible with native dp_align_update_no_share().

Store raw LeRobot observation windows (RGB uint8), but transfer native RGB
float CHW and normalized-image-free agent_pos to the policy only when sampling.
The denoising chain is deliberately *not* collapsed to the executed chunk.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from arbim.policy.inference_rl100_dp2d import prepare_observation


class OnlinePPOBuffer:
    def __init__(
        self, *, spec: Mapping[str, Any], n_obs_steps: int,
        prediction_steps: int, num_inference_steps: int, action_dim: int,
        device: torch.device | str,
    ) -> None:
        self.spec = spec
        self.n_obs_steps = n_obs_steps
        self.prediction_steps = prediction_steps
        self.num_inference_steps = num_inference_steps
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.records: list[dict[str, Any]] = []

    @property
    def count(self) -> int:
        return len(self.records)

    def clear(self) -> None:
        self.records.clear()

    def _window(self, obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
        result = {}
        for key in (self.spec["state_key"], *self.spec["camera_keys"]):
            if key not in obs:
                raise KeyError(f"Online window missing {key!r}.")
            value = np.asarray(obs[key])
            if value.shape[0] != self.n_obs_steps:
                raise ValueError(f"Window {key!r} has incorrect history length.")
            if key in self.spec["camera_keys"]:
                if value.dtype != np.uint8:
                    raise TypeError("Online buffer RGB must remain uint8.")
            elif not np.isfinite(value).all():
                raise ValueError("Robot state contains NaN/Inf.")
            result[key] = value.copy()
        return result

    @staticmethod
    def _cpu_numpy(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    def store(
        self, obs: Mapping[str, Any], action: Any, a_logprob: Any,
        reward: float, next_obs: Mapping[str, Any], done: bool, dw: bool,
    ) -> None:
        """Same seven positional fields as the upstream replay-buffer store()."""
        x = self._cpu_numpy(action)
        lp = self._cpu_numpy(a_logprob)
        shape = (self.num_inference_steps + 1, self.prediction_steps, self.action_dim)
        lp_shape = (self.num_inference_steps, self.prediction_steps, self.action_dim)
        if x.shape != shape or lp.shape != lp_shape:
            raise ValueError(f"DDIM trajectory/log-prob shape {x.shape}/{lp.shape}; expected {shape}/{lp_shape}.")
        if not np.isfinite(x).all() or not np.isfinite(lp).all() or not np.isfinite(reward):
            raise ValueError("Nonfinite online transition.")
        if bool(dw) and not bool(done):
            raise ValueError("dw=True requires done=True (true termination).")
        self.records.append({
            "obs": self._window(obs), "action": x.copy(), "logprob": lp.copy(),
            "reward": float(reward), "next_obs": self._window(next_obs),
            "done": bool(done), "dw": bool(dw),
        })

    def numpy_to_tensor(self):
        """Return exactly (s, a, a_logprob, r, s_, dw, done) for native PPO."""
        if not self.records:
            raise ValueError("Cannot convert an empty on-policy buffer.")
        keys = (self.spec["state_key"], *self.spec["camera_keys"])
        obs_np = {
            key: np.stack([record["obs"][key] for record in self.records])
            for key in keys
        }
        next_np = {
            key: np.stack([record["next_obs"][key] for record in self.records])
            for key in keys
        }
        s = prepare_observation(obs_np, self.spec, self.n_obs_steps, device=self.device)
        next_s = prepare_observation(next_np, self.spec, self.n_obs_steps, device=self.device)
        tensor = lambda name, dtype=torch.float32: torch.as_tensor(
            np.stack([record[name] for record in self.records]), dtype=dtype, device=self.device
        )
        action = tensor("action")
        logprob = tensor("logprob")
        reward = torch.as_tensor(
            [[record["reward"]] for record in self.records],
            dtype=torch.float32, device=self.device,
        )
        done = torch.as_tensor(
            [[record["done"]] for record in self.records], dtype=torch.float32, device=self.device
        )
        dw = torch.as_tensor(
            [[record["dw"]] for record in self.records], dtype=torch.float32, device=self.device
        )
        return s, action, logprob, reward, next_s, dw, done
