"""Multi-episode DuoBench evaluation of a frozen, native RL100 DP-2D policy.

The Task 5.3 episode loop remains the single source of action scheduling.
Three policy-facing RGB streams are written incrementally as MP4 files.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from arbim.rollout.single import run_single_episode


class EpisodeVideoRecorder:
    """Stream uint8 HWC RGB observations to one MP4 per mapped camera."""

    def __init__(self, folder: Path, camera_map: Mapping[str, str], fps: float):
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "Video recording requires OpenCV (cv2); install opencv-python-headless "
                "or set output.record_video=false."
            ) from exc
        self.cv2 = cv2
        self.folder = folder
        self.camera_map = dict(camera_map)
        self.fps = fps
        self.writers: dict[str, Any] = {}
        self.frame_count = 0
        self.folder.mkdir(parents=True, exist_ok=False)

    def write(self, observation: Mapping[str, Any], step: int) -> None:
        if step != self.frame_count:
            raise ValueError(f"Expected video frame index {self.frame_count}, got {step}.")
        for key, camera in self.camera_map.items():
            frame = np.asarray(observation[key])
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"Video camera {key!r} must be HWC uint8 RGB, got {frame.shape} {frame.dtype}.")
            height, width = frame.shape[:2]
            if camera not in self.writers:
                writer = self.cv2.VideoWriter(
                    str(self.folder / f"{camera}.mp4"),
                    self.cv2.VideoWriter_fourcc(*"mp4v"),
                    self.fps,
                    (width, height),
                )
                if not writer.isOpened():
                    writer.release()
                    raise RuntimeError(f"Could not open MP4 encoder for camera {camera!r}.")
                self.writers[camera] = (writer, (height, width))
            writer, expected_shape = self.writers[camera]
            if (height, width) != expected_shape:
                raise ValueError(f"Video camera {camera!r} changed size within the episode.")
            writer.write(self.cv2.cvtColor(np.ascontiguousarray(frame), self.cv2.COLOR_RGB2BGR))
        self.frame_count += 1

    def close(self) -> None:
        for writer, _ in self.writers.values():
            writer.release()
        self.writers.clear()


def summarize_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Success comes from env info; inference latency is weighted by inference count."""
    if not episodes:
        raise ValueError("Cannot summarize an empty evaluation.")
    n = len(episodes)
    successes = sum(bool(item["success"]) for item in episodes)
    inferences = sum(item["inference_count"] for item in episodes)
    return {
        "num_episodes": n,
        "success_count": successes,
        "success_rate": successes / n,
        "mean_episode_steps": sum(item["episode_steps"] for item in episodes) / n,
        "mean_episode_return": sum(item["episode_return"] for item in episodes) / n,
        "mean_elapsed_time": sum(item["elapsed_time"] for item in episodes) / n,
        "total_inference_count": inferences,
        "mean_inference_ms": (
            sum(item["mean_inference_ms"] * item["inference_count"] for item in episodes) / inferences
            if inferences else 0.0
        ),
    }


def evaluate_episodes(
    adapter: Any,
    policy: Any,
    spec: Mapping[str, Any],
    *,
    num_episodes: int,
    output_dir: str | Path,
    camera_map: Mapping[str, str],
    execute_steps: int,
    max_env_steps: int,
    seed: int | None = None,
    deterministic: bool = True,
    record_video: bool = True,
    video_fps: float = 30.0,
) -> dict[str, Any]:
    """Run sequential episodes reusing Policy and environment.

    The caller owns adapter and policy lifetime. Frames are written on reset
    and after every env step, without storing complete RGB trajectories.
    """
    if type(num_episodes) is not int or num_episodes < 1:
        raise ValueError("num_episodes must be a positive integer.")
    if seed is not None and (type(seed) is not int or seed < 0 or seed + num_episodes > 2**63):
        raise ValueError("seed and per-episode increments must be nonnegative and below 2**63.")
    if type(record_video) is not bool:
        raise ValueError("record_video must be boolean.")
    if not isinstance(camera_map, Mapping) or set(camera_map.values()) != {
        "head", "left_wrist", "right_wrist"
    } or set(camera_map) != set(spec["camera_keys"]):
        raise ValueError("camera_map must match all three checkpoint RGB features.")
    if type(video_fps) not in (int, float) or not math.isfinite(video_fps) or video_fps <= 0:
        raise ValueError("video_fps must be positive and finite.")
    # Validate before creating any output or resetting the environment.
    if type(execute_steps) is not int or not 1 <= execute_steps <= policy.n_action_steps:
        raise ValueError("execute_steps must be in [1, policy.n_action_steps].")
    if type(max_env_steps) is not int or max_env_steps < 1:
        raise ValueError("max_env_steps must be positive.")

    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with (folder / "episodes.jsonl").open("w", encoding="utf-8") as episodes_file, (
        folder / "evaluation.log"
    ).open("w", encoding="utf-8") as log_file:
        for index in range(num_episodes):
            episode_seed = seed + index if seed is not None else None
            recorder = (
                EpisodeVideoRecorder(folder / f"episode_{index:03d}", camera_map, video_fps)
                if record_video else None
            )
            try:
                result = run_single_episode(
                    adapter, policy, spec,
                    execute_steps=execute_steps,
                    max_env_steps=max_env_steps,
                    seed=episode_seed,
                    deterministic=deterministic,
                    on_frame=recorder.write if recorder is not None else None,
                )
                if recorder is not None and recorder.frame_count != result["episode_steps"] + 1:
                    raise RuntimeError("Recorded video frame count disagrees with executed environment steps.")
                record = {
                    "episode_index": index,
                    "seed": episode_seed,
                    **result,
                    "recorded_frames": recorder.frame_count if recorder is not None else 0,
                }
            finally:
                if recorder is not None:
                    recorder.close()
            records.append(record)
            line = json.dumps(record, ensure_ascii=False, allow_nan=False)
            episodes_file.write(line + "\n")
            episodes_file.flush()
            message = (
                f"episode={index} seed={episode_seed} success={record['success']} "
                f"stage={record['final_stage']} steps={record['episode_steps']} "
                f"return={record['episode_return']:.4f} stop={record['stop_reason']}"
            )
            print(message, flush=True)
            log_file.write(message + "\n")
            log_file.flush()

    summary = summarize_episodes(records)
    with (folder / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    import yaml
    from arbim.envs.duobench import DuoBenchAdapter, create_duobench_env
    from arbim.policy.inference_rl100_dp2d import load_inference_policy, seed_inference

    parser = argparse.ArgumentParser(description="Evaluate native RL100 DP-2D on DuoBench Hinge-Chest")
    parser.add_argument("--config", type=Path, default=Path("configs/rollout/hinge_chest.yaml"))
    parser.add_argument("--checkpoint", type=Path, help="Override policy checkpoint_path")
    parser.add_argument("--device", help="Override policy device")
    parser.add_argument("--episodes", type=int, help="Override rollout.num_episodes")
    parser.add_argument("--max-env-steps", type=int, help="Override rollout.max_env_steps")
    parser.add_argument("--no-video", action="store_true", help="Disable MP4 recording for a quick smoke test")
    args = parser.parse_args()

    with args.config.expanduser().open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Rollout config must be a mapping.")
    env_cfg, policy_cfg, rollout_cfg, output_cfg = (
        config.get(name) for name in ("env", "policy", "rollout", "output")
    )
    if any(not isinstance(section, dict) for section in (env_cfg, policy_cfg, rollout_cfg, output_cfg)):
        raise ValueError("Rollout config must contain env, policy, rollout and output mappings.")
    if env_cfg.get("task") != "hinge_chest":
        raise ValueError("Task 5.4 evaluates only hinge_chest.")

    policy_cfg["checkpoint_path"] = str(args.checkpoint) if args.checkpoint else policy_cfg.get("checkpoint_path")
    policy_cfg["device"] = args.device or policy_cfg.get("device", "cuda:0")
    if not policy_cfg["checkpoint_path"]:
        raise ValueError("Set policy.checkpoint_path or --checkpoint.")
    if args.episodes is not None:
        rollout_cfg["num_episodes"] = args.episodes
    if args.max_env_steps is not None:
        rollout_cfg["max_env_steps"] = args.max_env_steps
    if args.no_video:
        output_cfg["record_video"] = False

    num_episodes = rollout_cfg.get("num_episodes", 10)
    seed = rollout_cfg.get("seed")
    record_video = output_cfg.get("record_video", True)
    video_fps = (env_cfg.get("frequency", 30) if output_cfg.get("video_fps") is None
                 else output_cfg["video_fps"])
    root = Path(output_cfg.get("root", "outputs/eval")).expanduser()
    run_id = output_cfg.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("output.run_id must contain only letters, digits, underscore or hyphen.")
    run_dir = root / "hinge_chest" / run_id

    policy, spec, _ = load_inference_policy(
        policy_cfg["checkpoint_path"],
        device=policy_cfg["device"],
        weights=policy_cfg.get("weights", "raw"),
        n_action_steps=policy_cfg.get("n_action_steps"),
        num_inference_steps=policy_cfg.get("num_inference_steps"),
    )
    seed_inference(seed)
    env = create_duobench_env(env_cfg)
    try:
        adapter = DuoBenchAdapter(
            env, spec, env_cfg["camera_map"],
            binarize_state_gripper=env_cfg.get("binarize_state_gripper", True),
            gripper_threshold=env_cfg.get("gripper_threshold", 0.9),
        )
        # An existing run must never be silently overwritten.
        run_dir.mkdir(parents=True, exist_ok=False)
        rollout_cfg["num_episodes"] = num_episodes
        rollout_cfg["execute_steps"] = rollout_cfg.get("execute_steps", policy.n_action_steps)
        rollout_cfg["max_env_steps"] = rollout_cfg.get("max_env_steps", 1000)
        output_cfg.update(run_id=run_id, run_dir=str(run_dir), record_video=record_video, video_fps=video_fps)
        with (run_dir / "config_resolved.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
        evaluate_episodes(
            adapter, policy, spec,
            num_episodes=num_episodes,
            output_dir=run_dir,
            camera_map=env_cfg["camera_map"],
            execute_steps=rollout_cfg.get("execute_steps", policy.n_action_steps),
            max_env_steps=rollout_cfg.get("max_env_steps", 1000),
            seed=seed,
            deterministic=policy_cfg.get("deterministic", True),
            record_video=record_video,
            video_fps=video_fps,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
