"""Task 5.4 isolated evaluation tests (no DuoBench, GPU or real codec required)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from arbim.rollout import eval as evaluation


@pytest.fixture
def camera_map():
    return {
        "observation.images.head": "head",
        "observation.images.left_wrist": "left_wrist",
        "observation.images.right_wrist": "right_wrist",
    }


@pytest.fixture
def spec(camera_map):
    return {"camera_keys": tuple(camera_map)}


def _frame(camera_map, value=20):
    return {
        key: np.full((4, 5, 3), [value, 40, 60], dtype=np.uint8)
        for key in camera_map
    }


class FakeVideoWriter:
    writers = []

    def __init__(self, path, fourcc, fps, size):
        self.path, self.fps, self.size = path, fps, size
        self.frames = []
        self.released = False
        self.writers.append(self)

    def isOpened(self):
        return True

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        self.released = True


@pytest.fixture
def fake_cv2(monkeypatch):
    FakeVideoWriter.writers = []
    fake = SimpleNamespace(
        VideoWriter=FakeVideoWriter,
        VideoWriter_fourcc=lambda *args: 1234,
        COLOR_RGB2BGR=4,
        cvtColor=lambda image, _: image[:, :, ::-1].copy(),
    )
    monkeypatch.setitem(sys.modules, "cv2", fake)
    return FakeVideoWriter.writers


def test_video_stream_writes_rgb_as_bgr_and_closes(tmp_path, camera_map, fake_cv2):
    recorder = evaluation.EpisodeVideoRecorder(tmp_path / "episode_000", camera_map, 30)
    recorder.write(_frame(camera_map, 20), 0)
    recorder.write(_frame(camera_map, 21), 1)
    with pytest.raises(ValueError, match="frame index"):
        recorder.write(_frame(camera_map), 3)
    recorder.close()
    assert recorder.frame_count == 2
    assert len(fake_cv2) == 3
    assert {Path(item.path).name for item in fake_cv2} == {
        "head.mp4", "left_wrist.mp4", "right_wrist.mp4"
    }
    for item in fake_cv2:
        assert item.size == (5, 4) and item.fps == 30
        assert len(item.frames) == 2 and item.released
        np.testing.assert_array_equal(item.frames[0][0, 0], [60, 40, 20])


def test_video_frame_contract_rejects_shape_changes(tmp_path, camera_map, fake_cv2):
    recorder = evaluation.EpisodeVideoRecorder(tmp_path / "episode_000", camera_map, 30)
    try:
        recorder.write(_frame(camera_map), 0)
        frame = _frame(camera_map)
        frame["observation.images.head"] = np.zeros((5, 5, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="changed size"):
            recorder.write(frame, 1)
    finally:
        recorder.close()
    assert all(w.released for w in fake_cv2)


def test_summary_uses_weighted_inference_latency():
    result = evaluation.summarize_episodes([
        {"success": True, "episode_steps": 10, "episode_return": 3,
         "elapsed_time": 2, "inference_count": 1, "mean_inference_ms": 10},
        {"success": False, "episode_steps": 20, "episode_return": 7,
         "elapsed_time": 4, "inference_count": 3, "mean_inference_ms": 30},
    ])
    assert result["success_count"] == 1
    assert result["success_rate"] == pytest.approx(0.5)
    assert result["mean_episode_steps"] == pytest.approx(15)
    assert result["mean_episode_return"] == pytest.approx(5)
    assert result["mean_elapsed_time"] == pytest.approx(3)
    assert result["total_inference_count"] == 4
    assert result["mean_inference_ms"] == pytest.approx(25)


def test_evaluation_reuses_objects_logs_every_episode(tmp_path, spec, camera_map, monkeypatch):
    adapter, policy = object(), SimpleNamespace(n_action_steps=16)
    calls = []

    def fake_episode(a, p, s, **kwargs):
        assert (a, p, s) == (adapter, policy, spec)
        calls.append(kwargs)
        idx = len(calls) - 1
        assert kwargs["on_frame"] is None
        return {
            "success": idx == 1, "terminated": idx == 1, "truncated": False,
            "max_steps_reached": idx != 1, "stop_reason": "success" if idx == 1 else "max_env_steps",
            "episode_steps": 3, "episode_return": float(idx),
            "final_stage": 3 if idx == 1 else 1, "inference_count": 2,
            "mean_inference_ms": 12.0, "elapsed_time": 1.5,
        }

    monkeypatch.setattr(evaluation, "run_single_episode", fake_episode)
    summary = evaluation.evaluate_episodes(
        adapter, policy, spec, num_episodes=3, output_dir=tmp_path,
        camera_map=camera_map, execute_steps=16, max_env_steps=3,
        seed=42, deterministic=True, record_video=False,
    )
    assert [c["seed"] for c in calls] == [42, 43, 44]
    assert summary["num_episodes"] == 3
    assert summary["success_rate"] == pytest.approx(1/3)
    episodes = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    assert [e["episode_index"] for e in episodes] == [0, 1, 2]
    assert all(e["recorded_frames"] == 0 for e in episodes)
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    assert len((tmp_path / "evaluation.log").read_text().splitlines()) == 3


def test_video_per_episode_streams_without_retaining_frames(
    tmp_path, spec, camera_map, fake_cv2, monkeypatch
):
    def fake_episode(adapter, policy, saved_spec, **kwargs):
        assert kwargs["on_frame"] is not None
        for step in range(4):
            kwargs["on_frame"](_frame(camera_map, 10 + step), step)
        return {
            "success": False, "terminated": False, "truncated": False,
            "max_steps_reached": True, "stop_reason": "max_env_steps",
            "episode_steps": 3, "episode_return": 1.0, "final_stage": 1,
            "inference_count": 1, "mean_inference_ms": 10.0, "elapsed_time": 0.5,
        }

    monkeypatch.setattr(evaluation, "run_single_episode", fake_episode)
    evaluation.evaluate_episodes(
        object(), SimpleNamespace(n_action_steps=16), spec,
        num_episodes=2, output_dir=tmp_path, camera_map=camera_map,
        execute_steps=16, max_env_steps=3, record_video=True, video_fps=30,
    )
    assert len(fake_cv2) == 6
    assert all(len(writer.frames) == 4 and writer.released for writer in fake_cv2)
    lines = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    assert [item["recorded_frames"] for item in lines] == [4, 4]
    assert (tmp_path / "episode_001").is_dir()


def test_video_encoder_released_on_episode_error(tmp_path, spec, camera_map, fake_cv2, monkeypatch):
    def fail_episode(adapter, policy, saved_spec, **kwargs):
        kwargs["on_frame"](_frame(camera_map), 0)
        raise RuntimeError("environment step failed")
    monkeypatch.setattr(evaluation, "run_single_episode", fail_episode)
    with pytest.raises(RuntimeError, match="environment step failed"):
        evaluation.evaluate_episodes(
            object(), SimpleNamespace(n_action_steps=16), spec,
            num_episodes=1, output_dir=tmp_path, camera_map=camera_map,
            execute_steps=16, max_env_steps=3, record_video=True,
        )
    assert len(fake_cv2) == 3
    assert all(writer.released for writer in fake_cv2)


@pytest.mark.parametrize("overrides", [
    {"num_episodes": 0}, {"num_episodes": True},
    {"seed": 2**63}, {"seed": True},
    {"execute_steps": 17}, {"execute_steps": 0},
    {"max_env_steps": 0}, {"video_fps": 0}, {"video_fps": float("nan")},
    {"record_video": 1},
])
def test_reject_invalid_inputs_without_outputs(tmp_path, spec, camera_map, overrides):
    options = dict(
        num_episodes=2, output_dir=tmp_path / "run", camera_map=camera_map,
        execute_steps=16, max_env_steps=3, seed=42, record_video=False,
    )
    options.update(overrides)
    with pytest.raises(ValueError):
        evaluation.evaluate_episodes(
            object(), SimpleNamespace(n_action_steps=16), spec, **options
        )
    assert not (tmp_path / "run").exists()


def test_cli_resolves_config_and_closes_environment(tmp_path, monkeypatch, spec, camera_map):
    from arbim.envs import duobench
    from arbim.policy import inference_rl100_dp2d
    import sys

    config = {
        "env": {"task": "hinge_chest", "camera_map": camera_map, "frequency": 30},
        "policy": {"checkpoint_path": "/mock/best.ckpt", "device": "cpu"},
        "rollout": {"num_episodes": 2, "max_env_steps": 10, "execute_steps": 2, "seed": 42},
        "output": {"root": str(tmp_path / "runs"), "run_id": "test_run", "record_video": False,
                   "video_fps": None},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["eval", "--config", str(path), "--episodes", "3",
                                     "--max-env-steps", "5"])
    policy = SimpleNamespace(n_action_steps=16)
    monkeypatch.setattr(inference_rl100_dp2d, "load_inference_policy",
                        lambda *a, **kw: (policy, spec, {}))
    seeded = []
    monkeypatch.setattr(inference_rl100_dp2d, "seed_inference", seeded.append)
    class NativeEnv:
        closed = False
        def close(self):
            self.closed = True
    env = NativeEnv()
    monkeypatch.setattr(duobench, "create_duobench_env", lambda x: env)
    monkeypatch.setattr(duobench, "DuoBenchAdapter", lambda *a, **kw: object())
    called = []
    monkeypatch.setattr(evaluation, "evaluate_episodes", lambda *a, **kw: called.append(kw))
    evaluation.main()
    resolved = yaml.safe_load((tmp_path / "runs" / "hinge_chest" / "test_run" /
                               "config_resolved.yaml").read_text())
    assert env.closed and seeded == [42]
    assert called[0]["num_episodes"] == 3 and called[0]["max_env_steps"] == 5
    assert resolved["rollout"]["num_episodes"] == 3
    assert resolved["rollout"]["max_env_steps"] == 5
    assert resolved["output"]["video_fps"] == 30
    assert resolved["output"]["run_id"] == "test_run"


@pytest.mark.skipif(os.environ.get("ARBIM_VIDEO_CODEC_SMOKE") != "1",
                    reason="Set ARBIM_VIDEO_CODEC_SMOKE=1 to test the installed OpenCV MP4 codec.")
def test_real_mp4_codec_writes_readable_three_view_video(tmp_path, camera_map):
    cv2 = pytest.importorskip("cv2")
    recorder = evaluation.EpisodeVideoRecorder(tmp_path / "episode_000", camera_map, 30)
    try:
        for step in range(3):
            # Even-sized image supports mp4v encoder.
            frame = {key: np.full((224, 224, 3), 20 + step, dtype=np.uint8)
                     for key in camera_map}
            recorder.write(frame, step)
    finally:
        recorder.close()
    for camera in camera_map.values():
        path = tmp_path / "episode_000" / f"{camera}.mp4"
        assert path.exists() and path.stat().st_size > 0
        reader = cv2.VideoCapture(str(path))
        try:
            assert reader.isOpened()
            decoded = 0
            while reader.read()[0]:
                decoded += 1
            assert decoded == 3
        finally:
            reader.release()


def test_existing_episode_log_is_never_overwritten(tmp_path, spec, camera_map):
    path = tmp_path / "episodes.jsonl"
    path.write_text("previous run\n")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        evaluation.evaluate_episodes(
            object(), SimpleNamespace(n_action_steps=16), spec,
            num_episodes=1, output_dir=tmp_path, camera_map=camera_map,
            execute_steps=16, max_env_steps=3, record_video=False,
        )
    assert path.read_text() == "previous run\n"



def test_native_video_uses_raw_resolution_not_policy_window(
    tmp_path, spec, camera_map, fake_cv2, monkeypatch
):
    class Adapter:
        last_raw_observation = None

    adapter = Adapter()

    def fake_episode(instance, policy, saved_spec, **kwargs):
        assert instance is adapter
        for step in range(3):
            instance.last_raw_observation = {
                "frames": {
                    camera: {"rgb": {"data": np.full((6, 10, 3), 20 + step, dtype=np.uint8)}}
                    for camera in camera_map.values()
                }
            }
            kwargs["on_frame"](_frame(camera_map), step)  # policy RGB remains 4x5
        return {
            "success": False, "terminated": False, "truncated": False,
            "max_steps_reached": True, "stop_reason": "max_env_steps",
            "episode_steps": 2, "episode_return": 0.0, "final_stage": 0,
            "inference_count": 1, "mean_inference_ms": 1.0, "elapsed_time": 0.5,
        }

    monkeypatch.setattr(evaluation, "run_single_episode", fake_episode)
    evaluation.evaluate_episodes(
        adapter, SimpleNamespace(n_action_steps=16), spec,
        num_episodes=1, output_dir=tmp_path,
        camera_map=camera_map, execute_steps=16, max_env_steps=2,
        video_source="native", record_video=True,
    )
    assert len(fake_cv2) == 3
    assert all(writer.size == (10, 6) and len(writer.frames) == 3 for writer in fake_cv2)
    assert all(writer.released for writer in fake_cv2)
    assert all(writer.frames[0].shape == (6, 10, 3) for writer in fake_cv2)


def test_invalid_video_source_rejected_before_run(tmp_path, spec, camera_map):
    with pytest.raises(ValueError, match="video_source"):
        evaluation.evaluate_episodes(
            object(), SimpleNamespace(n_action_steps=16), spec,
            num_episodes=1, output_dir=tmp_path / "run",
            camera_map=camera_map, execute_steps=16, max_env_steps=2,
            record_video=True, video_source="upscaled",
        )
    assert not (tmp_path / "run").exists()
