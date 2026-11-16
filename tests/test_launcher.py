"""Task 5.5 launcher tests; no checkpoint, simulator or GPU required."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from arbim.rollout import launch


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "hinge_chest.yaml"
    doc = {
        "env": {"task": "hinge_chest", "camera_resolution": None},
        "policy": {"checkpoint_path": "fake.ckpt"},
        "rollout": {"num_episodes": 2},
        "output": {"record_video": True, "video_source": "native"},
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def launched(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
    monkeypatch.setattr(launch.subprocess, "run", fake_run)
    return calls


def test_noninteractive_launcher_delegates_to_eval(config_path, launched, capsys):
    status = launch.main(["--task", "hinge_chest", "--mode", "normal",
                          "--config", str(config_path), "--yes"])
    assert status == 0
    assert launched == [(
        [sys.executable, "-m", "arbim.rollout.eval", "--config", str(config_path.resolve())],
        {"check": True},
    )]
    output = capsys.readouterr().out
    assert "Parsed YAML preview" in output
    assert "video_source: native" in output
    assert "config_resolved.yaml" in output


def test_full_interactive_menu_and_confirmation(config_path, launched, monkeypatch):
    answers = iter(["1", "1", str(config_path), "y"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    assert launch.main([]) == 0
    assert len(launched) == 1


def test_interactive_default_selection(config_path, launched, monkeypatch):
    monkeypatch.setattr(launch, "DEFAULT_CONFIG", config_path)
    answers = iter(["", "", "", "yes"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    assert launch.main([]) == 0
    assert len(launched) == 1


def test_cancel_does_not_start_eval(config_path, launched, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert launch.main(["--task", "hinge_chest", "--mode", "normal",
                        "--config", str(config_path)]) == 0
    assert not launched


def test_routed_rejected_without_eval(config_path, launched, capsys):
    assert launch.main(["--mode", "routed", "--yes", "--config", str(config_path)]) == 2
    assert "not implemented" in capsys.readouterr().out
    assert not launched


def test_task_mismatch_fails_before_launch(config_path, launched):
    doc = yaml.safe_load(config_path.read_text())
    doc["env"]["task"] = "other_task"
    config_path.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="differs"):
        launch.main(["--config", str(config_path), "--yes"])
    assert not launched


def test_missing_and_invalid_yaml_rejected(tmp_path, launched):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        launch.main(["--config", str(tmp_path / "missing.yaml"), "--yes"])
    bad = tmp_path / "bad.yaml"
    bad.write_text("env: []\n")
    with pytest.raises(ValueError, match="mapping"):
        launch.main(["--config", str(bad), "--yes"])
    assert not launched


def test_invalid_menu_choice_reprompts(config_path, launched, monkeypatch, capsys):
    answers = iter(["bad", "1", "2"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert launch.main([]) == 2
    assert "Invalid choice" in capsys.readouterr().out
    assert not launched


def test_subprocess_failure_is_propagated(config_path, monkeypatch):
    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(3, command)
    monkeypatch.setattr(launch.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError) as exc:
        launch.main(["--config", str(config_path), "--yes"])
    assert exc.value.returncode == 3


def test_ctrl_c_cancels_launcher(config_path, launched, monkeypatch):
    def interrupt(_):
        raise KeyboardInterrupt()
    monkeypatch.setattr("builtins.input", interrupt)
    assert launch.main([]) == 130
    assert not launched
