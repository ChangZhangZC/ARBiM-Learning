"""Output run-name template tests (no simulator or GPU required)."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from arbim.training.run_utils import check_config, resolve_run_directory
from arbim.rollout.eval import resolve_eval_run_id


def _training_config(tmp_path):
    path = Path(__file__).resolve().parents[1] / "configs/il/rl100_dp2d.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["output"]["root"] = str(tmp_path)
    return cfg


def test_training_run_name_expands_timestamp_and_preserves_prefix(tmp_path):
    cfg = _training_config(tmp_path)
    cfg["training"]["run_name"] = "experiment_A_{timestep}"
    check_config(cfg)
    path = resolve_run_directory(cfg, None)
    assert path.parent == tmp_path / cfg["training"]["task"] / cfg["training"]["method"]
    assert re.fullmatch(r"experiment_A_\\d{8}_\\d{6}", path.name)


def test_training_fixed_name_and_null_fallback(tmp_path):
    cfg = _training_config(tmp_path)
    cfg["training"]["run_name"] = "fixed_label"
    check_config(cfg)
    assert resolve_run_directory(cfg, None).name == "fixed_label"
    cfg["training"]["run_name"] = None
    check_config(cfg)
    assert re.fullmatch(r"\\d{8}_\\d{6}", resolve_run_directory(cfg, None).name)


@pytest.mark.parametrize("invalid", ["oops_{wrong}", "../escape_{timestep}", "bad/name", ""])
def test_training_rejects_invalid_run_names(tmp_path, invalid):
    cfg = _training_config(tmp_path)
    cfg["training"]["run_name"] = invalid
    with pytest.raises(ValueError, match="training.run_name"):
        check_config(cfg)


def test_eval_run_id_timestamp_template_and_fixed():
    assert re.fullmatch(r"eval_\\d{8}_\\d{6}_\\d{6}", resolve_eval_run_id("eval_{timestep}"))
    assert re.fullmatch(r"trial_\\d{8}_\\d{6}_\\d{6}", resolve_eval_run_id("trial_{timestep}"))
    assert re.fullmatch(r"\\d{8}_\\d{6}_\\d{6}", resolve_eval_run_id(None))
    assert resolve_eval_run_id("fixed_eval") == "fixed_eval"


@pytest.mark.parametrize("invalid", ["wrong_{date}", "bad/name", "../bad", "", 123])
def test_eval_rejects_invalid_run_ids(invalid):
    with pytest.raises(ValueError, match="output.run_id"):
        resolve_eval_run_id(invalid)
