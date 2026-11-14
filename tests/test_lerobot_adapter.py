"""Contract tests for ARBiM's LeRobot v3 -> RL-100 DP-2D adapter.

These tests mock upstream packages. They do not replace a smoke test with
real Parquet/video data and the pinned RL-100/LeRobot packages.
"""

import importlib
import sys
import types

import pytest
import torch


@pytest.fixture
def adapter(monkeypatch):
    rl = types.ModuleType("rl_100")
    rl.__path__ = []
    dataset_pkg = types.ModuleType("rl_100.dataset")
    dataset_pkg.__path__ = []
    base = types.ModuleType("rl_100.dataset.base_dataset")
    base.BaseDataset = type("BaseDataset", (torch.utils.data.Dataset,), {})
    model = types.ModuleType("rl_100.model")
    model.__path__ = []
    common = types.ModuleType("rl_100.model.common")
    common.__path__ = []
    norm = types.ModuleType("rl_100.model.common.normalizer")

    class Normalizer:
        def fit(self, data, last_n_dims, mode, **kwargs):
            self.data, self.mode = data, mode

    norm.LinearNormalizer = Normalizer
    lr = types.ModuleType("lerobot")
    lr.__path__ = []
    ds_module = types.ModuleType("lerobot.datasets")

    class Metadata:
        def __init__(self, repo_id, root=None):
            self.fps = 10
            self.total_episodes = 2
            self.features = {
                "observation.state": {"dtype": "float32", "shape": [7]},
                "action": {"dtype": "float32", "shape": [2, 7]},
                "head": {"dtype": "video", "shape": [8, 9, 3]},
                "wrist": {"dtype": "image", "shape": [8, 9, 3]},
                "depth": {"dtype": "video", "shape": [8, 9, 3], "video.is_depth_map": True},
            }

    class Columns:
        def __init__(self, episodes):
            self.episodes = episodes

        def __len__(self):
            return len(self.episodes) * 2

        def select_columns(self, columns):
            assert set(columns) == {"observation.state", "action"}
            return self

        def __getitem__(self, slc):
            idx = list(range(len(self)))[slc]
            return {
                "observation.state": [torch.full((7,), float(i)) for i in idx],
                "action": [torch.full((2, 7), float(i)) for i in idx],
            }

    class LeRobotDataset:
        def __init__(self, repo_id, root=None, episodes=None, delta_timestamps=None, video_backend=None):
            self.deltas = delta_timestamps
            self.episodes = episodes
            self.hf_dataset = Columns(episodes)

        def __len__(self):
            return len(self.episodes) * 2

        def __getitem__(self, idx):
            out = {}
            for key, steps in self.deltas.items():
                n = len(steps)
                if key == "observation.state":
                    out[key] = torch.zeros(n, 7)
                elif key == "action":
                    out[key] = torch.ones(n, 2, 7)
                else:
                    out[key] = torch.full((n, 3, 8, 9), 0.5)
            return out

    ds_module.LeRobotDataset = LeRobotDataset
    ds_module.LeRobotDatasetMetadata = Metadata
    stubs = {
        "rl_100": rl, "rl_100.dataset": dataset_pkg,
        "rl_100.dataset.base_dataset": base, "rl_100.model": model,
        "rl_100.model.common": common, "rl_100.model.common.normalizer": norm,
        "lerobot": lr, "lerobot.datasets": ds_module,
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules.pop("arbim.data.lerobot_adapter", None)
    imported = importlib.import_module("arbim.data.lerobot_adapter")
    yield imported
    sys.modules.pop("arbim.data.lerobot_adapter", None)


def test_meta_and_shape(adapter):
    from lerobot.datasets import LeRobotDatasetMetadata

    spec = adapter.infer_dataset_spec(LeRobotDatasetMetadata("fake"))
    assert spec["state_dim"] == 7
    assert spec["action_dim"] == 14
    assert spec["camera_keys"] == ("head", "wrist")
    meta = adapter.build_shape_meta(spec)
    assert meta["obs"]["head"]["shape"] == [3, 8, 9]
    assert meta["action"]["shape"] == [14]


def test_il_window_and_batch(adapter):
    from torch.utils.data import DataLoader

    ds = adapter.LeRobotDP2DDataset("fake", horizon=5, n_obs_steps=2, val_ratio=0)
    assert len(ds) == 4
    item = ds[0]
    assert item["obs"]["agent_pos"].shape == (2, 7)
    assert item["obs"]["head"].shape == (2, 3, 8, 9)
    assert item["action"].shape == (5, 14)
    assert next(iter(DataLoader(ds, batch_size=2)))["action"].shape == (2, 5, 14)
    assert ds.get_shape_info(4, 2)["action"] == (14,)


def test_episode_split_and_native_normalizer(adapter):
    ds = adapter.LeRobotDP2DDataset("fake", val_ratio=0.5)
    val = ds.get_validation_dataset()
    assert len(ds) == len(val) == 2
    assert set(ds._train_episodes).isdisjoint(set(val._val_episodes))
    norm = ds.get_normalizer()
    assert norm.mode == "limits"
    assert norm.data["agent_pos"].shape == (2, 7)
    assert norm.data["action"].shape == (2, 14)
    with pytest.raises(RuntimeError, match="training split"):
        val.get_normalizer()


def test_validation_rejects_bad_inputs(adapter):
    from lerobot.datasets import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata("fake")
    with pytest.raises(ValueError, match="depth map"):
        adapter.infer_dataset_spec(meta, camera_keys=["depth"])
    with pytest.raises(ValueError, match="horizon"):
        adapter.LeRobotDP2DDataset("fake", horizon=1, n_obs_steps=2)
    with pytest.raises(ValueError, match="NaN"):
        adapter._flat_window(torch.tensor([[float("nan")]]), "action", 1, 1)
    rgb = adapter._float_rgb(torch.full((1, 3, 8, 9), 255, dtype=torch.uint8), "head", 1, (3, 8, 9))
    assert (rgb == 1).all()
