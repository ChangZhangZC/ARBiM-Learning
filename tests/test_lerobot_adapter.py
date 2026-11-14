"""Mock upstream LeRobot/RL100 dependencies to test the dataset contract.

Real Parquet/video decoding and GPU forward/backward remain integration tests.
"""

import importlib
import sys
import types
from collections import UserDict

import numpy as np
import pytest
import torch


@pytest.fixture
def adapter(monkeypatch):
    rl = types.ModuleType("rl_100")
    rl.__path__ = []
    ds_pkg = types.ModuleType("rl_100.dataset")
    ds_pkg.__path__ = []
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
    lerobot = types.ModuleType("lerobot")
    lerobot.__path__ = []
    datasets_pkg = types.ModuleType("lerobot.datasets")
    lengths = [9, 8, 10]
    starts = np.cumsum([0] + lengths[:-1]).tolist()
    ends = np.cumsum(lengths).tolist()

    class Metadata:
        def __init__(self, repo_id, root=None):
            self.fps = 10
            self.total_episodes = 3
            self.episodes = [
                {"dataset_from_index": s, "dataset_to_index": e}
                for s, e in zip(starts, ends)
            ]
            self.features = {
                "observation.state": {"dtype": "float32", "shape": [2], "names": ["left", "right"]},
                "action": {"dtype": "float32", "shape": [2, 2]},
                "observation.images.head": {"dtype": "video", "shape": [8, 9, 3]},
                "observation.images.wrist": {"dtype": "image", "shape": [3, 8, 9]},
                "observation.images.depth": {
                    "dtype": "video", "shape": [8, 9, 3], "info": {"is_depth_map": True}
                },
                "observation.depth_h": {"dtype": "video", "shape": [8, 9, 3]},
            }

    class Column:
        def __init__(self, values):
            self.values = values

        def to_numpy(self, zero_copy_only=False):
            return np.asarray(self.values, dtype=np.int64)

    class Table:
        def __init__(self, ids):
            self.ids = ids

        def column(self, name):
            if name == "index":
                return Column(self.ids)
            return Column([
                next(i for i, end in enumerate(ends) if frame < end)
                for frame in self.ids
            ])

    class Raw:
        def __init__(self, ids, columns=None):
            self.ids, self.columns = ids, columns
            self.data = Table(ids)

        def __len__(self):
            return len(self.ids)

        def select_columns(self, keys):
            assert not any("images" in k for k in keys)
            return Raw(self.ids, keys)

        def __getitem__(self, slc):
            ids = self.ids[slc]
            return {
                k: [
                    torch.tensor([float(i), float(i + 1)])
                    if k == "observation.state"
                    else torch.full((2, 2), float(i))
                    for i in ids
                ]
                for k in self.columns
            }

    class LeRobotDataset:
        def __init__(
            self, repo_id, root=None, episodes=None, delta_timestamps=None,
            video_backend=None, return_uint8=False,
        ):
            assert return_uint8
            self.episodes = episodes
            self.deltas = delta_timestamps
            self.meta = Metadata(repo_id, root=root)
            self.ids = [i for ep in episodes for i in range(starts[ep], ends[ep])]
            self.hf_dataset = Raw(self.ids)

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, idx):
            anchor = self.ids[idx]
            ep = next(i for i, end in enumerate(ends) if anchor < end)
            result = {}
            for key, offsets in self.deltas.items():
                requested = [anchor + round(d * 10) for d in offsets]
                clamped = [min(ends[ep] - 1, max(starts[ep], i)) for i in requested]
                if key == "observation.state":
                    result[key] = torch.stack([
                        torch.tensor([float(i), float(i + 1)]) for i in clamped
                    ])
                elif key == "action":
                    result[key] = torch.stack([
                        torch.full((2, 2), float(i)) for i in clamped
                    ])
                    result["action_is_pad"] = torch.tensor([
                        i != j for i, j in zip(requested, clamped)
                    ])
                elif key.endswith("head"):
                    result[key] = torch.full(
                        (len(clamped), 3, 8, 9), 255, dtype=torch.uint8
                    )
                else:
                    result[key] = torch.full(
                        (len(clamped), 3, 8, 9), 0.5, dtype=torch.float32
                    )
            return result

    datasets_pkg.LeRobotDataset = LeRobotDataset
    datasets_pkg.LeRobotDatasetMetadata = Metadata
    stubs = {
        "rl_100": rl, "rl_100.dataset": ds_pkg,
        "rl_100.dataset.base_dataset": base, "rl_100.model": model,
        "rl_100.model.common": common,
        "rl_100.model.common.normalizer": norm,
        "lerobot": lerobot, "lerobot.datasets": datasets_pkg,
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules.pop("arbim.data.lerobot_adapter", None)
    imported = importlib.import_module("arbim.data.lerobot_adapter")
    yield imported
    sys.modules.pop("arbim.data.lerobot_adapter", None)


def test_metadata_shape_and_native_dict_apply(adapter):
    from lerobot.datasets import LeRobotDatasetMetadata

    spec = adapter.infer_dataset_spec(LeRobotDatasetMetadata("test"))
    assert spec["state_dim"] == 2 and spec["action_dim"] == 4
    assert spec["state_names"] == ("left", "right")
    assert spec["camera_keys"] == (
        "observation.images.head", "observation.images.wrist"
    )
    assert spec["camera_layouts"] == {
        "observation.images.head": "HWC", "observation.images.wrist": "CHW"
    }
    meta = adapter.build_shape_meta(spec)
    assert all(isinstance(x, UserDict) for x in meta["obs"].values())

    def dict_apply(x, func):
        return {
            k: dict_apply(v, func) if isinstance(v, dict) else func(v)
            for k, v in x.items()
        }

    assert dict_apply(meta["obs"], lambda x: x["shape"])["agent_pos"] == [2]
    assert meta["obs"]["observation.images.wrist"]["shape"] == [3, 8, 9]


def test_temporal_windows_uint8_and_padding(adapter):
    ds = adapter.LeRobotDP2DDataset("test", horizon=5, n_obs_steps=2, val_ratio=0)
    assert len(ds) == 27 - 3 * 3
    frame = ds[0]
    assert frame["obs"]["agent_pos"].shape == (2, 2)
    assert frame["action"].shape == (5, 4)
    assert frame["obs"]["observation.images.head"].shape == (2, 3, 8, 9)
    assert frame["obs"]["observation.images.head"].dtype == torch.uint8
    assert frame["obs"]["observation.images.wrist"].dtype == torch.float32
    assert not frame["action_is_pad"][1:].any()
    assert frame["action"][1, 0] == 0
    assert frame["obs"]["agent_pos"][-1, 0] == 0
    assert ds.prediction_horizon == 4
    assert ds.get_shape_info(4, 2)["action"] == (4,)
    from torch.utils.data import DataLoader
    batch = next(iter(DataLoader(ds, batch_size=2)))
    assert batch["action"].shape == (2, 5, 4)
    assert batch["action_is_pad"].shape == (2, 5)


def test_split_and_train_only_normalizer(adapter):
    ds = adapter.LeRobotDP2DDataset("test", horizon=5, n_obs_steps=2, val_ratio=0.33)
    val = ds.get_validation_dataset()
    assert set(ds._train_episodes).isdisjoint(val._val_episodes)
    assert val._val_episodes == ds._val_episodes
    assert len(ds) and len(val)
    normalizer = ds.get_normalizer()
    assert normalizer.mode == "limits"
    assert normalizer.data["agent_pos"].shape == (2, 2)
    assert normalizer.data["action"].shape == (2, 4)
    allowed = [
        i for ep in ds._train_episodes
        for i in range([0, 9, 17][ep], [9, 17, 27][ep])
    ]
    assert normalizer.data["agent_pos"][0, 0] == min(allowed)
    assert normalizer.data["agent_pos"][1, 0] == max(allowed)
    assert ds.get_all_actions().shape == (len(allowed), 4)
    with pytest.raises(RuntimeError, match="training split"):
        val.get_normalizer()
    with pytest.raises(ValueError, match="only"):
        ds.get_normalizer(mode="gaussian")


def test_optional_padding_and_empty_validation(adapter):
    ds = adapter.LeRobotDP2DDataset(
        "test", horizon=5, n_obs_steps=2, val_ratio=0,
        drop_padded_action_targets=False,
    )
    assert len(ds) == 27
    assert ds[8]["action_is_pad"][1:].any()
    with pytest.raises(ValueError, match="No validation episodes"):
        ds.get_validation_dataset()


def test_depth_layout_numeric_and_rgb_errors(adapter):
    from lerobot.datasets import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata("test")
    with pytest.raises(ValueError, match="depth"):
        adapter.infer_dataset_spec(meta, camera_keys=["observation.images.depth"])
    with pytest.raises(ValueError, match="depth"):
        adapter.infer_dataset_spec(meta, camera_keys=["observation.depth_h"])
    meta.features["bad"] = {"dtype": "image", "shape": [3, 6, 3]}
    with pytest.raises(ValueError, match="Ambiguous"):
        adapter.infer_dataset_spec(meta, camera_keys=["bad"])
    with pytest.raises(ValueError, match="NaN"):
        adapter._flat_window(
            torch.tensor([[float("nan"), 1]]), "action", 1, (2,)
        )
    with pytest.raises(ValueError, match="floating RGB"):
        adapter._rgb_window(
            torch.full((1, 3, 8, 9), 200.0), "cam", 1, (3, 8, 9)
        )
    image = adapter._rgb_window(
        torch.zeros((1, 8, 9, 3), dtype=torch.uint8), "cam", 1, (3, 8, 9)
    )
    assert image.shape == (1, 3, 8, 9) and image.dtype == torch.uint8


def test_invalid_horizon(adapter):
    with pytest.raises(ValueError, match="horizon"):
        adapter.LeRobotDP2DDataset("test", horizon=1, n_obs_steps=2)
