"""Generate unnormalized DA3 depth for every frame of selected expert episodes."""
import argparse
import shutil
from pathlib import Path
import tempfile
import numpy as np
from numpy.lib.format import open_memmap
from .common import SIDES, DepthEstimator, load_config, rgb_uint8, write_json, read_json
from .episode_selection import resolve_episode_ids

def check_existing(target, expected):
    """Reuse only complete, matching outputs; never overwrite."""
    meta = read_json(target / "meta.json")
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Existing episode metadata mismatch: {target}")
    n = meta["frames"]
    for side in SIDES:
        depth = np.load(target / f"depth_{side}.npy", mmap_mode="r")
        if depth.ndim != 3 or depth.shape[0] != n or depth.dtype != np.float32:
            raise ValueError(f"Invalid cached depth: {target}/{side}")
    q = np.load(target / "joint_state.npy", mmap_mode="r")
    if q.ndim != 2 or len(q) != n:
        raise ValueError(f"Invalid cached state: {target}")
    for key in ("timestamp", "frame_index", "global_index"):
        if np.load(target / f"{key}.npy", mmap_mode="r").shape != (n,):
            raise ValueError(f"Invalid cached {key}: {target}")

def generate(cfg, skip_existing=False):
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
    data = cfg["data"]
    if not data["dataset_root"]:
        raise ValueError("Set data.dataset_root")
    root = Path(data["dataset_root"]).expanduser().resolve()
    meta = LeRobotDatasetMetadata(root.name, root=root)
    ids = resolve_episode_ids(data["episode_ids"], int(meta.total_episodes))
    print(f"Selected {len(ids)} expert episodes; caller must ensure success", flush=True)
    fps = float(meta.fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid dataset FPS")
    out = Path(data["depth_root"]).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    pending = []
    for ep in ids:
        target = out / f"episode_{ep:06d}"
        if not target.exists():
            pending.append(ep)
            continue
        if not skip_existing:
            raise FileExistsError(f"{target}; use --skip-existing to reuse matching outputs")
        check_existing(target, {"episode_id": ep, "fps": fps,
                       "dataset_root": str(root), "state_key": data["state_key"],
                       "state_names": meta.features[data["state_key"]].get("names"),
                       "camera_keys": data["cameras"], "da3_model": cfg["da3"]["model"]})
        print(f"Reusing {target}", flush=True)
    if not pending:
        return
    estimator = DepthEstimator(cfg["da3"])
    for ep in pending:
        target = out / f"episode_{ep:06d}"
        dataset = LeRobotDataset(root.name, root=root, episodes=[ep],
                                 return_uint8=True, video_backend=data.get("video_backend"))
        n = len(dataset)
        if not n:
            raise ValueError("Empty episode")
        temp = Path(tempfile.mkdtemp(prefix=".partial-", dir=out))
        arrays = {}
        try:
            states, timestamps, frames, global_indices = [], [], [], []
            rgb_shapes = {}
            for i in range(n):
                row = dataset[i]
                if int(row["episode_index"]) != ep or int(row["frame_index"]) != i:
                    raise ValueError("Episode/frame ordering mismatch")
                state = np.asarray(row[data["state_key"]], dtype=np.float32).reshape(-1)
                if not np.isfinite(state).all():
                    raise ValueError("Invalid joint state")
                states.append(state)
                timestamps.append(float(row["timestamp"]))
                frames.append(i)
                global_indices.append(int(row["index"]))
                for side in SIDES:
                    rgb = rgb_uint8(row[data["cameras"][side]])
                    if i == 0:
                        rgb_shapes[side] = list(rgb.shape)
                    if list(rgb.shape) != rgb_shapes[side]:
                        raise ValueError("RGB resolution changed")
                    depth = estimator(rgb)
                    if i == 0:
                        arrays[side] = open_memmap(temp / f"depth_{side}.npy", mode="w+",
                                                   dtype=np.float32, shape=(n, *depth.shape))
                    if depth.shape != arrays[side].shape[1:]:
                        raise ValueError("Depth resolution changed")
                    arrays[side][i] = depth
                if i % 50 == 0:
                    print(f"episode={ep} frame={i}/{n}", flush=True)
            times = np.asarray(timestamps, dtype=np.float64)
            if not np.isfinite(times).all() or not np.allclose(np.diff(times), 1/fps, atol=1e-4, rtol=1e-3):
                raise ValueError("Non-uniform expert timestamps")
            for arr in arrays.values():
                arr.flush()
            arrays.clear()
            np.save(temp / "joint_state.npy", np.stack(states))
            np.save(temp / "timestamp.npy", times)
            np.save(temp / "frame_index.npy", np.asarray(frames))
            np.save(temp / "global_index.npy", np.asarray(global_indices))
            write_json(temp / "meta.json", {"episode_id": ep, "fps": fps, "frames": n,
                       "dataset_root": str(root), "state_key": data["state_key"],
                       "state_names": meta.features[data["state_key"]].get("names"),
                       "camera_keys": data["cameras"], "rgb_shapes": rgb_shapes,
                       "da3_model": cfg["da3"]["model"]})
            temp.rename(target)
        except BaseException:
            arrays.clear()
            shutil.rmtree(temp, ignore_errors=True)
            raise
        print(f"Saved {target}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/geometry_flow/config.yaml")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Reuse matching complete episode outputs without overwriting")
    args = parser.parse_args()
    generate(load_config(args.config), skip_existing=args.skip_existing)
