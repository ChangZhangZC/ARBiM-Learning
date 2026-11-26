"""Cache all-frame features; fit separate PCA and standardizers on train episodes."""
import argparse
from pathlib import Path
import shutil
import tempfile
import numpy as np
from sklearn.decomposition import PCA
from .common import SIDES, load_config, read_json, write_json
from .depth_encoder import DepthEncoder

def split_episodes(ids, ratio, seed):
    if len(ids) < 2 or not 0 < ratio < 1:
        raise ValueError("At least two episodes and 0 < val_ratio < 1 required")
    rng = np.random.default_rng(seed)
    n = min(len(ids)-1, max(1, round(len(ids)*ratio)))
    val = sorted(int(i) for i in rng.choice(ids,n,replace=False))
    return sorted(set(ids)-set(val)), val

def stats(x):
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.where(std > 1e-6,std,1.0).astype(np.float32)

def project(x, params, side):
    y = (x-params[f"{side}_pca_mean"]) @ params[f"{side}_components"].T
    return ((y-params[f"{side}_mean"])/params[f"{side}_std"]).astype(np.float32)

def extract(cfg):
    data, enc = cfg["data"], cfg["encoder"]
    ids = data["episode_ids"]
    train,val = split_episodes(ids,data["val_ratio"],data["seed"])
    source = Path(data["depth_root"]).expanduser()
    target = Path(data["feature_root"]).expanduser()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".features-",dir=target.parent))
    try:
        encoder = DepthEncoder(enc["device"])
        meta0 = None
        for ep in ids:
            name = f"episode_{ep:06d}"
            src,dst = source/name,temp/name
            dst.mkdir()
            meta = read_json(src/"meta.json")
            if meta["episode_id"] != ep or meta["da3_model"] != cfg["da3"]["model"]:
                raise ValueError("Expert metadata mismatch")
            contract = {k:meta[k] for k in ("fps","dataset_root","state_key","state_names","camera_keys","rgb_shapes","da3_model")}
            if meta0 is not None and contract != meta0:
                raise ValueError("Inconsistent expert metadata")
            meta0 = contract
            for side in SIDES:
                depth = np.load(src/f"depth_{side}.npy",mmap_mode="r")
                if len(depth) != meta["frames"]:
                    raise ValueError("Depth length mismatch")
                bs = int(enc["batch_size"])
                if bs < 1:
                    raise ValueError("Invalid batch size")
                features = np.concatenate([encoder(depth[i:i+bs],enc["roi"][side])
                                            for i in range(0,len(depth),bs)])
                if not np.isfinite(features).all():
                    raise ValueError("Nonfinite encoder features")
                np.save(dst/f"raw_{side}.npy",features)
            for key in ("joint_state","timestamp","frame_index","global_index"):
                shutil.copyfile(src/f"{key}.npy",dst/f"{key}.npy")
        params = {}
        for side in SIDES:
            x = np.concatenate([np.load(temp/f"episode_{ep:06d}"/f"raw_{side}.npy") for ep in train])
            pca = PCA(n_components=enc["feature_dim"],svd_solver="randomized",random_state=data["seed"])
            pca.fit(x)
            params[f"{side}_pca_mean"] = pca.mean_.astype(np.float32)
            params[f"{side}_components"] = pca.components_.astype(np.float32)
            y = (x-params[f"{side}_pca_mean"]) @ params[f"{side}_components"].T
            params[f"{side}_mean"],params[f"{side}_std"] = stats(y)
            for ep in ids:
                folder = temp/f"episode_{ep:06d}"
                raw = folder/f"raw_{side}.npy"
                np.save(folder/f"geometry_{side}.npy",project(np.load(raw),params,side))
                raw.unlink()
        q = np.concatenate([np.load(temp/f"episode_{ep:06d}"/"joint_state.npy") for ep in train])
        params["joint_mean"],params["joint_std"] = stats(q)
        np.savez(temp/"preprocess.npz",**params)
        write_json(temp/"meta.json",{"train_ids":train,"val_ids":val,"expert":meta0,
                  "roi":enc["roi"],"feature_dim":enc["feature_dim"],"state_dim":q.shape[1],
                  "encoder":"resnet18_IMAGENET1K_V1_pool2x2","depth_transform":"raw_imagenet_fixed"})
        temp.rename(target)
    except BaseException:
        shutil.rmtree(temp,ignore_errors=True)
        raise
    print(f"Saved {target}")

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="experiments/geometry_flow/config.yaml")
    extract(load_config(p.parse_args().config))
