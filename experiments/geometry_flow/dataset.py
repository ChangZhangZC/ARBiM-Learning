"""Fixed-length, causal, full-rate sliding anchors with strided observations."""
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from .common import read_json

def expert_stride(temporal, expert_fps):
    t,k,w = (temporal[key] for key in ("window_size","sampling_stride","window_stride"))
    if any(type(v) is not int or v < 1 for v in (t,k,w)):
        raise ValueError("Window size and strides must be positive integers")
    freq = float(temporal["rollout_frequency"])
    if not np.isfinite(freq) or freq <= 0:
        raise ValueError("Invalid rollout frequency")
    stride = k*expert_fps/freq
    if stride < 1 or not np.isclose(stride,round(stride),atol=1e-6,rtol=0):
        raise ValueError("Expert FPS cannot represent rollout interval exactly")
    return int(round(stride))

def window_indices(length,t,stride,window_stride):
    span = (t-1)*stride
    return [np.arange(end-span,end+1,stride) for end in range(span,length,window_stride)]

class GeometryDataset(Dataset):
    def __init__(self, root, temporal, split):
        root = Path(root)
        self.meta = read_json(root/"meta.json")
        if split not in ("train","val"):
            raise ValueError("split must be train or val")
        self.params = dict(np.load(root/"preprocess.npz"))
        self.stride = expert_stride(temporal,self.meta["expert"]["fps"])
        self.t = temporal["window_size"]
        self.episodes = {}
        self.samples = []
        for ep in self.meta[f"{split}_ids"]:
            path = root/f"episode_{ep:06d}"
            left,right,q = (np.load(path/f"{k}.npy",mmap_mode="r")
                            for k in ("geometry_left","geometry_right","joint_state"))
            n = len(q)
            if left.shape != (n,self.meta["feature_dim"]) or right.shape != left.shape or q.shape != (n,self.meta["state_dim"]):
                raise ValueError("Feature/state shape mismatch")
            self.episodes[ep] = left,right,q
            for indices in window_indices(n,self.t,self.stride,temporal["window_stride"]):
                self.samples.append((ep,int(indices[-1])))
        if not self.samples:
            raise ValueError(f"No complete {split} windows")
    def __len__(self):
        return len(self.samples)
    def __getitem__(self,i):
        ep,end = self.samples[i]
        idx = np.arange(end-(self.t-1)*self.stride,end+1,self.stride)
        l,r,q = self.episodes[ep]
        return {"left":np.asarray(l[idx],dtype=np.float32),
                "right":np.asarray(r[idx],dtype=np.float32),
                "joint":((q[idx]-self.params["joint_mean"])/self.params["joint_std"]).astype(np.float32)}
