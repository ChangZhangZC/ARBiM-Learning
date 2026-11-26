"""Chunk-boundary shadow monitor with checkpoint-defined preprocessing."""
from collections import deque
from pathlib import Path
import numpy as np
import torch
from .common import DepthEstimator, SIDES, read_json, rgb_uint8
from .depth_encoder import DepthEncoder
from .extract_features import project
from .conditional_flow import BimanualFlow

class WindowHistory:
    def __init__(self, t, stride):
        self.t,self.stride=t,stride
        self.reset()
    def reset(self):
        self.frames=deque(maxlen=self.t)
        self.last_step=None
    def append(self,step,left,right,joint):
        if self.last_step is None and step != 0:
            raise ValueError("New episode must start at step zero")
        if self.last_step is not None and step-self.last_step != self.stride:
            raise ValueError("Chunk interval differs from trained sampling stride")
        self.last_step=step
        self.frames.append((left,right,joint))
        if len(self.frames)<self.t:
            return None
        return [np.stack(items)[None] for items in zip(*self.frames)]

class GeometryMonitor:
    def __init__(self, model_dir, device, da3_device=None):
        path=Path(model_dir)
        self.contract=read_json(path/"contract.json")
        self.meta=self.contract["feature_meta"]
        self.expert=self.meta["expert"]
        tc=self.contract["temporal"]
        self.device=torch.device(device)
        self.history=WindowHistory(tc["window_size"],tc["sampling_stride"])
        self.params=dict(np.load(path/"preprocess.npz"))
        self.model=BimanualFlow(tc["window_size"],self.meta["feature_dim"],
                                self.meta["state_dim"],self.contract["flow"]).to(device).eval()
        for side in SIDES:
            self.model.flows[side].load_state_dict(torch.load(path/f"best_{side}.pt",map_location=device,weights_only=True))
        self.encoder=DepthEncoder(device)
        self.depth=DepthEstimator({"model":self.expert["da3_model"],"device":da3_device or device})
    def reset(self):
        self.history.reset()
    @torch.inference_mode()
    def observe(self,observation,step):
        features={}
        for side in SIDES:
            rgb=rgb_uint8(observation[self.expert["camera_keys"][side]])
            if list(rgb.shape)!=self.expert["rgb_shapes"][side]:
                raise ValueError(f"{side} expert/rollout RGB resolution mismatch")
            depth=self.depth(rgb)
            features[side]=project(self.encoder(depth,self.meta["roi"][side]),self.params,side)[0]
        q=np.asarray(observation[self.expert["state_key"]],dtype=np.float32).reshape(-1)
        if q.shape!=(self.meta["state_dim"],) or not np.isfinite(q).all():
            raise ValueError("Invalid robot state")
        q=(q-self.params["joint_mean"])/self.params["joint_std"]
        window=self.history.append(step,features["left"],features["right"],q)
        if window is None:
            return {"ready":False,"left_nll":None,"right_nll":None}
        scores=self.model.scores(*(torch.as_tensor(x,device=self.device) for x in window))
        result={f"{side}_nll":float(scores[side].item()) for side in SIDES}
        if not all(np.isfinite(v) for v in result.values()):
            raise RuntimeError("Nonfinite monitor NLL")
        return {"ready":True,**result}
