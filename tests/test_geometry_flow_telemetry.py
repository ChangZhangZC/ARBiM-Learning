import numpy as np
import torch
from experiments.geometry_flow.detector import GeometryMonitor, WindowHistory

class FakeModel:
    def scores(self, *args):
        return {"left": torch.tensor([10.]), "right": torch.tensor([20.])}

def make_monitor(capture):
    m = GeometryMonitor.__new__(GeometryMonitor)
    m.capture = capture
    m.last_sample = None
    m.expert = {"camera_keys":{"left":"l","right":"r"},
                "rgb_shapes":{"left":[2,2,3],"right":[2,2,3]},
                "state_key":"q"}
    m.meta = {"roi":{"left":None,"right":None},"state_dim":2}
    m.params = {"joint_mean":np.ones(2),"joint_std":np.ones(2)}
    for s in ("left","right"):
        m.params.update({f"{s}_pca_mean":np.zeros(2),f"{s}_components":np.eye(2),
                         f"{s}_mean":np.zeros(2),f"{s}_std":np.ones(2)})
    m.depth_calls = 0
    def depth(rgb):
        m.depth_calls += 1
        return rgb[:,:,0].astype(np.float32)
    m.depth = depth
    m.encoder = lambda d,roi: np.array([[1.,2.]],dtype=np.float32)
    m.history = WindowHistory(2,16)
    m.device = "cpu"
    m.model = FakeModel()
    return m

def test_capture_preserves_scores_and_reuses_depth():
    obs = {"l":np.ones((2,2,3),dtype=np.uint8),
           "r":np.full((2,2,3),2,dtype=np.uint8),"q":np.array([.4,.9],dtype=np.float32)}
    a,b = make_monitor(True),make_monitor(False)
    for step in (0,16):
        assert a.observe(obs,step)==b.observe(obs,step)
        assert a.last_sample["step"]==step
        assert a.last_sample["history_size"]==step//16+1
        np.testing.assert_array_equal(a.last_sample["joint_state"],obs["q"])
        np.testing.assert_array_equal(a.last_sample["depth"]["right"],obs["r"][:,:,0])
        assert b.last_sample is None
    assert a.depth_calls==b.depth_calls==4
    a.reset()
    assert a.last_sample is None
