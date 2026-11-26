import numpy as np
import pytest
from experiments.geometry_flow.depth_preprocess import crop_roi
from experiments.geometry_flow.dataset import window_indices, expert_stride

def test_roi():
    x=np.arange(100).reshape(10,10)
    np.testing.assert_array_equal(crop_roi(x,[.2,.3,.8,.9]),x[3:9,2:8])
    with pytest.raises(ValueError):
        crop_roi(x,[.8,0,.2,1])

def test_full_rate_anchors():
    windows=window_indices(115,8,16,1)
    assert len(windows)==3
    assert windows[0].tolist()==[0,16,32,48,64,80,96,112]
    assert windows[1][-1]==113
    assert windows[2][-1]==114

def test_fps_conversion():
    cfg=dict(window_size=8,sampling_stride=16,window_stride=1,rollout_frequency=30)
    assert expert_stride(cfg,30)==16
    assert expert_stride(cfg,60)==32
    with pytest.raises(ValueError):
        expert_stride(cfg,25)
