import numpy as np
import pytest
from experiments.geometry_flow.detector import WindowHistory

def test_warmup_reset_stride():
    h=WindowHistory(8,16)
    for i in range(7):
        assert h.append(i*16,np.ones(4),np.zeros(4),np.zeros(16)) is None
    l,r,q=h.append(112,np.ones(4),np.zeros(4),np.zeros(16))
    assert l.shape==(1,8,4) and q.shape==(1,8,16)
    with pytest.raises(ValueError):
        h.append(113,np.ones(4),np.zeros(4),np.zeros(16))
    h.reset()
    assert h.append(0,np.ones(4),np.zeros(4),np.zeros(16)) is None
