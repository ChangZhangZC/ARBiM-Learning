import numpy as np
import pytest
pytest.importorskip("cv2")
from experiments.geometry_flow.inspect_depth import panel,colorize

def test_panel_and_colorize():
    d=np.linspace(.1,2,100,dtype=np.float32).reshape(10,10)
    before=d.copy()
    assert colorize(d,.1,2).shape==(10,10,3)
    assert panel(d,[.1,.2,.9,1],.1,2,"left",0).shape==(504,1008,3)
    np.testing.assert_array_equal(d,before)
