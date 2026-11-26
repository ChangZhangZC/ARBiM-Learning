import sys

import numpy as np
import pytest

from experiments.geometry_flow.select_roi import pixel_bounds, check_display
from experiments.geometry_flow.depth_preprocess import crop_roi


@pytest.mark.parametrize("size",[504,224,17])
def test_pixel_round_trip(size):
    x=np.zeros((size,size))
    for p in range(size):
        b=pixel_bounds([p,p,size,size],size,size)
        assert crop_roi(x,b).shape==(size-p,size-p)
    assert pixel_bounds([0,0,size,size],size,size)==[0,0,1,1]


def test_no_display(monkeypatch):
    monkeypatch.setattr(sys,"platform","linux")
    monkeypatch.delenv("DISPLAY",raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY",raising=False)
    with pytest.raises(RuntimeError,match="No graphical display"):
        check_display()
