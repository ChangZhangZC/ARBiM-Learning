import numpy as np
import pytest
pytest.importorskip("cv2")
from experiments.geometry_flow.visualization import render_panel,sample_text
from experiments.geometry_flow.viewer import write_packet,live_overlay

def test_render_packet(tmp_path):
    sides=("left","right")
    sample=dict(step=112,depth={s:np.arange(64,dtype=np.float32).reshape(8,8) for s in sides},
                rgb={s:np.zeros((8,8,3),dtype=np.uint8) for s in sides},
                score=dict(ready=True,left_nll=750.,right_nll=800.),
                joint_state=np.ones(16),history_size=8,history_required=8)
    history={s:[(0,None),(112,sample["score"][s+"_nll"])] for s in sides}
    before=sample["depth"]["left"].copy()
    image=render_panel(sample,{"left":[.25,.25,1,1],"right":None},history)
    assert image.shape==(346,1040,3) and image.dtype==np.uint8
    np.testing.assert_array_equal(sample["depth"]["left"],before)
    labels=sample_text(sample,1,7,2,90.)
    assert "1 / 112 / 7" in labels[1]
    path=tmp_path/"latest.npz"
    write_packet(path,image,labels)
    with np.load(path,allow_pickle=False) as f:
        np.testing.assert_array_equal(f["image"],image)
        assert tuple(f["text"].tolist())==labels
    write_packet(path,image,("new","sample"))
    with np.load(path,allow_pickle=False) as f:
        assert tuple(f["text"].tolist())==("new","sample")
    assert not path.with_suffix(".tmp").exists()

def test_disabled_and_headless():
    with live_overlay(False,{}) as overlay:
        assert overlay is None
    with pytest.raises(RuntimeError,match="headless"):
        with live_overlay(True,{"headless":True}):
            pass
