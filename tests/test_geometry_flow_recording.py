import csv
import numpy as np
import pytest
pytest.importorskip("cv2")
from experiments.geometry_flow.recording import EpisodeRecording

def test_frame_chunk_alignment(tmp_path, monkeypatch):
    frames=[]
    def write(self,name,bgr):
        frames.append((name,bgr.shape))
    monkeypatch.setattr(EpisodeRecording,"_write",write)
    raw={"frames":{s:{"rgb":{"data":np.zeros((8,8,3),np.uint8)}} for s in ("head","left_wrist","right_wrist")}}
    def sample(step,ready):
        return {"step":step,"rgb":{s:np.zeros((8,8,3),np.uint8) for s in ("left","right")},
                "depth":{s:np.full((8,8),step+1,dtype=np.float32) for s in ("left","right")},
                "joint_state":np.ones(16,dtype=np.float32),
                "score":{"ready":ready,"left_nll":10. if ready else None,"right_nll":20. if ready else None},
                "history_size":2 if ready else 1,"history_required":2}
    path=tmp_path/"episode_000000"
    with EpisodeRecording(path,30,{"left":None,"right":None},lambda:np.zeros((12,16,3),np.uint8)) as recorder:
        for i in range(5):
            recorder.on_frame(raw,i,i//2)
            if i in (0,2):
                recorder.on_chunk(sample(i,i==2),i//2,i//2,90.)
    with (path/"frame_index.csv").open() as f:
        rows=list(csv.DictReader(f))
    assert [r["monitor_step"] for r in rows]==["0","0","2","2","2"]
    assert [r["env_step"] for r in rows]==["0","1","2","3","4"]
    assert [r["stage"] for r in rows]==["0","0","1","1","2"]
    assert sum(name=="synchronized" for name,_ in frames)==5
    np.testing.assert_array_equal(np.load(path/"env_steps.npy"),[0,2])
    assert np.load(path/"left_depth.npy").shape==(2,8,8)
    assert np.load(path/"left_depth.npy")[1,0,0]==3
    assert np.load(path/"joint_state.npy").shape==(2,16)
    assert (path/"left_nll.png").exists()

def test_no_chunk_episode(tmp_path,monkeypatch):
    monkeypatch.setattr(EpisodeRecording,"_write",lambda *args:None)
    raw={"frames":{s:{"rgb":{"data":np.zeros((8,8,3),np.uint8)}} for s in ("head","left_wrist","right_wrist")}}
    path=tmp_path/"ep"
    with EpisodeRecording(path,30,{"left":None,"right":None},lambda:np.zeros((12,16,3),np.uint8)) as recorder:
        recorder.on_frame(raw,0,3)
    assert np.load(path/"env_steps.npy").size==0
    with (path/"frame_index.csv").open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==1 and rows[0]["monitor_step"]==""
