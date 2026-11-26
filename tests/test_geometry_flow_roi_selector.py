import json
import re

import numpy as np
import pytest

pytest.importorskip("cv2")
from experiments.geometry_flow.select_roi import build_payload, export_roi


def test_headless_roi_export(tmp_path):
    source = tmp_path / "episode_000020"
    source.mkdir()
    (source / "meta.json").write_text(json.dumps(dict(episode_id=20, frames=4, fps=30)))
    data = np.arange(4*10*12, dtype=np.float32).reshape(4,10,12)
    for side in ("left", "right"):
        np.save(source / f"depth_{side}.npy", data)
    cfg = dict(data=dict(depth_root=str(tmp_path)),
               encoder=dict(roi=dict(left=[0.25,0.2,1,1],right=None)))
    p = build_payload(cfg, 20, stride=2)
    assert p["indices"] == [0,2,3]
    assert p["sides"]["left"]["roi"] == [3,2,12,10]
    assert p["sides"]["right"]["roi"] is None
    assert len(p["sides"]["left"]["images"]) == 3
    out = tmp_path / "roi.html"
    export_roi(cfg, 20, out, stride=2, color_mode="fixed")
    html = out.read_text()
    saved = json.loads(re.search(r'<script id="payload" type="application/json">(.*?)</script>', html).group(1))
    assert saved["indices"] == [0,2,3]
    assert saved["sides"]["left"]["original_bounds"] == [0.25,0.2,1,1]
    assert saved["sides"]["right"]["width"] == 12
    np.testing.assert_array_equal(np.load(source/"depth_left.npy"),data)
    with pytest.raises(FileExistsError):
        export_roi(cfg, 20, out)
