"""Small shared I/O and DA3 adapter; no policy changes."""
from pathlib import Path
import json
import tempfile
import numpy as np
import yaml

SIDES = ("left", "right")

def load_config(path):
    with Path(path).open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Expected YAML mapping")
    return cfg

def read_json(path):
    with Path(path).open() as f:
        return json.load(f)

def write_json(path, value):
    with Path(path).open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")

def rgb_uint8(image):
    x = np.asarray(image)
    if x.ndim != 3:
        raise ValueError(f"Expected RGB image, got {x.shape}")
    if x.shape[0] == 3 and x.shape[-1] != 3:
        x = np.moveaxis(x, 0, -1)
    if x.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB, got {x.shape}")
    if x.dtype != np.uint8:
        if not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
            raise ValueError("Floating RGB must be finite in [0,1]")
        x = np.rint(x * 255).astype(np.uint8)
    return np.ascontiguousarray(x)

class DepthEstimator:
    def __init__(self, cfg):
        from depth_anything_3.api import DepthAnything3
        self.model = DepthAnything3.from_pretrained(cfg["model"]).to(cfg["device"]).eval()
        self.model.requires_grad_(False)

    def __call__(self, rgb):
        import torch
        from PIL import Image
        # Same single-image path API as experiments/da3; no color map or normalization.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rgb.png"
            Image.fromarray(rgb_uint8(rgb)).save(path)
            with torch.inference_mode():
                prediction = self.model.inference(image=[str(path)])
        depth = np.asarray(prediction.depth, dtype=np.float32)
        if depth.ndim != 3 or depth.shape[0] != 1 or not np.isfinite(depth).all():
            raise ValueError(f"Invalid DA3 output {depth.shape}")
        return depth[0]
