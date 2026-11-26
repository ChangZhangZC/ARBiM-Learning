"""Fixed ROI and fixed ImageNet channel transform; no depth rescaling."""
import numpy as np
import torch
import torch.nn.functional as F

def crop_roi(depth, bounds):
    x = np.asarray(depth)
    if x.ndim < 2:
        raise ValueError("Expected depth image(s)")
    if bounds is None:
        return x
    if len(bounds) != 4:
        raise ValueError("ROI must be [x1,y1,x2,y2]")
    x1,y1,x2,y2 = map(float,bounds)
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("Invalid normalized ROI")
    h,w = x.shape[-2:]
    a,b,c,d = int(x1*w),int(y1*h),int(x2*w),int(y2*h)
    if a >= c or b >= d:
        raise ValueError("Empty ROI")
    return x[...,b:d,a:c]

def preprocess(depth, bounds, device):
    x = np.array(crop_roi(depth,bounds), dtype=np.float32, copy=True)
    if x.ndim == 2:
        x = x[None]
    if x.ndim != 3 or not np.isfinite(x).all():
        raise ValueError("Expected finite [B,H,W] depth")
    x = torch.from_numpy(x).to(device).unsqueeze(1)
    x = F.interpolate(x, size=(224,224), mode="bilinear", align_corners=False, antialias=True)
    x = x.repeat(1,3,1,1)
    mean = x.new_tensor([0.485,0.456,0.406])[None,:,None,None]
    std = x.new_tensor([0.229,0.224,0.225])[None,:,None,None]
    return (x-mean)/std
