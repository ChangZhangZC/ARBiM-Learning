"""Inspect cached raw depths without DA3 inference or modifying training data."""
import argparse
import csv
from pathlib import Path
import numpy as np
from .common import SIDES, load_config, read_json, write_json
from .depth_preprocess import crop_roi

def colorize(depth, lo, hi):
    import cv2
    if not np.isfinite(depth).all():
        raise ValueError("Nonfinite depth")
    scaled = np.clip((depth-lo)/max(hi-lo,1e-8),0,1)
    return cv2.applyColorMap(np.rint(scaled*255).astype(np.uint8),cv2.COLORMAP_TURBO)

def panel(depth, bounds, lo, hi, side, frame, size=504):
    import cv2
    original = colorize(depth,lo,hi)
    cropped = colorize(crop_roi(depth,bounds),lo,hi)
    if bounds is not None:
        h,w=depth.shape
        x1,y1,x2,y2=bounds
        cv2.rectangle(original,(int(x1*w),int(y1*h)),(int(x2*w)-1,int(y2*h)-1),(255,255,255),2)
    def fit(im):
        h,w=im.shape[:2]
        scale=min(size/h,size/w)
        resized=cv2.resize(im,(max(1,round(w*scale)),max(1,round(h*scale))))
        canvas=np.zeros((size,size,3),dtype=np.uint8)
        y=(size-resized.shape[0])//2
        x=(size-resized.shape[1])//2
        canvas[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
        return canvas
    result=np.concatenate([fit(original),fit(cropped)],axis=1)
    cv2.putText(result,f"{side} frame={frame} full | ROI", (8,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
    return result

def inspect(cfg, episode, output, mode="fixed", stride=1, keyframes=12):
    import cv2
    if stride<1 or keyframes<1:
        raise ValueError("Positive stride and keyframe count required")
    source=Path(cfg["data"]["depth_root"])/f"episode_{episode:06d}"
    meta=read_json(source/"meta.json")
    arrays={s:np.load(source/f"depth_{s}.npy",mmap_mode="r") for s in SIDES}
    n=meta["frames"]
    if n<1 or any(a.ndim!=3 or len(a)!=n for a in arrays.values()):
        raise ValueError("Invalid episode depth shape")
    bounds=cfg["encoder"]["roi"]
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    summary={}
    indices=set(np.linspace(0,n-1,min(keyframes,n),dtype=int).tolist())
    for side,a in arrays.items():
        crop_roi(a[0],bounds[side])
        rows=[]
        for i,d in enumerate(a):
            if not np.isfinite(d).all():
                raise ValueError(f"Nonfinite {side} frame {i}")
            rows.append((i,float(d.min()),float(d.max()),float(d.mean()),float(d.std())))
        stats=np.asarray(rows)
        lo,hi=float(stats[:,1].min()),float(stats[:,2].max())
        summary[side]={"shape":list(a.shape),"dtype":str(a.dtype),"min":lo,"max":hi,
                       "mean":float(stats[:,3].mean()),"frame_mean_std":float(stats[:,3].std()),
                       "roi":bounds[side]}
        with (output/f"{side}_stats.csv").open("w",newline="") as f:
            w=csv.writer(f);w.writerow(["frame","min","max","mean","std"]);w.writerows(rows)
        writer=cv2.VideoWriter(str(output/f"{side}_{mode}.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),
                               float(meta["fps"])/stride,(1008,504))
        if not writer.isOpened():
            writer.release()
            raise RuntimeError("Cannot open MP4 writer")
        thumbs=[]
        try:
            for i in range(n):
                if i%stride!=0 and i not in indices:
                    continue
                d=a[i]
                low,high=(lo,hi) if mode=="fixed" else (float(d.min()),float(d.max()))
                im=panel(d,bounds[side],low,high,side,i)
                if i%stride==0:
                    writer.write(im)
                if i in indices:
                    thumbs.append(cv2.resize(im,(504,252)))
        finally:
            writer.release()
        while len(thumbs)%2:
            thumbs.append(np.zeros_like(thumbs[0]))
        montage=np.concatenate([np.concatenate(thumbs[i:i+2],axis=1) for i in range(0,len(thumbs),2)],axis=0)
        if not cv2.imwrite(str(output/f"{side}_keyframes.png"),montage):
            raise RuntimeError("Cannot save keyframe montage")
        # Plot temporal raw-depth statistics using OpenCV; no plotting dependency.
        chart=np.full((320,900,3),255,dtype=np.uint8)
        cv2.rectangle(chart,(65,35),(875,280),(210,210,210),1)
        for j,label,color in [(1,"min",(180,80,30)),(2,"max",(30,60,200)),(3,"mean",(40,140,40))]:
            xs=65+np.arange(n)*810/max(n-1,1)
            ys=280-(stats[:,j]-lo)*245/max(hi-lo,1e-8)
            pts=np.stack([xs,ys],axis=1).astype(np.int32)
            if n>1: cv2.polylines(chart,[pts],False,color,1)
            cv2.putText(chart,label,(70+(j-1)*110,24),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
        cv2.putText(chart,f"{lo:.3f}",(3,280),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1)
        cv2.putText(chart,f"{hi:.3f}",(3,40),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1)
        cv2.putText(chart,f"Frame 0 to {n-1}",(350,310),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
        if not cv2.imwrite(str(output/f"{side}_stats.png"),chart):
            raise RuntimeError("Cannot save stats plot")
    write_json(output/"summary.json",{"episode_id":episode,"fps":meta["fps"],"color_mode":mode,"sides":summary})
    print(f"Saved {output}")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="experiments/geometry_flow/config.yaml")
    p.add_argument("--episode",type=int,required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--color-mode",choices=["fixed","frame"],default="fixed")
    p.add_argument("--stride",type=int,default=1)
    p.add_argument("--keyframes",type=int,default=12)
    a=p.parse_args()
    inspect(load_config(a.config),a.episode,a.output,a.color_mode,a.stride,a.keyframes)
