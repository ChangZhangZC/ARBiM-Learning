"""Chunk-aligned display panel. Visualization does not enter the Flow."""
import numpy as np
from .common import SIDES
from .depth_preprocess import crop_roi
from .inspect_depth import colorize


def _fit(image, width, height):
    import cv2
    h,w=image.shape[:2]
    scale=min(width/w,height/h)
    w2,h2=max(1,round(w*scale)),max(1,round(h*scale))
    image=cv2.resize(image,(w2,h2),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
    out=np.full((height,width,3),28,dtype=np.uint8)
    x,y=(width-w2)//2,(height-h2)//2
    out[y:y+h2,x:x+w2]=image
    return out


def _label(canvas,text,x,y,color=(230,230,230)):
    import cv2
    cv2.putText(canvas,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,.43,color,1,cv2.LINE_AA)


def _history_plot(canvas,history,side,current_step,x,y,w,h):
    import cv2
    values=[(step,score) for step,score in history[side] if score is not None]
    cv2.rectangle(canvas,(x,y),(x+w-1,y+h-1),(70,70,70),1)
    if not values:
        _label(canvas,"Waiting for history",x+12,y+30)
        return
    lo=min(v for _,v in values)
    hi=max(v for _,v in values)
    pad=max((hi-lo)*.08,1.)
    lo,hi=lo-pad,hi+pad
    first=history[side][0][0]
    points=np.asarray([(x+48+(step-first)*(w-62)/max(current_step-first,1),
                        y+h-20-(score-lo)*(h-28)/(hi-lo))
                       for step,score in values],dtype=np.int32)
    color=(80,200,255) if side=="left" else (255,185,85)
    if len(points)>1:
        cv2.polylines(canvas,[points],False,color,2,cv2.LINE_AA)
    cv2.circle(canvas,tuple(points[-1]),3,color,-1)
    _label(canvas,f"{hi:.0f}",x+3,y+13)
    _label(canvas,f"{lo:.0f}",x+3,y+h-20)
    _label(canvas,f"{first}",x+45,y+h-4)
    _label(canvas,f"{current_step}",x+w-48,y+h-4)


def render_panel(sample,roi,history):
    """Return RGB uint8 panel; all imagery is from the exact monitor sample."""
    import cv2
    canvas=np.full((346,1040,3),28,dtype=np.uint8)
    for index,side in enumerate(SIDES):
        y=index*173
        depth=sample["depth"][side]
        lo,hi=float(depth.min()),float(depth.max())
        full=colorize(depth,lo,hi)
        crop=colorize(crop_roi(depth,roi[side]),lo,hi)
        if roi[side] is not None:
            h,w=depth.shape
            a,b,c,d=roi[side]
            cv2.rectangle(full,(int(a*w),int(b*h)),(int(c*w)-1,int(d*h)-1),(255,255,255),2)
        rgb=cv2.cvtColor(sample["rgb"][side],cv2.COLOR_RGB2BGR)
        for x,im,title in [(8,rgb,f"{side.upper()} RGB"),(164,full,"DA3 + ROI"),(320,crop,"Cropped ROI")]:
            _label(canvas,title,x,y+16)
            canvas[y+22:y+164,x:x+148]=_fit(im,148,142)
        score=sample["score"][f"{side}_nll"]
        label="N/A (warmup)" if score is None else f"{score:.2f}"
        _label(canvas,f"{side.upper()} NLL: {label} | independent plot scale",480,y+16)
        _history_plot(canvas,history,side,sample["step"],480,y+22,550,142)
    return cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB)


def sample_text(sample,episode,chunk,stage,monitor_ms):
    def score(side):
        x=sample["score"][f"{side}_nll"]
        return "N/A" if x is None else f"{x:.2f}"
    q=sample["joint_state"]
    gripper=f"{q[7]:.3f} / {q[15]:.3f}" if len(q)==16 else "N/A"
    return (
        "Geometry Monitor (chunk sample)\nEpisode / Step / Chunk\nTask stage (sampled)\n"
        "Left / Right NLL\nHistory\nLeft / Right gripper\nMonitor time\nDepth display",
        f"{'ACTIVE' if sample['score']['ready'] else 'WARMING UP'}\n"
        f"{episode} / {sample['step']} / {chunk}\n{stage}\n"
        f"{score('left')} / {score('right')}\n"
        f"{sample['history_size']} / {sample['history_required']}\n{gripper}\n"
        f"{monitor_ms:.1f} ms\nDA3 relative; per-frame colors"
    )
