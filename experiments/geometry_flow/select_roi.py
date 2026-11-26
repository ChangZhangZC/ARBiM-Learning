"""Export a standalone browser ROI selector from cached depth (no desktop GUI)."""
import argparse
import base64
import json
from pathlib import Path

import numpy as np

from .common import SIDES, load_config, read_json
from .depth_preprocess import crop_roi
from .inspect_depth import colorize


def build_payload(cfg, episode, stride=1, color_mode="frame"):
    import cv2
    if type(stride) is not int or stride < 1 or episode < 0:
        raise ValueError("Expected nonnegative episode and positive stride")
    if color_mode not in ("frame", "fixed"):
        raise ValueError("Invalid color mode")
    source = Path(cfg["data"]["depth_root"]).expanduser() / f"episode_{episode:06d}"
    meta = read_json(source / "meta.json")
    n = int(meta["frames"])
    if n < 1 or meta["episode_id"] != episode:
        raise ValueError("Invalid episode metadata")
    fps = float(meta["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid FPS")
    indices = list(range(0, n, stride))
    if indices[-1] != n - 1:
        indices.append(n - 1)
    payload = dict(episode=episode, fps=fps, indices=indices, sides={})
    for side in SIDES:
        depth = np.load(source / f"depth_{side}.npy", mmap_mode="r")
        if depth.ndim != 3 or len(depth) != n or min(depth.shape[1:]) < 1:
            raise ValueError(f"Invalid {side} depth shape")
        bounds = cfg["encoder"]["roi"][side]
        crop_roi(depth[0], bounds)
        h, w = depth.shape[1:]
        roi = None if bounds is None else [
            int(bounds[0]*w), int(bounds[1]*h),
            int(bounds[2]*w), int(bounds[3]*h)]
        if color_mode == "fixed":
            lo, hi = float(depth.min()), float(depth.max())
            if not np.isfinite([lo, hi]).all():
                raise ValueError(f"Nonfinite {side} depth")
        images = []
        for i in indices:
            d = depth[i]
            if color_mode == "frame":
                lo, hi = float(d.min()), float(d.max())
            im = colorize(d, lo, hi)
            ok, encoded = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if not ok:
                raise RuntimeError("Could not encode depth preview")
            images.append(base64.b64encode(encoded.tobytes()).decode("ascii"))
        payload["sides"][side] = dict(width=w, height=h, roi=roi,
                                       original_bounds=bounds, images=images)
    return payload


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ARBiM Fixed ROI Selector</title>
<style>
body{font:16px system-ui,sans-serif;background:#f4f5f7;color:#17212e;max-width:1180px;margin:24px auto;padding:0 16px}
h1{font-size:24px} .grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
section{background:white;padding:12px;border:1px solid #d5dbe2;border-radius:8px}
canvas{display:block;width:100%;height:auto;touch-action:none;cursor:crosshair}
input[type=range]{width:100%} button{padding:8px 12px;margin:6px 4px 6px 0;cursor:pointer}
textarea{box-sizing:border-box;width:100%;min-height:110px;font:15px monospace}
small{color:#485566} @media(max-width:680px){.grid{grid-template-columns:1fr}}
</style></head><body>
<h1>ARBiM Fixed ROI Selector</h1>
<p>Drag a rectangle on each wrist image. Move the frame slider to verify the same fixed ROI across the episode. The yellow rectangle is the retained area; coordinates use the original depth pixels.</p>
<p id="frameLabel"></p>
<button id="prev">Previous frame</button><button id="next">Next frame</button>
<input id="slider" type="range" min="0" step="1" value="0" aria-label="Frame">
<div class="grid">
<section><h2>Left wrist</h2><canvas id="left"></canvas><p id="leftInfo"></p><button id="leftReset">Full image / null</button></section>
<section><h2>Right wrist</h2><canvas id="right"></canvas><p id="rightInfo"></p><button id="rightReset">Full image / null</button></section>
</div>
<h2>YAML</h2><p>Copy only the ROI block into your existing encoder configuration. The tool never edits the configuration or the raw NPY.</p>
<textarea id="yaml" readonly></textarea>
<button id="copy">Copy YAML</button><button id="download">Save ROI YAML</button>
<small id="status"></small>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
"use strict";
const data=JSON.parse(document.getElementById("payload").textContent);
const sides=["left","right"], el=id=>document.getElementById(id);
const slider=el("slider"), state={};
let index=0;
slider.max=data.indices.length-1;
function normalized(v,n){
  // Match Python int(bound*n), including floating-point roundoff.
  if(v===0||v===n) return v/n;
  return (v+0.000000001)/n;
}
function yaml(){
  return "encoder:\n  roi:\n"+sides.map(s=>{
    const a=state[s], r=a.roi;
    const b=a.original!==undefined?a.original:(r?[normalized(r[0],a.w),normalized(r[1],a.h),normalized(r[2],a.w),normalized(r[3],a.h)]:null);
    return "    "+s+": "+(b?JSON.stringify(b):"null");
  }).join("\n")+"\n";
}
function refresh(){
  el("yaml").value=yaml();
  for(const s of sides){
    const a=state[s],r=a.roi;
    el(s+"Info").textContent=r?"Pixel bounds: ["+r.join(", ")+"] | "+(r[2]-r[0])+" x "+(r[3]-r[1])+" px":"Full image: "+a.w+" x "+a.h+" px";
  }
}
function draw(s){
  const a=state[s],ctx=a.canvas.getContext("2d"),r=a.preview||a.roi;
  ctx.clearRect(0,0,a.w,a.h);
  if(a.img) ctx.drawImage(a.img,0,0,a.w,a.h);
  if(!r)return;
  const [x,y,X,Y]=r;
  ctx.fillStyle="rgba(0,0,0,0.48)";
  ctx.fillRect(0,0,a.w,y);ctx.fillRect(0,Y,a.w,a.h-Y);
  ctx.fillRect(0,y,x,Y-y);ctx.fillRect(X,y,a.w-X,Y-y);
  ctx.strokeStyle="#ffdc36";ctx.lineWidth=2;ctx.strokeRect(x,y,X-x,Y-y);
}
function showFrame(i){
  index=Math.max(0,Math.min(data.indices.length-1,i));
  slider.value=index;
  el("frameLabel").textContent="Episode "+data.episode+" | Frame "+data.indices[index]+" | "+(data.indices[index]/data.fps).toFixed(2)+" s";
  for(const s of sides){
    const a=state[s],im=new Image(),requested=index;
    im.onload=()=>{if(index===requested){a.img=im;draw(s);}};
    im.src="data:image/jpeg;base64,"+data.sides[s].images[index];
  }
}
function point(e,a){
  const r=a.canvas.getBoundingClientRect();
  return [Math.max(0,Math.min(a.w,Math.round((e.clientX-r.left)*a.w/r.width))),
          Math.max(0,Math.min(a.h,Math.round((e.clientY-r.top)*a.h/r.height)))];
}
function rectangle(p,q){return [Math.min(p[0],q[0]),Math.min(p[1],q[1]),Math.max(p[0],q[0]),Math.max(p[1],q[1])];}
for(const s of sides){
  const d=data.sides[s],canvas=el(s);
  const a=state[s]={canvas,w:d.width,h:d.height,roi:d.roi,original:d.original_bounds,img:null,preview:null,start:null,pointer:null};
  canvas.width=a.w;canvas.height=a.h;
  canvas.addEventListener("pointerdown",e=>{
    if(a.pointer!==null||e.button!==0)return;
    a.pointer=e.pointerId;a.start=point(e,a);a.preview=null;canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove",e=>{
    if(e.pointerId!==a.pointer)return;
    a.preview=rectangle(a.start,point(e,a));draw(s);
  });
  canvas.addEventListener("pointerup",e=>{
    if(e.pointerId!==a.pointer)return;
    const r=rectangle(a.start,point(e,a));
    if(r[2]>r[0]&&r[3]>r[1]){a.roi=r;a.original=undefined;}
    a.start=null;a.pointer=null;a.preview=null;draw(s);refresh();
  });
  canvas.addEventListener("pointercancel",e=>{
    if(e.pointerId!==a.pointer)return;
    a.start=null;a.pointer=null;a.preview=null;draw(s);
  });
  el(s+"Reset").onclick=()=>{a.roi=null;a.original=null;a.preview=null;draw(s);refresh();};
}
slider.oninput=()=>showFrame(Number(slider.value));
el("prev").onclick=()=>showFrame(index-1);
el("next").onclick=()=>showFrame(index+1);
el("copy").onclick=async()=>{
  el("yaml").focus();el("yaml").select();
  try{await navigator.clipboard.writeText(el("yaml").value);el("status").textContent="Copied.";}
  catch(e){el("status").textContent="Text selected. Press Ctrl+C or Cmd+C to copy.";}
};
el("download").onclick=()=>{
  const url=URL.createObjectURL(new Blob([yaml()],{type:"text/yaml"}));
  const link=document.createElement("a");link.href=url;link.download="roi.yaml";link.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
};
refresh();showFrame(0);
</script></body></html>"""


def export_roi(cfg, episode, output, stride=1, color_mode="frame"):
    output = Path(output).expanduser()
    if output.suffix.lower() != ".html":
        raise ValueError("Output must be an HTML file")
    if output.exists():
        raise FileExistsError(output)
    payload = build_payload(cfg, episode, stride, color_mode)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        f.write(HTML.replace("__PAYLOAD__", json.dumps(payload, allow_nan=False, separators=(",", ":"))))
    print(f"Saved {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/geometry_flow/config.yaml")
    parser.add_argument("--episode", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stride", default=1, type=int,
                        help="Preview every N frames; does not affect training")
    parser.add_argument("--color-mode", choices=["frame", "fixed"], default="frame")
    args = parser.parse_args()
    export_roi(load_config(args.config), args.episode, args.output, args.stride, args.color_mode)
