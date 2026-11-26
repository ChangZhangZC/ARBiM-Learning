"""Opt-in RCS viewer overlay using its native spawn loop, without modifying RCS."""
from contextlib import contextmanager
from functools import partial
import inspect
import os
from pathlib import Path
import tempfile
import warnings
import numpy as np

from .common import SIDES


def write_packet(path,image,text):
    """Single writer, atomic replacement; the viewer never sees partial data."""
    path=Path(path)
    tmp=path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez(f,image=image,text=np.asarray(text,dtype=str))
    os.replace(tmp,path)


class OverlayViewer:
    """Proxy used only in the RCS GUI child process."""
    def __init__(self,viewer,path):
        self.viewer=viewer
        self.path=Path(path)
        self.token=None
        self.packet=None
        self.viewport_key=None
        self.disabled=False

    def __getattr__(self,name):
        return getattr(self.viewer,name)

    def sync(self,*args,**kwargs):
        if not self.disabled:
            try:
                self._refresh()
            except Exception as exc:
                self.disabled=True
                warnings.warn(f"Geometry overlay disabled in GUI: {exc}",RuntimeWarning)
        return self.viewer.sync(*args,**kwargs)

    def _refresh(self):
        import mujoco as mj
        import cv2
        try:
            st=self.path.stat()
        except FileNotFoundError:
            return
        token=(st.st_ino,st.st_mtime_ns,st.st_size)
        changed=token!=self.token
        if changed:
            with np.load(self.path,allow_pickle=False) as f:
                self.packet=(f["image"],tuple(f["text"].tolist()))
            self.token=token
        viewport=self.viewer.viewport
        if viewport is None or viewport.width<100 or viewport.height<100:
            return
        key=(viewport.left,viewport.bottom,viewport.width,viewport.height)
        if not changed and key==self.viewport_key:
            return
        image,text=self.packet
        h,w=image.shape[:2]
        scale=min(.96*viewport.width/w,.40*viewport.height/h,1.0)
        width,height=max(1,round(w*scale)),max(1,round(h*scale))
        image=cv2.resize(image,(width,height),interpolation=cv2.INTER_AREA)
        rect=mj.MjrRect(viewport.left+(viewport.width-width)//2,
                        viewport.bottom+4,width,height)
        self.viewer.set_texts((mj.mjtFontScale.mjFONTSCALE_100,
                               mj.mjtGridPos.mjGRID_TOPLEFT,*text))
        self.viewer.set_images((rect,np.ascontiguousarray(image)))
        self.viewport_key=key


def run_overlay_gui(gui_uuid,close_event,*,packet_path):
    """In the spawned process, wrap launch_passive while retaining native gui_loop."""
    import rcs.sim.sim as rcs_sim
    import mujoco.viewer as mjviewer
    original_launch=mjviewer.launch_passive

    @contextmanager
    def launch(*args,**kwargs):
        with original_launch(*args,**kwargs) as handle:
            yield OverlayViewer(handle,packet_path)

    mjviewer.launch_passive=launch
    try:
        rcs_sim.gui_loop(gui_uuid,close_event)
    finally:
        mjviewer.launch_passive=original_launch


class LiveOverlay:
    def __init__(self,path):
        self.path=path
        self.history={s:[] for s in SIDES}

    def reset(self):
        self.history={s:[] for s in SIDES}

    def publish(self,sample,roi,episode,chunk,stage,monitor_ms):
        from .visualization import render_panel,sample_text
        for s in SIDES:
            self.history[s].append((sample["step"],sample["score"][f"{s}_nll"]))
        image=render_panel(sample,roi,self.history)
        text=sample_text(sample,episode,chunk,stage,monitor_ms)
        write_packet(self.path,image,text)


@contextmanager
def live_overlay(enabled,env_config):
    if not enabled:
        yield None
        return
    if env_config.get("headless",True):
        raise RuntimeError("--live-view requires env.headless: false")
    import mujoco.viewer as mjviewer
    import rcs.sim.sim as rcs_sim
    required=("set_texts","set_images","viewport")
    missing=[name for name in required if not hasattr(mjviewer.Handle,name)]
    if missing:
        raise RuntimeError("Installed MuJoCo viewer lacks: "+", ".join(missing)
                           +". Run without --live-view; do not upgrade blindly.")
    original=rcs_sim.gui_loop
    # Only use the verified RCS module-global spawn target pattern.
    try:
        source=inspect.getsource(rcs_sim.Sim.open_gui)
    except (OSError,TypeError) as exc:
        raise RuntimeError("Cannot verify installed RCS viewer launch contract") from exc
    if "target=gui_loop" not in "".join(source.split()) or tuple(inspect.signature(original).parameters)!=("gui_uuid","close_event"):
        raise RuntimeError("Unsupported installed RCS gui_loop contract; run without --live-view")
    with tempfile.TemporaryDirectory(prefix="arbim-geometry-view-") as directory:
        path=Path(directory)/"latest.npz"
        rcs_sim.gui_loop=partial(run_overlay_gui,packet_path=str(path))
        try:
            yield LiveOverlay(path)
        finally:
            rcs_sim.gui_loop=original
