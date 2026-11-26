"""Record exact environment-step frames with held, chunk-aligned monitor samples."""
import csv
from pathlib import Path
import numpy as np

from .common import SIDES, write_json
from .inspect_depth import panel as depth_panel
from .visualization import render_panel, _fit, _label


class SceneRenderer:
    """Independent fixed MuJoCo free-camera render, not desktop viewer capture."""
    def __init__(self, env):
        import mujoco as mj
        self.sim = env.get_wrapper_attr("sim")
        model = self.sim.model
        self.camera = mj.MjvCamera()
        mj.mjv_defaultFreeCamera(model, self.camera)
        width = min(960, model.vis.global_.offwidth)
        height = min(540, model.vis.global_.offheight)
        self.renderer = mj.Renderer(model, height=height, width=width)

    def __call__(self):
        self.renderer.update_scene(self.sim.data, camera=self.camera)
        return self.renderer.render().copy()

    def close(self):
        self.renderer.close()


class EpisodeRecording:
    """Stream videos at environment FPS; keep raw depth only at monitor cadence."""
    def __init__(self, folder, fps, roi, render_scene):
        import cv2
        self.cv2 = cv2
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        self.fps = float(fps)
        self.roi = roi
        self.render_scene = render_scene
        self.writers = {}
        self.history = {s: [] for s in SIDES}
        self.depth = {s: [] for s in SIDES}
        self.joints, self.steps = [], []
        self.pending = None
        self.sample = None
        self.panel = None
        self.depth_images = None
        self.frame_count = 0
        self.closed = False
        self.index_file = (self.folder / "frame_index.csv").open("w", newline="")
        self.index = csv.writer(self.index_file)
        self.index.writerow(["video_frame", "env_step", "stage", "monitor_step", "monitor_chunk", "ready"])

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if kind is not None:
            self.close()
            return False
        try:
            self.finish()
        finally:
            self.close()

    def _write(self, name, bgr):
        cv2 = self.cv2
        h, w = bgr.shape[:2]
        if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError(f"Invalid video frame: {name}")
        # MP4 encoders may truncate odd sizes; pad rather than crop.
        if h % 2 or w % 2:
            bgr = cv2.copyMakeBorder(bgr, 0, h % 2, 0, w % 2, cv2.BORDER_CONSTANT)
            h, w = bgr.shape[:2]
        if name not in self.writers:
            writer = cv2.VideoWriter(str(self.folder / (name + ".mp4")),
                                     cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            if not writer.isOpened():
                writer.release()
                raise RuntimeError(f"Cannot open MP4 writer: {name}")
            self.writers[name] = (writer, (h, w))
        writer, shape = self.writers[name]
        if shape != (h, w):
            raise ValueError(f"Video size changed: {name}")
        writer.write(np.ascontiguousarray(bgr))

    def on_frame(self, raw, step, stage):
        # on_frame precedes on_chunk at the same step; delay writing exactly
        # one frame so step K uses the new sample from K, never K-16.
        self._flush()
        if step != self.frame_count:
            raise ValueError(f"Expected consecutive frame {self.frame_count}, got {step}")
        cameras = {}
        for name, source in (("head_rgb","head"),("left_rgb","left_wrist"),("right_rgb","right_wrist")):
            rgb = np.asarray(raw["frames"][source]["rgb"]["data"])
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError(f"Invalid native RGB: {source}")
            cameras[name] = self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR)
        scene = self.render_scene()
        self.pending = (step, stage, cameras, self.cv2.cvtColor(scene, self.cv2.COLOR_RGB2BGR))

    def on_chunk(self, sample, chunk, stage, monitor_ms):
        if self.pending is None or sample["step"] != self.pending[0] or chunk != len(self.steps):
            raise ValueError("Monitor sample must match the pending frame and chunk index")
        self.steps.append(sample["step"])
        self.joints.append(sample["joint_state"].copy())
        self.sample = (sample, chunk, stage, monitor_ms)
        self.depth_images = {}
        for s in SIDES:
            d = sample["depth"][s]
            self.depth[s].append(d.copy())
            self.history[s].append((sample["step"], sample["score"][s+"_nll"]))
            self.depth_images[s] = depth_panel(d, self.roi[s], float(d.min()), float(d.max()),
                                               s, sample["step"])
        self.panel = self.cv2.cvtColor(render_panel(sample, self.roi, self.history),
                                       self.cv2.COLOR_RGB2BGR)

    def _flush(self):
        if self.pending is None:
            return
        step, stage, cameras, scene = self.pending
        for name, frame in cameras.items():
            self._write(name, frame)
        canvas = np.full((862,1040,3),28,dtype=np.uint8)
        _label(canvas, f"Episode {self.folder.name} | Env step {step} | Stage {stage} | t={step/self.fps:.3f}s", 12,19)
        _label(canvas, "MuJoCo fixed free camera (not desktop capture)", 12,39)
        _label(canvas, "Head RGB (current env step)", 700,39)
        canvas[45:505,8:688] = _fit(scene,680,460)
        canvas[45:505,696:1032] = _fit(cameras["head_rgb"],336,460)
        if self.sample is None:
            # Possible only for episodes already successful at reset.
            _label(canvas,"No monitor sample",12,535)
            monitor_step = chunk = ready = ""
            for s in SIDES:
                blank = np.zeros((504,1008,3),dtype=np.uint8)
                _label(blank,"No monitor sample",8,24)
                self._write(s+"_depth",blank)
        else:
            sample,chunk,sampled_stage,ms = self.sample
            monitor_step=sample["step"]
            ready=sample["score"]["ready"]
            if monitor_step>step:
                raise ValueError("Future monitor sample in video")
            canvas[516:862] = self.panel
            # Keep the monitor timestamp visible above the panel without overlap.
            _label(canvas,f"Monitor step {monitor_step} | chunk {chunk} | stage {sampled_stage} | age {step-monitor_step} | history {sample['history_size']}/{sample['history_required']} | {ms:.1f} ms",12,513)
            for s in SIDES:
                self._write(s+"_depth", self.depth_images[s])
        self._write("synchronized",canvas)
        self.index.writerow([self.frame_count,step,stage,monitor_step,chunk,ready])
        self.frame_count+=1
        self.pending=None

    def finish(self):
        self._flush()
        np.save(self.folder / "env_steps.npy",np.asarray(self.steps,dtype=np.int64))
        np.save(self.folder / "joint_state.npy",np.stack(self.joints) if self.joints else np.empty((0,16),dtype=np.float32))
        for i,s in enumerate(SIDES):
            np.save(self.folder / (s+"_depth.npy"),np.stack(self.depth[s]) if self.depth[s] else np.empty((0,0,0),dtype=np.float32))
            if self.panel is not None:
                # The final history panel is a snapshot of all observed scores.
                plot=self.panel[i*173:(i+1)*173,478:1040]
                if not self.cv2.imwrite(str(self.folder / (s+"_nll.png")),plot):
                    raise RuntimeError("Cannot save NLL plot")
        write_json(self.folder / "recording.json", {
            "fps":self.fps,"video_frames":self.frame_count,"monitor_samples":len(self.steps),
            "video_step_mapping":"frame_index.csv","raw_sample_step_mapping":"env_steps.npy",
            "scene":"fixed MuJoCo free camera, not native desktop window capture",
            "depth_color":"per monitor sample, visualization only",
            "monitor_values":"held until next chunk; no future samples",
            "complete":True,
        })

    def close(self):
        if self.closed:
            return
        for writer,_ in self.writers.values():
            writer.release()
        self.writers.clear()
        self.index_file.close()
        self.closed=True
