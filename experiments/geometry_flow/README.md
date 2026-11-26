# Geometry Flow V1

Experimental, independent of the policy. No recovery actions. No DA3 scale
alignment, per-frame normalization, SAM2 or task-stage prediction.

## Setup and smoke workflow

From repository root, in the existing ARBiM environment:

```bash
pip install -r experiments/geometry_flow/requirements.txt
python -m pytest tests/test_geometry_flow_data.py tests/test_geometry_flow_model.py tests/test_geometry_flow_history.py
```

Set data.dataset_root and 5–10 known successful episode IDs in config.yaml.
Both depth generation and feature extraction support a list of IDs, a quoted
half-open range (e.g. "0:100"), or all. All means every episode in the source
dataset, not automatic success filtering; the caller must verify success.
Use --skip-existing on generate_depth to reuse matching structurally complete
cached episodes. Default behavior rejects existing outputs; no overwrites.
Check storage: 2 * frames * H * W * 4 bytes for raw depth. Generate all frames:

```bash
python -m experiments.geometry_flow.generate_depth
```

Review full training episodes to select left/right normalized ROI bounds
[x1,y1,x2,y2] (null means full image). Do not use held-out episodes to tune ROI.
Inspect a cached training episode (requires OpenCV, already used by DA3 video eval):
```bash
python -m experiments.geometry_flow.inspect_depth --episode 20 --output outputs/geometry_flow/inspect_20
```
Use --config for your local YAML and --color-mode frame for per-frame visualization.
Default fixed color limits are computed separately for each side over the entire
episode. Full view with ROI outline and cropped view appear side by side;
MP4, keyframe montage, raw-depth stats CSV/PNG and summary JSON are saved.
Visualization never changes NPY or training transforms.
For local desktop ROI selection (no browser):
```bash
python -m experiments.geometry_flow.select_roi --config experiments/geometry_flow/config.yaml --episode 20 --frame 200
```
Run in a terminal opened on the graphical Ubuntu desktop, not a plain SSH
session. GUI-enabled OpenCV and an accessible display are required; a
subprocess preflight catches Qt initialization errors before the main GUI.
Drag with the mouse; 1/2 switches left/right, frame slider or A/D browses
frames, R resets current side to full image, S prints both ROI YAML entries
and exits, Q/Esc cancels. Optional --output path.yaml writes a new YAML
snippet; copy only the ROI values into the existing config. Existing ROI
bounds are loaded from config; they persist across frames. No NPY or config
is changed. --color-mode fixed is optional; default frame mode is for
visualization only. Legacy HTML export remains available only when
--output ends in .html.
Use a fresh feature_root for changed ROI. Raw NPY needs no regeneration.

```bash
python -m experiments.geometry_flow.extract_features
python -m experiments.geometry_flow.train
python -m experiments.geometry_flow.eval --model-dir outputs/geometry_flow/models --output outputs/geometry_flow/rollout_smoke --episodes 2
```

The eval entry point reuses the existing DA3 rollout hook but obtains
dataset-aligned RGB via the existing DuoBench observation conversion rather than native raw RGB.
The monitor explicitly disables observation gripper binarization to match the
continuous expert states observed in Hinge-Chest episode 20. Policy adapter
settings and native action handling are unchanged. Verify raw environment
observations still expose continuous gripper values during runtime testing.
It records NLL and timing and does not change policy actions. Optional synchronized recording is described below.
Check actual checkpoint and rollout config paths before running.

## Optional live MuJoCo monitor

Run with the existing rollout config (env.headless: false) and a new output directory:

```bash
python -m experiments.geometry_flow.eval --model-dir outputs/geometry_flow/models --output outputs/geometry_flow/rollout_live --episodes 2 --live-view
```

This overlays the latest chunk's episode/step/stage, left/right NLL, warmup,
gripper values and monitor timing, plus wrist RGB, full DA3 depth with ROI,
cropped depth and separate NLL plots in the native MuJoCo window.
Depth colors are per-frame visualization only; both plots have independent
vertical scales. No threshold or recovery action. These are chunk samples,
not per-environment-step scores. No extra DA3 inference.

Requires an installed MuJoCo viewer exposing set_texts, set_images and viewport,
and RCS using the verified v0.7.4-style gui_loop spawn target. The opt-in
adapter wraps the native launch in the GUI child, transfers only the latest
display packet via an atomic temporary file, and restores the parent launch
target after env.close(). No installed RCS or MuJoCo files are modified.
If incompatible, run without --live-view and check the installed versions;
do not upgrade blindly. Actual graphical runtime remains to be validated.
## Optional synchronized rollout recording

```bash
python -m experiments.geometry_flow.eval --model-dir outputs/geometry_flow/models --output outputs/geometry_flow/rollout_recorded --episodes 2 --live-view --record-video
```

Use --record-video independently of --live-view (including headless mode).
On Linux, the recording entry point automatically sets MUJOCO_GL=egl and
PYOPENGL_PLATFORM=egl before importing simulation/model modules. This avoids
the GLFW/GLX context creation conflict observed with RCS. The setting is
local to the evaluation process and its children; no shell exports or
installation changes are needed. Non-recording runs and non-Linux platforms
retain their existing backend configuration. If an incompatible GL backend
was already imported, start a fresh Python process.
Each episode subdirectory contains synchronized.mp4 (fixed MuJoCo free
camera, current head RGB, sampled wrist RGB/depth/ROI and NLL plots),
head_rgb.mp4, left_rgb.mp4, right_rgb.mp4, left_depth.mp4, right_depth.mp4,
left_depth.npy, right_depth.npy, joint_state.npy, env_steps.npy, frame_index.csv,
recording.json and final NLL plot PNGs when monitor samples exist.
The top-level episode CSV and episode summary remain unchanged.

The synchronized video is a deterministic step-indexed composition, **not
a screen capture of the interactive GUI**; moving the GUI camera does not
affect the fixed recording camera. It uses the existing simulation model and
data for offscreen rendering without stepping it. Native camera RGB and the
scene are captured at every environment observation (including reset and
terminal frame). Videos use environment FPS; monitor images and scores are
held until the next chunk and explicitly labeled with their sample step.
Raw depth/joints are saved once per chunk; env_steps.npy indexes them.
frame_index.csv maps every video frame to both environment and monitor steps.
No duplicate DA3 inference. MP4/GL support and actual rendering still need
local runtime verification. Recording adds render/encoding/I/O overhead
outside monitor_ms; compare performance with recording disabled.
Use a fresh output path. No recordings are created without --record-video.

For threshold calibration, collect a separate set of successful policy
rollouts with sufficient complete history, then run:

```bash
python -m experiments.geometry_flow.calibrate --rollout-dir outputs/geometry_flow/calibration --output outputs/geometry_flow/thresholds.json
```

Thresholds are empirical per-arm quantiles of episode maximum NLL; not
failure probabilities or conformal guarantees. Evaluate on independent
success/failure episodes with manually annotated local failure events.
Calibration is offline; no threshold-triggered intervention.

## Temporal contract

Default T=8, stride=16 environment steps at 30 Hz, 3.73 s history.
Training anchors slide one expert frame at a time; each window contains
T samples spaced by the deployment interval. Exact FPS conversion required.
No windows cross episode boundaries. Warm-up produces null scores until T
chunk observations exist. Changing T or sampling interval requires retraining
Flow only; features and raw depth remain reusable. Variable intervals rejected.
Both flows receive identical full bimanual state history, while modeled
geometry, PCA, weights, optimizer and score are side-specific.

Raw depths are float32; fixed ROI resize to 224x224, replicate channels,
fixed ImageNet mean/std (no per-frame depth rescaling). Frozen ImageNet
ResNet18 pool2x2, train-only PCA and feature/state standardizers.
Validation episodes excluded from PCA. This is a feature density, not depth
pixel density. Explicitly check raw numerical range and domain gap.

All output directories are non-overwriting. Use new paths for reruns.
Check actual expert joint ordering, camera mapping, state gripper convention,
FPS, raw depth range and runtime compatibility in integration testing.
The initial 5–10 episodes are for smoke testing, not performance claims.
