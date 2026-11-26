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
Use a fresh feature_root for changed ROI. Raw NPY needs no regeneration.

```bash
python -m experiments.geometry_flow.extract_features
python -m experiments.geometry_flow.train
python -m experiments.geometry_flow.eval --model-dir outputs/geometry_flow/models --output outputs/geometry_flow/rollout_smoke --episodes 2
```

The eval entry point reuses the existing DA3 rollout hook but obtains
dataset-aligned RGB and state from DuoBenchAdapter rather than native raw RGB.
It records NLL and timing, does not change policy actions, and does not record
video in this V1 entry point. Use existing DA3 eval separately for video.
Check actual checkpoint and rollout config paths before running.

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
