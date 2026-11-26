# Geometry Flow V1

Independent experiment, no changes to IL/PPO or original DA3 experiment.
Expert depth is generated at every frame, single-frame DA3MONO, float32 NPY,
without scale alignment, smoothing, or frame-wise normalization.
Select 5–10 confirmed successful demonstrations in config.yaml for smoke testing.
Run from repository root:

```bash
python -m experiments.geometry_flow.generate_depth --config experiments/geometry_flow/config.yaml
```

Each episode is written atomically and contains full-rate left/right depth,
state, frame/global index, timestamps, and metadata. Existing output is never
overwritten. Check free space first (two float32 depth maps per frame).
The raw depth files can be reused for any later ROI or temporal sampling.
ROI is intentionally null/full-frame until full training episodes are reviewed.

Target configuration: 30 Hz, execute_steps=16, T=8 (3.73 s), stride=16,
window_stride=1; T configurable. Inference at chunk start only.
Use identical RGB resolution and state convention for expert and rollout.
Feature and monitoring implementations will be integrated in separate tasks.
