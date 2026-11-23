# DA3 research experiment (placeholder)

This directory is reserved for early DA3 perception and failure-detection experiments. It is **not** part of the stable `arbim` Python package.

- `inference.py`: planned thin adapter around the pinned upstream DA3 Python API.
- `eval.py`: planned experimental entrypoint reusing the existing `arbim` policy, environment adapter, and rollout logic.
- `configs/experiments/da3.yaml`: separate experiment configuration placeholder.

No inference, evaluation, or detection logic is implemented yet. The existing IL/RL training, inference, and evaluation code remains unchanged. Once validated, reusable components may be promoted to `src/arbim/perception/` or `src/arbim/detection/` in a separate reviewed change.
