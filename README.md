# ARBiM-Learning

**ARBiM** (Asymmetric Reinforcement for Bimanual Manipulation) integrates native
**RL-100 DP-2D** with LeRobot datasets and DuoBench simulation. The current
baseline covers **IL training → checkpoint inference → Hinge-Chest rollout →
multi-episode evaluation and RGB video recording**. RL post-training and routed
rollout are not implemented.

## Setup

Use a prepared Python 3.12 environment with the native RL-100, LeRobot,
DuoBench/RCS, and GPU dependencies installed. `environment.yml` is a minimal
starting point, **not** a complete simulator installation.

```bash
git submodule update --init third_party/rl100 third_party/lerobot third_party/duobench
python -m pip install -e .
export PYTHONPATH="$PWD/src:$PWD/third_party/rl100/RL-100:$PWD/third_party/lerobot/src${PYTHONPATH:+:$PYTHONPATH}"
```

Do not use recursive submodule initialization unless separately required:
upstream RL-100 contains an unrelated nested submodule with an incomplete URL.

## Run

Run commands from the repository root. Edit the corresponding YAML to match
your dataset, checkpoint, GPU, and output directories.

**1. IL training** — `configs/il/rl100_dp2d.yaml`
(set `dataset.root`, `training.task`, and training limits):

```bash
python -m arbim.training.train_rl100_dp2d --config configs/il/rl100_dp2d.yaml
```

Training outputs use `outputs/il/<task>/<method>/<run_name>/`.
In `configs/il/rl100_dp2d.yaml`, the default `training.run_name: run_{timestep}`
expands the launch time (e.g. `run_20261117_130000`).
Use `my_experiment_{timestep}` for a custom label with a time suffix,
or a fixed name without `{timestep}`.

Resume an existing run using its `last.ckpt`:

```bash
python -m arbim.training.train_rl100_dp2d --resume outputs/il/YOUR_TASK/rl100_dp2d/YOUR_RUN
```

**2. Checkpoint inference smoke test** — `configs/inference/rl100_dp2d.yaml`.
Requires a checkpoint and access to its validation dataset:

```bash
python -m arbim.policy.inference_rl100_dp2d \
  --config configs/inference/rl100_dp2d.yaml \
  --checkpoint /path/to/best.ckpt
```

**3. DuoBench Hinge-Chest rollout** — `configs/rollout/hinge_chest.yaml`.
Set `policy.checkpoint_path` to the trained checkpoint first:

```bash
# Interactive launcher (normal mode; routed mode is not implemented)
python -m arbim.rollout.launch

# One closed-loop episode
python -m arbim.rollout.single --config configs/rollout/hinge_chest.yaml --max-env-steps 300

# Multi-episode evaluation + three-camera MP4 recording
python -m arbim.rollout.eval --config configs/rollout/hinge_chest.yaml --episodes 10
```

`hinge_chest.yaml` controls GUI/realtime, max steps, seeds, camera resolution,
and `output.video_source` (`native` for HD or `policy` for 224×224).
Evaluation outputs go to `outputs/eval/hinge_chest/<run_id>/` and include
`config_resolved.yaml`, `episodes.jsonl`, `summary.json`, and
`episode_*/{head,left_wrist,right_wrist}.mp4`.
The default `output.run_id: eval_{timestep}` can be renamed to, e.g.,
`test_policy_{timestep}`. Eval timestamps include microseconds.

For local TensorBoard, edit `path=` at the top of `scripts/tensorboard.sh`
and launch from the repository root:

```bash
bash scripts/tensorboard.sh
```

## Implementation

- `src/arbim/data/` — LeRobot → RL-100 dataset adapter.
- `src/arbim/policy/` — native model construction and checkpoint inference.
- `src/arbim/training/` — IL trainer, checkpoints, and logging.
- `src/arbim/envs/` and `src/arbim/rollout/` — DuoBench adapter and evaluation.
- `third_party/` — pinned RL-100, LeRobot, and DuoBench submodules.

Native RL-100 Actor, Normalizer, diffusion, and loss remain unchanged.
More training and Resume details: [IL training guide](docs/rl100_dp2d_training.md).

Development flow: `gpt/**` → `feature/**` → `dev` (Squash Merge after validation).
