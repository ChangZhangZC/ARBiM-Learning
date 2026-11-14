# ARBiM-Learning

Minimal, environment-independent learning integration for **ARBiM** (Asymmetric Reinforcement for Bimanual Manipulation).

Current scope: the native RL-100 DP-2D **IL training path** runs on LeRobot v3 data. Dataset adaption, native GPU single-batch smoke, and train-loop orchestration are in Feature-stage validation. RL post-training, simulation rollout and robot deployment are **not** implemented here.

## Repository layout

- `third_party/rl100/`: upstream RL-100 as a pinned Git submodule.
- `third_party/lerobot/`: upstream LeRobot as a pinned Git submodule.
- `src/arbim/policy/`: native RL1002D construction and weight-loading adapter.
- `src/arbim/data/`: LeRobot v3-to-RL1002D data contract and camera alias mapping.
- `src/arbim/training/`: native IL training, resume, checkpoint and telemetry.
- `src/arbim/runtime.py`: inference interface (planned).
- `src/arbim/integrations/`: external environment adapters (planned; DuoBench is deferred).
- `configs/il/` and `tests/`: RL100 DP-2D IL profiles and regression tests.
- `outputs/`: ignored local training artifacts.


## Native RL-100 DP-2D IL

See [IL training, exact-step Resume and experiment output guide](docs/rl100_dp2d_training.md).
The reusable `configs/il/rl100_dp2d.yaml` profile exposes independent Steps/Epochs
limits, step/epoch checkpoint/validation cadence, strict Resume, TensorBoard
and optional W&B. Update its LeRobot dataset path and `training.task` for a
new dataset; `policy.horizon/n_obs_steps/n_action_steps` are the native timing
settings and `training.batch_size` is the DataLoader batch size.

```bash
python -m arbim.training.train_rl100_dp2d --config configs/il/rl100_dp2d.yaml
# Resume the exact same training run:
python -m arbim.training.train_rl100_dp2d --resume outputs/il/<task>/<method>/<run>
```

RL-100's model, normalizer, `compute_loss()` and diffusion stack are left
unmodified. A full native GPU train + resume regression is required before Dev
integration.


## Pinned upstream versions

- RL-100: [Lei-Kun/RL-100](https://github.com/Lei-Kun/RL-100), commit `6b6afceb7eb12525398207b5d6e66908863dc385` (2026-09-28, RGB observation augmentation fix).
- LeRobot: [huggingface/lerobot](https://github.com/huggingface/lerobot), LeTools-Learning-aligned development commit `a07f22e22ce88cddff1f6eddced9ea008fbfc37c` (`pyproject.toml` version `0.5.2`; not claimed to be a tagged release).

After cloning the repository:

```bash
git submodule update --init third_party/rl100 third_party/lerobot
```

Upstream RL-100 Python sources live under `third_party/rl100/RL-100/`, rather than the outer submodule root. Submodule checkout does **not** install its dependencies. The pinned LeRobot source requires Python >=3.12. Runtime compatibility with the native RL-100 dependencies and DuoBench has not been validated; pinning a Git submodule does not install dependencies.

### Upstream behavior review

The RL-100 update from `7c5df9a` to `6b6afce` adds RGB augmentation key selection based on observation `shape_meta`. An intermediate commit changes the default `chunk_as_single_action` from `True` to `False` across 2D/3D diffusion and flow configurations. ARBiM experiment configurations must explicitly choose the desired chunk semantics before RL training. The update has been source-reviewed, **not** runtime-tested.

## Python environment (minimal, unvalidated)

The first-stage baseline is Python **3.12** with PyTorch **2.9.1** and torchvision **0.24.1**. `environment.yml` contains only a small common ML/configuration dependency set; it does **not** install the two Git submodules or any robot/simulation stack.

```bash
git submodule update --init third_party/rl100 third_party/lerobot
conda env create -f environment.yml
conda activate arbim-learning
python -m pip check
python -c "import sys, torch, torchvision; print(sys.version); print(torch.__version__, torchvision.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The PyTorch wheel/CUDA runtime must be checked on the actual Linux GPU host; for a specific CUDA build use the [official PyTorch 2.9.1 install matrix](https://pytorch.org/get-started/previous-versions/). This environment is **not yet validated** with the RL-100 native policy. Its original installation was verified on Python 3.8 / torch 2.4, and native RL-100 imports may require additional compiled PyTorch3D and image-encoder dependencies. The pinned LeRobot source uses Python >=3.12 and Hugging Face Hub >=1.0; RL-100's verified Hugging Face Hub version was 0.33.1, so the combination needs a real import test.

Once dataset work begins, install and test the pinned LeRobot dataset extras separately rather than pulling them into this initial baseline. DuoBench integration, ROS, robot hardware support, and RL training extras are intentionally deferred.

## Development workflow

`gpt/**` (implementation) → `feature/**` (review and testing) → `dev` (squash merge after approval).

Do not modify upstream submodule source, `main`, `dev`, LeTools-Learning, kdc_ARBiM, or DuoBench as part of this bootstrap task.
