# RL-100 DP-2D imitation-learning trainer

ARBiM-Learning uses pinned **native RL1002D Actor, compute_loss, Diffusion,
Normalizer, Scheduler and EMA**. Training orchestration is ARBiM-specific.
The pipeline does **not** wrap RL100 in LeRobot PreTrainedPolicy.

## New training run

Edit `configs/il/duobench_transfer_cube.yaml` for a real LeRobot v3 dataset.
From the repository root and the prepared `arbim` environment:

```bash
export PYTHONPATH="$PWD/src:$PWD/third_party/rl100/RL-100:$PWD/third_party/lerobot/src"
python -m arbim.training.il --config configs/il/duobench_transfer_cube.yaml
```

Output by default:

```text
outputs/il/<task>/<method>/<YYYYMMDD_HHMMSS>/
  config_resolved.yaml
  run_meta.json
  metrics.jsonl
  tensorboard/
  wandb_run_id.txt          # only if W&B enabled
  checkpoints/
    last.ckpt               # full resumable state
    step_00002000.ckpt      # retained snapshot, if scheduled
    best.ckpt               # lowest measured validation loss; raw Policy
```

Use `experiment.run_name` for an explicit run name or `output.output_dir`
for a full path. A fresh run rejects existing directories. There are no
batch-count smoke caps in the production YAML; all eligible training windows
are used per complete epoch.

## Steps, Epochs, and save triggers

`training.max_steps` and `training.max_epochs` each accept a positive
integer or null. At least one must be set. If both are set, stop at the
first reached limit. Each optimizer update is a Step. A full Epoch consumes
all training-window indices, counting the final partial batch as an update.
Effective budget is min(max_steps, max_epochs * ceil(N / batch_size)), omitting
disabled limits. A Step limit can end training mid-Epoch immediately.

`checkpoint` and `validation` both accept `every_steps` and
`every_epochs`; these triggers are OR-combined, firing once at a shared
Step/Epoch boundary. `checkpoint.save_on_finish` and
`validation.on_finish` allow final-step behavior. `checkpoint.keep_last`
bounds the number of periodic `step_*.ckpt` archives; `last.ckpt` and
`best.ckpt` are separate. Atomic replacement protects `last.ckpt`.
`best.ckpt` is **not** a Resume checkpoint.

## Exact Step Resume

```bash
python -m arbim.training.il --resume \
  outputs/il/duobench_transfer_cube/rl100_dp2d/20261115_103000
```

**Native RL-100 Normalizer load-order compatibility:** On Resume, the entire Policy
(including its dynamically reconstructed `DictOfTensorMixin` Normalizer) is
strict-loaded on CPU before moving to the training device. Only then are the
Optimizer and EMA constructed; the restored EMA state is separately moved to
the device after strict load. This is necessary because the native Normalizer
replaces registered ParameterDict objects when loading weights. The original
Checkpoint remains valid after a failed Resume before the next save.

With `--resume DIR`, the saved `config_resolved.yaml` is used unless an
updated `--config` is explicitly passed. This always uses
`DIR/checkpoints/last.ckpt`, continues logging in the **same** run directory,
restores native Policy/Normalizer, Optimizer, Scheduler, EMA, global Step and
Python/NumPy/Torch/CUDA RNG. Contract mismatch fails explicitly and never
starts a fresh run. The model, data contract, split, batch size, seed and
original budget/Scheduler plan are fixed; logging intervals/settings may change.

A deterministic StepBatchSampler uses a private CPU generator with
`seed + epoch_index`. Resume reconstructs the same permutation and skips
previously updated batches, independent of DataLoader prefetch. Random
**worker-side transforms** cannot be exactly replayed without tracking each
worker's RNG; keep dataset-side operations deterministic for bitwise resumes.
Current native visual augmentations happen in the Policy and use restored RNG.

Validation uses a fixed diffusion-noise seed, temporarily disables native
RandomShiftsAug, and restores training RNG afterward. It runs on **raw**
Policy weights as in the prior smoke trainer; the native EMA is kept in the
training checkpoint. Validation noise-prediction loss is not task success.

A v1 epoch-boundary `latest.ckpt` is not automatically interpreted as a v2
exact-Step checkpoint. Start a new run for this checkpoint format.

## Telemetry and preprocess/postprocess ownership

Terminal output shows Step/target, equivalent Epoch, Loss, LR, throughput and
ETA. Machine-readable `metrics.jsonl` is always available when enabled.
TensorBoard requires `python -m pip install tensorboard` and can be viewed
via `tensorboard --logdir outputs/il`. Optional W&B requires
`python -m pip install wandb`, and `mode: offline` avoids cloud access.
Resume trims metrics newer than the last durable checkpoint and retains the
W&B Run ID if it exists.

`run_meta.json` records Git and Submodule revisions, camera alias mapping,
state/action contract and preprocess/postprocess ownership. ARBiM adapts
LeRobot camera names, tensor layouts and RGB transport; native RL100 owns
state/action Normalizer, vision transforms, Diffusion and Action unnormalization.
Do **not** stack LeRobot Normalize/Unnormalize on top. Robot-specific control
mapping and deployable policy export are future integration tasks, not part of
this IL training PR.

## Acceptance tests

```bash
pytest -q tests/test_train_ops.py tests/test_il.py tests/test_lerobot_adapter.py tests/test_rl100_dp.py
```

CPU regression tests use a tiny fake Policy to compare uninterrupted training
to mid-Epoch and Epoch-boundary interruption and Resume. Real RL100 GPU
forward/backward, Video decoding and checkpoint restoration must also be
verified on the Feature branch before Dev integration.
