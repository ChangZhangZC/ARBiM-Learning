# Task 6 + Task 7 — RL100 DP-2D Online PPO (DuoBench Hinge-Chest)

## Scope

Start from a verified IL checkpoint, use native RL1002D Actor/Normalizer/DDIM,
collect on-policy DuoBench trajectories, and optimize with the pinned native
`BehaviorProximalPolicyOptimization.dp_align_update_no_share()`.

Task 7 adds run management, local TensorBoard, compact console summaries, a
real total environment-step budget and strict resumable checkpoints. It does
**not** add offline IQL, dynamics, distillation, BC/reconstruction loss, DDP,
asymmetric updates, or persistent online trajectory datasets. The native PPO
mathematics and diffusion sampler are not modified.

Only `hinge_chest` is supported by the current task adapter/reward evaluator.

## Environment setup

From the repo root in the tested `arbim` environment:

```bash
export PYTHONPATH="$PWD/third_party/rl100/RL-100/rl_100:$PWD/third_party/rl100/RL-100:${PYTHONPATH:-}"
python -m pip install tensorboard  # needed for torch.utils.tensorboard
```

`tensorboardX` is a separate *upstream import dependency* and does not replace
the `tensorboard` package required by the local IL-style TensorBoard sink.

## Start a short smoke test

The default online YAML is deliberately a **one-update smoke configuration**;
it does not silently launch a long job.

```bash
python -u -m arbim.training.train_online_rl100_dp2d \
  --config configs/online/rl100_dp2d_hinge_chest.yaml \
  --checkpoint /absolute/path/to/validated/Hinge-Chest/last.ckpt
```

For a longer run, copy the YAML into a separate experiment-specific YAML and set:

```yaml
rollout:
  max_env_steps: 992       # per-episode cap, divisible by effective n_action_steps
  max_total_env_steps: 10000
  max_updates: null
evaluation:
  seed: 100042            # same initial states across checkpoints
  num_episodes: 10
```

A more concise CLI override is also supported:

```bash
python -u -m arbim.training.train_online_rl100_dp2d \
  --config configs/online/rl100_dp2d_hinge_chest.yaml \
  --checkpoint /absolute/path/to/validated/last.ckpt \
  --max-total-env-steps 10000 \
  --max-updates 1000 \
  --run-name 'run_{timestep}'
```

When both limits are set, whichever is reached first stops the experiment
*after a complete PPO update*. At a total step budget of 10,000 with 992-step
episodes, the actual step counter may exceed 10,000. This preserves complete
chunks and native GAE semantics.

`policy.n_action_steps: null` and `policy.num_inference_steps: null` preserve
the validated IL checkpoint settings. Full chunk V1 uses
`ppo.ratio_mode: chunk_joint`. Upstream `per_action_step` is only supported
for single-action chunks (`n_action_steps=1`) due to a pinned upstream
broadcasting issue for longer chunks.

## Run directory and files

```text
outputs/online/hinge_chest/run_YYYYMMDD_HHMMSS/
  config_resolved.yaml      # authoritative full config
  config_history.jsonl      # budget/logging amendments on resume (when changed)
  run_meta.json             # IL path/SHA256, Git/submodule revisions, task/device
  metrics.jsonl             # structured rollout/ppo/evaluation/checkpoint events
  metrics.uncommitted.jsonl # optional archived events newer than last durable update
  train.log                 # compact ARBiM terminal summaries
  tensorboard/              # local scalar event files, when enabled
  diagnostics/ratio_stats.csv # append-only native ratio data, when enabled
  diagnostics/ratio.uncommitted.csv # optional archived uncommitted ratio rows
  checkpoints/
    last.ckpt               # actor, critic, optimizers, adaptive LR, RNG, counters
    step_XXXXXXXX.ckpt      # policy + inference contract, directly eval-able
  evaluation/
    update_000001/
      episodes.jsonl
      evaluation.log
      summary.json
```

The original IL `last.ckpt` is not copied into the Online run. Its path and
streaming SHA256 checksum are recorded; the source must remain accessible and
unaltered for strict Resume policy reconstruction. The new
Online `last.ckpt` format V2 stores a strict algorithm-contract SHA256 digest
instead of a duplicate full YAML; V1 smoke checkpoints remain readable with
their original strict full-config contract.

Policy snapshots preserve the existing ARBiM version-2 inference format
(Policy including its Normalizer plus the minimum existing inference contract),
not the full online optimizer state. `checkpoint.keep_last` prunes only older
`step_*.ckpt` snapshots within that run; `last.ckpt` and the source IL
checkpoint are never removed. No automatic `best.ckpt` exists: use fixed-seed
evaluation first before defining a statistically meaningful selection rule.

## Resume and interrupt behavior

```bash
python -u -m arbim.training.train_online_rl100_dp2d \
  --resume outputs/online/hinge_chest/run_YYYYMMDD_HHMMSS
```

This automatically loads the run's `config_resolved.yaml` and its
`checkpoints/last.ckpt`. You may use `--max-updates` or
`--max-total-env-steps` to **extend** a V2 run's budget, or `--device` to
move it to a different available CUDA device. Algorithm/environment parameters
remain strict; changing reward, PPO hyperparameters or rollout seed requires a
new run. V1 checkpoints require their original *complete* configuration,
including the budget.

The first Ctrl+C requests a stop after the current full on-policy batch
(which may span multiple complete episodes) and PPO update, then saves a
resumable checkpoint. A second Ctrl+C forces interruption.
An abrupt process crash discards all in-memory, not-yet-updated on-policy
data; Resume starts with an empty buffer and fresh sampling from the last
durable update. This is intentional; mid-episode replay/restore is unsupported.
If metrics were written beyond the last checkpoint, they are moved to
`metrics.uncommitted.jsonl` rather than silently duplicated.

Stop at a saved completed update boundary; no Offline RL or experience replay
is implied by this behavior.

## Monitoring

To view TensorBoard on the training machine:

```bash
tensorboard --logdir outputs/online --host 127.0.0.1 --port 6006
```

Use SSH port forwarding when accessing from another computer rather than
publicly exposing the service. The local TensorBoard sink reuses the IL
`MetricLogger`; metrics retain event-based JSONL separately.

Primary TensorBoard x-axis is cumulative `env_steps`:

- `rollout/`: reward/return, stage, success, episode steps, transitions
- `ppo/`: actor and critic losses
- `lr/`: actual actor/critic optimizer learning rates
- `eval/`: success rate, mean return/steps/stage with a fixed seed set
- `system/`: rollout and update/evaluation times, steps per second
- `diagnostics/`: upstream native ratio quantile/mean and logprob delta

The pinned native PPO code exposes detailed ratio diagnostics optionally.
The append-only `diagnostics/ratio_stats.csv` preserves those native records
without changing ratio calculation. Approx-KL, clip fraction and full
advantage/value distributions are **not yet exposed** by the V1 native
update return interface. Do not infer them from actor loss or fabricate them.
More detailed diagnostics would require a separate read-only instrumentation
review.

Training and evaluation differ: training samples stochastic DDIM and
Eval executes deterministic inference with fixed seeds, so individual
training returns should not be compared directly with Eval returns.
The current Reward mode is only `env_raw` and is not a validated research reward.

## Validation gates

```bash
python -m pytest -q tests/test_online_*.py
```

Follow with a genuine GPU/MuJoCo smoke run, check both actor and critic
optimizer update, frozen vision encoder, V2 full checkpoint round-trip,
policy snapshot Eval, two consecutive PPO updates, then a separately
interrupted-and-resumed run. Compare fixed-seed IL and PPO evaluations
before interpreting online reward increases as performance improvements.

Neither GitHub-source review nor CPU-only tests substitute for these GPU/
DuoBench validation steps. See
[rl100_dp2d_online_contract.md](rl100_dp2d_online_contract.md) for native
shape/ratio boundaries.
