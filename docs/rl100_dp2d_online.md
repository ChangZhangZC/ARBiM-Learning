# Task 6 + Task 7 — RL100 DP-2D Online PPO (DuoBench Hinge-Chest)

## Scope

Start from a verified IL checkpoint, use native RL1002D Actor/Normalizer/DDIM,
collect on-policy DuoBench trajectories, and optimize with the pinned native
`BehaviorProximalPolicyOptimization.dp_align_update_no_share()`.

Task 7 adds run management, local TensorBoard, compact console summaries,
an explicit PPO-update training budget and strict resumable checkpoints. It does
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

The default online YAML now sets **10,000 complete PPO updates** to match the
requested long-run budget. The RL-100-aligned 1024-transition PPO batch is
expensive in single-environment DuoBench; always use a short CLI override in
an isolated new run to validate GPU memory and runtime first.

```bash
python -u -m arbim.training.train_online_rl100_dp2d \
  --config configs/online/rl100_dp2d_hinge_chest.yaml \
  --checkpoint /absolute/path/to/validated/Hinge-Chest/last.ckpt \
  --max-updates 2
```

For an experiment using the full 10,000-update budget, the core parameters are:

```yaml
env:
  max_episode_steps: 992  # same cap for training rollout and evaluation episodes
ppo:
  max_updates: 10000     # 10000 complete PPO update cycles, NOT environment steps
  batch_size: 1024       # native RL-100 Chunk Online setting
  mini_batch_size: 128
  K_epochs: 5            # minibatch passes inside EACH PPO update
  epsilon: 0.2
  gamma: 0.997
  gae_lambda: 0.95
  actor_lr: 2.0e-6
  critic_lr: 3.0e-4
  linear_lr_decay: true
rollout:
  seed: 42
evaluation:
  every_updates: 50
  seed: 100042           # fixed seeds across checkpoint comparisons
  num_episodes: 10
checkpoint:
  every_updates: 50
```

A more concise CLI override is also supported:

```bash
python -u -m arbim.training.train_online_rl100_dp2d \
  --config configs/online/rl100_dp2d_hinge_chest.yaml \
  --checkpoint /absolute/path/to/validated/last.ckpt \
  --max-updates 10000 \
  --run-name 'run_{timestep}'
```

`ppo.max_updates` is the sole stopping budget. Each PPO update collects whole
training episodes until the buffer contains at least `ppo.batch_size` action-
chunk transitions, then runs `ppo.K_epochs` minibatch passes. The actual
number of environmental interactions is recorded separately as `env_steps`;
independent evaluation episodes do NOT contribute to this counter.

For example, with `max_episode_steps=992`, `n_action_steps=32`,
`batch_size=1024` and no early success, 31 transitions are collected per
episode, requiring 34 complete episodes per PPO update. The actual buffer is
**1,054** transitions, or 33,728 environment steps per update. A 10,000-update
run may therefore consume approximately **337.28 million** training
environment steps (when every episode runs to its cap). This is only a
capacity estimate, NOT a prediction of actual performance or episode length.
The full 1,054-transition batch is materialized for PPO; confirm host and GPU
memory before attempting a full training run. The YAML's existing
`evaluation.every_updates: 1` and `checkpoint.every_updates: 1` are kept
unchanged in this minimal algorithm-alignment patch; for a long run, override
both in a separate experiment YAML before starting (e.g. 50 and 10).

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
  tensorboard/              # PPO-update indexed scalar events (new runs)
  tensorboard_ppo_updates/  # only when migrating an old env-step indexed run
  diagnostics/ratio_stats.csv # append-only native ratio data, when enabled
  diagnostics/ratio.uncommitted.csv # optional archived uncommitted ratio rows
  checkpoints/
    last.ckpt               # actor, critic, optimizers, LR, RNG, decision counter
    update_000001.ckpt      # policy snapshot + inference contract, directly eval-able
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
(Policy including its Normalizer plus the existing inference contract), not
the full online optimizer state. The snapshot's `global_step` now denotes
completed **PPO updates**, and `env_steps` / `episodes` are separate metadata.
Existing `step_*.ckpt` files from old runs are not renamed, and the
snapshot-retention helper recognizes both old and new filenames.
`checkpoint.keep_last` never removes `last.ckpt` or the source IL checkpoint. No automatic `best.ckpt` exists: use fixed-seed
evaluation first before defining a statistically meaningful selection rule.

## Resume and interrupt behavior

```bash
python -u -m arbim.training.train_online_rl100_dp2d \
  --resume outputs/online/hinge_chest/run_YYYYMMDD_HHMMSS
```

This automatically loads the run's `config_resolved.yaml` and
`checkpoints/last.ckpt`. When **linear LR decay is enabled**, `ppo.max_updates`
determines the LR horizon, so changing it on Resume is rejected by the
checkpoint contract. An Adaptive-KL run without linear decay can still
extend its V2 update budget. Device relocation using `--device` is supported;
algorithm/environment parameters
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
`MetricLogger`; metrics retain event-based JSONL separately. An old pre-schema
run's TensorBoard files are kept unchanged, while new PPO-update-indexed
events from its Resume go into `tensorboard_ppo_updates/`.

All TensorBoard scalar groups use completed **PPO update count** as their
x-axis. Training Rollout episode-level records remain in `metrics.jsonl`;
`rollout/` curves are the **mean over episodes collected for each update**,
not one point per episode. This keeps Resume/purge semantics well defined.

- `rollout/`: mean return, success rate, stage, episode length, transitions
- `ppo/`: actor and critic losses
- `lr/`: actual actor/critic optimizer learning rates
- `eval/`: success rate, mean return/steps/stage with a fixed seed set
- `progress/`: PPO updates, total chunk decisions, actual batch size, total
  training environment steps and total episodes
- `system/`: rollout and update/evaluation times, train env steps per second
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

## Terminology and old-run compatibility

- `ppo.max_updates`: total number of **complete PPO update cycles**; 10,000
  here means exactly 10,000 PPO updates, barring manual interrupt.
- `ppo.K_epochs`: repeated optimization passes over one collected on-policy
  buffer, not the number of episodes or PPO update cycles.
- `env.max_episode_steps`: common per-episode cap used by training rollout
  and evaluation; it must be divisible by the effective action chunk size.
- `evaluation.every_updates` / `checkpoint.every_updates`: frequency in PPO
  updates, not environment steps. Evaluation interactions are excluded from
  training `env_steps`.
- `rollout.seed`: training episode seed; independent Eval uses its own seed.
- Native `ppo.max_train_steps` is a **decision-step LR-schedule horizon**,
  distinct from ARBiM `ppo.max_updates`. Rather than expose a second
  ambiguous training budget, the adapter derives it as
  `max_updates * ceil(batch_size / max_episode_decisions) * max_episode_decisions`,
  where `max_episode_decisions = env.max_episode_steps / n_action_steps`.
  With current defaults: `10000 * 34 * 31 = 10,540,000` decision steps.
  The unmodified RL-100 `lr_decay(total_steps)` is invoked after each PPO
  update with the cumulative **actual action-chunk decisions** collected,
  capped defensively at the derived horizon. The schedule may therefore
  end with nonzero LR if many episodes terminate early.
- New Online `last.ckpt` records `decision_steps` in addition to `updates`,
  `episodes` and `env_steps`. Linear-decay Resume requires this counter,
  and cannot silently rebase the LR horizon by extending `max_updates`.
- The native RL-100 Chunk Online example uses four parallel environments,
  a 16-step action chunk, an independently trainable Critic encoder and an
  unfrozen Actor encoder. ARBiM still uses a single DuoBench environment,
  keeps the IL checkpoint's 32-step chunk and frozen/shared encoder. These
  are explicitly **not yet equivalent**. The Critic is also random-initialized
  because no offline Q/V checkpoint is available.
- Old Task 7 `last.ckpt` V1/V2 Resume remains supported if its saved YAML had
  identical training/evaluation episode caps and no active environment-step
  training budget. A legacy run with differing caps or a non-null
  `rollout.max_total_env_steps` is rejected instead of silently changing
  semantics. Reorganized config is written after a successful new checkpoint.
- Legacy `step_XXXXXXXX.ckpt` filenames encoded environmental interactions,
  not PPO update count. They remain loadable as policy snapshots.
