# Task 6 + Task 7 — RL100 DP-2D Online PPO (DuoBench Hinge-Chest)

## Scope

Start from a verified IL checkpoint, use native RL1002D Actor/Normalizer/DDIM,
collect on-policy DuoBench trajectories, and optimize with the pinned native
`BehaviorProximalPolicyOptimization.dp_align_update_no_share()`.

Task 7 adds run management, local TensorBoard, compact console summaries,
an explicit PPO mini-batch-step training budget and strict resumable checkpoints. It does
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

## Smoke test (separate temporary YAML)

Never change the experiment-default `configs/online/rl100_dp2d_hinge_chest.yaml`
for smoke tests. Use `configs/online/tmp/rl100_dp2d_hinge_chest_smoke.yaml`:
small training batch, short episodes (including partial chunks), reduced epochs,
minimal evaluation, and isolated `/tmp/arbim_online_smoke` outputs.
This test-only YAML is **not** for scientific comparisons.

```bash
python -u -m arbim.training.train_online_rl100_dp2d \\
  --config configs/online/tmp/rl100_dp2d_hinge_chest_smoke.yaml \\
  --checkpoint /absolute/path/to/validated/Hinge-Chest/last.ckpt
```

The smoke budget counts PPO mini-batch steps, checked only at the end of a full
PPO rollout/update cycle. The number of cycles depends on the checkpoint's
actual chunk size; verify the `[PPO] rollout_update` count before using its output.

For a run using the 10,000 PPO mini-batch-step budget, the core parameters are:

```yaml
env:
  max_episode_steps: 500  # same cap for training rollout and evaluation episodes
ppo:
  max_steps: 10000       # PPO mini-batch passes, NOT environment steps or update cycles
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
  every_updates: 25     # complete PPO rollout/update cycles
  seed: 100042           # fixed seeds across checkpoint comparisons
  num_episodes: 30       # experiment default, matching native Online example
checkpoint:
  every_updates: 1      # recoverable after each complete PPO update
```

A more concise CLI override is also supported:

```bash
python -u -m arbim.training.train_online_rl100_dp2d \
  --config configs/online/rl100_dp2d_hinge_chest.yaml \
  --checkpoint /absolute/path/to/validated/last.ckpt \
  --max-steps 10000 \
  --run-name 'run_{timestep}'
```

New runs stop at `ppo.max_steps` completed mini-batch passes, checked only after a full native PPO update cycle. Legacy runs with `ppo.max_updates` retain their original whole-cycle budget. Each rollout/update cycle collects whole episodes until the buffer has at least `ppo.batch_size` chunk transitions, then runs `ppo.K_epochs` epochs of mini-batches. The actual
number of environmental interactions is recorded separately as `env_steps`;
independent evaluation episodes do NOT contribute to this counter.

For example, with `max_episode_steps=500`, `n_action_steps=32`,
`batch_size=1024` and no early success, 16 transitions are collected per
episode (the last chunk executes only 20 environment steps), requiring 64 complete episodes per rollout/update cycle. The actual buffer is
**1,024** transitions, or 32,000 environment steps per cycle. Five epochs with `mini_batch_size=128` yield **40 PPO mini-batch steps per cycle**. A 10,000-step budget corresponds to about **250 full cycles** and about **8 million training environment steps** in this no-early-success example, not 10,000 cycles. This is only a
capacity estimate, NOT a prediction of actual performance or episode length.
The full 1,024-transition batch is materialized for PPO; confirm host and GPU
memory before attempting a full training run. The experiment YAML evaluates 30 episodes every 25 complete PPO updates; the resumable checkpoint remains saved after every complete update. Test-only overrides stay in the separate smoke YAML.

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
  tensorboard_ppo_steps/    # new mini-batch-step indexed scalar events
  tensorboard/              # legacy whole-update indexed events
  tensorboard_ppo_updates/  # legacy env-step migration
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
`checkpoints/last.ckpt`. When **linear LR decay is enabled**, the configured training budget
(`ppo.max_steps`, or legacy `ppo.max_updates`) determines the native LR horizon, so changing it on Resume is rejected by the checkpoint contract. An Adaptive-KL run without linear decay can still extend its V2 budget. Device relocation using `--device` is supported;
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
publicly exposing the service. The local TensorBoard sink reuses the IL `MetricLogger`.
Existing TensorBoard files remain separate: new PPO-step runs write
`tensorboard_ppo_steps/`; legacy runs retain `tensorboard/` or
`tensorboard_ppo_updates/` after an env-step-to-update migration.

New PPO-step runs use completed **PPO mini-batch steps** as the TensorBoard x-axis; this is also the Resume purge point. Legacy `ppo.max_updates` runs retain whole-update x-axis semantics. Training Rollout episode-level records remain in `metrics.jsonl`;
`rollout/` curves are the **mean over episodes collected for each update**,
not one point per episode. This keeps Resume/purge semantics well defined.

- `rollout/`: mean return, success rate, stage, episode length, transitions
- `ppo/`: actor and critic losses
- `lr/`: actual actor/critic optimizer learning rates
- `eval/`: success rate, mean return/steps/stage with a fixed seed set
- `progress/`: PPO mini-batch steps, whole rollout/update cycles, Actor/Critic optimizer steps, chunk decisions, actual batch size, training environment steps and total episodes
- `system/`: rollout and update/evaluation times, train env steps per second
- `diagnostics/`: upstream native ratio quantile/mean and logprob delta

The pinned native PPO code exposes detailed ratio diagnostics optionally.
The append-only `diagnostics/ratio_stats.csv` preserves native records and adds `sample_size`, `approx_kl`, and `clip_fraction` without changing ratio calculation. An ARBiM-only read-only diagnostics wrapper now records `approx_kl` and
`clip_fraction` from the exact native Ratio tensors, without changing PPO loss.
Full advantage/value distributions are not exposed by the native update API.

Training and evaluation differ: training samples stochastic DDIM and
Eval executes deterministic inference with fixed seeds, so individual
training returns should not be compared directly with Eval returns.
The current Reward mode is only `env_raw` and is not a validated research reward.

## Validation gates

```bash
python -m pytest -q tests/test_online_*.py
```

Follow with a genuine GPU/MuJoCo smoke run. With native `false/false`,
check Actor Encoder **and** independent Critic Encoder gradients/parameter
updates, distinct optimizer ownership, V2 checkpoint round-trip, independent
policy-snapshot Eval, two PPO updates, and interrupted/resumed training.
Repeat small one-update tests for the other three flag combinations. Compare fixed-seed IL and PPO evaluations
before interpreting online reward increases as performance improvements.

Neither GitHub-source review nor CPU-only tests substitute for these GPU/
DuoBench validation steps. See
[rl100_dp2d_online_contract.md](rl100_dp2d_online_contract.md) for native
shape/ratio boundaries.

## Terminology and old-run compatibility

- `ppo.max_steps`: total number of **PPO mini-batch passes**; one 1024-transition cycle with mini-batch 128 and five epochs accounts for 40 PPO steps. Native PPO finishes its current cycle before checking the budget, so the final step counter may overshoot. Legacy `ppo.max_updates` still means complete rollout/update cycles.
- `ppo.K_epochs`: repeated optimization passes over the same on-policy
  buffer. Actor optimizer steps additionally multiply by DDIM inference steps; Critic optimizer steps equal PPO mini-batch steps.
- `env.max_episode_steps`: common per-episode cap used by training rollout
  and evaluation; it need not be divisible by the action chunk length. The final partial chunk uses its actual executed span for GAE.
- `evaluation.every_updates` / `checkpoint.every_updates`: frequency in PPO
  updates, not environment steps. Evaluation interactions are excluded from
  training `env_steps`.
- `rollout.seed`: training episode seed; independent Eval uses its own seed.
- Native `ppo.max_train_steps` is a **decision-step LR-schedule horizon**,
  distinct from ARBiM `ppo.max_steps` (and legacy `ppo.max_updates`). Rather than expose a second
  ambiguous training budget, the adapter derives it as
  `ceil(max_steps / (K_epochs * ceil(batch_size / mini_batch_size))) *
  (batch_size + ceil(max_episode_steps / n_action_steps) - 1)`
  decision steps in the new mode. This conservative bound includes the case
  where the last full episode overshoots the buffer threshold.
  With example defaults and H=32: `250 * (1024 + 16 - 1) = 259,750` decisions.
  The unmodified RL-100 `lr_decay(total_steps)` is invoked after each PPO
  update with the cumulative **actual action-chunk decisions** collected,
  capped defensively at the derived horizon. The schedule may therefore
  end with nonzero LR if many episodes terminate early.
- New Online `last.ckpt` records `decision_steps` and `ppo_steps` in addition to whole-cycle `updates`, `episodes` and `env_steps`. Linear-decay Resume requires the decision counter; new PPO-step-budget Resume also requires its minibatch counter. The LR horizon cannot be silently rebased by changing the training budget.
- The native RL-100 Chunk Online example uses four parallel environments,
  a 16-step action chunk, an independently trainable Critic Encoder and an
  unfrozen Actor Encoder. ARBiM now uses matching **false/false** Encoder
  modes by default, but keeps a single DuoBench environment and preserves
  the IL checkpoint's actual action chunk (32 is illustrative, not assumed). These remaining differences
  are explicitly **not yet equivalent**. The Value Head is randomly
  initialized because no offline Q/V checkpoint is available; the
  independent Critic Encoder is instead initialized from IL Actor weights.
- Old Task 7 `last.ckpt` V1/V2 Resume remains supported if its saved YAML had
  identical training/evaluation episode caps and no active environment-step
  training budget. A legacy run with differing caps or a non-null
  `rollout.max_total_env_steps` is rejected instead of silently changing
  semantics. Reorganized config is written after a successful new checkpoint.
- Legacy `step_XXXXXXXX.ckpt` filenames encoded environmental interactions,
  not PPO update count. They remain loadable as policy snapshots.

## Encoder modes: all four paths supported

Two independent PPO flags select the native RL-100 Actor/Value arrangement:

| `fix_encoder` | `share_encoder` | Actor RGB Encoder | Critic RGB Encoder |
|---|---|---|---|
| **false** | **false** | trainable, IL-initialized | independent, IL-initialized, trainable (**default**) |
| false | true | trainable | no separate Encoder; Critic uses detached Actor features |
| true | false | frozen | independent, IL-initialized, **still trainable** |
| true | true | frozen | no separate Encoder; Critic uses fixed Actor features |

`fix_encoder` applies to the **Actor Encoder only**. In RL-100's
`ValueLearner`, a separate Value Encoder is trainable even if the Actor
Encoder is fixed. Do not confuse `torch.nn.Module.eval()` (BatchNorm/Dropout
behavior) with `requires_grad_(False)`: native Online RL-100 puts the Actor
Encoder in eval mode even when it can receive PPO gradients. Both the
current and frozen `_old_policy` remain separate; `set_old_policy()` is
followed by explicit `requires_grad_(False)` to keep the old snapshot frozen.

With `share_encoder=false`, the native PPO route uses
`_policy.obs2this_nobs()` to supply normalized raw observations to
the **Critic's own** `ValueMLP._obs_encoder`. In the shared route,
`_policy.obs2latent()` provides detached features instead. This
selection is made by the pinned `dp_align_update_no_share()`, not a
new ARBiM PPO loss function.

The independent Critic Encoder is deep-copied from the verified IL
Actor Encoder as a necessary adaptation in the absence of offline
Critic weights. The independent Value Head is randomly initialized.
Changing `fix_encoder` or `share_encoder` between checkpoints changes
the architecture/optimization contract and is rejected on Resume.

**Validation matrix:** `tests/test_online_encoder_modes.py` exercises all four
combinations using a small native PPO/ValueMLP bridge. It verifies encoder
ownership, gradients, optimizer parameter groups, Value input shapes and
full checkpoint restoration without spawning DuoBench. The end-to-end GPU
PPO and MuJoCo smoke gates must still run before claiming native behavior
equivalence. `false/false` allocates another visual Encoder and will increase
VRAM and critic backward cost substantially relative to the previous V1.
