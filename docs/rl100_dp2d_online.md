# Task 6 — Native RL100 DP-2D Online PPO / DuoBench Hinge-Chest

## Scope
- Starts from an already validated Hinge-Chest IL checkpoint and preserves the entire RL1002D Actor/Normalizer/DDIM machinery.
- Reuses upstream BehaviorProximalPolicyOptimization.transfer2online() and dp_align_update_no_share(), not a new PPO loss.
- Starts a **new**, randomly initialized native ValueMLP critic (no Offline IQL warm-start).
- Uses the original DuoBenchAdapter for 16D left-then-right joint commands and three RGB views.
- Trains exclusively with environment-native reward in this first smoke-test version. No asymmetric reward, Q, model-based OPE, CM, residual policy, DDP, experience replay or offline data mixing.
- Task 5 Evaluation is reused and is never trained on its deterministic episodes.

## Starting a smoke run

Set the path to a real verified Hinge-Chest checkpoint. Do not use the example transfer_cube IL dataset path:

```bash
python -m arbim.training.train_online_rl100_dp2d \
  --config configs/online/rl100_dp2d_hinge_chest.yaml \
  --checkpoint /absolute/path/to/hinge_chest/checkpoints/best.ckpt
```

To exercise single-step sampling set `policy.n_action_steps: 1`, `ppo.ratio_mode: per_action_step`, and keep all environment step budgets divisible by one. Full Chunk V1 only supports `chunk_joint`. This restriction is required because the pinned upstream `per_action_step` branch has a [B,H,1] vs [B,1] broadcasting defect for H>1. No upstream code has been edited.

## Online data and PPO semantics

- `all_step_action_logprob()` samples native DDIM denoising transitions. The actual emitted physical action and each stored denoising trajectory come from the **same sampling call**.
- For each complete chunk, the reward is `sum(gamma ** j * r[j])`. Native GAE bootstraps with `gamma ** n_action_steps`; `done` resets GAE, while `dw` suppresses bootstrap for true termination.
- Only full-width chunks are supported for continuing or truncated episodes. Partial true-terminal chunks may be stored because their `dw` zeroes the bootstrap; partial truncated chunks are rejected instead of silently applying the wrong discount.
- The on-policy buffer stores raw RGB windows compactly as `uint8`, and converts to native float32 CHW images in [0,1] before PPO. Native Normalizer remains the only source for robot state normalization.
- The native encoder is frozen and evaluated without RandomShifts augmentation. ValueMLP receives native encoded RGB/state features via `obs2latent()`.
- Collection finishes entire episodes before PPO update. Old trajectories are never reused after an update; the buffer is cleared at update boundaries.

## Checkpoints
The run creates:
```text
outputs/online/hinge_chest/<run_name>/
  config_resolved.yaml
  metrics.jsonl
  checkpoints/
    last.ckpt          # actor, critic, optimizers, adaptive LR, RNG and counters
    step_XXXXXXXX.ckpt # policy-only: existing version-2 inference loader
  evaluation/
    update_000001/...
```
Resume only from a finished update boundary, with the **same** original IL checkpoint/config, using `--resume path/to/checkpoints/last.ckpt`. The initial IL checkpoint is never overwritten. EMA is not used by Task 6 V1 (do not misrepresent this as full RL-100 reproduction).

## Feature validation
1. Run `python -m pytest -q tests/test_online_*.py` in the pinned ARBiM environment.
2. Run all original tests and compare native IL deterministic inference outputs to the baseline.
3. Run a single GPU/MuJoCo collection-update-evaluation-checkpoint cycle.
4. Resume from a saved completed batch, confirm the existing policy-only snapshot loads in the Task 5 inference/evaluation entry point.
5. Record native success rate, environment return and training return separately; a successful smoke test is not evidence of policy performance improvement.

See [rl100_dp2d_online_contract.md](rl100_dp2d_online_contract.md) for shape contracts and explicit upstream risk boundaries.
