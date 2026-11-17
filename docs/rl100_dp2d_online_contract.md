# Task 6.0 — Native Online PPO contract review (2026-11-17)

Base: ARBiM-Learning dev a221cb1; RL-100 submodule 6b6afceb; DuoBench e57e72cb.
User confirms the Hinge-Chest IL checkpoint already runs inference. Do not claim GPU PPO tests are complete before Feature validation.

## Boundary
- Preserve RL1002D actor, compute_loss, Normalizer, DDIM scheduler and native dp_align_update_no_share loss.
- Actor observes agent_pos and rgb_0/rgb_1/rgb_2 (two time steps in the example); no point clouds.
- DuoBenchAdapter converts left seven joints+gripper, followed by right seven joints+gripper (16D).
- IL checkpoint contract determines H, T, K and camera mapping; no train-time normalization added to input adapter.

## Tensor contract

| Item | Shape |
| --- | --- |
| Native observation | agent_pos [B,To,16]; each rgb_i [B,To,3,H,W] |
| Sampled executed action | [B,H_exec,16], physical/unnormalized |
| all_x | [K+1,B,T,16], normalized full denoising trajectory |
| a_logprob | [K,B,T,16] |
| Buffer action | [N,K+1,T,16] |
| Buffer old log probability | [N,K,T,16] |
| Raw chunk reward/done/dw | [N,1] |
| Native PPO value and advantage | [N,1] |

Here T = horizon - n_obs_steps + 1 when no_pre_action=True, while H_exec = effective n_action_steps <= T. Reward is discounted within the executed chunk and the native GAE uses gamma ** H_exec between decisions. Actual true termination may occur earlier, with dw=True preventing bootstrapping. Timeout should preserve bootstrapping; partial, nonterminal chunks are explicitly unsupported in V1.

## Original Online ratio modes and discovered blocker

- chunk_as_single_action=True: sum action time and dimension, ratio [N].
- chunk_as_single_action=False: sum action dimension only. In pinned uni_ppo.py, sum(-1, keepdim=True).squeeze(1) produces [N,H,1] when H>1, which cannot correctly broadcast with advantage [N,1].
- No upstream edits, monkey patches, or duplicate PPO loss are permitted. Therefore V1 supports full chunk only in chunk_joint; per_action_step must have H_exec=1. This is a documented deviation from the original Task 6 v1.0 plan, pending explicit upstream patch approval.
- offline_chunk_ratio_mode, bppo_chunk_level_ratio, ratio_strategy and top_k must not be mistaken for Online ratio selectors.

## Value and experiment contract

- Online Value uses native ValueMLP(None, obs_feature_dim * n_obs_steps, ...), fed by frozen native obs2latent() on current/next RGB+state.
- share_encoder=True, fix_encoder=True; no separate training of vision encoder, Q, IQL, dynamics or distillation.
- The critic starts from scratch: NOT RL-100 offline critic warm-start.
- Use complete episodes per PPO minibatch to avoid GAE episode-boundary leakage and permit reliable checkpoint/resume at batch boundaries.
- Save policy-only inference snapshots in the existing version-2 format (original IL contract), and separate full Online training checkpoints with actor, critic, both optimizers, adaptive KL learning-rate state and RNG.
- Existing Task 5 evaluation remains independent; training returns, raw rewards and success rates must be separately recorded.

## Feature validation gates

1. Correct strict checkpoint load, image aliasing, DDIM sampling shapes, finite log probabilities, matching sampler/recomputed log-prob where applicable.
2. chunk_joint with H>1; per_action_step with H=1; the unsafe H>1 case must fail fast.
3. Buffer episode boundaries, image dtype/normalizer, actual environment action order, gamma-spanned reward, done/dw.
4. Native actor/critic gradient updates and finite loss; frozen vision encoder stays fixed.
5. GPU/MuJoCo single rollout -> PPO update -> independent fixed-seed evaluation -> snapshot -> resume.
6. Original IL training/inference/evaluation regression and no upstream submodule changes.
