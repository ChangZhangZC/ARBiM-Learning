# ARBiM-Learning

Minimal, environment-independent learning integration for **ARBiM** (Asymmetric Reinforcement for Bimanual Manipulation).

Current scope: establish the repository structure and preserve upstream sources. **IL, RL, runtime, and DuoBench integration have not yet been implemented.**

## Repository layout

- `third_party/rl100/`: upstream RL-100 as a pinned Git submodule.
- `third_party/lerobot/`: upstream LeRobot as a pinned Git submodule.
- `src/arbim/policy/`: model loading/construction adapter (planned).
- `src/arbim/data/`: LeRobot-to-RL-100 batch adapter (planned).
- `src/arbim/training/`: IL training entry (planned).
- `src/arbim/runtime.py`: inference interface (planned).
- `src/arbim/integrations/`: external environment adapters (planned; DuoBench is deferred).
- `configs/il/`, `scripts/`, `tests/`: experiments, CLI entry points, and tests (planned).
- `outputs/`: ignored local training artifacts.

## Pinned upstream versions

- RL-100: [Lei-Kun/RL-100](https://github.com/Lei-Kun/RL-100), commit `7c5df9a5d3111e5fa8d2fe814c4fdcb3acfa3f26`.
- LeRobot: [huggingface/lerobot](https://github.com/huggingface/lerobot), `v0.4.4` / commit `8fff0fde7c79f23a93d845d1a50e985de01f8b8a`.

After cloning the repository:

```bash
git submodule update --init third_party/rl100 third_party/lerobot
```

Upstream RL-100 Python sources live under `third_party/rl100/RL-100/`, rather than the outer submodule root. Submodule checkout does **not** install its dependencies. LeRobot `v0.4.4` is provisionally pinned for Python 3.11; its dependency compatibility with RL-100 is not yet validated.

## Development workflow

`gpt/**` (implementation) → `feature/**` (review and testing) → `dev` (squash merge after approval).

Do not modify upstream submodule source, `main`, `dev`, LeTools-Learning, kdc_ARBiM, or DuoBench as part of this bootstrap task.
