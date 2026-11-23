# DA3 experiment: official single-image smoke

This experimental directory is separate from the established `src/arbim` IL/RL training, checkpoint inference, and DuoBench evaluation code. DA3 itself remains unchanged in `third_party/da3`.

## Reproduce the official example

Starting from `gpt/da3-official-smoke` (or a later integration branch) in the existing `arbim` environment:

```bash
git submodule update --init third_party/da3
python experiments/da3/inference.py --config configs/experiments/da3.yaml
```

The smoke script imports the pinned submodule's Python API directly from `third_party/da3/src`, not from an older installed clone. It runs `depth-anything/DA3MONO-LARGE` on `third_party/da3/assets/examples/SOH/000.png`, with `process_res=504`, and uses the upstream `depth_vis` exporter. The environment must already have DA3's Python dependencies and an available CUDA device; this script does not install packages.

Paths in the YAML are resolved against the ARBiM-Learning repository root, regardless of the command's working directory. This is **not** an IL rollout or failure detector.

## Outputs and model cache

```text
outputs/experiments/da3/
├── hf_cache/                         # Local HF_HOME; model downloads here
│   └── hub/models--depth-anything--DA3MONO-LARGE/
└── official_mono_<timestamp>/
    ├── depth_vis/0000.jpg            # Official side-by-side RGB/depth export
    ├── depth.npy                     # Raw [1,H,W] depth prediction
    ├── result.json                   # Shapes, paths, load/inference timings
    └── config_resolved.yaml
```

`/outputs/*` is already ignored by the repository's `.gitignore`. No model weights or generated images should be committed.

Unlike the previous standalone-repository run, this script *explicitly* sets `HF_HOME` and `HF_HUB_CACHE` to the project-local output cache **before** importing DA3. Its first run therefore downloads the checkpoint into that new cache instead of silently reusing a model in the old user-global cache.

To delete just the project-local model snapshot and download again:

```bash
python experiments/da3/inference.py --redownload
```

`--redownload` clears only the `models--depth-anything--DA3MONO-LARGE` directory within this experiment's HF Hub cache. Hugging Face's separate Xet chunk cache may still reuse already-downloaded chunks; no user-global caches are affected.

## Previous standalone cache (manual cleanup)

The old run used Hugging Face caching, but its absolute path was not recorded. Inspect the old/current user-level Hub cache before deleting anything:

```bash
python - <<'PY'
from pathlib import Path
from huggingface_hub.constants import HF_HUB_CACHE
cache = Path(HF_HUB_CACHE) / 'models--depth-anything--DA3MONO-LARGE'
print(cache, '(exists:', cache.exists(), ')')
PY
```

If you want to remove that original copy too, verify the printed path and remove **that exact model directory only** (for example with `rm -rI -- /absolute/verified/path`). Do not remove the entire Hugging Face cache; it may contain unrelated models.

## Planned next steps (not implemented)

- `eval.py` remains a placeholder for a future experiment combining DA3 with the existing ARBiM rollout hooks.
- Do not add an `experiments/da3/src/` package for a single smoke script. Split helpers into a local module only when multiple experiments need to share them.
- Source code pinned as a submodule does not pin the Hugging Face model-weight revision; pin the HF revision separately for strict future reproducibility.
