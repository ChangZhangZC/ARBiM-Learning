"""Single-image DA3MONO-LARGE smoke test using the pinned official Python API.

This is an isolated research experiment, not a DuoBench/IL evaluation entrypoint.
Run from any working directory:
    python experiments/da3/inference.py --config configs/experiments/da3.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "depth-anything/DA3MONO-LARGE"


def repo_path(value: str) -> Path:
    """Resolve experiment paths relative to the ARBiM-Learning repository root."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="DA3 official single-image example in ARBiM-Learning")
    parser.add_argument(
        "--config",
        default="configs/experiments/da3.yaml",
        help="YAML configuration path (relative to the repository root)",
    )
    parser.add_argument(
        "--redownload",
        action="store_true",
        help="Delete only this model's project-local HF Hub cache before loading it",
    )
    args = parser.parse_args()

    import yaml

    config_path = repo_path(args.config)
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("DA3 configuration must be a mapping.")

    model_cfg = config["model"]
    model_id = model_cfg["id"]
    if model_id != MODEL_ID:
        raise ValueError(f"This official-example test supports only {MODEL_ID!r}.")
    device = model_cfg.get("device", "cuda:0")
    process_res = model_cfg.get("process_res", 504)
    if type(process_res) is not int or process_res <= 0:
        raise ValueError("model.process_res must be a positive integer.")

    image_path = repo_path(config["input"]["image"])
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Official example image not found: {image_path}. "
            "Run 'git submodule update --init third_party/da3'."
        )

    output_root = repo_path(config["output"]["root"])
    cache_root = output_root / "hf_cache"
    output_root.mkdir(parents=True, exist_ok=True)

    # Set these BEFORE importing DA3 or huggingface_hub. This explicitly avoids
    # silently reusing an old user-global HF cache.
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_HUB_CACHE"] = str(cache_root / "hub")
    os.environ["HF_XET_CACHE"] = str(cache_root / "xet")

    model_cache = cache_root / "hub" / "models--depth-anything--DA3MONO-LARGE"
    if args.redownload and model_cache.is_dir():
        if model_cache.is_symlink():
            raise RuntimeError(f"Refusing to remove symlinked model cache: {model_cache}")
        shutil.rmtree(model_cache)

    # Import the code from THIS pinned submodule, not a previously installed DA3 checkout.
    da3_src = REPO_ROOT / "third_party" / "da3" / "src"
    if not (da3_src / "depth_anything_3" / "api.py").is_file():
        raise FileNotFoundError("DA3 submodule is not initialized; run git submodule update --init third_party/da3")
    sys.path.insert(0, str(da3_src))

    import numpy as np
    import torch
    from depth_anything_3.api import DepthAnything3

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but not available: {device}")

    run_dir = output_root / f"official_mono_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir.mkdir(parents=False, exist_ok=False)

    loaded_at = time.perf_counter()
    model = DepthAnything3.from_pretrained(model_id).to(device)
    model.eval()
    load_seconds = time.perf_counter() - loaded_at

    started = time.perf_counter()
    prediction = model.inference(
        image=[str(image_path)],
        process_res=process_res,
        export_dir=str(run_dir),
        export_format="depth_vis",
    )
    inference_seconds = time.perf_counter() - started

    depth = np.asarray(prediction.depth)
    if depth.ndim != 3 or depth.shape[0] != 1:
        raise ValueError(f"Unexpected DA3 depth shape: {depth.shape}")
    np.save(run_dir / "depth.npy", depth)
    visualization = run_dir / "depth_vis" / "0000.jpg"
    if not visualization.is_file():
        raise RuntimeError(f"DA3 depth visualization was not exported: {visualization}")

    record = {
        "model_id": model_id,
        "da3_source": str(da3_src),
        "image": str(image_path),
        "device": str(device),
        "process_res": process_res,
        "cache_root": str(cache_root),
        "run_dir": str(run_dir),
        "depth_shape": list(depth.shape),
        "is_metric": bool(getattr(prediction, "is_metric", False)),
        "model_load_seconds": round(load_seconds, 4),
        "inference_and_export_seconds": round(inference_seconds, 4),
        "depth_visualization": str(visualization),
        "depth_array": str(run_dir / "depth.npy"),
    }
    with (run_dir / "result.json").open("w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    with (run_dir / "config_resolved.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
    print(json.dumps(record, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
