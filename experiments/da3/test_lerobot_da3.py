#!/usr/bin/env python3
"""Extract LeRobot RGB frames and run a static DA3 depth inference test."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frames", type=int, nargs="+", default=[0])
    parser.add_argument("--camera", default="observation.images.left_wrist")
    parser.add_argument("--model", default="depth-anything/DA3MONO-LARGE")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--output-path", "--output-dir", dest="output_path", type=Path,
        default=Path("outputs/da3_static_test"),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from depth_anything_3.api import DepthAnything3

    dataset_root = args.dataset_root.expanduser().resolve()
    dataset = LeRobotDataset(
        repo_id=dataset_root.name,
        root=dataset_root,
        episodes=[args.episode],
    )
    if args.camera not in dataset.features:
        raise ValueError(f"Camera not found: {args.camera}; available: {list(dataset.features)}")
    if min(args.frames) < 0 or max(args.frames) >= len(dataset):
        raise IndexError(f"Episode {args.episode} contains {len(dataset)} frames")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    output = (
        args.output_path.expanduser()
        / f"episode_{args.episode:06d}"
        / args.camera.split(".")[-1]
    )
    output.mkdir(parents=True, exist_ok=True)

    model = DepthAnything3.from_pretrained(args.model).to(device).eval()

    for frame_index in args.frames:
        rgb = dataset[frame_index][args.camera]
        rgb = rgb.cpu().numpy() if isinstance(rgb, torch.Tensor) else np.asarray(rgb)
        if rgb.shape[0] == 3:  # LeRobot: [C, H, W] -> [H, W, C]
            rgb = np.moveaxis(rgb, 0, -1)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * 255 if rgb.max() <= 1 else rgb, 0, 255).astype(np.uint8)

        stem = f"frame_{frame_index:06d}"
        image_path = output / f"{stem}_rgb.png"
        Image.fromarray(rgb).save(image_path)

        # Single-frame inference; save native depth values, not a grayscale image.
        with torch.inference_mode():
            prediction = model.inference(image=[str(image_path)])

        depth = np.asarray(prediction.depth)
        if depth.ndim != 3 or depth.shape[0] != 1:
            raise ValueError(f"Unexpected DA3 depth shape: {depth.shape}")
        np.save(output / f"{stem}_depth.npy", depth[0])
        print(f"Saved {stem}: RGB {rgb.shape}, depth {depth[0].shape}")

    print(f"Output: {output.resolve()}")


if __name__ == "__main__":
    main()
