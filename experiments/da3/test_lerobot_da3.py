#!/usr/bin/env python3
"""Static LeRobot RGB -> Depth Anything 3 inference smoke test.

Run from the ARBiM-Learning repository root:

    python experiments/da3/test_lerobot_da3.py \
        --dataset-root /path/to/lerobot_dataset \
        --episode 0 \
        --frames 0 20 40 \
        --camera observation.images.wrist_cam_l \
        --model depth-anything/DA3MONO-LARGE \
        --output-path outputs/da3_static_test

Requires LeRobot, DA3 and their native dependencies installed in the environment.
Does not modify the dataset or any training pipeline.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_CAMERA = "observation.images.wrist_cam_l"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Local LeRobot dataset directory")
    parser.add_argument("--episode", type=int, default=0, help="Episode index in the dataset")
    parser.add_argument(
        "--frames", type=int, nargs="+", default=[0],
        help="0-based frame indices within the selected episode",
    )
    parser.add_argument(
        "--camera", default=DEFAULT_CAMERA,
        help="LeRobot visual feature key, e.g. observation.images.wrist_cam_l",
    )
    parser.add_argument(
        "--model", default="depth-anything/DA3MONO-LARGE",
        help="DA3 Hugging Face model ID or local checkpoint directory",
    )
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default="auto",
    )
    parser.add_argument(
        "--output-path", "--output-dir", dest="output_path", type=Path,
        default=Path("outputs/da3_static_test"),
        help="Root directory for saved RGB, depth, and visualizations",
    )
    return parser.parse_args()


def image_to_rgb_uint8(value):
    """Convert a LeRobot [3,H,W] float tensor or [H,W,3] image into RGB uint8."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)

    if array.ndim != 3:
        raise ValueError(f"Expected a 3D RGB image, got shape={array.shape}")
    if array.shape[0] == 3:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] != 3:
        raise ValueError(f"Expected RGB with three channels, got shape={array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("Input RGB image contains NaN/Inf")

    if array.dtype == np.uint8:
        return array

    array = array.astype(np.float32)
    if array.min() >= 0.0 and array.max() <= 1.0:
        array = array * 255.0
    elif array.min() < 0.0 or array.max() > 255.0:
        raise ValueError(
            f"Unexpected RGB range [{array.min()}, {array.max()}]; "
            "expected [0,1] or [0,255] (not normalized policy tensors)"
        )
    return np.clip(np.rint(array), 0, 255).astype(np.uint8)


def depth_visualization(depth):
    """8-bit per-image percentile visualization; NEVER a metric depth encoding."""
    finite = np.isfinite(depth)
    if not finite.any():
        raise ValueError("DA3 returned a depth map with no finite values")
    valid = depth[finite]
    low, high = np.percentile(valid, [2, 98])
    if high <= low:
        grayscale = np.zeros(depth.shape, dtype=np.uint8)
    else:
        normalized = np.clip((depth - low) / (high - low), 0, 1)
        grayscale = np.where(finite, normalized, 0)
        grayscale = (grayscale * 255).astype(np.uint8)
    return Image.fromarray(grayscale, mode="L")


def main():
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not (dataset_root / "meta").is_dir():
        raise FileNotFoundError(f"Not a local LeRobot dataset (missing meta/): {dataset_root}")
    if args.episode < 0 or any(i < 0 for i in args.frames):
        raise ValueError("--episode and --frames must be non-negative")
    if len(set(args.frames)) != len(args.frames):
        raise ValueError("--frames cannot contain duplicates")

    # Import lazily: --help works without heavy robot/model dependencies.
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from depth_anything_3.api import DepthAnything3

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    # The episodes filter makes frame 0 the first frame of this episode.
    dataset = LeRobotDataset(
        repo_id=dataset_root.name, root=dataset_root, episodes=[args.episode]
    )
    cameras = list(getattr(dataset.meta, "camera_keys", []))
    if args.camera not in dataset.features:
        raise KeyError(
            f"Camera {args.camera!r} is not in dataset features. "
            f"Available camera keys: {cameras}"
        )
    if cameras and args.camera not in cameras:
        raise ValueError(f"{args.camera!r} is not an image/video camera key: {cameras}")
    if max(args.frames) >= len(dataset):
        raise IndexError(
            f"Episode {args.episode} has {len(dataset)} frames, "
            f"requested frame {max(args.frames)}"
        )

    # Only write a small, fixed set of selected RGB images.
    camera_name = args.camera.split(".")[-1]
    output = args.output_path.expanduser() / f"episode_{args.episode:06d}" / camera_name
    output.mkdir(parents=True, exist_ok=True)
    image_paths = []
    rgb_images = []
    for frame_index in args.frames:
        frame = dataset[frame_index]
        rgb = image_to_rgb_uint8(frame[args.camera])
        rgb_image = Image.fromarray(rgb, mode="RGB")
        rgb_path = output / f"frame_{frame_index:06d}_rgb.png"
        rgb_image.save(rgb_path)
        image_paths.append(str(rgb_path))
        rgb_images.append(rgb_image)

    print(f"Loaded {len(image_paths)} image(s) from {dataset_root}")
    print(f"Camera: {args.camera}, episode: {args.episode}, frames: {args.frames}")
    print(f"DA3 model: {args.model} on {device}")
    model = DepthAnything3.from_pretrained(args.model).to(device).eval()
    with torch.inference_mode():
        prediction = model.inference(image=image_paths)

    depth_maps = np.asarray(prediction.depth)
    if depth_maps.ndim != 3 or depth_maps.shape[0] != len(image_paths):
        raise ValueError(
            f"Unexpected DA3 depth shape {depth_maps.shape}; "
            f"expected (N,H,W), N={len(image_paths)}"
        )

    manifest = {
        "dataset_root": str(dataset_root),
        "episode": args.episode,
        "camera": args.camera,
        "model": args.model,
        "device": device,
        "note": "Depth visualization is normalized per frame; not calibrated metric depth.",
        "frames": [],
    }
    for frame_index, rgb_image, depth in zip(args.frames, rgb_images, depth_maps):
        stem = f"frame_{frame_index:06d}"
        depth = np.asarray(depth, dtype=np.float32)
        if not np.isfinite(depth).any():
            raise ValueError(f"Frame {frame_index}: DA3 depth contains no finite pixels")

        depth_file = output / f"{stem}_depth.npy"
        vis_file = output / f"{stem}_depth_vis.png"
        compare_file = output / f"{stem}_compare.png"
        np.save(depth_file, depth)

        depth_vis = depth_visualization(depth)
        depth_vis.save(vis_file)
        comparison = Image.new("RGB", (rgb_image.width * 2, rgb_image.height))
        comparison.paste(rgb_image, (0, 0))
        comparison.paste(depth_vis.convert("RGB").resize(rgb_image.size), (rgb_image.width, 0))
        comparison.save(compare_file)

        manifest["frames"].append(
            {
                "frame_index": frame_index,
                "rgb": f"{stem}_rgb.png",
                "depth_npy": depth_file.name,
                "depth_visualization": vis_file.name,
                "side_by_side": compare_file.name,
                "depth_shape": list(depth.shape),
                "depth_min": float(depth[np.isfinite(depth)].min()),
                "depth_max": float(depth[np.isfinite(depth)].max()),
            }
        )
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved RGB/depth/visualizations to: {output.resolve()}")


if __name__ == "__main__":
    main()
