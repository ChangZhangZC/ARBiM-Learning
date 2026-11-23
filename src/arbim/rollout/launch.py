"""Small interactive entrypoint for the existing DuoBench evaluation CLI.

No task-specific rollout logic lives here; eval.py remains the sole runner.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path("configs/rollout/hinge_chest.yaml")


def _choice(label: str, options: dict[str, str]) -> str:
    """Present a short fixed menu; unvalidated tasks/modes are not advertised."""
    print(label)
    for key, value in options.items():
        print(f"  {key}) {value}")
    while True:
        answer = input("Choice: ").strip()
        if answer in options:
            return options[answer]
        print(f"Invalid choice: {answer!r}.")


def _read_config(path: Path, task: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Rollout YAML does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or any(
        not isinstance(config.get(section), dict)
        for section in ("env", "policy", "rollout", "output")
    ):
        raise ValueError("Rollout YAML must contain env, policy, rollout and output mappings.")
    actual = config["env"].get("task")
    if actual != task:
        raise ValueError(f"Selected task {task!r} differs from YAML env.task={actual!r}.")
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ARBiM DuoBench evaluation launcher")
    parser.add_argument("--task", choices=("hinge_chest",), help="Supported DuoBench task")
    parser.add_argument("--mode", choices=("normal", "routed"), help="Evaluation mode")
    parser.add_argument("--config", type=Path, help="Rollout YAML file")
    parser.add_argument("--yes", action="store_true", help="Use defaults and skip confirmation")
    args = parser.parse_args(argv)

    try:
        task = args.task or (
            "hinge_chest" if args.yes else _choice("DuoBench task:", {"1": "hinge_chest"})
        )
        mode = args.mode or (
            "normal" if args.yes else _choice(
                "Rollout mode:", {"1": "normal", "2": "routed"
            })
        )
        if mode == "routed":
            print("Routed Rollout is not implemented; no evaluation was started.")
            return 2

        if args.config is not None:
            path = args.config
        elif args.yes:
            path = DEFAULT_CONFIG
        else:
            raw_path = ""
            while not raw_path:
                raw_path = input("Rollout YAML: ").strip()
                if not raw_path:
                    print("Please enter a YAML path.")
            path = Path(raw_path)
        path = path.expanduser().resolve()
        config = _read_config(path, task)
        print(f"\nTask: {task}\nMode: {mode}\nYAML: {path}")
        print("\nParsed YAML preview (eval.py writes final config_resolved.yaml at runtime):")
        print(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        if not args.yes and input("Start evaluation? [y/N]: ").strip().lower() not in ("y", "yes"):
            print("Evaluation cancelled.")
            return 0

    except (KeyboardInterrupt, EOFError):
        print("\nEvaluation cancelled.")
        return 130

    # Run the original evaluation CLI and return its exit status unchanged.
    result = subprocess.run(
        [sys.executable, "-m", "arbim.rollout.eval", "--config", str(path)],
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
