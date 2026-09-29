#!/usr/bin/env python3
"""Launch one released task with explicit, reviewable Hydra overrides."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

if __package__:
    from .download_checkpoints import checkpoint_path, check_bundle
else:
    from download_checkpoints import checkpoint_path, check_bundle

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "stack_three": ("mimicgen", "diffusion", 200),
    "threading": ("mimicgen", "diffusion", 30),
    "three_piece_assembly": ("mimicgen", "diffusion", 200),
    "tool_hang": ("robomimic", "flow", 20),
}


def build_command(args, overrides):
    if any(value.lstrip("+~").split("=", 1)[0] == "base_policy_path" for value in overrides):
        raise ValueError("Use --checkpoint to select a custom BC checkpoint.")
    if any(value.lstrip("+~").split("=", 1)[0] == "seed" for value in overrides):
        raise ValueError("Use --seed so the launcher selects the matching task/seed settings.")
    _, _, demos = TASKS[args.task]
    if args.stage == "pretrain":
        config = f"{args.task}_pretrain"
        recipe = [f"train_dataset.max_n_episodes={demos}"]
    else:
        config = f"{args.task}_finetune"
        if args.task == "threading" and args.seed == 42:
            config = "threading_finetune_seed42"
        checkpoint = (Path(args.checkpoint).expanduser().resolve()
                      if args.checkpoint else checkpoint_path(args.task))
        # Quote for Hydra's grammar, independently from shell argument quoting.
        checkpoint_value = str(checkpoint).replace('\\', '\\\\').replace('"', '\\"')
        recipe = [
            f'base_policy_path="{checkpoint_value}"',
        ]
        recipe += [f"reward={args.reward}"]
    return [sys.executable, str(ROOT / "scripts/run.py"),
            f"--config-path={ROOT / 'configs'}",
            f"--config-name=experiment/{config}", f"seed={args.seed}", *recipe, *overrides]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["pretrain", "finetune"])
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--checkpoint", help="Override the released BC checkpoint; keep its sibling .hydra/config.yaml")
    parser.add_argument("--reward", choices=["hire", "sparse"], default="hire")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", choices=["disabled", "offline", "online"], help="W&B logging mode for the new run")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without starting training")
    parser.add_argument("--check", action="store_true", help="Check data, BC bundle, and device without starting training")
    parser.add_argument("--config-only", action="store_true", help="Print resolved Hydra configuration without loading models or data")
    args, overrides = parser.parse_known_args()
    if overrides and overrides[0] == "--":
        overrides = overrides[1:]
    if any("=" not in value for value in overrides):
        parser.error("Additional arguments must be Hydra key=value overrides, optionally after --.")
    try:
        cmd = build_command(args, overrides)
    except ValueError as exc:
        parser.error(str(exc))
    if args.config_only:
        cmd += ["--cfg", "job", "--resolve"]
    elif args.check:
        cmd += ["+check_inputs_only=true"]
    prefix = f"WANDB_MODE={args.wandb} " if args.wandb else ""
    print(prefix + shlex.join(cmd), flush=True)
    if args.dry_run:
        return 0
    if args.stage == "finetune" and not args.config_only:
        if args.checkpoint:
            checkpoint = Path(args.checkpoint).expanduser().resolve()
            if not checkpoint.is_file():
                parser.error(f"BC checkpoint not found: {checkpoint}")
            if not (checkpoint.parent.parent / ".hydra/config.yaml").is_file():
                parser.error("Keep the complete BC run directory: .hydra/config.yaml and checkpoint/*.pt.")
        else:
            try:
                check_bundle(args.task)
            except (OSError, ValueError) as exc:
                parser.error(
                    f"{exc}\nDownload the released bundle with: "
                    f"python scripts/download_checkpoints.py --tasks {args.task}\n"
                    "Alternatively, pass --checkpoint to use your own BC run."
                )
    run_env = os.environ.copy()
    run_env.setdefault("HIRE_DATA_DIR", run_env.get("DICE_RL_DATA_DIR", str(ROOT / "data_dir")))
    run_env.setdefault("DICE_RL_DATA_DIR", run_env["HIRE_DATA_DIR"])
    run_env.setdefault("HIRE_LOG_DIR", run_env.get("DICE_RL_LOG_DIR", str(ROOT / "log_dir")))
    run_env.setdefault("DICE_RL_LOG_DIR", run_env["HIRE_LOG_DIR"])
    if args.wandb:
        run_env["WANDB_MODE"] = args.wandb
    else:
        run_env.setdefault("WANDB_MODE", "disabled")
    run_env.setdefault("MUJOCO_GL", "egl")
    return subprocess.call(cmd, cwd=ROOT, env=run_env)


if __name__ == "__main__":
    raise SystemExit(main())
