#!/usr/bin/env python3
"""
Discover MimicGen image tasks, check processed data under data_dir/mimicgen,
download missing core HDF5 files, and run pretrain + finetune NPZ conversion.

Tasks run in parallel across environments; within each task, download ->
pretrain -> finetune are sequential.

Examples (from HiRE-Dice_RL repo root):
  # Status only (the three supported MimicGen D0 tasks)
  python scripts/dataset/prepare_mimicgen_datasets.py --check-only

  # Prepare all configured tasks, 2 in parallel
  python scripts/dataset/prepare_mimicgen_datasets.py --jobs 2

  # Specific tasks
  python scripts/dataset/prepare_mimicgen_datasets.py --tasks stack_three threading three_piece_assembly
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MIMICGEN_ROOT = Path(os.environ.get("MIMICGEN_ROOT", REPO_ROOT / "external" / "mimicgen"))
DEFAULT_DATA_DIR = Path(
    os.environ.get("HIRE_DATA_DIR", os.environ.get("DICE_RL_DATA_DIR", REPO_ROOT / "data_dir"))
)
DEFAULT_CAMERAS = ("agentview", "robot0_eye_in_hand")

# Default demonstration subsets match scripts/launch.py.
DEFAULT_CORE_HDF5: Dict[str, str] = {
    "stack_three": "stack_three_d0",
    "threading": "threading_d0",
    "three_piece_assembly": "three_piece_assembly_d0",
}
DEFAULT_DEMOS = {"stack_three": 200, "threading": 30, "three_piece_assembly": 200}


@dataclass(frozen=True)
class TaskSpec:
    env_name: str
    core_hdf5: str
    cameras: Tuple[str, ...] = DEFAULT_CAMERAS
    max_episodes: int = -1  # -1 = no limit in process_robomimic_dataset.py


def _load_mimicgen_registry(mimicgen_root: Path) -> Dict[str, Dict[str, dict]]:
    mimicgen_root = str(mimicgen_root)
    if mimicgen_root not in sys.path:
        sys.path.insert(0, mimicgen_root)
    from mimicgen import DATASET_REGISTRY  # noqa: WPS433

    return DATASET_REGISTRY


def _default_core_hdf5(env_name: str, registry_core: Dict[str, dict]) -> str:
    if env_name in DEFAULT_CORE_HDF5:
        candidate = DEFAULT_CORE_HDF5[env_name]
        if candidate in registry_core:
            return candidate
    preferred = f"{env_name}_d0"
    if preferred in registry_core:
        return preferred
    matches = sorted(t for t in registry_core if t.startswith(env_name + "_"))
    if matches:
        return matches[0]
    raise KeyError(
        f"No core HDF5 registered for env '{env_name}'. "
        f"Known core tasks: {sorted(registry_core.keys())[:8]}..."
    )


def _read_max_episodes_from_cfg(env_name: str) -> Optional[int]:
    """Return the released demonstration subset size."""
    return DEFAULT_DEMOS.get(env_name)


def discover_tasks_from_cfg(registry_core: Dict[str, dict]) -> List[TaskSpec]:
    return [TaskSpec(env_name=name, core_hdf5=_default_core_hdf5(name, registry_core),
                     max_episodes=DEFAULT_DEMOS[name]) for name in DEFAULT_CORE_HDF5]


def dataset_paths(data_dir: Path, env_name: str) -> Dict[str, Path]:
    base = data_dir / "mimicgen" / f"{env_name}-img"
    return {
        "pretrain_train": base / "ph_pretrain" / "train.npz",
        "pretrain_norm": base / "ph_pretrain" / "normalization.npz",
        "finetune_train": base / "ph_finetune" / "train.npz",
        "finetune_norm": base / "ph_finetune" / "normalization.npz",
    }


def check_task_complete(data_dir: Path, env_name: str) -> Tuple[bool, List[str]]:
    paths = dataset_paths(data_dir, env_name)
    missing = [k for k, p in paths.items() if not p.is_file()]
    return len(missing) == 0, missing


def hdf5_path(mimicgen_root: Path, core_hdf5: str) -> Path:
    return mimicgen_root / "datasets" / "core" / f"{core_hdf5}.hdf5"


def download_core_hdf5(
    mimicgen_root: Path,
    core_hdf5: str,
    python_exe: str,
    dry_run: bool,
) -> None:
    hdf5 = hdf5_path(mimicgen_root, core_hdf5)
    if hdf5.is_file():
        print(f"  [download] already exists: {hdf5}")
        return
    download_script = mimicgen_root / "mimicgen" / "scripts" / "download_datasets.py"
    if not download_script.is_file():
        raise FileNotFoundError(f"MimicGen download script not found: {download_script}")
    cmd = [
        python_exe,
        str(download_script),
        "--dataset_type",
        "core",
        "--tasks",
        core_hdf5,
        "--download_dir",
        str(mimicgen_root / "datasets"),
    ]
    if dry_run:
        print(f"  [download] dry-run: {' '.join(cmd)}")
        return
    print(f"  [download] {core_hdf5} -> {hdf5}")
    subprocess.run(cmd, check=True, cwd=str(mimicgen_root))


def run_process(
    python_exe: str,
    hdf5: Path,
    save_dir: Path,
    cameras: Sequence[str],
    truncate: bool,
    max_episodes: int,
    dry_run: bool,
) -> None:
    cmd = [
        python_exe,
        str(REPO_ROOT / "scripts" / "dataset" / "process_robomimic_dataset.py"),
        "--load_path",
        str(hdf5),
        "--save_dir",
        str(save_dir),
        "--normalize",
        "--cameras",
        *cameras,
        "--use_hdf5_images",
    ]
    if truncate:
        cmd.append("--truncate")
    if max_episodes > 0:
        cmd.extend(["--max_episodes", str(max_episodes)])
    label = "finetune" if truncate else "pretrain"
    if dry_run:
        print(f"  [process:{label}] dry-run: {' '.join(cmd)}")
        return
    print(f"  [process:{label}] {save_dir}")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def process_one_task(
    spec: TaskSpec,
    mimicgen_root: str,
    data_dir: str,
    python_exe: str,
    dry_run: bool,
    force: bool,
    max_episodes_override: int,
) -> Tuple[str, str]:
    """Worker entrypoint. Returns (env_name, status)."""
    mg_root = Path(mimicgen_root)
    d_dir = Path(data_dir)
    max_ep = max_episodes_override if max_episodes_override > 0 else spec.max_episodes

    complete, missing = check_task_complete(d_dir, spec.env_name)
    if complete and not force:
        return spec.env_name, "skip_complete"

    hdf5 = hdf5_path(mg_root, spec.core_hdf5)
    pretrain_dir = d_dir / "mimicgen" / f"{spec.env_name}-img" / "ph_pretrain"
    finetune_dir = d_dir / "mimicgen" / f"{spec.env_name}-img" / "ph_finetune"

    print(f"\n=== task={spec.env_name} core={spec.core_hdf5} max_episodes={max_ep} ===")
    if missing:
        print(f"  missing: {missing}")

    download_core_hdf5(mg_root, spec.core_hdf5, python_exe, dry_run)
    if not dry_run and not hdf5.is_file():
        return spec.env_name, f"error_missing_hdf5:{hdf5}"

    need_pretrain = force or not (pretrain_dir / "train.npz").is_file()
    need_finetune = force or not (finetune_dir / "train.npz").is_file()

    if need_pretrain:
        run_process(
            python_exe, hdf5, pretrain_dir, spec.cameras, False, max_ep, dry_run
        )
    else:
        print(f"  [process:pretrain] skip (exists) {pretrain_dir}")

    if need_finetune:
        run_process(
            python_exe, hdf5, finetune_dir, spec.cameras, True, max_ep, dry_run
        )
    else:
        print(f"  [process:finetune] skip (exists) {finetune_dir}")

    if dry_run:
        return spec.env_name, "dry_run"
    complete, missing = check_task_complete(d_dir, spec.env_name)
    if complete:
        return spec.env_name, "ok"
    return spec.env_name, f"incomplete:{missing}"


def print_status_table(
    specs: Sequence[TaskSpec],
    data_dir: Path,
    mimicgen_root: Path,
) -> List[TaskSpec]:
    """Print status; return specs that are not fully prepared."""
    incomplete: List[TaskSpec] = []
    print(f"\n{'env_name':<22} {'core_hdf5':<28} {'hdf5':^6} {'pretrain':^10} {'finetune':^10}")
    print("-" * 82)
    for spec in specs:
        paths = dataset_paths(data_dir, spec.env_name)
        h5 = hdf5_path(mimicgen_root, spec.core_hdf5)
        h5_ok = "yes" if h5.is_file() else "no"
        pre_ok = "yes" if paths["pretrain_train"].is_file() else "no"
        ft_ok = "yes" if paths["finetune_train"].is_file() else "no"
        complete, _ = check_task_complete(data_dir, spec.env_name)
        mark = "OK" if complete else "NEED"
        print(
            f"{spec.env_name:<22} {spec.core_hdf5:<28} {h5_ok:^6} {pre_ok:^10} {ft_ok:^10}  [{mark}]"
        )
        if not complete:
            incomplete.append(spec)
    return incomplete


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tasks",
        choices=sorted(DEFAULT_CORE_HDF5),
        nargs="+",
        default=None,
        help="Env names to process (default: all three supported MimicGen D0 tasks)",
    )
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Print status table and exit (no download/process)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without downloading or processing",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run processing even if train.npz already exists",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=int(os.environ.get("PREPARE_JOBS", "2")),
        help="Max parallel tasks (default: 2, env PREPARE_JOBS)",
    )
    p.add_argument(
        "--max-episodes",
        type=int,
        default=int(os.environ.get("MAX_EPISODES", "-1")),
        help="Override max episodes for all tasks (-1 = use the task recipe)",
    )
    p.add_argument(
        "--mimicgen-root",
        type=Path,
        default=DEFAULT_MIMICGEN_ROOT,
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )
    p.add_argument(
        "--python",
        default=os.environ.get("DICE_PY", sys.executable),
        help="Python executable (default: DICE_PY or current interpreter)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    registry = _load_mimicgen_registry(args.mimicgen_root)
    registry_core = registry["core"]

    specs = discover_tasks_from_cfg(registry_core)

    if args.tasks:
        wanted = set(args.tasks)
        specs = [s for s in specs if s.env_name in wanted]
        unknown = wanted - {s.env_name for s in specs}
        for env in sorted(unknown):
            specs.append(
                TaskSpec(
                    env_name=env,
                    core_hdf5=_default_core_hdf5(env, registry_core),
                    max_episodes=_read_max_episodes_from_cfg(env) or -1,
                )
            )
        specs = sorted(specs, key=lambda s: s.env_name)

    if not specs:
        print("No tasks discovered.", file=sys.stderr)
        return 1

    print(f"Repo:          {REPO_ROOT}")
    print(f"MimicGen root: {args.mimicgen_root}")
    print(f"Data dir:      {args.data_dir}")
    print(f"Python:        {args.python}")
    print(f"Tasks ({len(specs)}): {[s.env_name for s in specs]}")

    incomplete = print_status_table(specs, args.data_dir, args.mimicgen_root)
    if args.check_only:
        return 0 if not incomplete else 1

    if not incomplete and not args.force:
        print("\nAll tasks complete. Use --force to reprocess.")
        return 0

    to_run = specs if args.force else incomplete
    if not to_run:
        return 0

    print(f"\nPreparing {len(to_run)} task(s) with jobs={args.jobs}")

    worker_args = [
        (
            spec,
            str(args.mimicgen_root),
            str(args.data_dir),
            args.python,
            args.dry_run,
            args.force,
            args.max_episodes,
        )
        for spec in to_run
    ]

    results: Dict[str, str] = {}
    if args.jobs <= 1:
        for wa in worker_args:
            env, status = process_one_task(*wa)
            results[env] = status
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(process_one_task, *wa): wa[0].env_name for wa in worker_args
            }
            for fut in as_completed(futures):
                env = futures[fut]
                try:
                    _, status = fut.result()
                    results[env] = status
                except Exception as exc:
                    results[env] = f"error:{exc}"

    print("\n=== Summary ===")
    failed = False
    for env in sorted(results.keys()):
        status = results[env]
        print(f"  {env}: {status}")
        if not status.startswith(("ok", "skip_complete", "dry_run")):
            failed = True

    print_status_table(specs, args.data_dir, args.mimicgen_root)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
