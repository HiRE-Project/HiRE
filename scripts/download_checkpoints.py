#!/usr/bin/env python3
"""Download the released BC checkpoints and Hydra configs for HiRE finetuning."""
import argparse
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RELEASE = json.loads((ROOT / "configs/checkpoints.json").read_text())
CHECKPOINTS = RELEASE["checkpoints"]


def checkpoint_directory():
    return Path(os.environ.get("HIRE_CHECKPOINT_DIR", ROOT / "checkpoints/HiRE-release")).expanduser().resolve()


def checkpoint_path(task):
    if task not in CHECKPOINTS:
        raise ValueError(
            f"No released checkpoint is available for {task}. "
            "Pass --checkpoint /path/to/bc_run/checkpoint/state_<epoch>.pt."
        )
    return checkpoint_directory() / CHECKPOINTS[task]["checkpoint"]


def bundle_files(task):
    item = CHECKPOINTS[task]
    return [item["checkpoint"], item["config"]]


def check_bundle(task, directory=None):
    directory = checkpoint_directory() if directory is None else Path(directory)
    for relative in bundle_files(task):
        path = directory / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty checkpoint bundle file: {path}")
    return directory / CHECKPOINTS[task]["checkpoint"]


def download_file(relative, directory, force=False):
    destination = Path(directory) / relative
    if not force and destination.is_file() and destination.stat().st_size:
        print(f"Using existing: {destination}", flush=True)
        return
    url = f"https://huggingface.co/{RELEASE['repo_id']}/resolve/main/" + quote(relative, safe="/")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as handle:
            temporary = Path(handle.name)
            print(f"Downloading: {relative}", flush=True)
            with urlopen(Request(url, headers={"User-Agent": "HiRE-checkpoint-downloader"}), timeout=120) as response:
                for chunk in iter(lambda: response.read(8 * 1024 * 1024), b""):
                    handle.write(chunk)
        if temporary.stat().st_size == 0:
            raise OSError(f"Empty download: {relative}")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def download_task(task, directory=None, force=False):
    directory = checkpoint_directory() if directory is None else Path(directory).expanduser().resolve()
    for relative in bundle_files(task):
        download_file(relative, directory, force=force)
    return check_bundle(task, directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=CHECKPOINTS, default=list(CHECKPOINTS))
    parser.add_argument("--output-dir", type=Path, default=checkpoint_directory())
    parser.add_argument("--dry-run", action="store_true", help="List selected files without downloading")
    parser.add_argument("--force", action="store_true", help="Redownload existing files")
    args = parser.parse_args()
    directory = args.output_dir.expanduser().resolve()
    print(f"Source: {RELEASE['repo_id']}", flush=True)
    try:
        for task in dict.fromkeys(args.tasks):
            if args.dry_run:
                for relative in bundle_files(task):
                    print(directory / relative)
            else:
                print(f"Ready: {download_task(task, directory, force=args.force)}", flush=True)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Download failed: {exc}\nRerun the command to retry; completed files are reused.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
