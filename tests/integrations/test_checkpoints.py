"""Checkpoint selection, complete downloads, and launcher failure behavior."""
import argparse
import io
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts import download_checkpoints as downloads
from scripts import launch


def args(task="stack_three", checkpoint=None, stage="finetune"):
    return argparse.Namespace(task=task, checkpoint=checkpoint, stage=stage, reward="hire", seed=42)


@pytest.mark.parametrize("task,epoch", [("stack_three", 75), ("threading", 100), ("three_piece_assembly", 75)])
def test_default_checkpoint_matches_published_task(task, epoch, tmp_path, monkeypatch):
    monkeypatch.setenv("HIRE_CHECKPOINT_DIR", str(tmp_path / "weights with spaces"))
    command = launch.build_command(args(task), [])
    expected = tmp_path / "weights with spaces" / task / "checkpoint" / f"state_{epoch}.pt"
    with initialize_config_dir(config_dir=str(launch.ROOT / "configs"), version_base=None):
        cfg = compose(config_name=f"experiment/{task}_finetune", overrides=command[4:])
    assert cfg.base_policy_path == str(expected.resolve())
    assert cfg.model.pretrained_flow_policy_path == str(expected.resolve())
    assert cfg.expert_dataset.max_n_episodes == downloads.CHECKPOINTS[task]["demonstrations"]


def test_tool_hang_requires_custom_checkpoint_but_pretraining_is_available():
    with pytest.raises(ValueError, match="No released checkpoint"):
        launch.build_command(args("tool_hang"), [])
    assert launch.build_command(args("tool_hang", "/tmp/custom.pt"), [])
    assert launch.build_command(args("tool_hang", stage="pretrain"), [])


def test_custom_checkpoint_takes_precedence_over_release_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HIRE_CHECKPOINT_DIR", str(tmp_path / "release"))
    custom = tmp_path / "own BC run/checkpoint/state_50.pt"
    command = launch.build_command(args(checkpoint=str(custom)), [])
    assert any(str(custom.resolve()) in value for value in command)
    assert not any(str(tmp_path / "release") in value for value in command)
    with pytest.raises(ValueError, match="Use --checkpoint"):
        launch.build_command(args(), ["base_policy_path=/tmp/another.pt"])


@pytest.fixture
def tiny_bundle(monkeypatch):
    payloads = {
        "stack_three/checkpoint/state_75.pt": b"checkpoint payload",
        "stack_three/.hydra/config.yaml": b"seed: 42\n",
    }
    entry = {
        "checkpoint": "stack_three/checkpoint/state_75.pt",
        "config": "stack_three/.hydra/config.yaml",
    }
    monkeypatch.setitem(downloads.CHECKPOINTS, "stack_three", entry)
    return payloads, entry


def test_download_selects_only_required_files_and_reuses_downloads(tmp_path, monkeypatch, tiny_bundle):
    payloads, _ = tiny_bundle
    requested = []
    def serve(request, timeout):
        requested.append(request.full_url)
        prefix = f"https://huggingface.co/{downloads.RELEASE['repo_id']}/resolve/main/"
        assert request.full_url.startswith(prefix)
        return io.BytesIO(payloads[request.full_url.removeprefix(prefix)])
    monkeypatch.setattr(downloads, "urlopen", serve)
    checkpoint = downloads.download_task("stack_three", tmp_path)
    assert checkpoint.is_file()
    assert len(requested) == 2
    downloads.download_task("stack_three", tmp_path)
    assert len(requested) == 2
    assert not list(tmp_path.rglob("*.part"))


def test_interrupted_download_does_not_replace_existing_file(tmp_path, monkeypatch, tiny_bundle):
    _, entry = tiny_bundle
    target = tmp_path / entry["checkpoint"]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing file")
    def failed(*args, **kwargs):
        raise OSError("connection interrupted")
    monkeypatch.setattr(downloads, "urlopen", failed)
    with pytest.raises(OSError, match="connection interrupted"):
        downloads.download_task("stack_three", tmp_path, force=True)
    assert target.read_bytes() == b"existing file"
    assert not list(tmp_path.rglob("*.part"))


def test_empty_hydra_config_is_reported(tmp_path, tiny_bundle):
    payloads, entry = tiny_bundle
    for name, content in payloads.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (tmp_path / entry["config"]).write_text("")
    with pytest.raises(FileNotFoundError, match="Missing or empty"):
        downloads.check_bundle("stack_three", tmp_path)


def test_dry_run_does_not_access_network_or_create_files(tmp_path, monkeypatch):
    def unexpected(*a, **k):
        pytest.fail("Dry run accessed the network")
    monkeypatch.setattr(downloads, "urlopen", unexpected)
    monkeypatch.setattr(sys, "argv", ["download_checkpoints.py", "--tasks", "stack_three", "--output-dir", str(tmp_path), "--dry-run"])
    assert downloads.main() == 0
    assert list(tmp_path.iterdir()) == []


def test_missing_default_bundle_stops_before_training(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HIRE_CHECKPOINT_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["launch.py", "finetune", "stack_three"])
    monkeypatch.setattr(launch.subprocess, "call", lambda *a, **k: pytest.fail("Training started without checkpoint files"))
    with pytest.raises(SystemExit) as error:
        launch.main()
    assert error.value.code == 2
    assert "download_checkpoints.py --tasks stack_three" in capsys.readouterr().err


def test_seed_override_uses_the_task_aware_launcher_option():
    with pytest.raises(ValueError, match="Use --seed"):
        launch.build_command(args("threading"), ["seed=123"])
