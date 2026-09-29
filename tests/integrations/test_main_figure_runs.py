"""Compare executable task/seed presets with the exported main-figure W&B settings."""
import argparse
import json
from pathlib import Path
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from scripts.launch import ROOT, build_command
from hire_dice_rl.integration.config import configure_reward
from hire_dice_rl.agent.finetune.train_distill_residual_flow_agent import TrainDistillResidualFlowAgent
from hire_dice_rl.check_inputs import check_inputs

REFERENCE = json.loads((ROOT / "tests/fixtures/main_figure_runs.json").read_text())
CASES = [(task, run) for task, reference in REFERENCE.items() for run in reference["runs"]]


def compose_run(task, run, monkeypatch, tmp_path):
    for key in ("HIRE_DATA_DIR", "HIRE_LOG_DIR", "DICE_RL_DATA_DIR", "DICE_RL_LOG_DIR"):
        monkeypatch.setenv(key, str(tmp_path))
    args = argparse.Namespace(stage="finetune", task=task, seed=run["seed"], reward="hire",
                              checkpoint=str(tmp_path / "bc/checkpoint" / run["bc_checkpoint"]))
    command = build_command(args, [])
    config_name = next(value.split("=", 1)[1] for value in command if value.startswith("--config-name="))
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name=config_name, overrides=command[4:])
    cfg = configure_reward(cfg)
    # Match the image trainer's augmentation and actual wrapper conversion.
    cfg.obs_dim = 137
    trainer = TrainDistillResidualFlowAgent.__new__(TrainDistillResidualFlowAgent)
    return trainer._setup_upsampling_wrapper(cfg)


@pytest.mark.parametrize("task,run", CASES, ids=[f"{task}-seed{r['seed']}" for task, r in CASES])
def test_main_figure_run_configuration(task, run, monkeypatch, tmp_path):
    cfg = compose_run(task, run, monkeypatch, tmp_path)
    expected = {**REFERENCE[task]["common_settings"], **run["settings"]}
    for key, value in expected.items():
        actual = OmegaConf.select(cfg, key)
        if OmegaConf.is_config(actual):
            actual = OmegaConf.to_container(actual, resolve=True)
        assert actual == value, f"{run['url']}: {key}"
    assert Path(cfg.base_policy_path).name == run["bc_checkpoint"]
    metadata = ROOT / cfg.robomimic_env_cfg_path
    assert metadata.is_file() and metadata.name == run["environment_metadata"]
    assert f"seed{run['seed']}_" in cfg.wandb.run
    assert "id" not in cfg.wandb and "resume" not in cfg.wandb


def test_threading_seed42_uses_high_resolution_rendering(monkeypatch, tmp_path):
    cfg = compose_run("threading", REFERENCE["threading"]["runs"][0], monkeypatch, tmp_path)
    metadata = json.loads((ROOT / cfg.robomimic_env_cfg_path).read_text())
    assert metadata["env_kwargs"]["camera_heights"] == 256
    assert list(cfg.shape_meta.obs.rgb.shape) == [96, 96, 6]


def test_input_check_reports_missing_assets_before_environment_creation(monkeypatch, tmp_path):
    cfg = compose_run("stack_three", REFERENCE["stack_three"]["runs"][0], monkeypatch, tmp_path)
    cfg.device = "cpu"
    with pytest.raises(RuntimeError) as error:
        check_inputs(cfg)
    message = str(error.value)
    assert "normalizer" in message and "BC checkpoint" in message
    assert "expert replay dataset" in message and "positive-reference dataset" in message


def test_input_check_accepts_matching_files_and_rejects_wrong_bc_shape(monkeypatch, tmp_path):
    cfg = compose_run("stack_three", REFERENCE["stack_three"]["runs"][0], monkeypatch, tmp_path)
    cfg.device = "cpu"
    bc_path = Path(cfg.base_policy_path).parent.parent / ".hydra/config.yaml"
    for path in [cfg.base_policy_path, cfg.normalization_path, cfg.expert_dataset.dataset_path,
                 cfg.env.wrappers.robomimic_image.dino_positive_buffer.dataset_path]:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"test fixture")
    bc_path.parent.mkdir(parents=True)
    config = OmegaConf.create({"action_dim": 7, "horizon_steps": 8, "train_dataset": {"max_n_episodes": 200}})
    OmegaConf.save(config, bc_path)
    assert check_inputs(cfg)
    config.action_dim = 10
    OmegaConf.save(config, bc_path)
    with pytest.raises(ValueError, match="action dimensions"):
        check_inputs(cfg)


@pytest.mark.parametrize("stage", ["pretrain", "finetune"])
def test_wandb_receives_requested_entity_before_environment_start(stage, monkeypatch, tmp_path):
    from types import SimpleNamespace
    if stage == "finetune":
        from hire_dice_rl.agent.finetune import train_agent as module
        cls = module.TrainAgent
        monkeypatch.setattr(module.HydraConfig, "get", lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir=str(tmp_path))))
    else:
        from hire_dice_rl.agent.pretrain import train_agent as module
        cls = module.PreTrainAgent
    captured = {}
    class StopAfterLogging(Exception):
        pass
    def capture(**kwargs):
        captured.update(kwargs)
        raise StopAfterLogging
    monkeypatch.setattr(module, "init_wandb", capture)
    cfg = OmegaConf.create({"device": "cpu", "seed": 42, "env": {},
                            "wandb": {"project": "hire-test", "entity": "selected-team", "run": "seed42-test"}})
    with pytest.raises(StopAfterLogging):
        cls(cfg)
    assert captured["entity"] == "selected-team"
    assert captured["name"] == "seed42-test"
    assert "id" not in captured and "resume" not in captured
