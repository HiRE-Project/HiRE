"""CPU release contracts: task composition, checkpoint portability, and reward math."""
import argparse
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import get_class
from omegaconf import OmegaConf

from scripts.launch import ROOT, TASKS, build_command
from hire_dice_rl.util.checkpoint_config import load_checkpoint_config, prepare_evaluation_config
from hire_dice_rl.util.dino_prompt_buffer import OnlineDinoEmbeddingBuffer
from hire_dice_rl.util.dino_reward_shaper import DinoRewardShaper


@pytest.fixture(autouse=True)
def storage_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("HIRE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HIRE_LOG_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("DICE_RL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DICE_RL_LOG_DIR", str(tmp_path / "runs"))


def task_config(stage, task, reward="hire", checkpoint="/tmp/base run/checkpoint/state_100.pt"):
    args = argparse.Namespace(stage=stage, task=task, reward=reward, checkpoint=checkpoint, seed=123)
    cmd = build_command(args, [])
    config_dir = next(x.split("=", 1)[1] for x in cmd if x.startswith("--config-path="))
    config_name = next(x.split("=", 1)[1] for x in cmd if x.startswith("--config-name="))
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=cmd[4:])
        OmegaConf.resolve(cfg)
    from hire_dice_rl.integration.config import configure_reward
    return configure_reward(cfg)


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("stage", ["pretrain", "finetune"])
def test_task_configs_resolve_and_targets_import(stage, task):
    cfg = task_config(stage, task)
    assert (ROOT / cfg.robomimic_env_cfg_path).is_file()
    assert cfg.shape_meta.obs.rgb.shape[-1] == 6
    assert cfg.action_dim == 7 and cfg.horizon_steps == 8
    assert cfg.seed == 123
    if stage == "pretrain":
        assert cfg.train_dataset.max_n_episodes == TASKS[task][2]
    else:
        assert cfg.base_policy_path == str(Path("/tmp/base run/checkpoint/state_100.pt").resolve())
        assert cfg.env.wrappers.robomimic_image.adaptive_dense_weight_max == 1
        assert cfg.replay_buffer.relabel_success_episodes_sparse_only == (task == "tool_hang")
        assert cfg.expert_dataset.use_env_rewards_only
        assert cfg.expert_dataset.max_n_episodes == TASKS[task][2]
    def inspect(value):
        if isinstance(value, dict):
            if "_target_" in value:
                assert get_class(value["_target_"])
            for child in value.values():
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)
    inspect(OmegaConf.to_container(cfg))


@pytest.mark.parametrize("task", TASKS)
def test_sparse_recipe_disables_encoder(task):
    cfg = task_config("finetune", task, "sparse")
    wrapper = cfg.env.wrappers.robomimic_image
    assert wrapper.dino_goal_source_mode == "none"
    assert not wrapper.dino_compute_in_main
    assert wrapper.adaptive_dense_weight_max == 0


def test_checkpoint_config_prefers_original_hydra_and_evaluation_is_sparse(tmp_path):
    cfg = task_config("finetune", "stack_three")
    run = tmp_path / "run"
    (run / ".hydra").mkdir(parents=True)
    (run / "checkpoint").mkdir()
    OmegaConf.save(cfg, run / ".hydra/config.yaml")
    checkpoint_path = run / "checkpoint/model.pth"
    stored = {"config": {"obs_dim": 137}}
    loaded = load_checkpoint_config(checkpoint_path, stored)
    assert loaded.obs_dim == 9
    # Both root-level RL checkpoints and nested BC checkpoints are supported.
    assert load_checkpoint_config(run / "model.pth", stored).obs_dim == 9
    evaluated = prepare_evaluation_config(loaded, "cpu", "/tmp/moved/base.pt", "/tmp/moved/normalization.npz")
    wrapper = evaluated.env.wrappers.robomimic_image
    assert wrapper.dino_goal_source_mode == "none" and wrapper.dino_reward_weight == 0
    assert evaluated.model.device == "cpu"
    assert evaluated.model.pretrained_flow_policy_path == str(Path("/tmp/moved/base.pt").resolve())
    assert wrapper.normalization_path == str(Path("/tmp/moved/normalization.npz").resolve())
    assert loaded.env.wrappers.robomimic_image.dino_goal_source_mode == "contrastive_prompt"


def test_checkpoint_config_fallback_and_missing(tmp_path):
    assert load_checkpoint_config(tmp_path / "a.pt", {"cfg": {"seed": 7}}).seed == 7
    with pytest.raises(FileNotFoundError):
        load_checkpoint_config(tmp_path / "a.pt", {})


def shaper_without_encoder():
    # Bypass model downloads; exercise the production math and buffer operations.
    shaper = DinoRewardShaper.__new__(DinoRewardShaper)
    shaper.device = "cpu"
    shaper.logsumexp_beta = 10.0
    shaper.contrastive_lambda_mode = "fixed"
    shaper.contrastive_lambda = 0.9
    shaper.contrastive_use_positive = True
    shaper.rel_diff_pi = 0.99
    shaper.adaptive_dense_weight = 0.7
    shaper._prev_potential = {}
    return shaper


def test_contrastive_similarity_and_failure_penalty():
    shaper = shaper_without_encoder()
    current = torch.tensor([[[1.0, 0.0]]])
    targets = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    score = shaper._max_sim_to_targets(current, targets)
    assert score.item() == pytest.approx(math.log(math.exp(10) + 1) / 10)
    assert shaper._max_sim_to_targets(current, None).item() == 0
    positive = torch.tensor([1.0])
    no_failure = shaper._contrastive_sim(positive, torch.tensor([0.0]))
    near_failure = shaper._contrastive_sim(positive, torch.tensor([1.0]))
    assert near_failure.item() < no_failure.item()
    assert near_failure.item() == pytest.approx(0.1)


def test_pbrs_telescopes_across_action_chunks_and_separates_envs():
    shaper = shaper_without_encoder()
    phi = np.array([0.2, 0.5, 0.4])
    first = shaper._apply_pbrs(0, phi[:2], "train", 2)
    last = shaper._apply_pbrs(0, phi[2:], "train", 2)
    rewards = np.concatenate([first, last])
    discounts = shaper.rel_diff_pi ** np.arange(3)
    assert np.dot(discounts, rewards) == pytest.approx(0.7 * shaper.rel_diff_pi ** 3 * phi[-1])
    assert shaper._get_prev_potential("train", 2)[1] == 0
    assert shaper._get_prev_potential("eval", 2)[0] == 0


def test_online_fifo_evicts_old_references_and_samples_reproducibly():
    buffers = [OnlineDinoEmbeddingBuffer(["camera"], max_size=3, seed=5) for _ in range(2)]
    for buffer in buffers:
        buffer.add_embeddings("camera", torch.arange(2.0).reshape(2, 1, 1))
        buffer.add_embeddings("camera", torch.arange(2.0, 5.0).reshape(3, 1, 1))
        assert buffer.size("camera") == 3
        assert set(buffer._buffers["camera"].flatten().tolist()) == {2.0, 3.0, 4.0}
    assert torch.equal(buffers[0].sample_batch("camera", 20), buffers[1].sample_batch("camera", 20))
    with pytest.raises(KeyError):
        buffers[0].add_embeddings("missing", torch.zeros(1, 1, 1))



@pytest.mark.parametrize("task", ["stack_three", "tool_hang"])
def test_released_policy_loss_and_sampling_on_cpu(task):
    from hydra.utils import instantiate
    torch.set_num_threads(1)
    cfg = task_config("pretrain", task)
    cfg.device = "cpu"
    cfg.model.device = "cpu"
    if task == "stack_three":
        cfg.model.denoising_steps = 2
    else:
        cfg.model.flow_steps = 2
    model = instantiate(cfg.model)
    height, width, _ = cfg.shape_meta.obs.rgb.shape
    cond = {
        "state": torch.zeros(2, 1, 9),
        "rgb": torch.randint(0, 256, (2, 1, 6, height, width)).float(),
    }
    actions = torch.zeros(2, 8, 7)
    loss = model.loss(actions, cond)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    sample = model(cond).trajectories
    assert sample.shape == actions.shape
    assert torch.isfinite(sample).all()


def test_bc_loader_honors_device_and_preserves_model_weight_selection(tmp_path):
    from hire_dice_rl.model.rl.distill_residual_rl import DistillResidualRLModel
    run = tmp_path / "bc_run"
    (run / ".hydra").mkdir(parents=True)
    (run / "checkpoint").mkdir()
    cfg = OmegaConf.create({"device": "cuda:99", "model": {
        "_target_": "torch.nn.Linear", "in_features": 1, "out_features": 1,
        "device": "${device}",
    }})
    OmegaConf.save(cfg, run / ".hydra/config.yaml")
    checkpoint = run / "checkpoint/state_1.pt"
    torch.save({"model": {"weight": torch.ones(1, 1), "bias": torch.ones(1)},
                "ema": {"weight": torch.zeros(1, 1), "bias": torch.zeros(1)}}, checkpoint)
    model = DistillResidualRLModel._load_pretrained_policy(None, str(checkpoint), "cpu")
    assert model.weight.device.type == "cpu"
    assert model.weight.item() == 1
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert not model.training


def test_saved_legacy_targets_and_environment_metadata_migrate():
    from hire_dice_rl.util.checkpoint_config import migrate_checkpoint_config
    old = OmegaConf.create({
        "_target_": "agent.pretrain.train_diffusion_img_agent.TrainDiffusionImgAgent",
        "model": {"_target_": "model.diffusion.diffusion.DiffusionModel"},
        "robomimic_env_cfg_path": "cfg/mimicgen/env_meta/stack_three-img.json",
    })
    migrated = migrate_checkpoint_config(old)
    assert migrated._target_.startswith("hire_dice_rl.agent.")
    assert migrated.model._target_.startswith("hire_dice_rl.model.")
    assert (ROOT / migrated.robomimic_env_cfg_path).is_file()
    assert migrate_checkpoint_config(migrated) == migrated
    assert old.model._target_.startswith("model.")


def test_public_reward_and_encoder_configuration_are_independent():
    from hire_dice_rl.integration.config import configure_reward
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="experiment/stack_three_finetune", overrides=[
            "base_policy_path=/tmp/base.pt", "encoder=siglip", "reward.contrastive_weight=0.3"])
    assert "dino_contrastive_lambda" not in cfg.env.wrappers.robomimic_image
    adapted = configure_reward(cfg)
    assert adapted.env.wrappers.robomimic_image.sim_encoder == "siglip"
    assert adapted.env.wrappers.robomimic_image.dino_contrastive_lambda == 0.3
    assert "dino_contrastive_lambda" not in cfg.env.wrappers.robomimic_image
