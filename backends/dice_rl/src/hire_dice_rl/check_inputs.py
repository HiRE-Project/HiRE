"""Check experiment inputs before creating environments or starting W&B."""
from pathlib import Path
import torch
from omegaconf import OmegaConf


def check_inputs(cfg):
    paths = {
        "environment metadata": cfg.robomimic_env_cfg_path,
        "normalizer": cfg.normalization_path,
    }
    if "base_policy_path" in cfg:
        checkpoint = Path(cfg.base_policy_path).expanduser()
        paths["BC checkpoint"] = str(checkpoint)
        paths["BC configuration"] = str(checkpoint.parent.parent / ".hydra/config.yaml")
        if cfg.get("use_rlpd", False):
            paths["expert replay dataset"] = cfg.expert_dataset.dataset_path
        reward = cfg.env.wrappers.robomimic_image
        if reward.get("dino_compute_in_main", False):
            positive = reward.get("dino_positive_buffer", {})
            buffer = Path(positive.get("buffer_path", ""))
            if not buffer.is_file():
                if positive.get("build_if_missing", False):
                    paths["positive-reference dataset"] = positive.get("dataset_path", "")
                else:
                    paths["positive-reference buffer"] = str(buffer)
    else:
        paths["pretraining dataset"] = cfg.train_dataset_path
    problems = []
    for label, path in paths.items():
        local_path = Path(path).expanduser()
        if not path or not local_path.is_file() or local_path.stat().st_size == 0:
            problems.append(f"{label}: {path}")
    device = torch.device(cfg.device)
    if device.type == "cuda":
        index = 0 if device.index is None else device.index
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            problems.append(f"CUDA device unavailable: {cfg.device}")
    if problems:
        raise RuntimeError("Experiment inputs are not ready:\n  " + "\n  ".join(problems))
    if "base_policy_path" in cfg:
        bc = OmegaConf.load(paths["BC configuration"])
        if bc.get("env_name", cfg.env_name) != cfg.env_name:
            raise ValueError("BC task does not match the selected finetuning task")
        bc_shape = OmegaConf.select(bc, "shape_meta.obs.rgb.shape")
        if bc_shape is not None and list(bc_shape) != list(cfg.shape_meta.obs.rgb.shape):
            raise ValueError("BC camera shape does not match the policy observation shape")
        if int(bc.action_dim) != int(cfg.action_dim) or int(bc.horizon_steps) != int(cfg.horizon_steps):
            raise ValueError("BC action dimensions/horizon do not match the finetuning task")
        if int(bc.train_dataset.max_n_episodes) != int(cfg.expert_dataset.max_n_episodes):
            raise ValueError("BC and finetuning demonstration counts do not match")
    return paths
