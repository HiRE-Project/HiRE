"""Portable loading of Hydra configurations stored next to checkpoints."""
from pathlib import Path
from omegaconf import OmegaConf, open_dict


def load_checkpoint_config(checkpoint_path, checkpoint):
    """Prefer the original, unmodified run config over the runtime checkpoint copy."""
    parent = Path(checkpoint_path).expanduser().resolve().parent
    for directory in (parent, parent.parent):
        candidate = directory / ".hydra/config.yaml"
        if candidate.is_file():
            return migrate_checkpoint_config(OmegaConf.load(candidate))
    for key in ("config", "cfg"):
        if key in checkpoint:
            return migrate_checkpoint_config(OmegaConf.create(checkpoint[key]))
    raise FileNotFoundError(
        f"No .hydra/config.yaml or embedded configuration found for {checkpoint_path}"
    )


def prepare_evaluation_config(cfg, device, base_policy_path=None, normalization_path=None):
    """Use sparse task success for evaluation and allow relocated checkpoint bundles."""
    from hire_dice_rl.integration.config import configure_reward
    cfg = configure_reward(cfg)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    with open_dict(cfg):
        cfg.device = device
        cfg.model.device = device
        if base_policy_path:
            cfg.base_policy_path = str(Path(base_policy_path).expanduser().resolve())
            cfg.model.pretrained_flow_policy_path = cfg.base_policy_path
        if normalization_path:
            cfg.normalization_path = str(Path(normalization_path).expanduser().resolve())
            cfg.env.wrappers.robomimic_image.normalization_path = cfg.normalization_path
        wrapper = cfg.env.wrappers.robomimic_image
        wrapper.dino_goal_source_mode = "none"
        wrapper.dino_compute_in_main = False
        wrapper.dino_reward_weight = 0.0
        if "robometer_compute_in_main" in wrapper:
            wrapper.robometer_compute_in_main = False
        if "robometer_reward_weight" in wrapper:
            wrapper.robometer_reward_weight = 0.0
    return cfg


def migrate_checkpoint_config(cfg):
    """Translate saved Python targets without global import aliases."""
    data = OmegaConf.to_container(cfg, resolve=False)
    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "_target_" and isinstance(item, str):
                    if item.split(".")[0] in {"agent", "model", "util", "env"}:
                        value[key] = "hire_dice_rl." + item
                elif key == "robomimic_env_cfg_path" and isinstance(item, str) and item.startswith("cfg/"):
                    value[key] = "configs/task/" + item[4:].replace("/env_meta/", "/")
                else:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    walk(data)
    return OmegaConf.create(data)
