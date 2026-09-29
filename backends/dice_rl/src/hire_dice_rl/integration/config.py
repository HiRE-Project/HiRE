"""Translate public HiRE configuration at the DICE-RL boundary only."""
from omegaconf import OmegaConf, open_dict


def configure_reward(cfg):
    if "reward" not in cfg:
        return cfg  # Saved configurations already contain wrapper settings.
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    reward = cfg.reward
    active = reward.name == "hire"
    if reward.name not in ("hire", "sparse"):
        raise ValueError(f"Unsupported reward: {reward.name}")
    mapping = {
        "dino_goal_source_mode": {"contrastive": "contrastive_prompt", "positive": "positive_buffer", "oracle": "oracle"}[reward.mode] if active else "none",
        "sim_encoder": cfg.encoder.name,
        "dino_device": cfg.encoder.device,
        "dino_compute_in_main": active,
        "dino_contrastive_lambda": reward.contrastive_weight,
        "dino_contrastive_lambda_mode": reward.contrastive_weight_mode,
        "dino_contrastive_kappa_eps": reward.contrastive_epsilon,
        "dino_contrastive_use_positive": reward.use_positive,
        "dino_logsumexp_beta": reward.temperature,
        "dino_reward_weight": reward.potential_scale if active else 0.0,
        "dino_rel_diff_decay_enabled": reward.shaping.pbrs,
        "dino_rel_diff_pi": reward.shaping.discount,
        "dino_shaping_per_step_mean": reward.shaping.normalize,
        "dino_shaping_step_norm": reward.shaping.normalizer,
        "adaptive_dense_weight_max": reward.schedule.maximum if active else 0.0,
        "adaptive_dense_weight_min": reward.schedule.minimum,
        "adaptive_dense_weight_alpha": reward.schedule.exponent,
        "adaptive_success_rate_window_size": reward.schedule.window,
        "adaptive_success_rate_ema_decay": reward.schedule.ema_decay,
        "adaptive_success_rate_norm_cap": reward.schedule.success_cap,
        "dino_positive_buffer": OmegaConf.to_container(reward.positive_buffer, resolve=True),
        "dino_negative_buffer": OmegaConf.to_container(reward.negative_buffer, resolve=True),
    }
    if "oracle_image" in reward:
        mapping["oracle_image"] = OmegaConf.to_container(reward.oracle_image, resolve=True)
    wrapper = cfg.env.wrappers.robomimic_image
    mapping["dino_positive_buffer"]["camera_keys"] = list(wrapper.image_keys)
    with open_dict(cfg):
        cfg.sim_encoder = cfg.encoder.name
        cfg.env.wrappers.robomimic_image = OmegaConf.merge(wrapper, mapping)
    return cfg
