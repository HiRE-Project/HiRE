"""Offline DINO reward precomputation for processed Robomimic / MimicGen NPZ files."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from hire_dice_rl.util.dino_reward_shaper import DinoRewardShaper

log = logging.getLogger(__name__)

STATE_KEY = "offline"


def camera_frames_from_npz_images(
    images: np.ndarray, camera_index: int, num_cameras: int
) -> np.ndarray:
    """Extract one camera view as (T, H, W, 3) HWC uint8/float from NPZ images."""
    if images.ndim != 4:
        raise ValueError(f"Expected images with 4 dims, got {images.shape}")
    if images.shape[1] == num_cameras * 3:
        chw = images[:, camera_index * 3 : (camera_index + 1) * 3]
        return np.transpose(chw, (0, 2, 3, 1))
    if images.shape[-1] == num_cameras * 3:
        return images[..., camera_index * 3 : (camera_index + 1) * 3]
    if images.shape[1] == 3 and images.shape[-1] % num_cameras == 0:
        hwc = np.transpose(images, (0, 2, 3, 1))
        width = hwc.shape[2] // num_cameras
        return hwc[:, :, camera_index * width : (camera_index + 1) * width]
    raise ValueError(
        f"Unsupported images shape {images.shape} for {num_cameras} cameras."
    )


def split_npz_images_to_hwc(
    images: np.ndarray, camera_keys: List[str]
) -> Dict[str, np.ndarray]:
    """Split concatenated NPZ images into per-camera HWC arrays (T, H, W, 3)."""
    num_cameras = len(camera_keys)
    return {
        camera_key: camera_frames_from_npz_images(images, idx, num_cameras)
        for idx, camera_key in enumerate(camera_keys)
    }


def _trajectory_slices(traj_lengths: Iterable[int]) -> Iterable[Tuple[int, int, int]]:
    start = 0
    for episode_id, length in enumerate(traj_lengths):
        end = start + int(length)
        yield episode_id, start, end
        start = end


def _encode_camera_frames_batched(
    shaper: DinoRewardShaper,
    frames_hwc: np.ndarray,
    encode_batch_size: int,
) -> torch.Tensor:
    """Encode (T, H, W, 3) frames in batches -> (T, P, D) on CPU."""
    tensors = []
    with torch.no_grad():
        for start in range(0, frames_hwc.shape[0], encode_batch_size):
            chunk = frames_hwc[start : start + encode_batch_size]
            emb = shaper.encode_frames_hwc(chunk)
            tensors.append(emb.detach().cpu())
    return torch.cat(tensors, dim=0)


def compute_episode_dino_shaping(
    shaper: DinoRewardShaper,
    images_by_camera: Dict[str, np.ndarray],
    traj_len: int,
    *,
    horizon_steps: int,
    encode_batch_size: int = 64,
    env_idx: int = 0,
    n_envs: int = 1,
    state_key: str = STATE_KEY,
    apply_adaptive_dense_weight: bool = True,
) -> np.ndarray:
    """Per-timestep DINO shaping only, aligned with online chunk + PBRS semantics.

    When ``apply_adaptive_dense_weight`` is False, PBRS deltas are cached without
    the adaptive scalar; callers multiply by ``shaper.adaptive_dense_weight`` at
    read time (expert RLPD online cache).
    """
    if shaper.reward_weight == 0.0:
        return np.zeros(traj_len, dtype=np.float64)

    camera_keys = shaper.dino_camera_keys
    embeddings_by_camera: Dict[str, torch.Tensor] = {}
    for camera_key in camera_keys:
        frames = images_by_camera[camera_key]
        if frames.shape[0] != traj_len:
            raise ValueError(
                f"Camera {camera_key}: expected {traj_len} frames, got {frames.shape[0]}"
            )
        embeddings_by_camera[camera_key] = _encode_camera_frames_batched(
            shaper, frames, encode_batch_size
        )

    dino_shaping = np.zeros(traj_len, dtype=np.float64)
    step_norm = (
        shaper.shaping_step_norm
        if shaper.shaping_step_norm > 0.0
        else float(horizon_steps)
    )

    for chunk_start in range(0, traj_len, horizon_steps):
        chunk_end = min(chunk_start + horizon_steps, traj_len)
        chunk_len = chunk_end - chunk_start

        sim_sum = np.zeros(chunk_len, dtype=np.float64)
        sim_count = np.zeros(chunk_len, dtype=np.int64)

        for camera_key in camera_keys:
            emb = embeddings_by_camera[camera_key][chunk_start:chunk_end].to(
                shaper.device
            )
            pos_targets = None
            neg_targets = None
            if shaper.mode == "contrastive_prompt":
                if shaper.contrastive_use_positive:
                    pos_targets = shaper._sample_positive(camera_key)
                if (
                    shaper.dino_negative_buffer is not None
                    and shaper.dino_negative_buffer.size(camera_key) > 0
                ):
                    neg_targets = shaper.dino_negative_buffer.sample_batch(
                        camera_key=camera_key,
                        batch_size=shaper.dino_negative_sample_batch_size,
                        device=shaper.device,
                    )
            elif shaper.mode == "positive_buffer":
                pos_targets = shaper._sample_positive(camera_key)

            sim = shaper.compute_similarity_from_embeddings(
                emb,
                camera_key,
                pos_targets=pos_targets,
                neg_targets=neg_targets,
            )
            sim_np = sim.detach().cpu().numpy().astype(np.float64)
            sim_sum += sim_np
            sim_count += 1

        mean_per_sub = np.where(sim_count > 0, sim_sum / np.maximum(sim_count, 1), 0.0)
        phi_per_sub = shaper.reward_weight * mean_per_sub

        if shaper.rel_diff_decay_enabled:
            per_sub_dino = shaper._apply_pbrs(
                env_idx,
                phi_per_sub,
                state_key,
                n_envs,
                multiply_adaptive_dense_weight=apply_adaptive_dense_weight,
            )
        else:
            per_sub_dino = phi_per_sub.copy()

        if shaper.shaping_per_step_mean and per_sub_dino.size > 0 and step_norm > 0.0:
            per_sub_dino = per_sub_dino / step_norm

        dino_shaping[chunk_start:chunk_end] = per_sub_dino

    return dino_shaping


def precompute_dino_rewards_npz(
    dataset_path: str,
    wrapper_cfg: dict,
    *,
    horizon_steps: int = 8,
    output_path: Optional[str] = None,
    inplace: bool = False,
    encode_batch_size: int = 64,
    max_episodes: Optional[int] = None,
    overwrite: bool = False,
) -> str:
    """Add DINO-shaped rewards to a processed NPZ (finetune / expert data).

    Writes:
      - ``env_rewards``: original sparse env rewards (preserved once)
      - ``dino_shaping``: per-timestep DINO-only component
      - ``rewards``: env_rewards + dino_shaping (used by RLPD expert sampling)
      - ``dino_reward_precomputed``: scalar flag (1.0)

    Args:
        dataset_path: Path to train.npz with ``images``, ``rewards``, ``traj_lengths``.
        wrapper_cfg: Same dict as ``env.wrappers.robomimic_image`` in finetune yaml.
        horizon_steps: Chunk size for positive-buffer resampling (match ``horizon_steps``).
        output_path: If set, write here; else update ``dataset_path`` when ``inplace``.
        inplace: When True and ``output_path`` is None, overwrite ``dataset_path``.
        encode_batch_size: GPU batch size for DINO encoding.
        max_episodes: Limit trajectories (None = all).
        overwrite: Recompute even if ``dino_reward_precomputed`` is already set.

    Returns:
        Path to the written NPZ file.
    """
    data = np.load(dataset_path, allow_pickle=False)
    if "images" not in data.files:
        raise KeyError(f"{dataset_path} missing `images` (image NPZ required)")
    if "rewards" not in data.files or "traj_lengths" not in data.files:
        raise KeyError(f"{dataset_path} missing `rewards` or `traj_lengths`")

    if (
        not overwrite
        and "dino_reward_precomputed" in data.files
        and float(np.asarray(data["dino_reward_precomputed"]).reshape(-1)[0]) > 0.5
    ):
        log.info("DINO rewards already precomputed in %s; skipping.", dataset_path)
        return output_path or dataset_path

    traj_lengths_full = np.asarray(data["traj_lengths"])
    n_episodes_process = len(traj_lengths_full)
    if max_episodes is not None:
        n_episodes_process = min(n_episodes_process, int(max_episodes))
    traj_lengths = traj_lengths_full[:n_episodes_process]
    total_steps_process = int(np.sum(traj_lengths))
    total_steps_full = int(np.sum(traj_lengths_full))

    images = np.asarray(data["images"][:total_steps_process])
    if "env_rewards" in data.files:
        env_rewards_full = np.asarray(data["env_rewards"], dtype=np.float64)
    else:
        env_rewards_full = np.asarray(data["rewards"], dtype=np.float64)
    env_rewards = env_rewards_full[:total_steps_full].copy()

    positive_cfg = wrapper_cfg.get("dino_positive_buffer", {}) or {}
    camera_keys = list(
        positive_cfg.get("camera_keys", wrapper_cfg.get("image_keys", []))
    )
    if not camera_keys:
        raise ValueError("wrapper_cfg must specify dino_positive_buffer.camera_keys")

    shaper = DinoRewardShaper(wrapper_cfg)
    shaper.reset_offline_state(state_key=STATE_KEY, n_envs=1)

    dino_shaping = np.zeros(total_steps_full, dtype=np.float64)
    episodes = list(_trajectory_slices(traj_lengths))

    for _ep_id, start, end in tqdm(episodes, desc="Offline DINO rewards"):
        traj_len = end - start
        images_by_camera = split_npz_images_to_hwc(images[start:end], camera_keys)
        ep_dino = compute_episode_dino_shaping(
            shaper,
            images_by_camera,
            traj_len,
            horizon_steps=horizon_steps,
            encode_batch_size=encode_batch_size,
            env_idx=0,
            n_envs=1,
        )
        dino_shaping[start:end] = ep_dino
        env_sum = float(env_rewards[start:end].sum())
        shaper.finish_offline_episode(
            env_sum, state_key=STATE_KEY, env_idx=0, n_envs=1
        )

    total_rewards = env_rewards + dino_shaping

    out_path = output_path or (dataset_path if inplace else None)
    if out_path is None:
        raise ValueError("Specify output_path or inplace=True")

    payload = {key: data[key] for key in data.files}
    payload["env_rewards"] = env_rewards.astype(np.float32)
    payload["dino_shaping"] = dino_shaping.astype(np.float32)
    payload["rewards"] = total_rewards.astype(np.float32)
    payload["dino_reward_precomputed"] = np.array(1.0, dtype=np.float32)

    np.savez_compressed(out_path, **payload)
    log.info(
        "Wrote DINO rewards to %s: env mean=%.4f dino mean=%.4f total mean=%.4f",
        out_path,
        float(env_rewards.mean()),
        float(dino_shaping.mean()),
        float(total_rewards.mean()),
    )
    return out_path


def _expand_path(path: str) -> str:
    return os.path.abspath(os.path.expandvars(os.path.expanduser(str(path))))


def _wrapper_cfg_for_precompute(cfg: Any) -> dict:
    if not hasattr(cfg, "env") or not hasattr(cfg.env, "wrappers"):
        raise ValueError("dino_reward_precompute requires cfg.env.wrappers.robomimic_image")
    wrapper = cfg.env.wrappers.robomimic_image
    wrapper_cfg = OmegaConf.to_container(wrapper, resolve=True)
    if not isinstance(wrapper_cfg, dict):
        raise TypeError("robomimic_image wrapper cfg must be a dict")
    pos = wrapper_cfg.get("dino_positive_buffer", {}) or {}
    if isinstance(pos, dict) and "online" in pos:
        online = dict(pos.get("online") or {})
        online["enabled"] = False
        pos["online"] = online
        wrapper_cfg["dino_positive_buffer"] = pos
    step_norm = float(wrapper_cfg.get("dino_shaping_step_norm", 0) or 0)
    if step_norm <= 0.0 and hasattr(cfg, "horizon_steps"):
        wrapper_cfg["dino_shaping_step_norm"] = int(cfg.horizon_steps)
    return wrapper_cfg


def _precompute_split_if_needed(
    src_path: str,
    dst_path: str,
    wrapper_cfg: dict,
    *,
    horizon_steps: int,
    encode_batch_size: int,
    max_episodes: Optional[int],
    overwrite: bool,
    label: str,
) -> Optional[str]:
    src_path = _expand_path(src_path)
    dst_path = _expand_path(dst_path)
    if not os.path.isfile(src_path):
        log.warning(
            "DINO precompute [%s]: source missing, skip: %s", label, src_path
        )
        return None
    if os.path.isfile(dst_path) and not overwrite:
        try:
            flag = np.load(dst_path, allow_pickle=False).get(
                "dino_reward_precomputed"
            )
            if flag is not None and float(np.asarray(flag).reshape(-1)[0]) > 0.5:
                log.info(
                    "DINO precompute [%s]: already exists, skip: %s",
                    label,
                    dst_path,
                )
                return dst_path
        except Exception:
            pass
    log.info("DINO precompute [%s]: %s -> %s", label, src_path, dst_path)
    return precompute_dino_rewards_npz(
        src_path,
        wrapper_cfg,
        horizon_steps=horizon_steps,
        output_path=dst_path,
        inplace=False,
        encode_batch_size=encode_batch_size,
        max_episodes=max_episodes,
        overwrite=overwrite,
    )


def maybe_run_dino_reward_precompute_from_cfg(cfg: Any) -> None:
    """Run train/val DINO NPZ export when ``cfg.dino_reward_precompute.enabled``."""
    if not hasattr(cfg, "dino_reward_precompute"):
        return
    pc = cfg.dino_reward_precompute
    if not bool(pc.get("enabled", False)):
        log.info("dino_reward_precompute.enabled=false; using paths from yaml as-is.")
        return

    data_dir = _expand_path(pc.data_dir)
    train_src = os.path.join(data_dir, str(pc.train_src))
    train_dst = os.path.join(data_dir, str(pc.train_dst))
    val_src = os.path.join(data_dir, str(pc.val_src))
    val_dst = os.path.join(data_dir, str(pc.val_dst))

    wrapper_cfg = _wrapper_cfg_for_precompute(cfg)
    horizon_steps = int(cfg.get("horizon_steps", 8))
    encode_batch_size = int(pc.get("encode_batch_size", 64))
    overwrite = bool(pc.get("overwrite", False))
    max_ep = pc.get("max_episodes", 0)
    max_episodes = None if int(max_ep or 0) == 0 else int(max_ep)

    _precompute_split_if_needed(
        train_src,
        train_dst,
        wrapper_cfg,
        horizon_steps=horizon_steps,
        encode_batch_size=encode_batch_size,
        max_episodes=max_episodes,
        overwrite=overwrite,
        label="train",
    )
    _precompute_split_if_needed(
        val_src,
        val_dst,
        wrapper_cfg,
        horizon_steps=horizon_steps,
        encode_batch_size=encode_batch_size,
        max_episodes=max_episodes,
        overwrite=overwrite,
        label="val",
    )

    if hasattr(cfg, "expert_dataset"):
        expected = _expand_path(train_dst)
        configured = _expand_path(cfg.expert_dataset.dataset_path)
        if configured != expected:
            log.warning(
                "expert_dataset.dataset_path (%s) != dino train_dst (%s); "
                "set dataset_path to ${dino_reward_precompute.data_dir}/"
                "${dino_reward_precompute.train_dst}",
                configured,
                expected,
            )
