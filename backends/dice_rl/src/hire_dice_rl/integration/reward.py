"""DICE-RL adapter for HiRE reward computation and chunk bookkeeping.

When the env wrapper runs `dino_compute_in_main: true`, env workers emit raw
RGB images per substep into `info["dino_chunk_images"]`. The agent runs this
shaper in the main process so DINOv2 stays on GPU instead of being forced onto
CPU inside forked env workers.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Mapping
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF

from hire.encoders import create_encoder as build_similarity_encoder
from hire.potential import smooth_max, reference_similarity, contrastive_score
from hire.shaping import potential_difference
from hire_dice_rl.util.dino_prompt_buffer import (
    DinoPositiveBufferDataset,
    HybridDinoPositiveSampler,
    OnlineDinoNegativeBuffer,
    RobomimicNpzDinoPositiveBufferBuilder,
    build_online_positive_buffer,
    parse_online_positive_buffer_cfg,
)


log = logging.getLogger(__name__)


def _resolve_oracle_path(cfg, image_key: str) -> Optional[str]:
    if not isinstance(cfg, Mapping):
        return None
    value = cfg.get(image_key)
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return value.get("path")
    return None


class HiRERewardAdapter:
    """Compute DINO-shaped reward in the main process on GPU.

    Reads the same fields from `cfg.env.wrappers.robomimic_image` as the env
    wrapper used to: `dino_goal_source_mode`, `dino_reward_weight`,
    `dino_contrastive_lambda`, `dino_logsumexp_beta`, `oracle_image`,
    `dino_positive_buffer` (offline + optional `online` sub-config),
    `dino_negative_buffer`, `image_keys`, `dino_device`,
    `sim_encoder` (dino | liv | siglip).
    """

    def __init__(self, wrapper_cfg, device: str = "cuda"):
        if wrapper_cfg is None:
            raise ValueError("wrapper_cfg required for DinoRewardShaper")

        mode = wrapper_cfg.get("dino_goal_source_mode")
        self.mode = (
            str(mode).strip().lower() if mode not in (None, "", "none") else None
        )
        if self.mode not in ("oracle", "positive_buffer", "contrastive_prompt"):
            raise ValueError(
                f"DinoRewardShaper: unsupported dino_goal_source_mode={mode}"
            )

        self.reward_weight = float(wrapper_cfg.get("dino_reward_weight", 0.0))
        self.contrastive_lambda = float(
            wrapper_cfg.get("dino_contrastive_lambda", 1.0)
        )
        self.contrastive_lambda_mode = str(
            wrapper_cfg.get("dino_contrastive_lambda_mode", "fixed")
        ).strip().lower()
        if self.contrastive_lambda_mode not in ("fixed", "kappa_ratio"):
            raise ValueError(
                "dino_contrastive_lambda_mode must be 'fixed' or 'kappa_ratio', "
                f"got {self.contrastive_lambda_mode!r}"
            )
        self.contrastive_kappa_eps = float(
            wrapper_cfg.get("dino_contrastive_kappa_eps", 1e-6)
        )
        self.logsumexp_beta = float(wrapper_cfg.get("dino_logsumexp_beta", 10.0))

        # PBRS follows the visual-reward convention used in RLinf:
        # shaped_i = adaptive_dense_weight * (gamma * Phi_i - Phi_{i-1}).
        # Previous potentials are carried across chunks and reset per episode.
        self.rel_diff_decay_enabled = bool(
            wrapper_cfg.get("dino_rel_diff_decay_enabled", False)
        )
        self.rel_diff_pi = float(wrapper_cfg.get("dino_rel_diff_pi", 0.99))

        # Adaptive dense weight: w = max · (1 − sr)^alpha + min.
        # Refresh from recent completed episodes before each RL update.
        self.adaptive_dense_weight_max = float(
            wrapper_cfg.get("adaptive_dense_weight_max", 1.0)
        )
        self.adaptive_dense_weight_min = float(
            wrapper_cfg.get("adaptive_dense_weight_min", 0.0)
        )
        self.adaptive_dense_weight_alpha = float(
            wrapper_cfg.get("adaptive_dense_weight_alpha", 1.0)
        )
        self.adaptive_success_rate_ema_decay = float(
            wrapper_cfg.get("adaptive_success_rate_ema_decay", 0.95)
        )
        self.adaptive_success_rate_norm_cap = float(
            wrapper_cfg.get("adaptive_success_rate_norm_cap", 0.8)
        )
        window_size = int(wrapper_cfg.get("adaptive_success_rate_window_size", 100))
        self.adaptive_success_rate_window_size = window_size
        self._adaptive_sr_use_update_window = window_size > 0
        self.adaptive_success_rate_ema: float = 0.0
        self.latest_train_success_rate: float = 0.0
        self.adaptive_dense_weight: float = self._compute_adaptive_dense_weight()
        # Per-env Φ(s_{t-1}) carried across chunks. Stored separately for the
        # training and evaluation venvs so that eval rollouts cannot poison the
        # PBRS state used by the training collector.
        self._prev_potential: Dict[str, np.ndarray] = {}

        # Scale substep shaping before replay sums it into action-chunk rewards.
        # A configured divisor remains fixed even when a terminal chunk is shorter.
        self.shaping_per_step_mean = bool(
            wrapper_cfg.get("dino_shaping_per_step_mean", False)
        )
        # Default normalizer: horizon_steps if available, else 1 (no-op). The
        # value is fetched lazily in `shape()` so that callers can override via
        # cfg without needing the shaper to know about the outer training cfg.
        self.shaping_step_norm = float(
            wrapper_cfg.get("dino_shaping_step_norm", 0.0)
        )

        self.image_keys: List[str] = list(wrapper_cfg.get("image_keys", []))
        positive_cfg = wrapper_cfg.get("dino_positive_buffer", {}) or {}
        self.dino_camera_keys: List[str] = list(
            positive_cfg.get("camera_keys", self.image_keys)
        )
        self.oracle_image_cfg = wrapper_cfg.get("oracle_image", {}) or {}

        requested_device = wrapper_cfg.get("dino_device", device)
        self.device = torch.device(requested_device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            log.warning(
                "DinoRewardShaper requested cuda but CUDA unavailable; using cpu"
            )
            self.device = torch.device("cpu")

        self.sim_encoder_kind = str(
            wrapper_cfg.get("sim_encoder", "dino") or "dino"
        ).strip().lower()
        log.info(
            "DinoRewardShaper init: mode=%s encoder=%s device=%s",
            self.mode,
            self.sim_encoder_kind,
            self.device,
        )
        self.encoder = build_similarity_encoder(
            self.sim_encoder_kind, device=str(self.device)
        )

        self.contrastive_use_positive = bool(
            wrapper_cfg.get("dino_contrastive_use_positive", True)
        )
        if self.mode != "contrastive_prompt":
            self.contrastive_use_positive = True
        self.use_positive_buffer = self.mode == "positive_buffer" or (
            self.mode == "contrastive_prompt" and self.contrastive_use_positive
        )
        self.use_negative_buffer = self.mode == "contrastive_prompt"
        self._episode_online_frames: Dict[str, List[List[dict]]] = {}
        self._episode_online_frame_indices: Dict[str, List[List[int]]] = {}

        self.oracle_goal_embeddings: Dict[str, torch.Tensor] = {}
        self.dino_positive_buffer: Optional[DinoPositiveBufferDataset] = None
        self.dino_online_positive_buffer: Optional[OnlineDinoNegativeBuffer] = None
        self.dino_positive_sampler: Optional[HybridDinoPositiveSampler] = None
        self.dino_negative_buffer: Optional[OnlineDinoNegativeBuffer] = None
        self.dino_online_positive_add_mode = "last_frame"
        self.dino_online_positive_sampling_mode = "random"
        self.dino_online_frame_stride = max(
            1, int(positive_cfg.get("frame_stride", 5))
        )
        self.dino_positive_sample_batch_size = int(
            positive_cfg.get("sample_batch_size", 64)
        )
        self.dino_positive_sampling_mode = str(
            positive_cfg.get("sampling_mode", "random")
        )

        if self.use_positive_buffer:
            self._init_positive_buffer(positive_cfg)
        elif self.use_negative_buffer:
            pass
        else:
            self._load_oracle_goal_images()
        if self.use_negative_buffer:
            self._init_negative_buffer(
                wrapper_cfg.get("dino_negative_buffer", {}) or {}
            )
        if not self.use_positive_buffer and not self.use_negative_buffer:
            if not self.oracle_goal_embeddings:
                log.warning(
                    "DinoRewardShaper: oracle mode but no oracle images loaded; "
                    "DINO reward will be zero."
                )

    # ------------------------------------------------------------------
    # init helpers
    # ------------------------------------------------------------------
    def _init_positive_buffer(self, cfg):
        buffer_path = cfg.get("buffer_path")
        if not buffer_path:
            raise ValueError(
                f"dino_goal_source_mode={self.mode} requires "
                "dino_positive_buffer.buffer_path"
            )
        if not os.path.exists(buffer_path):
            if not bool(cfg.get("build_if_missing", False)):
                raise FileNotFoundError(
                    f"DINO positive buffer not found: {buffer_path}"
                )
            dataset_path = cfg.get("dataset_path")
            if not dataset_path:
                raise ValueError(
                    "dino_positive_buffer.build_if_missing=true requires dataset_path"
                )
            os.makedirs(os.path.dirname(buffer_path), exist_ok=True)
            lock_path = f"{buffer_path}.lock"
            wait_timeout_s = int(cfg.get("build_wait_timeout_s", 7200))
            poll_interval_s = float(cfg.get("build_wait_poll_s", 2.0))
            lock_fd = None
            start = time.time()
            while lock_fd is None and not os.path.exists(buffer_path):
                try:
                    lock_fd = os.open(
                        lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR
                    )
                except FileExistsError:
                    if time.time() - start > wait_timeout_s:
                        raise TimeoutError(
                            f"Timed out waiting for DINO positive buffer build "
                            f"lock: {lock_path}"
                        )
                    time.sleep(poll_interval_s)
            if lock_fd is not None:
                try:
                    builder = RobomimicNpzDinoPositiveBufferBuilder(
                        dataset_path=dataset_path,
                        output_path=buffer_path,
                        camera_keys=self.dino_camera_keys,
                        encoder=self.encoder,
                        device=str(self.device),
                        frame_stride=int(cfg.get("frame_stride", 5)),
                        max_episodes=cfg.get("max_episodes", None),
                        max_frames_per_episode=cfg.get(
                            "max_frames_per_episode", None
                        ),
                        encode_batch_size=int(cfg.get("encode_batch_size", 64)),
                        encoder_kind=self.sim_encoder_kind,
                        save_images_in_buffer=bool(
                            cfg.get("save_images_in_buffer", False)
                        ),
                    )
                    builder.build()
                finally:
                    os.close(lock_fd)
                    if os.path.exists(lock_path):
                        os.remove(lock_path)
            elif not os.path.exists(buffer_path):
                raise FileNotFoundError(
                    f"Expected DINO positive buffer after waiting: {buffer_path}"
                )

        self.dino_positive_buffer = DinoPositiveBufferDataset(
            buffer_path=buffer_path,
            camera_keys=list(self.dino_camera_keys),
            seed=int(cfg.get("seed", 0)),
            sampling_mode=self.dino_positive_sampling_mode,
        )
        online_enabled, online_cfg = parse_online_positive_buffer_cfg(cfg)
        if online_enabled:
            self.dino_online_positive_buffer = build_online_positive_buffer(
                camera_keys=self.dino_camera_keys,
                online_cfg=online_cfg,
            )
            self.dino_online_positive_add_mode = str(
                online_cfg.get("add_mode", "last_frame")
            ).strip().lower()
            self.dino_online_positive_sampling_mode = str(
                online_cfg.get("sampling_mode", "random")
            ).strip().lower()
            self.dino_online_frame_stride = max(
                1, int(online_cfg.get("frame_stride", 5))
            )
            if self.dino_online_positive_add_mode not in (
                "last_frame",
                "all_substeps",
                "episode_stride",
            ):
                raise ValueError(
                    "dino_positive_buffer.online.add_mode must be "
                    "'last_frame', 'all_substeps', or 'episode_stride', "
                    f"got {self.dino_online_positive_add_mode}"
                )
            log.info(
                "DinoRewardShaper online positive buffer: max_size=%s "
                "online_mix_ratio=%s add_mode=%s frame_stride=%s",
                online_cfg.get("buffer_size", 128),
                online_cfg.get("online_mix_ratio", 0.5),
                self.dino_online_positive_add_mode,
                self.dino_online_frame_stride,
            )
        else:
            self.dino_online_frame_stride = max(
                1, int(cfg.get("frame_stride", 5))
            )
        self.dino_positive_sampler = HybridDinoPositiveSampler(
            offline=self.dino_positive_buffer,
            online=self.dino_online_positive_buffer,
            online_mix_ratio=float(
                (online_cfg if online_enabled else {}).get("online_mix_ratio", 0.5)
            ),
        )

    def _init_negative_buffer(self, cfg):
        max_size = int(cfg.get("buffer_size", 4096))
        self.dino_negative_sample_batch_size = int(cfg.get("sample_batch_size", 64))
        self.dino_negative_buffer = OnlineDinoNegativeBuffer(
            camera_keys=self.dino_camera_keys,
            max_size=max_size,
            seed=int(cfg.get("seed", 0)),
            store_images=bool(cfg.get("store_images", False)),
        )
        self.buffer_snapshot_recorder = None

    def _load_oracle_goal_images(self):
        for image_key in self.dino_camera_keys:
            path = _resolve_oracle_path(self.oracle_image_cfg, image_key)
            if not path:
                continue
            log.info(
                "DinoRewardShaper loading oracle image for %s from %s",
                image_key,
                path,
            )
            img = Image.open(path).convert("RGB")
            goal = TF.to_tensor(img).mul(255.0).unsqueeze(0).to(self.device)
            self.oracle_goal_embeddings[image_key] = self.encoder.compute_embeddings(
                goal
            )

    # ------------------------------------------------------------------
    # similarity helpers
    # ------------------------------------------------------------------
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(x, dim=-1)

    def _logsumexp_smooth_max(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        beta = float(self.logsumexp_beta)
        return smooth_max(x, beta, dim)

    def _contrastive_sim(
        self, pos_sim: torch.Tensor, neg_sim: torch.Tensor
    ) -> torch.Tensor:
        """Contrastive score κ₊ − λ·κ₋ with fixed or data-dependent λ.

        ``kappa_ratio``: λ = κ₋ / (κ₊ + ε), score = κ₊ − λ·κ₋.
        When ``contrastive_use_positive`` is false (κ₊ ≡ 0), falls back to −κ₋.
        """
        if self.contrastive_lambda_mode == "fixed":
            return contrastive_score(pos_sim, neg_sim, self.contrastive_lambda)
        eps = float(self.contrastive_kappa_eps)
        if self.contrastive_use_positive:
            lam = neg_sim / (pos_sim + eps)
            return pos_sim - lam * neg_sim
        lam = neg_sim / (neg_sim + eps)
        return -lam * neg_sim

    def _max_sim_to_targets(
        self, current: torch.Tensor, targets: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return reference_similarity(current, targets, self.logsumexp_beta)

    def _sim_to_oracle(
        self, current: torch.Tensor, image_key: str
    ) -> torch.Tensor:
        goal = self.oracle_goal_embeddings.get(image_key)
        if goal is None:
            return torch.zeros(current.shape[0], device=self.device)
        cur = self._normalize(current)
        goal = self._normalize(goal)  # [1, P, D]
        patch = (cur * goal).sum(dim=-1)  # [B, P]
        return patch.mean(dim=-1)

    # ------------------------------------------------------------------
    # PBRS / adaptive-dense-weight helpers
    # ------------------------------------------------------------------
    def _compute_adaptive_dense_weight(
        self, success_rate: Optional[float] = None
    ) -> float:
        if success_rate is None:
            sr = max(0.0, min(1.0, float(self.adaptive_success_rate_ema)))
        else:
            sr = max(0.0, min(1.0, float(success_rate)))
        return (
            self.adaptive_dense_weight_max
            * ((1.0 - sr) ** self.adaptive_dense_weight_alpha)
            + self.adaptive_dense_weight_min
        )

    def refresh_adaptive_dense_weight_before_update(
        self, success_rate: Optional[float] = None
    ) -> float:
        """Set ``adaptive_dense_weight`` from policy success rate.

        Called once per RL ``update_networks``. In update-window mode the agent
        passes ``success_rate`` (same value as wandb ``train/success_rate``).
        If omitted, uses success_rate=0 (full ``max``).
        """
        if self.adaptive_dense_weight_max <= 0.0:
            self.adaptive_dense_weight = 0.0
            return 0.0
        if not self._adaptive_sr_use_update_window:
            return self.adaptive_dense_weight

        sr_recent = 0.0 if success_rate is None else float(success_rate)
        self.latest_train_success_rate = max(0.0, min(1.0, sr_recent))

        self.adaptive_success_rate_ema = sr_recent
        self.adaptive_dense_weight = self._compute_adaptive_dense_weight(sr_recent)
        log.info(
            "DINO adaptive weight refresh: sr_recent=%.4f (train/success_rate) -> w=%.6f",
            sr_recent,
            self.adaptive_dense_weight,
        )
        return self.adaptive_dense_weight

    def _get_prev_potential(self, state_key: str, n_envs: int) -> np.ndarray:
        prev = self._prev_potential.get(state_key)
        if prev is None or prev.shape[0] != n_envs:
            prev = np.zeros(n_envs, dtype=np.float64)
            self._prev_potential[state_key] = prev
        return prev

    def _apply_pbrs(
        self,
        env_idx: int,
        phi_per_sub: np.ndarray,
        state_key: str,
        n_envs: int,
        *,
        multiply_adaptive_dense_weight: bool = True,
    ) -> np.ndarray:
        """Compute γ·Φ_i − Φ_{i-1} per substep, optionally scaled by adaptive_dense_weight.

        Updates the per-env carry-over potential in-place. Episode-boundary
        resets are handled in `shape()` after this call (to use the post-step
        terminated/truncated arrays).
        """
        prev_arr = self._get_prev_potential(state_key, n_envs)
        prev = float(prev_arr[env_idx])
        shaped = np.zeros_like(phi_per_sub)
        gamma = self.rel_diff_pi
        for i in range(phi_per_sub.shape[0]):
            phi_i = float(phi_per_sub[i])
            shaped[i] = potential_difference(prev, phi_i, gamma)
            prev = phi_i
        prev_arr[env_idx] = prev
        if multiply_adaptive_dense_weight:
            return shaped * float(self.adaptive_dense_weight)
        return shaped

    def _update_adaptive_dense_weight_from_chunk(
        self,
        info_venv: Sequence[dict],
        terminated_venv: Optional[np.ndarray],
        truncated_venv: Optional[np.ndarray],
    ) -> None:
        """Update success-rate EMA from envs that just finished an episode.

        Uses `env_reward_chunk_sum > 0`, matching the reference-buffer updates.
        """
        if self.adaptive_dense_weight_max <= 0.0:
            return
        if terminated_venv is None and truncated_venv is None:
            return
        n_envs = len(info_venv)
        n_finished = 0
        n_success = 0
        for env_idx in range(n_envs):
            terminated = (
                bool(terminated_venv[env_idx]) if terminated_venv is not None else False
            )
            truncated = (
                bool(truncated_venv[env_idx]) if truncated_venv is not None else False
            )
            if not (terminated or truncated):
                continue
            n_finished += 1
            env_reward_sum = float(info_venv[env_idx].get("env_reward_chunk_sum", 0.0))
            if env_reward_sum > 0.0:
                n_success += 1
        if n_finished == 0:
            return
        batch_sr = n_success / n_finished
        cap = max(self.adaptive_success_rate_norm_cap, 1e-8)
        batch_sr_for_ema = max(0.0, min(1.0, batch_sr / cap))
        decay = self.adaptive_success_rate_ema_decay
        self.adaptive_success_rate_ema = (
            decay * self.adaptive_success_rate_ema
            + (1.0 - decay) * batch_sr_for_ema
        )
        self.adaptive_dense_weight = self._compute_adaptive_dense_weight()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def shape(
        self,
        reward_venv: np.ndarray,
        info_venv: Sequence[dict],
        terminated_venv: Optional[np.ndarray] = None,
        truncated_venv: Optional[np.ndarray] = None,
        state_key: str = "train",
    ) -> np.ndarray:
        """Add DINO-shaped reward in-place. Returns the updated reward array.

        Each `info_venv[i]` is expected to contain `info["dino_chunk_images"]`
        as a list of dicts mapping camera_key -> uint8 HWC ndarray (one per env
        substep).
        """
        if self.reward_weight == 0.0:
            return reward_venv

        n_envs = len(info_venv)
        # Collect all (env_idx, substep_idx, camera_key, image) into batches
        # per camera_key for a single GPU forward.
        per_camera_images: Dict[str, List[np.ndarray]] = {
            k: [] for k in self.dino_camera_keys
        }
        per_camera_index: Dict[str, List[int]] = {k: [] for k in self.dino_camera_keys}
        env_substep_count = np.zeros(n_envs, dtype=np.int64)

        for env_idx in range(n_envs):
            chunk = info_venv[env_idx].get("dino_chunk_images")
            if not chunk:
                continue
            env_substep_count[env_idx] = len(chunk)
            for sub_idx, img_dict in enumerate(chunk):
                if not isinstance(img_dict, Mapping):
                    continue
                for key in self.dino_camera_keys:
                    img = img_dict.get(key)
                    if img is None:
                        continue
                    per_camera_images[key].append(img)
                    per_camera_index[key].append((env_idx, sub_idx))

        if not any(per_camera_images.values()):
            # Even when no images arrived this chunk, propagate PBRS state for
            # episodes that finished, so the next chunk's first shaped reward
            # starts from Φ=0 (Ng 1999 PBRS terminal convention).
            self._post_chunk_pbrs_bookkeeping(
                n_envs, info_venv, terminated_venv, truncated_venv, state_key
            )
            return reward_venv

        # Encode batches and compute similarities per camera.
        # Map (env_idx, sub_idx) -> per-camera similarity sum across cameras.
        substep_sim_sum: Dict[int, np.ndarray] = {}
        substep_sim_count: Dict[int, np.ndarray] = {}
        for env_idx in range(n_envs):
            count = int(env_substep_count[env_idx])
            if count == 0:
                continue
            substep_sim_sum[env_idx] = np.zeros(count, dtype=np.float64)
            substep_sim_count[env_idx] = np.zeros(count, dtype=np.int64)

        for camera_key, imgs in per_camera_images.items():
            if not imgs:
                continue
            batch = self._stack_images_to_tensor(imgs)  # [B, 3, H, W] float on device
            current = self.encoder.compute_embeddings(batch)  # [B, P, D]

            if self.mode == "oracle":
                sim = self._sim_to_oracle(current, camera_key)
            elif self.mode == "positive_buffer":
                pos = self._sample_positive(camera_key)
                sim = self._max_sim_to_targets(current, pos)
            elif self.mode == "contrastive_prompt":
                if self.contrastive_use_positive:
                    pos = self._sample_positive(camera_key)
                    pos_sim = self._max_sim_to_targets(current, pos)
                else:
                    pos_sim = torch.zeros(current.shape[0], device=self.device)
                if (
                    self.dino_negative_buffer is not None
                    and self.dino_negative_buffer.size(camera_key) > 0
                ):
                    neg = self.dino_negative_buffer.sample_batch(
                        camera_key=camera_key,
                        batch_size=self.dino_negative_sample_batch_size,
                        device=self.device,
                    )
                    neg_sim = self._max_sim_to_targets(current, neg)
                else:
                    neg_sim = torch.zeros_like(pos_sim)
                sim = self._contrastive_sim(pos_sim, neg_sim)
            else:
                raise RuntimeError(f"Unhandled mode={self.mode}")

            sim_np = sim.detach().cpu().numpy().astype(np.float64)
            for (env_idx, sub_idx), s in zip(per_camera_index[camera_key], sim_np):
                substep_sim_sum[env_idx][sub_idx] += float(s)
                substep_sim_count[env_idx][sub_idx] += 1

        for env_idx, sums in substep_sim_sum.items():
            counts = substep_sim_count[env_idx]
            if counts.sum() == 0:
                continue
            mean_per_sub = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
            # Φ_i = reward_weight · sim_i (per substep, after camera averaging).
            phi_per_sub = self.reward_weight * mean_per_sub
            if self.rel_diff_decay_enabled:
                # PBRS: shaped_i = adaptive_dense_weight · (γ · Φ_i − Φ_{i-1}).
                # `prev_potential` is carried across chunks for this env; it is
                # reset to 0 at episode boundaries below (after this loop).
                per_sub_dino_reward = self._apply_pbrs(
                    env_idx, phi_per_sub, state_key, n_envs
                )
            else:
                per_sub_dino_reward = phi_per_sub
            # Per-step normalization: divide each substep's shaping by the chunk
            # length so the downstream replay-buffer `chunk_reward = Σ r_{t+i}`
            # over horizon_steps substeps becomes a *mean* of per-substep
            # shapings (per-step granularity), not a length-dependent sum.
            if self.shaping_per_step_mean and per_sub_dino_reward.size > 0:
                norm = (
                    self.shaping_step_norm
                    if self.shaping_step_norm > 0.0
                    else float(per_sub_dino_reward.shape[0])
                )
                if norm > 0.0:
                    per_sub_dino_reward = per_sub_dino_reward / norm
            chunk_dino_reward = float(per_sub_dino_reward.sum())
            reward_venv[env_idx] = float(reward_venv[env_idx]) + chunk_dino_reward
            info_venv[env_idx]["dino_reward_chunk"] = chunk_dino_reward
            info_venv[env_idx]["dino_similarity_chunk_mean"] = float(
                mean_per_sub.mean()
            )
            if self.rel_diff_decay_enabled:
                info_venv[env_idx]["dino_adaptive_dense_weight"] = float(
                    self.adaptive_dense_weight
                )
                info_venv[env_idx]["dino_adaptive_success_rate_ema"] = float(
                    self.adaptive_success_rate_ema
                )
            # Critical: also distribute the shaped reward across the per-substep
            # rewards stored in info['full_trajectory']['rewards']. The replay
            # buffer (HybridReplayBuffer) builds chunk_rewards / n-step returns
            # from these substep rewards, NOT from the chunk-aggregated
            # `reward_venv` returned here. Without this, DINO shaping is dropped
            # from the Q-learning target and `dino_reward_weight` has no effect
            # on training. We also stash a copy under
            # `dino_per_substep_reward` for downstream introspection.
            traj = info_venv[env_idx].get("full_trajectory")
            if isinstance(traj, dict):
                traj_rewards = traj.get("rewards")
                if (
                    traj_rewards is not None
                    and len(traj_rewards) == per_sub_dino_reward.shape[0]
                ):
                    traj["rewards"] = [
                        float(traj_rewards[i]) + float(per_sub_dino_reward[i])
                        for i in range(len(traj_rewards))
                    ]
                    info_venv[env_idx]["dino_per_substep_reward"] = (
                        per_sub_dino_reward.tolist()
                    )

        if self.dino_online_positive_add_mode == "episode_stride":
            self._accumulate_online_episode_frames(
                info_venv, env_substep_count, state_key
            )

        if terminated_venv is not None or truncated_venv is not None:
            # Negative buffer: failed terminated chunks.
            if self.use_negative_buffer and self.dino_negative_buffer is not None:
                self._maybe_update_negative_buffer(
                    info_venv, terminated_venv, truncated_venv, env_substep_count
                )
            # Online positive buffer: successful terminated chunks.
            if self.dino_online_positive_buffer is not None:
                self._maybe_update_online_positive_buffer(
                    info_venv,
                    terminated_venv,
                    truncated_venv,
                    env_substep_count,
                    state_key=state_key,
                )

        self._post_chunk_pbrs_bookkeeping(
            n_envs, info_venv, terminated_venv, truncated_venv, state_key
        )

        return reward_venv

    def _post_chunk_pbrs_bookkeeping(
        self,
        n_envs: int,
        info_venv: Sequence[dict],
        terminated_venv: Optional[np.ndarray],
        truncated_venv: Optional[np.ndarray],
        state_key: str,
    ) -> None:
        """Reset Φ(s_{t-1}) on episode boundaries and update the success-rate EMA.

        Resetting Φ to 0 at terminal/truncated states realises Ng 1999's
        Φ(terminal)=0 convention so the per-episode shaped return reduces to
        the telescoping  γ·Φ_T + (γ−1)·ΣΦ_i,  bounded by ±max|Φ|. The
        adaptive-dense-weight EMA is only updated on the train collector so
        eval rollouts cannot poison the schedule used for training.
        """
        if not self.rel_diff_decay_enabled:
            return
        if terminated_venv is None and truncated_venv is None:
            return
        prev_arr = self._get_prev_potential(state_key, n_envs)
        for env_idx in range(n_envs):
            terminated = (
                bool(terminated_venv[env_idx])
                if terminated_venv is not None
                else False
            )
            truncated = (
                bool(truncated_venv[env_idx])
                if truncated_venv is not None
                else False
            )
            if terminated or truncated:
                prev_arr[env_idx] = 0.0
        if state_key == "train" and not self._adaptive_sr_use_update_window:
            self._update_adaptive_dense_weight_from_chunk(
                info_venv, terminated_venv, truncated_venv
            )

    def _collect_terminal_chunk_images(
        self,
        info_venv: Sequence[dict],
        terminated_venv: Optional[np.ndarray],
        truncated_venv: Optional[np.ndarray],
        env_substep_count: np.ndarray,
        *,
        require_success: bool,
        add_mode: str,
    ) -> Dict[str, List[np.ndarray]]:
        per_camera_images: Dict[str, List[np.ndarray]] = {
            k: [] for k in self.dino_camera_keys
        }
        for env_idx in range(len(info_venv)):
            count = int(env_substep_count[env_idx])
            if count == 0:
                continue
            terminated = (
                bool(terminated_venv[env_idx])
                if terminated_venv is not None
                else False
            )
            truncated = (
                bool(truncated_venv[env_idx])
                if truncated_venv is not None
                else False
            )
            if not (terminated or truncated):
                continue
            env_reward_sum = float(
                info_venv[env_idx].get("env_reward_chunk_sum", 0.0)
            )
            if require_success:
                if env_reward_sum <= 0:
                    continue
            elif env_reward_sum > 0:
                continue
            chunk = info_venv[env_idx].get("dino_chunk_images")
            if not chunk:
                continue
            frames = chunk if add_mode == "all_substeps" else [chunk[-1]]
            for frame in frames:
                if not isinstance(frame, Mapping):
                    continue
                for key in self.dino_camera_keys:
                    img = frame.get(key)
                    if img is not None:
                        per_camera_images[key].append(img)
        return per_camera_images

    def _maybe_update_negative_buffer(
        self,
        info_venv: Sequence[dict],
        terminated_venv: Optional[np.ndarray],
        truncated_venv: Optional[np.ndarray],
        env_substep_count: np.ndarray,
    ):
        from hire_dice_rl.util.dino_buffer_snapshot import apply_terminal_buffer_updates

        apply_terminal_buffer_updates(
            self,
            info_venv,
            terminated_venv,
            truncated_venv,
            env_substep_count,
            buffer_attr="dino_negative_buffer",
            require_success=False,
            add_mode="last_frame",
            event_kind="negative",
        )

    def _get_episode_online_storage(
        self, state_key: str, n_envs: int
    ) -> tuple:
        frames = self._episode_online_frames.get(state_key)
        indices = self._episode_online_frame_indices.get(state_key)
        if frames is None or len(frames) != n_envs:
            frames = [[] for _ in range(n_envs)]
            indices = [[] for _ in range(n_envs)]
            self._episode_online_frames[state_key] = frames
            self._episode_online_frame_indices[state_key] = indices
        return frames, indices

    def _accumulate_online_episode_frames(
        self,
        info_venv: Sequence[dict],
        env_substep_count: np.ndarray,
        state_key: str,
    ) -> None:
        n_envs = len(info_venv)
        frames_acc, indices_acc = self._get_episode_online_storage(state_key, n_envs)
        for env_idx in range(n_envs):
            if int(env_substep_count[env_idx]) == 0:
                continue
            chunk = info_venv[env_idx].get("dino_chunk_images") or []
            chunk_indices = list(
                info_venv[env_idx].get("dino_chunk_env_frame_indices") or []
            )
            for sub_idx, frame in enumerate(chunk):
                if isinstance(frame, Mapping):
                    frames_acc[env_idx].append(frame)
                    if sub_idx < len(chunk_indices):
                        indices_acc[env_idx].append(int(chunk_indices[sub_idx]))

    def _clear_episode_online_storage(self, state_key: str, env_idx: int) -> None:
        frames = self._episode_online_frames.get(state_key)
        indices = self._episode_online_frame_indices.get(state_key)
        if frames is None or indices is None:
            return
        if 0 <= env_idx < len(frames):
            frames[env_idx] = []
            indices[env_idx] = []

    def _maybe_update_online_positive_buffer(
        self,
        info_venv: Sequence[dict],
        terminated_venv: Optional[np.ndarray],
        truncated_venv: Optional[np.ndarray],
        env_substep_count: np.ndarray,
        *,
        state_key: str = "train",
    ):
        if self.dino_online_positive_add_mode == "episode_stride":
            self._maybe_update_online_positive_from_episode_stride(
                info_venv,
                terminated_venv,
                truncated_venv,
                env_substep_count,
                state_key=state_key,
            )
            return
        from hire_dice_rl.util.dino_buffer_snapshot import apply_terminal_buffer_updates

        apply_terminal_buffer_updates(
            self,
            info_venv,
            terminated_venv,
            truncated_venv,
            env_substep_count,
            buffer_attr="dino_online_positive_buffer",
            require_success=True,
            add_mode=self.dino_online_positive_add_mode,
            event_kind="online_positive",
        )

    def _maybe_update_online_positive_from_episode_stride(
        self,
        info_venv: Sequence[dict],
        terminated_venv: Optional[np.ndarray],
        truncated_venv: Optional[np.ndarray],
        env_substep_count: np.ndarray,
        *,
        state_key: str,
    ) -> None:
        from hire_dice_rl.util.dino_buffer_snapshot import apply_terminal_buffer_updates

        buffer = self.dino_online_positive_buffer
        if buffer is None:
            return
        n_envs = len(info_venv)
        frames_acc, indices_acc = self._get_episode_online_storage(state_key, n_envs)
        stride = max(1, int(self.dino_online_frame_stride))
        for env_idx in range(n_envs):
            if int(env_substep_count[env_idx]) == 0:
                continue
            terminated = (
                bool(terminated_venv[env_idx])
                if terminated_venv is not None
                else False
            )
            truncated = (
                bool(truncated_venv[env_idx])
                if truncated_venv is not None
                else False
            )
            if not (terminated or truncated):
                continue
            env_reward_sum = float(
                info_venv[env_idx].get("env_reward_chunk_sum", 0.0)
            )
            if env_reward_sum <= 0:
                self._clear_episode_online_storage(state_key, env_idx)
                continue
            episode_frames = frames_acc[env_idx]
            episode_indices = indices_acc[env_idx]
            if not episode_frames:
                continue
            selected_frames = episode_frames[::stride]
            selected_indices = episode_indices[::stride]
            synthetic_info = [
                {
                    "env_reward_chunk_sum": env_reward_sum,
                    "dino_chunk_images": selected_frames,
                    "dino_chunk_env_frame_indices": selected_indices,
                }
            ]
            synthetic_substep_count = np.array([len(selected_frames)], dtype=np.int64)
            apply_terminal_buffer_updates(
                self,
                synthetic_info,
                np.array([True]),
                None,
                synthetic_substep_count,
                buffer_attr="dino_online_positive_buffer",
                require_success=True,
                add_mode="all_substeps",
                event_kind="online_positive",
            )
            self._clear_episode_online_storage(state_key, env_idx)

    def _sample_positive(self, camera_key: str) -> Optional[torch.Tensor]:
        if self.dino_positive_sampler is not None:
            return self.dino_positive_sampler.sample_batch(
                camera_key=camera_key,
                batch_size=self.dino_positive_sample_batch_size,
                device=self.device,
                sampling_mode=self.dino_positive_sampling_mode,
            )
        if self.dino_positive_buffer is None:
            return None
        return self.dino_positive_buffer.sample_batch(
            camera_key=camera_key,
            batch_size=self.dino_positive_sample_batch_size,
            device=self.device,
            sampling_mode=self.dino_positive_sampling_mode,
        )

    def _stack_images_to_tensor(self, imgs: List[np.ndarray]) -> torch.Tensor:
        # imgs: list of (H, W, 3) uint8 numpy arrays.
        arr = np.stack([np.asarray(im) for im in imgs], axis=0)  # [B, H, W, 3]
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise ValueError(
                f"Expected list of HWC RGB images, got stacked shape={arr.shape}"
            )
        tensor = torch.from_numpy(arr).to(self.device)
        tensor = tensor.permute(0, 3, 1, 2).contiguous()
        if tensor.dtype == torch.uint8:
            tensor = tensor.float()
        return tensor

    def encode_frames_hwc(self, frames_hwc: np.ndarray) -> torch.Tensor:
        """Encode a batch of HWC RGB frames -> [B, P, D] embeddings."""
        if frames_hwc.ndim != 4 or frames_hwc.shape[-1] != 3:
            raise ValueError(
                f"Expected frames_hwc (B, H, W, 3), got shape={frames_hwc.shape}"
            )
        with torch.no_grad():
            batch = self._stack_images_to_tensor(
                [frames_hwc[i] for i in range(frames_hwc.shape[0])]
            )
            return self.encoder.compute_embeddings(batch)

    def compute_similarity_from_embeddings(
        self,
        current: torch.Tensor,
        camera_key: str,
        *,
        pos_targets: Optional[torch.Tensor] = None,
        neg_targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-frame similarity [B] for one camera (matches online `shape()`)."""
        if self.mode == "oracle":
            return self._sim_to_oracle(current, camera_key)
        if self.mode == "positive_buffer":
            targets = pos_targets
            if targets is None:
                targets = self._sample_positive(camera_key)
            return self._max_sim_to_targets(current, targets)
        if self.mode == "contrastive_prompt":
            if self.contrastive_use_positive:
                pos = pos_targets
                if pos is None:
                    pos = self._sample_positive(camera_key)
                pos_sim = self._max_sim_to_targets(current, pos)
            else:
                pos_sim = torch.zeros(current.shape[0], device=self.device)
            if (
                self.dino_negative_buffer is not None
                and self.dino_negative_buffer.size(camera_key) > 0
            ):
                neg = neg_targets
                if neg is None:
                    neg = self.dino_negative_buffer.sample_batch(
                        camera_key=camera_key,
                        batch_size=self.dino_negative_sample_batch_size,
                        device=self.device,
                    )
                neg_sim = self._max_sim_to_targets(current, neg)
            else:
                neg_sim = torch.zeros_like(pos_sim)
            return self._contrastive_sim(pos_sim, neg_sim)
        raise RuntimeError(f"Unhandled mode={self.mode}")

    def reset_episode_pbrs_state(self, state_key: str, n_envs: int = 1) -> None:
        """Reset Φ carry-over for one episode without touching adaptive-weight EMA."""
        self._prev_potential[state_key] = np.zeros(n_envs, dtype=np.float64)

    def reset_offline_state(self, state_key: str = "offline", n_envs: int = 1) -> None:
        """Reset PBRS carry-over and adaptive EMA (offline precompute entry point)."""
        self.reset_episode_pbrs_state(state_key, n_envs)
        self.adaptive_success_rate_ema = 0.0
        self.adaptive_dense_weight = self._compute_adaptive_dense_weight()

    def finish_offline_episode(
        self,
        env_reward_sum: float,
        state_key: str = "offline",
        env_idx: int = 0,
        n_envs: int = 1,
    ) -> None:
        """Episode boundary: reset Φ and update adaptive dense weight (offline)."""
        if self.rel_diff_decay_enabled:
            prev_arr = self._get_prev_potential(state_key, n_envs)
            prev_arr[env_idx] = 0.0
        if self.adaptive_dense_weight_max <= 0.0 or self._adaptive_sr_use_update_window:
            return
        batch_sr = 1.0 if env_reward_sum > 0.0 else 0.0
        cap = max(self.adaptive_success_rate_norm_cap, 1e-8)
        batch_sr_for_ema = max(0.0, min(1.0, batch_sr / cap))
        decay = self.adaptive_success_rate_ema_decay
        self.adaptive_success_rate_ema = (
            decay * self.adaptive_success_rate_ema
            + (1.0 - decay) * batch_sr_for_ema
        )
        self.adaptive_dense_weight = self._compute_adaptive_dense_weight()
