"""
Environment wrapper for Robomimic environments with image observations.

Also return done=False since we do not terminate episode early.

Modified from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/env/robomimic/robomimic_image_wrapper.py

"""

import numpy as np
import gym
from gym import spaces
import imageio
import xml.etree.ElementTree as ET
import os
import logging
import time
from collections.abc import Mapping

import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF

from hire_dice_rl.util.dino_prompt_buffer import (
    DinoPositiveBufferDataset,
    HybridDinoPositiveSampler,
    OnlineDinoNegativeBuffer,
    RobomimicNpzDinoPositiveBufferBuilder,
    build_online_positive_buffer,
    parse_online_positive_buffer_cfg,
)
from hire_dice_rl.util.similarity_encoder import DinoV2Encoder, build_similarity_encoder


log = logging.getLogger(__name__)


def convert_10d_to_7d(action_10d):
    """
    Convert 10D action with 6D rotation to 7D action with axis-angle.
    
    Args:
        action_10d: (batch_size, 10) or (10,) with [pos(3), rot6d(6), gripper(1)]
    
    Returns:
        action_7d: Same shape but 7D with [pos(3), axis_angle(3), gripper(1)]
    """
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from hire_dice_rl.util.rotation_conversion import RotationTransformer
    
    rot_transformer = RotationTransformer(from_rep="rotation_6d", to_rep="axis_angle")
    
    # Handle both single action and batch of actions
    single_action = False
    if action_10d.ndim == 1:
        action_10d = action_10d[np.newaxis, :]
        single_action = True
    
    # Extract components
    pos = action_10d[:, :3]
    rot6d = action_10d[:, 3:9]
    gripper = action_10d[:, 9:10]
    
    # Convert 6D rotation to axis-angle
    axis_angle = rot_transformer.forward(rot6d)
    
    # Reconstruct 7D action
    action_7d = np.concatenate([pos, axis_angle, gripper], axis=-1)
    
    if single_action:
        action_7d = action_7d[0]
    
    return action_7d


class RobomimicImageWrapper(gym.Env):
    def __init__(
        self,
        env,
        shape_meta: dict,
        normalization_path=None,
        low_dim_keys=[
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ],
        image_keys=[
            "agentview_image",
            "robot0_eye_in_hand_image",
        ],
        clamp_obs=False,
        init_state=None,
        render_hw=(256, 256),
        render_camera_name="robot0_eye_in_hand",
        success_steps_before_termination=5,
        use_6d_rot=False,  # Whether incoming actions use 6D rotation representation
        dino_goal_source_mode=None,
        sim_encoder="dino",
        dino_contrastive_lambda=1.0,
        dino_contrastive_lambda_mode="fixed",
        dino_contrastive_kappa_eps=1e-6,
        dino_contrastive_use_positive=True,
        dino_logsumexp_beta=10.0,
        dino_reward_weight=0.0,
        oracle_image=None,
        dino_positive_buffer=None,
        dino_negative_buffer=None,
        dino_device=None,
        dino_compute_in_main=False,
        # Potential-based reward shaping knobs. These are consumed exclusively
        # by `util.dino_reward_shaper.DinoRewardShaper` in the main process; the
        # env wrapper itself does not perform PBRS. We accept them here so that
        # `make_async` can `**`-unpack the wrapper config dict without raising.
        dino_rel_diff_decay_enabled=False,
        dino_rel_diff_pi=0.99,
        adaptive_dense_weight_max=1.0,
        adaptive_dense_weight_min=0.0,
        adaptive_dense_weight_alpha=1.0,
        adaptive_success_rate_ema_decay=0.95,
        adaptive_success_rate_norm_cap=0.8,
        adaptive_success_rate_window_size=100,
        # Per-step shaping knobs. Same rationale: only consumed by the main
        # process shaper; accepted here just to survive `**` unpacking of the
        # wrapper cfg dict.
        dino_shaping_per_step_mean=False,
        dino_shaping_step_norm=0.0,
        # Robometer-4B dense reward (computed in main process via HTTP client).
        robometer_compute_in_main=False,
        robometer_reward_weight=0.0,
        robometer_server_url=None,
        robometer_task_instruction=None,
        robometer_camera_key=None,
        robometer_use_frame_steps=None,
        robometer_max_frames=None,
        robometer_request_timeout_s=None,
        robometer_bgr_to_rgb=None,
        robometer_rel_diff_decay_enabled=None,
        robometer_rel_diff_pi=None,
        robometer_use_relative_rewards=None,
        robometer_use_success_detection=None,
        robometer_success_detection_duration=None,
        robometer_success_detection_threshold=None,
        robometer_shaping_per_step_mean=None,
        robometer_shaping_step_norm=None,
        # Performance knobs (main-process RobometerRewardShaper only).
        robometer_query_every_n_chunks=None,
        robometer_query_fill_mode=None,
        robometer_batch_env_queries=None,
        robometer_max_batch_size=None,
        robometer_parallel_requests=None,
    ):
        self.env = env
        self.init_state = init_state
        self.has_reset_before = False
        self.camera_modified = False
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.video_writers = {}
        self.clamp_obs = clamp_obs
        self.success_steps_before_termination = success_steps_before_termination
        self.use_6d_rot = use_6d_rot
        self.dino_goal_source_mode = (
            str(dino_goal_source_mode).strip().lower()
            if dino_goal_source_mode is not None
            else None
        )
        self.sim_encoder_kind = str(sim_encoder or "dino").strip().lower()
        self.dino_reward_weight = float(dino_reward_weight)
        self.dino_contrastive_lambda = float(dino_contrastive_lambda)
        self.dino_contrastive_lambda_mode = str(
            dino_contrastive_lambda_mode or "fixed"
        ).strip().lower()
        self.dino_contrastive_kappa_eps = float(dino_contrastive_kappa_eps)
        self.dino_contrastive_use_positive = bool(dino_contrastive_use_positive)
        self.dino_logsumexp_beta = float(dino_logsumexp_beta)
        self.oracle_image_cfg = oracle_image or {}
        self.dino_positive_cfg = dino_positive_buffer or {}
        self.dino_negative_cfg = dino_negative_buffer or {}
        self.dino_device = dino_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dino_encoder = None
        self.oracle_goal_embeddings = {}
        self.dino_positive_buffer = None
        self.dino_online_positive_buffer = None
        self.dino_positive_sampler = None
        self.dino_negative_buffer = None
        self.dino_online_positive_add_mode = "last_frame"
        self.dino_camera_keys = list(self.dino_positive_cfg.get("camera_keys", image_keys))
        self.dino_compute_in_main = bool(dino_compute_in_main)
        self.robometer_compute_in_main = bool(robometer_compute_in_main)
        self.robometer_reward_weight = float(robometer_reward_weight)
        self.robometer_use_success_detection = bool(
            robometer_use_success_detection or False
        )
        self.use_dino_reward = (
            self.dino_goal_source_mode not in (None, "", "none")
            and not self.dino_compute_in_main
        )
        # Whether to ship per-step raw images to the main process for DINO / Robometer.
        self._export_dino_images = (
            self.dino_compute_in_main
            and self.dino_goal_source_mode not in (None, "", "none")
        ) or (self.robometer_compute_in_main and self.robometer_reward_weight > 0.0)
        self.use_dino_positive_buffer = self.dino_goal_source_mode in (
            "positive_buffer",
            "contrastive_prompt",
        )
        self.use_dino_contrastive_prompt = (
            self.dino_goal_source_mode == "contrastive_prompt"
        )
        self.dino_positive_sample_batch_size = int(
            self.dino_positive_cfg.get("sample_batch_size", 64)
        )
        self.dino_positive_sampling_mode = str(
            self.dino_positive_cfg.get("sampling_mode", "random")
        )
        self.dino_negative_sample_batch_size = int(
            self.dino_negative_cfg.get("sample_batch_size", 64)
        )
        self.dino_negative_buffer_max_size = int(
            self.dino_negative_cfg.get("buffer_size", 4096)
        )
        self.dino_buffer_debug_cfg = self.dino_positive_cfg.get("debug", {})
        self.last_raw_obs = None
        self.last_episode_succeeded = False
        
        # Initialize tracking variables for success-based termination
        self.success_count = 0
        self.episode_reward = 0.0
        self.step_count = 0
        self.ever_succeeded = False

        # set up normalization
        self.normalize = normalization_path is not None
        if self.normalize:
            normalization = np.load(normalization_path)
            self.obs_min = normalization["obs_min"]
            self.obs_max = normalization["obs_max"]
            self.action_min = normalization["action_min"]
            self.action_max = normalization["action_max"]

        # setup spaces
        low = np.full(env.action_dimension, fill_value=-1)
        high = np.full(env.action_dimension, fill_value=1)
        self.action_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype,
        )
        self.low_dim_keys = low_dim_keys
        self.image_keys = image_keys
        self.obs_keys = low_dim_keys + image_keys
        rgb_meta = shape_meta.get("obs", {}).get("rgb", {})
        rgb_shape = rgb_meta.get("shape")
        self._target_rgb_hw = (
            (int(rgb_shape[0]), int(rgb_shape[1])) if rgb_shape is not None else None
        )
        if self.use_dino_reward:
            self._init_dino_reward()
        observation_space = spaces.Dict()
        for key, value in shape_meta["obs"].items():
            shape = value["shape"]
            if key.endswith("rgb"):
                min_value, max_value = 0, 1
            elif key.endswith("state"):
                min_value, max_value = -1, 1
            else:
                raise RuntimeError(f"Unsupported type {key}")
            this_space = spaces.Box(
                low=min_value,
                high=max_value,
                shape=shape,
                dtype=np.float32,
            )
            observation_space[key] = this_space
        self.observation_space = observation_space

    def normalize_obs(self, obs):
        obs = 2 * (
            (obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5
        )  # -> [-1, 1]
        if self.clamp_obs:
            obs = np.clip(obs, -1, 1)
        return obs

    def unnormalize_action(self, action):
        action = (action + 1) / 2  # [-1, 1] -> [0, 1]
        return action * (self.action_max - self.action_min) + self.action_min

    def _init_dino_reward(self):
        if self.dino_goal_source_mode not in (
            "oracle",
            "positive_buffer",
            "contrastive_prompt",
        ):
            raise ValueError(
                f"Unsupported dino_goal_source_mode={self.dino_goal_source_mode}. "
                "Expected one of: oracle, positive_buffer, contrastive_prompt."
            )
        log.info(
            "Initializing %s encoder for dino_goal_source_mode=%s",
            self.sim_encoder_kind.upper(),
            self.dino_goal_source_mode,
        )
        self.dino_encoder = build_similarity_encoder(
            self.sim_encoder_kind, device=self.dino_device
        )
        # Encoder may have downgraded to CPU (e.g. forked worker); keep all
        # downstream tensors on the same device.
        self.dino_device = str(self.dino_encoder.device)
        if self.use_dino_positive_buffer:
            self._init_dino_positive_buffer()
            if self.use_dino_contrastive_prompt:
                self._init_dino_negative_buffer()
        else:
            self._load_oracle_goal_images()
        if (
            self.dino_goal_source_mode == "oracle"
            and not self.oracle_goal_embeddings
        ):
            log.warning(
                "dino_goal_source_mode=oracle is enabled, but no oracle images were loaded. "
                "DINO reward will be zero."
            )

    def _init_dino_positive_buffer(self):
        buffer_path = self.dino_positive_cfg.get("buffer_path")
        if not buffer_path:
            raise ValueError(
                f"dino_goal_source_mode={self.dino_goal_source_mode} requires "
                "dino_positive_buffer.buffer_path."
            )
        if not os.path.exists(buffer_path):
            if not bool(self.dino_positive_cfg.get("build_if_missing", False)):
                raise FileNotFoundError(
                    f"DINO positive buffer not found: {buffer_path}. "
                    "Set dino_positive_buffer.build_if_missing=true or build it first."
                )
            dataset_path = self.dino_positive_cfg.get("dataset_path")
            if not dataset_path:
                raise ValueError(
                    "dino_positive_buffer.build_if_missing=true requires dataset_path."
                )
            os.makedirs(os.path.dirname(buffer_path), exist_ok=True)
            lock_path = f"{buffer_path}.lock"
            wait_timeout_s = int(self.dino_positive_cfg.get("build_wait_timeout_s", 7200))
            poll_interval_s = float(self.dino_positive_cfg.get("build_wait_poll_s", 2.0))
            lock_fd = None
            start = time.time()
            while lock_fd is None and not os.path.exists(buffer_path):
                try:
                    lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                except FileExistsError:
                    if time.time() - start > wait_timeout_s:
                        raise TimeoutError(
                            f"Timed out waiting for DINO positive buffer build lock: {lock_path}"
                        )
                    time.sleep(poll_interval_s)
            if lock_fd is not None:
                try:
                    builder = RobomimicNpzDinoPositiveBufferBuilder(
                        dataset_path=dataset_path,
                        output_path=buffer_path,
                        camera_keys=self.dino_camera_keys,
                        encoder=self.dino_encoder,
                        device=self.dino_device,
                        frame_stride=int(self.dino_positive_cfg.get("frame_stride", 5)),
                        max_episodes=self.dino_positive_cfg.get("max_episodes", None),
                        max_frames_per_episode=self.dino_positive_cfg.get(
                            "max_frames_per_episode", None
                        ),
                        encode_batch_size=int(
                            self.dino_positive_cfg.get("encode_batch_size", 64)
                        ),
                        encoder_kind=self.sim_encoder_kind,
                        save_images_in_buffer=bool(
                            self.dino_positive_cfg.get("save_images_in_buffer", False)
                        ),
                    )
                    builder.build()
                finally:
                    os.close(lock_fd)
                    if os.path.exists(lock_path):
                        os.remove(lock_path)
            elif not os.path.exists(buffer_path):
                raise FileNotFoundError(
                    f"Expected DINO positive buffer after waiting, but missing: {buffer_path}"
                )

        camera_keys = self.dino_camera_keys
        self.dino_positive_buffer = DinoPositiveBufferDataset(
            buffer_path=buffer_path,
            camera_keys=list(camera_keys),
            seed=int(self.dino_positive_cfg.get("seed", 0)),
            sampling_mode=self.dino_positive_sampling_mode,
        )
        online_enabled, online_cfg = parse_online_positive_buffer_cfg(
            self.dino_positive_cfg
        )
        if online_enabled:
            self.dino_online_positive_buffer = build_online_positive_buffer(
                camera_keys=self.dino_camera_keys,
                online_cfg=online_cfg,
            )
            self.dino_online_positive_add_mode = str(
                online_cfg.get("add_mode", "last_frame")
            ).strip().lower()
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
                "Initialized online DINO positive buffer: max_size=%s "
                "online_mix_ratio=%s add_mode=%s",
                online_cfg.get("buffer_size", 128),
                online_cfg.get("online_mix_ratio", 0.5),
                self.dino_online_positive_add_mode,
            )
        self.dino_positive_sampler = HybridDinoPositiveSampler(
            offline=self.dino_positive_buffer,
            online=self.dino_online_positive_buffer,
            online_mix_ratio=float(
                (online_cfg if online_enabled else {}).get("online_mix_ratio", 0.5)
            ),
        )
        log.info(
            "Loaded DINO positive buffer from %s: %s",
            buffer_path,
            {
                key: {
                    "num_embeddings": self.dino_positive_buffer.num_samples(key),
                    "num_unique_timesteps": self.dino_positive_buffer.num_unique_timesteps(key),
                    "online_embeddings": (
                        self.dino_online_positive_buffer.size(key)
                        if self.dino_online_positive_buffer is not None
                        else 0
                    ),
                }
                for key in camera_keys
            },
        )

    def _init_dino_negative_buffer(self):
        self.dino_negative_buffer = OnlineDinoNegativeBuffer(
            camera_keys=self.dino_camera_keys,
            max_size=self.dino_negative_buffer_max_size,
            seed=int(self.dino_negative_cfg.get("seed", 0)),
            store_images=False,
        )
        log.info(
            "Initialized DINO negative buffer: max_size=%s sample_batch_size=%s lambda=%s",
            self.dino_negative_buffer_max_size,
            self.dino_negative_sample_batch_size,
            self.dino_contrastive_lambda,
        )

    def _get_oracle_image_path(self, image_key):
        cfg = self.oracle_image_cfg
        if not isinstance(cfg, Mapping):
            return None
        value = cfg.get(image_key)
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping):
            return value.get("path")
        return None

    def _load_oracle_goal_images(self):
        for image_key in self.dino_camera_keys:
            path = self._get_oracle_image_path(image_key)
            if not path:
                continue
            log.info("Loading DINO oracle image for %s from %s", image_key, path)
            img = Image.open(path).convert("RGB")
            goal = TF.to_tensor(img).mul(255.0).unsqueeze(0).to(self.dino_device)
            self.oracle_goal_embeddings[image_key] = self.dino_encoder.compute_embeddings(goal)

    def _image_to_dino_tensor(self, image):
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected HWC image with 3 channels, got shape={image.shape}")
        tensor = torch.from_numpy(image).to(self.dino_device)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        if tensor.dtype == torch.uint8:
            tensor = tensor.float()
        return tensor

    def _compute_dino_similarity(self, current_image, goal_embeddings):
        if goal_embeddings is None:
            return torch.zeros(1, device=self.dino_device, dtype=torch.float32)
        current = self.dino_encoder.compute_embeddings(current_image)
        return self._compute_dino_similarity_from_embeddings(current, goal_embeddings)

    def _compute_dino_similarity_from_embeddings(self, current, goal_embeddings):
        current = torch.nn.functional.normalize(current, dim=-1)
        goal = torch.nn.functional.normalize(goal_embeddings, dim=-1)
        patch_sim = (current * goal).sum(dim=-1)
        return patch_sim.mean(dim=-1)

    def _logsumexp_smooth_max(self, x, dim):
        beta = float(self.dino_logsumexp_beta)
        return torch.logsumexp(beta * x, dim=dim) / beta

    def _contrastive_sim(self, pos_sim, neg_sim):
        if self.dino_contrastive_lambda_mode == "fixed":
            return pos_sim - self.dino_contrastive_lambda * neg_sim
        eps = float(self.dino_contrastive_kappa_eps)
        if self.dino_contrastive_use_positive:
            lam = neg_sim / (pos_sim + eps)
            return pos_sim - lam * neg_sim
        lam = neg_sim / (neg_sim + eps)
        return -lam * neg_sim

    def _compute_max_dino_similarity_to_embeddings(self, current_embeddings, target_embeddings):
        if target_embeddings is None or target_embeddings.numel() == 0:
            return torch.zeros(1, device=self.dino_device, dtype=torch.float32)
        current = torch.nn.functional.normalize(current_embeddings, dim=-1)
        target = torch.nn.functional.normalize(target_embeddings, dim=-1)
        pairwise_patch_sim = torch.einsum("bpd,kpd->bkp", current, target)
        pairwise_sim = pairwise_patch_sim.mean(dim=-1)
        return self._logsumexp_smooth_max(pairwise_sim, dim=-1)

    def _sample_positive_embeddings(self, image_key):
        if self.dino_positive_sampler is not None:
            return self.dino_positive_sampler.sample_batch(
                camera_key=image_key,
                batch_size=self.dino_positive_sample_batch_size,
                device=torch.device(self.dino_device),
                sampling_mode=self.dino_positive_sampling_mode,
            )
        if self.dino_positive_buffer is None:
            return None
        return self.dino_positive_buffer.sample_batch(
            camera_key=image_key,
            batch_size=self.dino_positive_sample_batch_size,
            device=torch.device(self.dino_device),
            sampling_mode=self.dino_positive_sampling_mode,
        )

    def _add_last_obs_to_positive_buffer_if_succeeded(self):
        if self.dino_compute_in_main:
            return
        if (
            self.dino_online_positive_buffer is None
            or self.dino_encoder is None
            or self.last_raw_obs is None
            or self.step_count <= 0
            or not self.last_episode_succeeded
        ):
            return
        for image_key in self.dino_camera_keys:
            if image_key not in self.last_raw_obs:
                continue
            current_image = self._image_to_dino_tensor(self.last_raw_obs[image_key])
            embeddings = self.dino_encoder.compute_embeddings(current_image).detach()
            self.dino_online_positive_buffer.add_embeddings(image_key, embeddings)
        log.debug(
            "Added succeeded episode final observation to online DINO positive buffer."
        )

    def _add_last_obs_to_negative_buffer_if_failed(self):
        if self.dino_compute_in_main:
            # Buffer updates are handled by DinoRewardShaper in the main process.
            return
        if (
            not self.use_dino_contrastive_prompt
            or self.dino_negative_buffer is None
            or self.dino_encoder is None
            or self.last_raw_obs is None
            or self.step_count <= 0
            or self.last_episode_succeeded
        ):
            return
        for image_key in self.dino_camera_keys:
            if image_key not in self.last_raw_obs:
                continue
            current_image = self._image_to_dino_tensor(self.last_raw_obs[image_key])
            embeddings = self.dino_encoder.compute_embeddings(current_image).detach()
            self.dino_negative_buffer.add_embeddings(image_key, embeddings)
        log.debug("Added failed episode final observation to DINO negative buffer.")

    def _compute_dino_reward(self, raw_obs, info):
        if not self.use_dino_reward or self.dino_encoder is None:
            return 0.0
        similarities = []
        image_keys = (
            self.dino_camera_keys
            if self.use_dino_positive_buffer
            else self.oracle_goal_embeddings.keys()
        )
        for image_key in image_keys:
            if image_key not in raw_obs:
                continue
            current_image = self._image_to_dino_tensor(raw_obs[image_key])
            current_embeddings = self.dino_encoder.compute_embeddings(current_image)
            if self.dino_goal_source_mode == "oracle":
                goal_embeddings = self.oracle_goal_embeddings.get(image_key)
                sim = self._compute_dino_similarity_from_embeddings(
                    current_embeddings, goal_embeddings
                )
            elif self.dino_goal_source_mode == "positive_buffer":
                pos = self._sample_positive_embeddings(image_key)
                sim = self._compute_max_dino_similarity_to_embeddings(
                    current_embeddings, pos
                )
            elif self.dino_goal_source_mode == "contrastive_prompt":
                pos = self._sample_positive_embeddings(image_key)
                pos_sim = self._compute_max_dino_similarity_to_embeddings(
                    current_embeddings, pos
                )
                if (
                    self.dino_negative_buffer is not None
                    and self.dino_negative_buffer.size(image_key) > 0
                ):
                    neg = self.dino_negative_buffer.sample_batch(
                        camera_key=image_key,
                        batch_size=self.dino_negative_sample_batch_size,
                        device=torch.device(self.dino_device),
                    )
                    neg_sim = self._compute_max_dino_similarity_to_embeddings(
                        current_embeddings, neg
                    )
                else:
                    neg_sim = torch.zeros_like(pos_sim)
                info[f"dino_similarity_pos_{image_key}"] = float(pos_sim.item())
                info[f"dino_similarity_neg_{image_key}"] = float(neg_sim.item())
                sim = self._contrastive_sim(pos_sim, neg_sim)
            else:
                raise RuntimeError(
                    f"Unexpected dino_goal_source_mode={self.dino_goal_source_mode}"
                )
            sim_value = float(sim.item())
            info[f"dino_similarity_{image_key}"] = sim_value
            info[f"dino_reward_{image_key}"] = self.dino_reward_weight * sim_value
            similarities.append(sim)
        if not similarities:
            return 0.0
        dino_similarity = torch.stack(similarities, dim=0).mean()
        dino_reward = self.dino_reward_weight * float(dino_similarity.item())
        info["dino_similarity"] = float(dino_similarity.item())
        info["dino_reward"] = dino_reward
        return dino_reward

    def _resize_policy_rgb(self, rgb_hwc: np.ndarray) -> np.ndarray:
        """Resize stacked camera views to ``shape_meta`` (e.g. 84x84 env -> 96x96 policy)."""
        if self._target_rgb_hw is None:
            return rgb_hwc
        target_h, target_w = self._target_rgb_hw
        h, w, c = rgb_hwc.shape
        if h == target_h and w == target_w:
            return rgb_hwc
        if c % 3 != 0:
            raise ValueError(
                f"rgb observation has {c} channels; expected a multiple of 3 "
                f"(one RGB view per camera)."
            )
        resized_views = []
        for i in range(c // 3):
            view_bgr = rgb_hwc[..., i * 3 : (i + 1) * 3]
            view_rgb = view_bgr[..., ::-1]
            view_rgb = np.array(
                Image.fromarray(view_rgb).resize(
                    (target_w, target_h), Image.BILINEAR
                )
            )
            resized_views.append(view_rgb[..., ::-1])
        return np.concatenate(resized_views, axis=-1).astype(np.uint8)

    def get_observation(self, raw_obs):
        obs = {"rgb": None, "state": None}  # stack rgb if multiple cameras
        for key in self.obs_keys:
            if key in self.image_keys:
                raw_img = raw_obs[key]  # (H, W, 3) in BGR
                rgb_img = raw_img  # keep BGR for now since robomimic uses BGR
                if obs["rgb"] is None:
                    obs["rgb"] = rgb_img
                else:
                    obs["rgb"] = np.concatenate(
                        [obs["rgb"], rgb_img], axis=-1
                    )  # H W C
            else:
                if obs["state"] is None:
                    obs["state"] = raw_obs[key]
                else:
                    obs["state"] = np.concatenate([obs["state"], raw_obs[key]], axis=0)
        if self.normalize:
            obs["state"] = self.normalize_obs(obs["state"])
        obs["rgb"] = self._resize_policy_rgb(obs["rgb"].astype(np.uint8))
        return obs

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        else:
            np.random.seed()

    def reset(self, options={}, **kwargs):
        """Ignore passed-in arguments like seed"""
        self._add_last_obs_to_negative_buffer_if_failed()
        self._add_last_obs_to_positive_buffer_if_succeeded()
        self._close_video_writers()
        if "video_path" in options:
            self._open_video_writers(options["video_path"])

        # Call reset
        new_seed = options.get(
            "seed", None
        )  # used to set all environments to specified seeds
        
        # Check if init_state is passed through options (for AsyncVectorEnv compatibility)
        init_state_from_options = options.get("init_state", None)
        
        effective_init_state = init_state_from_options if init_state_from_options is not None else self.init_state
        if effective_init_state is not None:
            if not self.has_reset_before:
                # the env must be fully reset at least once to ensure correct rendering
                self.env.reset()
                self.has_reset_before = True

            # always reset to the same state to be compatible with gym
            raw_obs = self.env.reset_to({"states": effective_init_state})
        elif new_seed is not None:
            self.seed(seed=new_seed)
            raw_obs = self.env.reset()
        else:
            # random reset
            raw_obs = self.env.reset()
        
        # Modified camera fov for tool hang
        # NOTE: If using default cam settings, bc+rl still works, but rl costs roughly 1.5x samples
        env_name = None
        if hasattr(self.env, 'env') and hasattr(self.env.env, '__class__'):
            env_name = self.env.env.__class__.__name__
        elif hasattr(self.env, '__class__'):
            env_name = self.env.__class__.__name__
        
        if env_name == 'ToolHang' and not self.camera_modified:
                print('Modifying camera for ToolHang environment (first reset only)...')
                current_state = self.env.env.sim.get_state().flatten()
                
                # Load modified XML  
                modified_xml_path = "configs/task/robomimic/tool_hang_model_modified.xml"
                if os.path.exists(modified_xml_path):
                    print(f"Loading pre-modified XML from {modified_xml_path}")
                    with open(modified_xml_path, "r") as f:
                        modified_xml = f.read()
                    
                    reset_state = {
                        "states": current_state,
                        "model": modified_xml
                    }

                    raw_obs = self.env.reset_to(reset_state)
                    self.camera_modified = True 
                else:
                    print(f"Warning: Modified XML not found at {modified_xml_path}, using original camera")
        
        # Reset tracking variables for new episode
        self.success_count = 0
        self.episode_reward = 0.0
        self.step_count = 0
        self.ever_succeeded = False
        self.last_episode_succeeded = False
        self.last_raw_obs = raw_obs
        if self.video_writers:
            self._append_video_frames()

        return self.get_observation(raw_obs)

    def step(self, action):
        # If using 6D rotation, first unnormalize then convert to 7D
        if self.use_6d_rot:
            if action.shape[-1] != 10:
                raise ValueError(f"Expected 10D action when use_6d_rot=True, got {action.shape[-1]}D")
            
            # Unnormalize 10D action first (if needed)
            if self.normalize:   
                action = self.unnormalize_action(action)
                
            # Convert 10D (with 6D rotation) to 7D (with axis-angle)
            action = convert_10d_to_7d(action)
            
        else:
            # Standard path: unnormalize 7D action
            if self.normalize:
                action = self.unnormalize_action(action)
        
        raw_obs, reward, done, info = self.env.step(action)
        self.last_raw_obs = raw_obs
        env_reward = reward
        if env_reward > 0:
            self.last_episode_succeeded = True
        info["env_step_reward"] = float(env_reward)
        if self.use_dino_reward:
            dino_reward = self._compute_dino_reward(raw_obs, info)
            info["env_reward"] = env_reward
            reward = reward + dino_reward
        if self._export_dino_images:
            step_images = {
                key: np.ascontiguousarray(raw_obs[key])
                for key in self.dino_camera_keys
                if key in raw_obs
            }
            info["dino_step_images"] = step_images
            if self.robometer_compute_in_main and self.robometer_reward_weight > 0.0:
                info["robometer_step_images"] = step_images
        obs = self.get_observation(raw_obs)
        
        # Update tracking variables
        self.step_count += 1
        self.episode_reward += reward
        terminated = False
        
        # Check for success-based termination
        if env_reward > 0:  # Success detected from the sparse Robomimic task reward
            if not hasattr(self, 'success_count'):
                self.success_count = 0
            self.success_count += 1
            self.ever_succeeded = True  # Track if we ever succeeded
            if self.success_count >= self.success_steps_before_termination:  # Terminate after configured steps
                terminated = True
                # print(f"DEBUG: Episode terminating at step {self.step_count} with total_reward={self.episode_reward:.1f}")
                self.success_count = 0
                self.episode_reward = 0.0
                self.step_count = 0
                self.ever_succeeded = False
            else:
                terminated = False
        else:
            # Only reset success count if we haven't succeeded yet (enforce "once success, always success")
            if not hasattr(self, 'ever_succeeded'):
                self.ever_succeeded = False
            
            if not self.ever_succeeded:
                self.success_count = 0
            # else:
            #     print(f"WARNING: Got reward={reward} after success at step {self.step_count}! Not resetting success_count={self.success_count}")
            terminated = False
            
        if self.video_writers:
            self._append_video_frames()

        return obs, reward, terminated, info

    @staticmethod
    def _image_key_to_camera_name(image_key: str) -> str:
        if image_key.endswith("_image"):
            return image_key[: -len("_image")]
        return image_key

    def _close_video_writers(self) -> None:
        for writer in self.video_writers.values():
            try:
                writer.close()
            except Exception:
                pass
        self.video_writers = {}

    def _open_video_writers(self, video_path: str) -> None:
        """One mp4 per policy camera under ``<render_dir>/<image_key>/<stem>.mp4``."""
        root = os.path.dirname(video_path) or "."
        stem = os.path.splitext(os.path.basename(video_path))[0]
        self.video_writers = {}
        for key in self.image_keys:
            view_dir = os.path.join(root, key)
            os.makedirs(view_dir, exist_ok=True)
            out_path = os.path.join(view_dir, f"{stem}.mp4")
            self.video_writers[key] = imageio.get_writer(out_path, fps=30)

    @staticmethod
    def _pad_frame_for_ffmpeg(frame: np.ndarray) -> np.ndarray:
        pad_w = (-frame.shape[1]) % 16
        pad_h = (-frame.shape[0]) % 16
        if not pad_w and not pad_h:
            return frame
        return np.pad(
            frame,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="edge",
        )

    def _view_frame_rgb(self, image_key: str) -> np.ndarray:
        """Single-camera frame for video or ``render`` (HWC uint8, same order as NPZ/HDF5)."""
        h, w = self.render_hw
        raw_obs = self.last_raw_obs
        if raw_obs is not None and image_key in raw_obs:
            img = np.asarray(raw_obs[image_key])
        else:
            cam = self._image_key_to_camera_name(image_key)
            img = self.env.render(
                mode="rgb_array", height=h, width=w, camera_name=cam
            )
            img = np.asarray(img)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return self._pad_frame_for_ffmpeg(img)

    def _append_video_frames(self) -> None:
        for key, writer in self.video_writers.items():
            writer.append_data(self._view_frame_rgb(key))

    def render(self, mode="rgb_array"):
        if mode != "rgb_array":
            return self.env.render(mode=mode)
        if self.image_keys:
            return self._view_frame_rgb(self.image_keys[0])
        h, w = self.render_hw
        img = self.env.render(
            mode="rgb_array",
            height=h,
            width=w,
            camera_name=self.render_camera_name,
        )
        img = np.asarray(img)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return self._pad_frame_for_ffmpeg(img)
