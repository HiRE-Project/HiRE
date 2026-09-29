import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

log = logging.getLogger(__name__)


class DinoPositiveBufferDataset:
    """Positive DINO embedding buffer stored as a torch checkpoint."""

    def __init__(
        self,
        buffer_path: str,
        camera_keys: Optional[List[str]] = None,
        seed: int = 0,
        sampling_mode: str = "random",
    ):
        if not os.path.exists(buffer_path):
            raise FileNotFoundError(
                f"DINO positive buffer not found: {buffer_path}. Build it first."
            )
        payload = torch.load(buffer_path, map_location="cpu")
        camera_embeddings = payload.get("camera_embeddings", {})
        camera_frame_indices = payload.get("camera_frame_indices", {})
        camera_images = payload.get("camera_images", {})
        if not camera_embeddings:
            raise ValueError(f"Buffer has no camera embeddings: {buffer_path}")
        if not camera_frame_indices:
            raise ValueError(
                "Buffer missing `camera_frame_indices`; rebuild it with the latest builder."
            )

        if camera_keys is None:
            camera_keys = sorted(camera_embeddings.keys())
        self.camera_keys = list(camera_keys)
        self.embeddings_by_camera = {}
        self.frame_indices_by_camera = {}
        self.images_by_camera = {}
        self.unique_frame_values_by_camera = {}
        self.frame_to_row_indices_by_camera = {}
        for camera_key in self.camera_keys:
            if camera_key not in camera_embeddings:
                raise KeyError(
                    f"Camera key '{camera_key}' not found in buffer. "
                    f"Available keys: {list(camera_embeddings.keys())}"
                )
            embeddings = camera_embeddings[camera_key]
            frame_indices = camera_frame_indices.get(camera_key)
            if not torch.is_tensor(embeddings) or embeddings.ndim != 3:
                raise ValueError(
                    f"Expected embeddings [{camera_key}] with shape [N,P,D], "
                    f"got {type(embeddings)}"
                )
            if not torch.is_tensor(frame_indices) or frame_indices.ndim != 1:
                raise ValueError(
                    f"Expected frame indices [{camera_key}] with shape [N], "
                    f"got {type(frame_indices)}"
                )
            if int(embeddings.shape[0]) != int(frame_indices.shape[0]):
                raise ValueError(
                    f"Length mismatch for {camera_key}: embeddings={embeddings.shape[0]} "
                    f"frame_indices={frame_indices.shape[0]}"
                )

            embeddings = embeddings.contiguous().float()
            frame_indices = frame_indices.contiguous().long()
            self.embeddings_by_camera[camera_key] = embeddings
            self.frame_indices_by_camera[camera_key] = frame_indices
            if camera_key in camera_images:
                images = camera_images[camera_key]
                if (
                    torch.is_tensor(images)
                    and images.ndim == 4
                    and int(images.shape[0]) == int(embeddings.shape[0])
                ):
                    self.images_by_camera[camera_key] = images.contiguous().to(torch.uint8)

            unique_frames = torch.unique(frame_indices, sorted=True)
            self.unique_frame_values_by_camera[camera_key] = unique_frames
            self.frame_to_row_indices_by_camera[camera_key] = {
                int(t.item()): torch.where(frame_indices == t)[0].contiguous()
                for t in unique_frames
            }

        self.metadata = payload.get("metadata", {})
        self.random_generator = torch.Generator(device="cpu")
        self.random_generator.manual_seed(int(seed))
        self.sampling_mode = str(sampling_mode).strip().lower()
        if self.sampling_mode not in ("random", "uniform"):
            raise ValueError(
                f"Unsupported sampling_mode={self.sampling_mode}. "
                "Expected one of: random, uniform."
            )

    def num_samples(self, camera_key: str) -> int:
        return int(self.embeddings_by_camera[camera_key].shape[0])

    def num_unique_timesteps(self, camera_key: str) -> int:
        return int(self.unique_frame_values_by_camera[camera_key].shape[0])

    def has_images(self, camera_key: str) -> bool:
        return camera_key in self.images_by_camera

    def dump_images(
        self,
        camera_key: str,
        save_dir: str,
        max_images: int = 64,
        filename_prefix: str = "pos",
    ) -> int:
        if camera_key not in self.images_by_camera:
            return 0
        images = self.images_by_camera[camera_key]
        n = min(int(max_images), int(images.shape[0]))
        os.makedirs(save_dir, exist_ok=True)
        for i in range(n):
            Image.fromarray(images[i].cpu().numpy()).save(
                os.path.join(save_dir, f"{filename_prefix}_{i:06d}.png")
            )
        return n

    def sample_batch(
        self,
        camera_key: str,
        batch_size: int,
        device: Optional[torch.device] = None,
        sampling_mode: Optional[str] = None,
    ) -> torch.Tensor:
        if camera_key not in self.embeddings_by_camera:
            raise KeyError(
                f"Unknown camera key '{camera_key}'. Available: {self.camera_keys}"
            )
        source = self.embeddings_by_camera[camera_key]
        n = int(source.shape[0])
        if n <= 0:
            raise RuntimeError(f"No embeddings found for camera '{camera_key}'")
        k = int(batch_size)
        if k <= 0:
            raise ValueError(f"batch_size must be > 0, got {k}")

        mode = (sampling_mode or self.sampling_mode).strip().lower()
        if mode == "random":
            indices = torch.randint(
                low=0, high=n, size=(k,), generator=self.random_generator
            ).long()
        elif mode == "uniform":
            unique_frames = self.unique_frame_values_by_camera[camera_key]
            num_unique = int(unique_frames.shape[0])
            if k > num_unique:
                raise ValueError(
                    f"Cannot uniformly sample {k} timesteps from {camera_key}; "
                    f"buffer only has {num_unique} unique timesteps."
                )
            chosen_pos = torch.floor(
                torch.arange(k, dtype=torch.float32) * (float(num_unique) / float(k))
            ).long()
            chosen_pos = torch.clamp(chosen_pos, min=0, max=num_unique - 1)
            chosen_t = unique_frames.index_select(0, chosen_pos)
            rows = []
            frame_to_rows = self.frame_to_row_indices_by_camera[camera_key]
            for t in chosen_t.tolist():
                candidates = frame_to_rows[int(t)]
                ridx = torch.randint(
                    low=0,
                    high=int(candidates.shape[0]),
                    size=(1,),
                    generator=self.random_generator,
                ).item()
                rows.append(candidates[ridx])
            indices = torch.stack(rows, dim=0).long()
        else:
            raise ValueError(
                f"Unsupported sampling_mode={mode}. Expected one of: random, uniform."
            )

        batch = source.index_select(0, indices)
        if device is not None:
            batch = batch.to(device=device)
        return batch


from hire.buffers import EmbeddingBuffer as OnlineDinoEmbeddingBuffer

# Backward-compatible alias used by contrastive negative-buffer code paths.
OnlineDinoNegativeBuffer = OnlineDinoEmbeddingBuffer
OnlineDinoPositiveBuffer = OnlineDinoEmbeddingBuffer


def _cfg_mapping_to_dict(cfg: Any) -> dict:
    """Normalize plain dict or OmegaConf DictConfig to a resolved dict."""
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    if hasattr(cfg, "items"):
        try:
            from omegaconf import OmegaConf

            container = OmegaConf.to_container(cfg, resolve=True)
            if isinstance(container, dict):
                return container
        except Exception:
            pass
        try:
            return dict(cfg)
        except Exception:
            return {}
    return {}


def parse_online_positive_buffer_cfg(
    positive_cfg: Optional[dict],
) -> Tuple[bool, dict]:
    """Return (enabled, options) for nested `dino_positive_buffer.online` config."""
    positive_cfg = _cfg_mapping_to_dict(positive_cfg)
    if not positive_cfg:
        return False, {}
    online_cfg = _cfg_mapping_to_dict(positive_cfg.get("online", {}) or {})
    if not online_cfg:
        return False, {}
    enabled = bool(online_cfg.get("enabled", False))
    return enabled, online_cfg


def build_online_positive_buffer(
    camera_keys: List[str],
    online_cfg: dict,
) -> OnlineDinoEmbeddingBuffer:
    return OnlineDinoEmbeddingBuffer(
        camera_keys=list(camera_keys),
        max_size=int(online_cfg.get("buffer_size", 128)),
        seed=int(online_cfg.get("seed", 0)),
        store_images=bool(online_cfg.get("store_images", False)),
    )


class HybridDinoPositiveSampler:
    """Mix offline pretrain positives with online success embeddings."""

    def __init__(
        self,
        offline: DinoPositiveBufferDataset,
        online: Optional[OnlineDinoEmbeddingBuffer] = None,
        online_mix_ratio: float = 0.5,
    ):
        if not (0.0 <= float(online_mix_ratio) <= 1.0):
            raise ValueError(
                f"online_mix_ratio must be in [0, 1], got {online_mix_ratio}"
            )
        self.offline = offline
        self.online = online
        self.online_mix_ratio = float(online_mix_ratio)
        self.camera_keys = list(offline.camera_keys)

    def num_samples(self, camera_key: str) -> int:
        n = int(self.offline.num_samples(camera_key))
        if self.online is not None:
            n += int(self.online.size(camera_key))
        return n

    def sample_batch(
        self,
        camera_key: str,
        batch_size: int,
        device: Optional[torch.device] = None,
        sampling_mode: Optional[str] = None,
    ) -> torch.Tensor:
        k = int(batch_size)
        if k <= 0:
            raise ValueError(f"batch_size must be > 0, got {k}")

        online_n = int(self.online.size(camera_key)) if self.online is not None else 0
        ratio = float(self.online_mix_ratio)
        if self.online is None or online_n <= 0 or ratio <= 0.0:
            return self.offline.sample_batch(
                camera_key=camera_key,
                batch_size=k,
                device=device,
                sampling_mode=sampling_mode,
            )

        k_online = int(round(k * ratio))
        k_online = max(0, min(k, k_online))
        k_offline = k - k_online
        parts: List[torch.Tensor] = []

        if k_offline > 0:
            parts.append(
                self.offline.sample_batch(
                    camera_key=camera_key,
                    batch_size=k_offline,
                    device=device,
                    sampling_mode=sampling_mode,
                )
            )
        if k_online > 0:
            parts.append(
                self.online.sample_batch(
                    camera_key=camera_key,
                    batch_size=k_online,
                    device=device,
                )
            )
        if not parts:
            return self.offline.sample_batch(
                camera_key=camera_key,
                batch_size=k,
                device=device,
                sampling_mode=sampling_mode,
            )
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=0)


class RobomimicNpzDinoPositiveBufferBuilder:
    """Build positive DINO embeddings directly from processed DICE-RL NPZ images."""

    def __init__(
        self,
        dataset_path: str,
        output_path: str,
        camera_keys: List[str],
        encoder,
        device: str = "cuda",
        frame_stride: int = 5,
        max_episodes: Optional[int] = None,
        max_frames_per_episode: Optional[int] = None,
        encode_batch_size: int = 64,
        save_images_in_buffer: bool = False,
        encoder_kind: str = "dino",
    ):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.camera_keys = list(camera_keys)
        self.encoder = encoder
        self.device = device
        self.frame_stride = max(1, int(frame_stride))
        self.max_episodes = None if max_episodes in (None, 0) else int(max_episodes)
        self.max_frames_per_episode = (
            None
            if max_frames_per_episode in (None, 0)
            else int(max_frames_per_episode)
        )
        self.encode_batch_size = max(1, int(encode_batch_size))
        self.save_images_in_buffer = bool(save_images_in_buffer)
        self.encoder_kind = (encoder_kind or "dino").strip().lower()

    @staticmethod
    def _trajectory_slices(traj_lengths: Iterable[int]) -> Iterable[Tuple[int, int, int]]:
        start = 0
        for episode_id, length in enumerate(traj_lengths):
            end = start + int(length)
            yield episode_id, start, end
            start = end

    def _camera_frames(self, images: np.ndarray, camera_index: int) -> np.ndarray:
        if images.ndim != 4:
            raise ValueError(f"Expected images with 4 dims, got {images.shape}")
        num_cameras = len(self.camera_keys)
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

    def _selected_indices(self, traj_lengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rows = []
        frame_indices = []
        selected_episodes = list(self._trajectory_slices(traj_lengths))
        if self.max_episodes is not None:
            selected_episodes = selected_episodes[: self.max_episodes]
        for _episode_id, start, end in selected_episodes:
            local = np.arange(0, end - start, self.frame_stride, dtype=np.int64)
            if self.max_frames_per_episode is not None:
                local = local[: self.max_frames_per_episode]
            rows.append(start + local)
            frame_indices.append(local)
        if not rows:
            raise ValueError("No trajectory frames selected for DINO positive buffer.")
        return np.concatenate(rows), np.concatenate(frame_indices)

    def _encode_frames(self, frames_hwc: np.ndarray) -> torch.Tensor:
        tensors = torch.from_numpy(np.ascontiguousarray(frames_hwc)).permute(0, 3, 1, 2)
        tensors = tensors.contiguous().to(device=self.device).float()
        chunks = torch.split(tensors, self.encode_batch_size, dim=0)
        embeddings = []
        with torch.no_grad():
            for chunk in chunks:
                embeddings.append(self.encoder.compute_embeddings(chunk).detach().cpu())
        return torch.cat(embeddings, dim=0).contiguous().float()

    def build(self) -> Dict[str, torch.Tensor]:
        data = np.load(self.dataset_path, allow_pickle=True)
        if "images" not in data.files:
            raise KeyError(f"{self.dataset_path} does not contain an `images` array")
        if "traj_lengths" not in data.files:
            raise KeyError(f"{self.dataset_path} does not contain `traj_lengths`")

        images = data["images"]
        traj_lengths = data["traj_lengths"]
        selected_rows, frame_indices = self._selected_indices(traj_lengths)
        frame_idx_t = torch.from_numpy(frame_indices).long().contiguous()
        episode_idx_t = []
        for episode_id, start, end in self._trajectory_slices(traj_lengths):
            if self.max_episodes is not None and episode_id >= self.max_episodes:
                break
            local = np.arange(0, end - start, self.frame_stride, dtype=np.int64)
            if self.max_frames_per_episode is not None:
                local = local[: self.max_frames_per_episode]
            episode_idx_t.append(torch.full((len(local),), int(episode_id), dtype=torch.long))
        episode_idx_t = torch.cat(episode_idx_t, dim=0).contiguous()

        camera_embeddings = {}
        camera_frame_indices = {}
        camera_episode_indices = {}
        camera_images = {}
        metadata = {
            "buffer_path": os.path.abspath(self.output_path),
            "dataset_path": self.dataset_path,
            "encoder_kind": self.encoder_kind,
            "frame_stride": self.frame_stride,
            "max_episodes": self.max_episodes,
            "max_frames_per_episode": self.max_frames_per_episode,
            "encode_batch_size": self.encode_batch_size,
            "save_images_in_buffer": self.save_images_in_buffer,
            "camera_stats": {},
        }

        for camera_index, camera_key in enumerate(self.camera_keys):
            frames = self._camera_frames(images, camera_index)[selected_rows]
            embeddings = self._encode_frames(frames)
            camera_embeddings[camera_key] = embeddings
            camera_frame_indices[camera_key] = frame_idx_t.clone()
            camera_episode_indices[camera_key] = episode_idx_t.clone()
            if self.save_images_in_buffer:
                camera_images[camera_key] = torch.from_numpy(
                    np.ascontiguousarray(frames)
                ).to(torch.uint8)
            cam_stat = {
                "num_embeddings": int(embeddings.shape[0]),
                "num_unique_timesteps": int(torch.unique(frame_idx_t).shape[0]),
                "embedding_shape": list(embeddings.shape[1:]),
            }
            if self.save_images_in_buffer:
                cam_stat["image_shape"] = list(frames.shape[1:])
            metadata["camera_stats"][camera_key] = cam_stat
            log.info(
                "Built DINO positive buffer for %s: embeddings=%s unique_timesteps=%s",
                camera_key,
                int(embeddings.shape[0]),
                int(torch.unique(frame_idx_t).shape[0]),
            )

        payload = {
            "camera_embeddings": camera_embeddings,
            "camera_frame_indices": camera_frame_indices,
            "camera_episode_indices": camera_episode_indices,
            "metadata": metadata,
        }
        if self.save_images_in_buffer:
            payload["camera_images"] = camera_images
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        torch.save(payload, self.output_path)
        log.info("Saved DINO positive buffer to %s", self.output_path)
        return camera_embeddings


