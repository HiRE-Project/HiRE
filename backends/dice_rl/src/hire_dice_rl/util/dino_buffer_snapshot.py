"""Periodic export of DINO pos/neg buffers during finetune (additive; does not change reward math).

Checkpoints use format ``version=1`` (flat dict, see ``outputs/checkpoint_step_*.pt``):
``negative_buffer`` / ``online_positive_buffer`` ring exports, reward hyperparameters,
``offline_positive_buffer_path``, and ``env_step``. Merge with disk offline buffer via
``combine_online_snapshot_with_offline``.

Legacy v2 snapshots (``format_version=2``, event log) remain readable via
``normalize_snapshot_dict``.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

log = logging.getLogger(__name__)

CHECKPOINT_VERSION = 1
LEGACY_FORMAT_VERSION = 2
DEFAULT_CHECKPOINT_FILENAME = "checkpoint_step_{step:08d}.pt"


def _ordered_ring_tensor(
    storage: torch.Tensor, write_ptr: int, size: int
) -> torch.Tensor:
    """Return ring-buffer contents in chronological order as a contiguous clone."""
    if size <= 0:
        return storage[:0].clone()
    if size < storage.shape[0]:
        return storage[:size].clone()
    ptr = int(write_ptr) % int(storage.shape[0])
    if ptr == 0:
        return storage.clone()
    return torch.cat([storage[ptr:], storage[:ptr]], dim=0)


def _export_generator_state(buffer: Any) -> torch.Tensor:
    gen = getattr(buffer, "random_generator", None)
    if gen is None:
        return torch.empty(0, dtype=torch.uint8)
    return gen.get_state()


def export_ring_buffer_checkpoint(
    buffer: Any, *, buffer_type: str
) -> Dict[str, Any]:
    """Export ``OnlineDinoEmbeddingBuffer`` in ``checkpoint_step_*.pt`` layout."""
    if buffer is None:
        return {}
    out: Dict[str, Any] = {
        "buffer_type": str(buffer_type),
        "camera_keys": list(getattr(buffer, "camera_keys", [])),
        "max_size": int(getattr(buffer, "max_size", 0)),
        "store_images": bool(getattr(buffer, "store_images", False)),
        "per_camera": {},
        "random_generator_state": _export_generator_state(buffer),
    }
    for camera_key in out["camera_keys"]:
        size = int(buffer._sizes[camera_key])
        ptr = int(buffer._write_ptr[camera_key])
        emb_storage = buffer._buffers.get(camera_key)
        if emb_storage is None:
            out["per_camera"][camera_key] = {
                "size": 0,
                "write_ptr": 0,
                "embeddings": torch.empty(0),
            }
            continue
        cam_payload: Dict[str, Any] = {
            "size": size,
            "write_ptr": ptr,
            "embeddings": emb_storage.clone(),
        }
        if buffer.store_images:
            img_storage = buffer._image_buffers.get(camera_key)
            if img_storage is not None:
                cam_payload["images"] = img_storage.clone()
        frame_storage = buffer._frame_index_buffers.get(camera_key)
        if frame_storage is not None:
            cam_payload["env_frame_indices"] = frame_storage.clone()
        out["per_camera"][camera_key] = cam_payload
    return out


def export_online_buffer_full(buffer: Any) -> Dict[str, Any]:
    """Chronological ring export (legacy v2 / combine helpers)."""
    if buffer is None:
        return {}
    out: Dict[str, Any] = {
        "camera_keys": list(getattr(buffer, "camera_keys", [])),
        "max_size": int(getattr(buffer, "max_size", 0)),
        "store_images": bool(getattr(buffer, "store_images", False)),
        "seed": int(getattr(buffer, "seed", 0)),
        "per_camera": {},
    }
    for camera_key in out["camera_keys"]:
        size = int(buffer._sizes[camera_key])
        ptr = int(buffer._write_ptr[camera_key])
        emb_storage = buffer._buffers.get(camera_key)
        if emb_storage is None or size <= 0:
            out["per_camera"][camera_key] = {
                "size": 0,
                "write_ptr": ptr,
                "embeddings": torch.empty(0),
            }
            continue
        cam_payload: Dict[str, Any] = {
            "size": size,
            "write_ptr": ptr,
            "embeddings": _ordered_ring_tensor(emb_storage, ptr, size),
        }
        if buffer.store_images:
            img_storage = buffer._image_buffers.get(camera_key)
            if img_storage is not None:
                cam_payload["images"] = _ordered_ring_tensor(
                    img_storage, ptr, size
                )
        frame_storage = buffer._frame_index_buffers.get(camera_key)
        if frame_storage is not None:
            cam_payload["env_frame_indices"] = _ordered_ring_tensor(
                frame_storage, ptr, size
            )
        out["per_camera"][camera_key] = cam_payload
    return out


def _buffer_has_entries(buffer: Any) -> bool:
    if buffer is None:
        return False
    for camera_key in getattr(buffer, "camera_keys", []):
        if int(buffer._sizes[camera_key]) > 0:
            return True
    return False


def _snapshot_seed(shaper: Any) -> int:
    for attr in ("dino_negative_buffer", "dino_online_positive_buffer"):
        buf = getattr(shaper, attr, None)
        if buf is not None:
            return int(getattr(buf, "seed", 0))
    positive = getattr(shaper, "dino_positive_buffer", None)
    if positive is not None:
        return int(getattr(positive, "seed", 0))
    return 0


def export_offline_positive_ref(shaper: Any) -> Dict[str, Any]:
    """Lightweight pointers for merging offline buffer from disk later."""
    positive_buffer = getattr(shaper, "dino_positive_buffer", None)
    if positive_buffer is None:
        return {}
    meta = dict(getattr(positive_buffer, "metadata", {}) or {})
    return {
        "buffer_path": str(meta.get("buffer_path") or ""),
        "dataset_path": str(meta.get("dataset_path") or ""),
        "camera_keys": list(getattr(positive_buffer, "camera_keys", [])),
        "encoder_kind": str(
            meta.get("encoder_kind", getattr(shaper, "sim_encoder_kind", ""))
        ),
    }


def export_offline_positive_buffer(
    positive_buffer: Any, *, buffer_path: Optional[str] = None
) -> Dict[str, Any]:
    """Export full offline ``DinoPositiveBufferDataset`` tensors."""
    if positive_buffer is None:
        return {}
    payload: Dict[str, Any] = {
        "buffer_path": buffer_path,
        "camera_keys": list(getattr(positive_buffer, "camera_keys", [])),
        "metadata": dict(getattr(positive_buffer, "metadata", {}) or {}),
        "camera_embeddings": {},
        "camera_frame_indices": {},
        "camera_episode_indices": {},
    }
    for key in payload["camera_keys"]:
        payload["camera_embeddings"][key] = positive_buffer.embeddings_by_camera[
            key
        ].clone()
        payload["camera_frame_indices"][key] = (
            positive_buffer.frame_indices_by_camera[key].clone()
        )
        if key in getattr(positive_buffer, "images_by_camera", {}):
            payload.setdefault("camera_images", {})[key] = (
                positive_buffer.images_by_camera[key].clone()
            )
        if hasattr(positive_buffer, "frame_indices_by_camera"):
            ep = getattr(positive_buffer, "episode_indices_by_camera", None)
            if ep and key in ep:
                payload["camera_episode_indices"][key] = ep[key].clone()
    return payload


def _shaper_reward_config_snapshot(shaper: Any) -> Dict[str, Any]:
    return {
        "mode": str(getattr(shaper, "mode", "")),
        "sim_encoder_kind": str(getattr(shaper, "sim_encoder_kind", "")),
        "reward_weight": float(getattr(shaper, "reward_weight", 0.0)),
        "contrastive_lambda": float(getattr(shaper, "contrastive_lambda", 0.0)),
        "logsumexp_beta": float(getattr(shaper, "logsumexp_beta", 0.0)),
        "rel_diff_decay_enabled": bool(
            getattr(shaper, "rel_diff_decay_enabled", False)
        ),
        "rel_diff_pi": float(getattr(shaper, "rel_diff_pi", 0.0)),
        "adaptive_dense_weight_max": float(
            getattr(shaper, "adaptive_dense_weight_max", 0.0)
        ),
        "adaptive_dense_weight_min": float(
            getattr(shaper, "adaptive_dense_weight_min", 0.0)
        ),
        "adaptive_dense_weight_alpha": float(
            getattr(shaper, "adaptive_dense_weight_alpha", 0.0)
        ),
        "adaptive_success_rate_ema_decay": float(
            getattr(shaper, "adaptive_success_rate_ema_decay", 0.0)
        ),
        "adaptive_success_rate_norm_cap": float(
            getattr(shaper, "adaptive_success_rate_norm_cap", 0.0)
        ),
        "adaptive_success_rate_window_size": int(
            getattr(shaper, "adaptive_success_rate_window_size", 0)
        ),
        "shaping_per_step_mean": bool(
            getattr(shaper, "shaping_per_step_mean", False)
        ),
        "shaping_step_norm": float(getattr(shaper, "shaping_step_norm", 0.0)),
        "dino_camera_keys": list(getattr(shaper, "dino_camera_keys", [])),
        "dino_positive_sample_batch_size": int(
            getattr(shaper, "dino_positive_sample_batch_size", 0)
        ),
        "dino_positive_sampling_mode": str(
            getattr(shaper, "dino_positive_sampling_mode", "")
        ),
        "dino_negative_sample_batch_size": int(
            getattr(shaper, "dino_negative_sample_batch_size", 0)
        ),
        "dino_online_positive_add_mode": str(
            getattr(shaper, "dino_online_positive_add_mode", "")
        ),
    }


def build_shaper_snapshot(
    shaper: Any,
    train_step: int,
    *,
    train_success_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """Assemble a ``checkpoint_step_*.pt``-compatible snapshot (``version=1``)."""
    offline_ref = export_offline_positive_ref(shaper)
    neg_buf = getattr(shaper, "dino_negative_buffer", None)
    pos_buf = getattr(shaper, "dino_online_positive_buffer", None)
    sampler = getattr(shaper, "dino_positive_sampler", None)
    online_mix_ratio = (
        float(sampler.online_mix_ratio) if sampler is not None else 0.0
    )
    if train_success_rate is None:
        train_success_rate = float(
            getattr(shaper, "latest_train_success_rate", 0.0)
        )
    else:
        train_success_rate = float(train_success_rate)
    online_pos_export: Optional[Dict[str, Any]] = None
    if _buffer_has_entries(pos_buf):
        online_pos_export = export_ring_buffer_checkpoint(
            pos_buf, buffer_type="online_positive"
        )

    payload: Dict[str, Any] = {
        "version": CHECKPOINT_VERSION,
        "env_step": int(train_step),
        "timestamp": float(time.time()),
        "seed": _snapshot_seed(shaper),
        "camera_keys": list(getattr(shaper, "dino_camera_keys", [])),
        "sim_encoder": str(getattr(shaper, "sim_encoder_kind", "dino")),
        "dino_goal_source_mode": str(getattr(shaper, "mode", "")),
        "dino_contrastive_lambda": float(getattr(shaper, "contrastive_lambda", 0.0)),
        "dino_logsumexp_beta": float(getattr(shaper, "logsumexp_beta", 0.0)),
        "dino_positive_sample_batch_size": int(
            getattr(shaper, "dino_positive_sample_batch_size", 0)
        ),
        "dino_positive_sampling_mode": str(
            getattr(shaper, "dino_positive_sampling_mode", "")
        ),
        "dino_negative_sample_batch_size": int(
            getattr(shaper, "dino_negative_sample_batch_size", 0)
        ),
        "dino_online_positive_sample_ratio": online_mix_ratio,
        "dino_online_positive_sampling_mode": str(
            getattr(
                shaper,
                "dino_online_positive_sampling_mode",
                getattr(shaper, "dino_positive_sampling_mode", "uniform"),
            )
        ),
        "offline_positive_buffer_path": str(offline_ref.get("buffer_path") or ""),
        "train_success_rate": train_success_rate,
        "negative_buffer": export_ring_buffer_checkpoint(
            neg_buf, buffer_type="online_negative"
        ),
        "online_positive_buffer": online_pos_export,
    }
    return payload


def normalize_snapshot_dict(snap: Dict[str, Any]) -> Dict[str, Any]:
    """Map v1 checkpoint or legacy v2 keys to a common view for combine tooling."""
    if not isinstance(snap, dict):
        raise TypeError(f"Expected dict snapshot, got {type(snap)}")
    if snap.get("version") == CHECKPOINT_VERSION:
        out = dict(snap)
        out.setdefault("train_step", out.get("env_step"))
        out["offline_positive_ref"] = {
            "buffer_path": out.get("offline_positive_buffer_path", ""),
            "camera_keys": list(out.get("camera_keys") or []),
            "encoder_kind": str(out.get("sim_encoder") or ""),
        }
        out["online_negative"] = export_online_buffer_full_from_checkpoint(
            out.get("negative_buffer")
        )
        pos_ckpt = out.get("online_positive_buffer")
        out["online_positive"] = (
            export_online_buffer_full_from_checkpoint(pos_ckpt)
            if pos_ckpt
            else {}
        )
        return out
    if int(snap.get("format_version", 0)) == LEGACY_FORMAT_VERSION:
        out = dict(snap)
        out.setdefault("env_step", out.get("train_step"))
        out.setdefault(
            "offline_positive_buffer_path",
            (out.get("offline_positive_ref") or {}).get("buffer_path", ""),
        )
        return out
    out = dict(snap)
    out.setdefault("train_step", out.get("env_step", out.get("train_step")))
    out.setdefault("env_step", out.get("train_step"))
    return out


def export_online_buffer_full_from_checkpoint(
    buffer_ckpt: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Convert v1 ``negative_buffer`` / ``online_positive_buffer`` to legacy chronological view."""
    if not buffer_ckpt:
        return {}
    out: Dict[str, Any] = {
        "camera_keys": list(buffer_ckpt.get("camera_keys") or []),
        "max_size": int(buffer_ckpt.get("max_size", 0)),
        "store_images": bool(buffer_ckpt.get("store_images", False)),
        "per_camera": {},
    }
    for camera_key, cam in (buffer_ckpt.get("per_camera") or {}).items():
        size = int(cam.get("size", 0))
        ptr = int(cam.get("write_ptr", 0))
        emb_storage = cam.get("embeddings")
        if emb_storage is None or not torch.is_tensor(emb_storage) or size <= 0:
            out["per_camera"][camera_key] = {
                "size": 0,
                "write_ptr": ptr,
                "embeddings": torch.empty(0),
            }
            continue
        cam_payload: Dict[str, Any] = {
            "size": size,
            "write_ptr": ptr,
            "embeddings": _ordered_ring_tensor(emb_storage, ptr, size),
        }
        if out["store_images"] and cam.get("images") is not None:
            cam_payload["images"] = _ordered_ring_tensor(
                cam["images"], ptr, size
            )
        if cam.get("env_frame_indices") is not None:
            cam_payload["env_frame_indices"] = _ordered_ring_tensor(
                cam["env_frame_indices"], ptr, size
            )
        out["per_camera"][camera_key] = cam_payload
    return out


def checkpoint_filename_for_step(
    step: int, template: str = DEFAULT_CHECKPOINT_FILENAME
) -> str:
    return str(template).format(step=int(step))


def save_shaper_snapshot(
    shaper: Any,
    train_step: int,
    output_path: str,
    *,
    train_success_rate: Optional[float] = None,
) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    payload = build_shaper_snapshot(
        shaper, train_step, train_success_rate=train_success_rate
    )
    torch.save(payload, output_path)
    log.info("Saved DINO buffer snapshot to %s (train_step=%s)", output_path, train_step)
    return output_path


def resolve_offline_positive_buffer_path(
    *,
    data_dir: str,
    benchmark: str,
    env_name: str,
    sim_encoder: str = "dino",
) -> str:
    """Standard path: ``<data_dir>/<benchmark>/<env>-img/ph_pretrain/<encoder>_positive_buffer.pt``."""
    return os.path.join(
        os.path.expandvars(data_dir),
        benchmark,
        f"{env_name}-img",
        "ph_pretrain",
        f"{sim_encoder}_positive_buffer.pt",
    )


def load_offline_positive_from_path(buffer_path: str) -> Dict[str, Any]:
    """Load full offline positive export from a ``*_positive_buffer.pt`` checkpoint."""
    from hire_dice_rl.util.dino_prompt_buffer import DinoPositiveBufferDataset

    path = os.path.expandvars(os.path.expanduser(str(buffer_path)))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Offline DINO positive buffer not found: {path}")
    dataset = DinoPositiveBufferDataset(buffer_path=path)
    return export_offline_positive_buffer(dataset, buffer_path=path)


def combine_online_snapshot_with_offline(
    online_snapshot_path: str,
    *,
    offline_buffer_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    benchmark: Optional[str] = None,
    env_name: Optional[str] = None,
    sim_encoder: str = "dino",
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Merge online-only finetune snapshot with offline positive buffer from disk.

    v1 snapshots may still contain inline ``offline_positive``; v2 stores
    ``offline_positive_ref`` only. Pass ``offline_buffer_path`` explicitly, or
    provide ``data_dir`` + ``benchmark`` + ``env_name`` to resolve the default path.
  """
    snap = torch.load(
        os.path.expandvars(online_snapshot_path), map_location="cpu", weights_only=False
    )
    if not isinstance(snap, dict):
        raise TypeError(f"Expected dict snapshot, got {type(snap)}")
    snap = normalize_snapshot_dict(snap)

    offline: Dict[str, Any] = {}
    if snap.get("format_version", 1) < 2 and snap.get("offline_positive"):
        log.info("Using inline offline_positive from v1 snapshot")
        offline = snap["offline_positive"]
    else:
        ref = snap.get("offline_positive_ref") or {}
        path = (
            offline_buffer_path
            or snap.get("offline_positive_buffer_path")
            or ref.get("buffer_path")
            or ""
        )
        path = str(path).strip()
        if not path:
            if not all([data_dir, benchmark, env_name]):
                raise ValueError(
                    "Provide offline_buffer_path, or data_dir + benchmark + env_name, "
                    "or a snapshot with offline_positive_ref.buffer_path"
                )
            path = resolve_offline_positive_buffer_path(
                data_dir=str(data_dir),
                benchmark=str(benchmark),
                env_name=str(env_name),
                sim_encoder=str(ref.get("encoder_kind") or sim_encoder),
            )
        log.info("Loading offline positive buffer from %s", path)
        offline = load_offline_positive_from_path(path)

    combined: Dict[str, Any] = dict(snap)
    combined["offline_positive"] = offline
    combined["combined_at_unix"] = float(time.time())
    if output_path:
        out = os.path.expandvars(os.path.expanduser(output_path))
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        torch.save(combined, out)
        log.info("Wrote combined DINO buffer snapshot to %s", out)
    return combined


class DinoBufferSnapshotRecorder:
    """Append-only log of pos/neg buffer insertions (failure metadata + tensors)."""

    def __init__(self, *, clear_events_after_save: bool = True):
        self.clear_events_after_save = bool(clear_events_after_save)
        self._events: List[Dict[str, Any]] = []
        self.events_since_step: int = 0
        self.current_train_step: int = 0

    def set_train_step(self, train_step: int) -> None:
        self.current_train_step = int(train_step)

    def _entry_from_images(
        self,
        camera_key: str,
        imgs: Sequence[np.ndarray],
        embeddings: torch.Tensor,
    ) -> Dict[str, Any]:
        emb = embeddings.detach().cpu()
        entry: Dict[str, Any] = {"embedding": emb}
        if len(imgs) > 0:
            stack = np.stack([np.asarray(im) for im in imgs], axis=0)
            entry["image_hwc"] = np.ascontiguousarray(stack, dtype=np.uint8)
        return entry

    def record_negative_buffer_update(
        self,
        *,
        env_idx: int,
        env_reward_chunk_sum: float,
        env_substep_count: int,
        add_mode: str,
        per_camera_imgs: Mapping[str, Sequence[np.ndarray]],
        per_camera_embeddings: Mapping[str, torch.Tensor],
        failure_substep_indices: Optional[Sequence[int]] = None,
        env_frame_indices: Optional[Sequence[int]] = None,
    ) -> None:
        per_camera: Dict[str, Any] = {}
        for camera_key, emb in per_camera_embeddings.items():
            imgs = list(per_camera_imgs.get(camera_key, []))
            per_camera[camera_key] = self._entry_from_images(camera_key, imgs, emb)
        self._events.append(
            {
                "kind": "negative",
                "train_step": int(self.current_train_step),
                "env_idx": int(env_idx),
                "env_reward_chunk_sum": float(env_reward_chunk_sum),
                "env_substep_count": int(env_substep_count),
                "add_mode": str(add_mode),
                "failure_substep_indices": (
                    list(failure_substep_indices)
                    if failure_substep_indices is not None
                    else None
                ),
                "env_frame_indices": (
                    list(env_frame_indices)
                    if env_frame_indices is not None
                    else None
                ),
                "per_camera": per_camera,
            }
        )

    def record_online_positive_buffer_update(
        self,
        *,
        env_idx: int,
        env_reward_chunk_sum: float,
        env_substep_count: int,
        add_mode: str,
        per_camera_imgs: Mapping[str, Sequence[np.ndarray]],
        per_camera_embeddings: Mapping[str, torch.Tensor],
        success_substep_indices: Optional[Sequence[int]] = None,
        env_frame_indices: Optional[Sequence[int]] = None,
    ) -> None:
        per_camera: Dict[str, Any] = {}
        for camera_key, emb in per_camera_embeddings.items():
            imgs = list(per_camera_imgs.get(camera_key, []))
            per_camera[camera_key] = self._entry_from_images(camera_key, imgs, emb)
        self._events.append(
            {
                "kind": "online_positive",
                "train_step": int(self.current_train_step),
                "env_idx": int(env_idx),
                "env_reward_chunk_sum": float(env_reward_chunk_sum),
                "env_substep_count": int(env_substep_count),
                "add_mode": str(add_mode),
                "success_substep_indices": (
                    list(success_substep_indices)
                    if success_substep_indices is not None
                    else None
                ),
                "env_frame_indices": (
                    list(env_frame_indices)
                    if env_frame_indices is not None
                    else None
                ),
                "per_camera": per_camera,
            }
        )

    def export_events(self) -> List[Dict[str, Any]]:
        return list(self._events)

    def clear_events(self) -> None:
        self._events.clear()
        self.events_since_step = int(self.current_train_step)


def parse_snapshot_cfg(cfg: Any) -> Dict[str, Any]:
    """Read ``dino_buffer_snapshot`` from Hydra cfg (dict or OmegaConf)."""
    raw = {}
    if cfg is None:
        return raw
    if hasattr(cfg, "get"):
        raw = cfg.get("dino_buffer_snapshot", {}) or {}
    if hasattr(raw, "items") and not isinstance(raw, dict):
        try:
            from omegaconf import OmegaConf

            raw = OmegaConf.to_container(raw, resolve=True) or {}
        except Exception:
            raw = dict(raw)
    if not isinstance(raw, dict):
        raw = {}
    return raw


def _images_list_to_uint8_tensor(imgs: Sequence[np.ndarray]) -> torch.Tensor:
    stack = np.stack([np.asarray(im) for im in imgs], axis=0)
    return torch.from_numpy(np.ascontiguousarray(stack, dtype=np.uint8))


def apply_terminal_buffer_updates(
    shaper: Any,
    info_venv: Sequence[dict],
    terminated_venv: Optional[np.ndarray],
    truncated_venv: Optional[np.ndarray],
    env_substep_count: np.ndarray,
    *,
    buffer_attr: str,
    require_success: bool,
    add_mode: str,
    event_kind: str,
) -> None:
    """Mirror ``DinoRewardShaper`` terminal buffer updates (per-env), with optional snapshot log."""
    buffer = getattr(shaper, buffer_attr, None)
    if buffer is None:
        return
    recorder = getattr(shaper, "buffer_snapshot_recorder", None)
    camera_keys = list(getattr(shaper, "dino_camera_keys", []))

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
        env_reward_sum = float(info_venv[env_idx].get("env_reward_chunk_sum", 0.0))
        if require_success:
            if env_reward_sum <= 0:
                continue
        elif env_reward_sum > 0:
            continue
        chunk = info_venv[env_idx].get("dino_chunk_images")
        if not chunk:
            continue
        frames = chunk if add_mode == "all_substeps" else [chunk[-1]]
        chunk_frame_indices = list(
            info_venv[env_idx].get("dino_chunk_env_frame_indices") or []
        )
        if add_mode == "all_substeps":
            env_frame_indices = chunk_frame_indices[: len(frames)]
        else:
            env_frame_indices = (
                [int(chunk_frame_indices[-1])] if chunk_frame_indices else []
            )
        substep_indices = (
            list(range(len(chunk)))
            if add_mode == "all_substeps"
            else [len(chunk) - 1]
        )
        per_camera_imgs: Dict[str, List[np.ndarray]] = {k: [] for k in camera_keys}
        for frame in frames:
            if not isinstance(frame, Mapping):
                continue
            for key in camera_keys:
                img = frame.get(key)
                if img is not None:
                    per_camera_imgs[key].append(np.asarray(img))
        per_camera_embeddings: Dict[str, torch.Tensor] = {}
        for camera_key, imgs in per_camera_imgs.items():
            if not imgs:
                continue
            batch = shaper._stack_images_to_tensor(imgs)
            embeddings = shaper.encoder.compute_embeddings(batch).detach()
            if buffer.store_images:
                buffer.add_embeddings(
                    camera_key,
                    embeddings,
                    images=_images_list_to_uint8_tensor(imgs),
                    env_frame_indices=env_frame_indices,
                )
            else:
                buffer.add_embeddings(
                    camera_key,
                    embeddings,
                    env_frame_indices=env_frame_indices,
                )
            per_camera_embeddings[camera_key] = embeddings
        if recorder is None or not per_camera_embeddings:
            continue
        if event_kind == "negative":
            recorder.record_negative_buffer_update(
                env_idx=env_idx,
                env_reward_chunk_sum=env_reward_sum,
                env_substep_count=count,
                add_mode=add_mode,
                per_camera_imgs=per_camera_imgs,
                per_camera_embeddings=per_camera_embeddings,
                failure_substep_indices=substep_indices,
                env_frame_indices=env_frame_indices,
            )
        elif event_kind == "online_positive":
            recorder.record_online_positive_buffer_update(
                env_idx=env_idx,
                env_reward_chunk_sum=env_reward_sum,
                env_substep_count=count,
                add_mode=add_mode,
                per_camera_imgs=per_camera_imgs,
                per_camera_embeddings=per_camera_embeddings,
                success_substep_indices=substep_indices,
                env_frame_indices=env_frame_indices,
            )


class DinoBufferSnapshotManager:
    """Save DINO buffer snapshots on an interval independent of ``train.save_freq``."""

    def __init__(
        self,
        shaper: Any,
        *,
        logdir: str,
        save_interval: int,
        output_subdir: str = "dino_buffer_snapshots",
        clear_events_after_save: bool = True,
        save_at_end: bool = True,
        checkpoint_filename_template: str = DEFAULT_CHECKPOINT_FILENAME,
    ):
        if shaper is None:
            raise ValueError("DinoBufferSnapshotManager requires a DinoRewardShaper")
        self.shaper = shaper
        self.logdir = str(logdir)
        self.save_interval = max(1, int(save_interval))
        self.output_dir = os.path.join(self.logdir, output_subdir)
        self.save_at_end = bool(save_at_end)
        self.checkpoint_filename_template = str(
            checkpoint_filename_template or DEFAULT_CHECKPOINT_FILENAME
        )
        self._last_saved_step = -1
        self.recorder = DinoBufferSnapshotRecorder(
            clear_events_after_save=clear_events_after_save
        )
        shaper.buffer_snapshot_recorder = self.recorder
        os.makedirs(self.output_dir, exist_ok=True)
        log.info(
            "DINO buffer snapshots enabled: dir=%s interval=%s",
            self.output_dir,
            self.save_interval,
        )
        self._warn_if_images_not_stored()

    def _warn_if_images_not_stored(self) -> None:
        neg = getattr(self.shaper, "dino_negative_buffer", None)
        pos = getattr(self.shaper, "dino_online_positive_buffer", None)
        if neg is not None and not bool(getattr(neg, "store_images", False)):
            log.warning(
                "dino_buffer_snapshot: online negative buffer store_images=false; "
                "ring snapshot will omit raw images (event log still has HWC if recorded)."
            )
        if pos is not None and not bool(getattr(pos, "store_images", False)):
            log.warning(
                "dino_buffer_snapshot: online positive buffer store_images=false; "
                "ring snapshot will omit raw images (event log still has HWC if recorded)."
            )

    def set_train_step(self, train_step: int) -> None:
        self.recorder.set_train_step(train_step)

    def maybe_save(
        self,
        train_step: int,
        *,
        force: bool = False,
        train_success_rate: Optional[float] = None,
    ) -> Optional[str]:
        step = int(train_step)
        self.set_train_step(step)
        if not force:
            if step <= 0:
                return None
            if step % self.save_interval != 0:
                return None
            if step == self._last_saved_step:
                return None
        fname = checkpoint_filename_for_step(
            step, self.checkpoint_filename_template
        )
        path = os.path.join(self.output_dir, fname)
        save_shaper_snapshot(
            self.shaper,
            step,
            path,
            train_success_rate=train_success_rate,
        )
        self._last_saved_step = step
        if self.recorder.clear_events_after_save:
            self.recorder.clear_events()
        return path

    def save_final(
        self,
        train_step: int,
        *,
        train_success_rate: Optional[float] = None,
    ) -> Optional[str]:
        if not self.save_at_end:
            return None
        if int(train_step) == self._last_saved_step:
            return None
        return self.maybe_save(
            train_step,
            force=True,
            train_success_rate=train_success_rate,
        )
