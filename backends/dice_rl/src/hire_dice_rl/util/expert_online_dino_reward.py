"""On-the-fly DINO shaping for RLPD expert samples using the live ``DinoRewardShaper``."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from hire_dice_rl.util.offline_dino_reward import (
    compute_episode_dino_shaping,
    split_npz_images_to_hwc,
)

log = logging.getLogger(__name__)


def shaper_buffer_state_version(shaper: Any) -> Tuple[Any, ...]:
    """Hashable tag for cache invalidation when pos/neg buffers or shaper config change.

    ``adaptive_dense_weight`` is intentionally excluded: it only scales PBRS output
    linearly and is applied at sample time (see ``_adaptive_dense_scale``).
    """
    keys = list(getattr(shaper, "dino_camera_keys", []) or [])
    neg_sizes = ()
    if getattr(shaper, "dino_negative_buffer", None) is not None:
        neg_sizes = tuple(
            int(shaper.dino_negative_buffer.size(k)) for k in keys
        )
    pos_online_sizes = ()
    if getattr(shaper, "dino_online_positive_buffer", None) is not None:
        pos_online_sizes = tuple(
            int(shaper.dino_online_positive_buffer.size(k)) for k in keys
        )
    return (
        neg_sizes,
        pos_online_sizes,
        str(getattr(shaper, "mode", "")),
        float(getattr(shaper, "contrastive_lambda", 0.0)),
        float(getattr(shaper, "reward_weight", 0.0)),
        bool(getattr(shaper, "rel_diff_decay_enabled", False)),
        float(getattr(shaper, "rel_diff_pi", 0.0)),
        bool(getattr(shaper, "shaping_per_step_mean", False)),
        float(getattr(shaper, "shaping_step_norm", 0.0)),
    )


def _adaptive_dense_scale(shaper: Any) -> float:
    """Scalar applied to cached unscaled expert DINO (matches online PBRS semantics)."""
    if not bool(getattr(shaper, "rel_diff_decay_enabled", False)):
        return 1.0
    return float(getattr(shaper, "adaptive_dense_weight", 0.0))


class ExpertOnlineDinoRewardComputer:
    """Add DINO dense reward to expert RLPD samples with the **current** pos/neg buffers."""

    def __init__(
        self,
        expert_dataset: Any,
        *,
        horizon_steps: int,
        gamma: float = 0.99,
        use_n_step: bool = False,
        n_step: int = 1,
        encode_batch_size: int = 64,
    ):
        if not getattr(expert_dataset, "use_img", False):
            raise ValueError(
                "ExpertOnlineDinoRewardComputer requires an image expert dataset "
                "(use_img=true)."
            )
        if not hasattr(expert_dataset, "images"):
            raise ValueError("Expert dataset must expose `images` for online DINO.")

        self.dataset = expert_dataset
        self.horizon_steps = int(horizon_steps)
        self.gamma = float(gamma)
        self.use_n_step = bool(use_n_step)
        self.n_step = int(n_step)
        self.encode_batch_size = int(encode_batch_size)

        traj_lengths = np.asarray(expert_dataset.traj_lengths, dtype=np.int64)
        starts = [0]
        for length in traj_lengths:
            starts.append(starts[-1] + int(length))
        self._episode_starts = np.asarray(starts[:-1], dtype=np.int64)
        self._episode_ends = np.asarray(starts[1:], dtype=np.int64)

        self._cache_version: Optional[Tuple[Any, ...]] = None
        self._episode_dino: Dict[int, np.ndarray] = {}

    def _episode_id_for_global_step(self, global_start: int) -> int:
        idx = int(
            np.searchsorted(self._episode_ends, global_start, side="right")
        )
        if idx >= len(self._episode_starts):
            idx = len(self._episode_starts) - 1
        ep_start = int(self._episode_starts[idx])
        ep_end = int(self._episode_ends[idx])
        if global_start < ep_start or global_start >= ep_end:
            raise ValueError(
                f"global_start={global_start} outside episode {idx} "
                f"[{ep_start}, {ep_end})"
            )
        return idx

    def _invalidate_if_needed(self, shaper: Any) -> None:
        version = shaper_buffer_state_version(shaper)
        if version != self._cache_version:
            self._cache_version = version
            self._episode_dino.clear()
            log.info(
                "Expert online DINO cache cleared (contrastive buffer / shaper config changed)."
            )

    def _get_episode_dino(self, shaper: Any, episode_id: int) -> np.ndarray:
        if episode_id in self._episode_dino:
            return self._episode_dino[episode_id]

        ep_start = int(self._episode_starts[episode_id])
        ep_end = int(self._episode_ends[episode_id])
        traj_len = ep_end - ep_start

        images_np = self.dataset.images[ep_start:ep_end]
        if isinstance(images_np, torch.Tensor):
            images_np = images_np.detach().cpu().numpy()

        images_by_camera = split_npz_images_to_hwc(
            images_np, list(shaper.dino_camera_keys)
        )
        state_key = f"expert_ep_{episode_id}"
        shaper.reset_episode_pbrs_state(state_key=state_key, n_envs=1)

        ep_dino = compute_episode_dino_shaping(
            shaper,
            images_by_camera,
            traj_len,
            horizon_steps=self.horizon_steps,
            encode_batch_size=self.encode_batch_size,
            env_idx=0,
            n_envs=1,
            state_key=state_key,
            apply_adaptive_dense_weight=False,
        )
        self._episode_dino[episode_id] = ep_dino
        return ep_dino

    def _sum_dino_window(self, shaper: Any, global_start: int, global_end: int) -> float:
        episode_id = self._episode_id_for_global_step(global_start)
        ep_start = int(self._episode_starts[episode_id])
        local_start = global_start - ep_start
        local_end = global_end - ep_start
        ep_dino = self._get_episode_dino(shaper, episode_id)
        base_sum = float(ep_dino[local_start:local_end].sum())
        return base_sum * _adaptive_dense_scale(shaper)

    def chunk_dino_reward(self, shaper: Any, global_start: int) -> float:
        """DINO contribution for one expert chunk (matches online chunk sum semantics)."""
        if shaper is None or float(getattr(shaper, "reward_weight", 0.0)) == 0.0:
            return 0.0
        self._invalidate_if_needed(shaper)
        global_end = min(
            global_start + self.horizon_steps,
            int(self._episode_ends[self._episode_id_for_global_step(global_start)]),
        )
        return self._sum_dino_window(shaper, global_start, global_end)

    def total_reward_for_transition(
        self, shaper: Any, global_start: int, env_reward: float
    ) -> float:
        """Env (possibly n-step) reward plus live DINO shaping for this expert sample."""
        if shaper is None or float(getattr(shaper, "reward_weight", 0.0)) == 0.0:
            return float(env_reward)

        if self.use_n_step and self.n_step > 1:
            total_dino = 0.0
            for step in range(self.n_step):
                chunk_start = global_start + step * self.horizon_steps
                chunk_end = chunk_start + self.horizon_steps
                if chunk_end > len(self.dataset.env_rewards):
                    break
                dino_chunk = self._sum_dino_window(shaper, chunk_start, chunk_end)
                total_dino += (self.gamma ** step) * dino_chunk
            return float(env_reward) + total_dino

        global_end = global_start + self.horizon_steps
        dino = self._sum_dino_window(shaper, global_start, global_end)
        return float(env_reward) + dino

    def augment_expert_rewards(
        self,
        shaper: Any,
        expert_batch: Sequence[Any],
        env_rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Return (B, 1) rewards = env + DINO using current shaper buffers."""
        self._invalidate_if_needed(shaper)
        out = env_rewards.clone()
        for i, transition in enumerate(expert_batch):
            cond = transition.conditions
            global_start = cond.get("expert_global_start")
            if global_start is None:
                log.warning(
                    "Expert transition missing expert_global_start; skipping DINO."
                )
                continue
            total = self.total_reward_for_transition(
                shaper, int(global_start), float(env_rewards[i, 0].item())
            )
            out[i, 0] = total
        return out
