"""Main-process Robometer-4B reward shaper (HTTP client to eval server).

Aligned with ``robometer/scripts/example_libero_robometer_wrapper.py``:

- Step reward from **progress** head: one scalar per query = ``progress_pred[-1]``
  (``extract_rewards_from_output``), applied to the **last substep** of each action chunk.
- Optional ``robometer_use_relative_rewards``: weight * (progress_t - progress_{t-1}).
- Optional ``robometer_use_success_detection``: early success via success_prob window
  (threshold 0.65, duration 2) — does **not** add success to step reward.
- No binary success term in reward (unlike deprecated ``robometer_add_binary_reward``).
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from hire_dice_rl.util.robometer_client import (
    extract_progress_from_outputs,
    extract_success_probs_from_outputs,
    health_check,
    make_progress_sample,
    post_evaluate_batch_npy,
    subsample_trajectory_frames,
)

log = logging.getLogger(__name__)

MIMICGEN_TASK_INSTRUCTIONS: Dict[str, str] = {
    "coffee": "Prepare coffee by placing the mug under the machine and operating it.",
    "coffee_preparation": "Prepare coffee with all required objects arranged correctly.",
    "hammer_cleanup": "Pick up the hammer and place it in the target container.",
    "kitchen": "Complete the kitchen manipulation task.",
    "mug_cleanup": "Pick up the mug and place it in the bin.",
    "nut_assembly": "Assemble the nut onto the peg.",
    "pick_place": "Pick up the object and place it at the target location.",
    "square": "Pick up the square nut and place it on the peg.",
    "stack": "Stack the blocks in the correct order.",
    "stack_three": "Stack three blocks in the correct order.",
    "threading": "Thread the needle through the ring.",
    "three_piece_assembly": "Assemble the three parts in the correct order.",
}


def resolve_robometer_task_instruction(wrapper_cfg, env_name: Optional[str] = None) -> str:
    explicit = wrapper_cfg.get("robometer_task_instruction")
    if explicit not in (None, ""):
        return str(explicit)
    if env_name and env_name in MIMICGEN_TASK_INSTRUCTIONS:
        return MIMICGEN_TASK_INSTRUCTIONS[env_name]
    if env_name:
        return f"Complete the {env_name.replace('_', ' ')} robot manipulation task."
    return "Complete the robot manipulation task."


def clamp_progress(values: np.ndarray) -> np.ndarray:
    """Match ``extract_rewards_from_output`` clipping to [0, 1]."""
    if values.size == 0:
        return values
    return np.clip(values.astype(np.float64), 0.0, 1.0)


class RobometerRewardShaper:
    """Dense reward from Robometer-4B via eval server (LIBERO-style)."""

    def __init__(self, wrapper_cfg, env_name: Optional[str] = None):
        if wrapper_cfg is None:
            raise ValueError("wrapper_cfg required for RobometerRewardShaper")

        self.reward_weight = float(wrapper_cfg.get("robometer_reward_weight", 0.0))
        self.server_url = str(
            wrapper_cfg.get("robometer_server_url", "http://127.0.0.1:8000")
        )
        self.task_instruction = resolve_robometer_task_instruction(
            wrapper_cfg, env_name=env_name
        )
        self.camera_key = str(
            wrapper_cfg.get("robometer_camera_key", "agentview_image")
        )
        # LIBERO local wrapper: single forward on prefix, not frame-step expansion.
        self.use_frame_steps = bool(wrapper_cfg.get("robometer_use_frame_steps", False))
        # LIBERO ``raw_dict_to_sample(..., max_frames=16)``: linspace subsample before VLM.
        self.max_frames = int(wrapper_cfg.get("robometer_max_frames", 16))
        if self.max_frames < 0:
            raise ValueError(f"robometer_max_frames must be >= 0, got {self.max_frames}")
        self.request_timeout_s = float(
            wrapper_cfg.get("robometer_request_timeout_s", 120.0)
        )
        self.bgr_to_rgb = bool(wrapper_cfg.get("robometer_bgr_to_rgb", True))

        # LIBERO: use_relative_rewards -> pred_t - pred_{t-1}
        self.use_relative_rewards = bool(
            wrapper_cfg.get("robometer_use_relative_rewards", False)
        )

        # LIBERO: success head for early termination only (not step reward).
        self.use_success_detection = bool(
            wrapper_cfg.get("robometer_use_success_detection", False)
        )
        self.success_detection_duration = int(
            wrapper_cfg.get("robometer_success_detection_duration", 2)
        )
        self.success_detection_threshold = float(
            wrapper_cfg.get("robometer_success_detection_threshold", 0.65)
        )

        # Deprecated DiceRL extension (off by default; not used in LIBERO).
        if bool(wrapper_cfg.get("robometer_add_binary_reward", False)):
            log.warning(
                "robometer_add_binary_reward is deprecated and ignored; "
                "use robometer_use_success_detection for LIBERO-style success handling."
            )

        self.shaping_per_step_mean = bool(
            wrapper_cfg.get("robometer_shaping_per_step_mean", False)
        )
        self.shaping_step_norm = float(wrapper_cfg.get("robometer_shaping_step_norm", 0.0))

        # Performance: query server every N action chunks (1 = every chunk, LIBERO-like).
        self.query_every_n_chunks = max(
            1, int(wrapper_cfg.get("robometer_query_every_n_chunks", 1))
        )
        fill_mode = str(wrapper_cfg.get("robometer_query_fill_mode", "hold")).lower()
        if fill_mode not in ("hold", "linear"):
            raise ValueError(
                f"robometer_query_fill_mode must be 'hold' or 'linear', got {fill_mode!r}"
            )
        self.query_fill_mode = fill_mode

        # Batch all envs that need a query into one HTTP POST (server batch_collator).
        self.batch_env_queries = bool(
            wrapper_cfg.get("robometer_batch_env_queries", True)
        )
        max_bs = wrapper_cfg.get("robometer_max_batch_size", None)
        self.max_batch_size = (
            None if max_bs is None else max(1, int(max_bs))
        )
        self.parallel_requests = max(
            1, int(wrapper_cfg.get("robometer_parallel_requests", 4))
        )

        self._episode_frames: Dict[str, List[List[np.ndarray]]] = {}
        self._prev_progress: Dict[str, np.ndarray] = {}
        self._success_windows: Dict[str, List[Deque[float]]] = {}
        self._chunk_counters: Dict[str, np.ndarray] = {}
        self._cached_progress: Dict[str, np.ndarray] = {}
        self._cached_success: Dict[str, np.ndarray] = {}
        self._progress_rate: Dict[str, np.ndarray] = {}
        self._interval_start_progress: Dict[str, np.ndarray] = {}

        if self.reward_weight > 0.0:
            if health_check(self.server_url, timeout_s=5.0):
                log.info(
                    "RobometerRewardShaper: server healthy at %s", self.server_url
                )
            else:
                log.warning(
                    "RobometerRewardShaper: server not reachable at %s "
                    "(training will fail on first shape() unless server starts)",
                    self.server_url,
                )
            log.info(
                "RobometerRewardShaper (LIBERO-aligned): task=%r camera=%s "
                "progress_w=%.4f relative=%s success_det=%s (dur=%d thr=%.2f) "
                "frame_steps=%s max_frames=%d step_mean_norm=%s "
                "query_every_n_chunks=%d fill=%s batch_env=%s max_batch=%s parallel_http=%d",
                self.task_instruction,
                self.camera_key,
                self.reward_weight,
                self.use_relative_rewards,
                self.use_success_detection,
                self.success_detection_duration,
                self.success_detection_threshold,
                self.use_frame_steps,
                self.max_frames,
                self.shaping_per_step_mean,
                self.query_every_n_chunks,
                self.query_fill_mode,
                self.batch_env_queries,
                self.max_batch_size,
                self.parallel_requests,
            )

    # Adaptive dense weight hooks (no-op; kept for agent refresh API compatibility).
    adaptive_success_rate_window_size: int = 0

    def refresh_adaptive_dense_weight_before_update(self, success_rate: float) -> None:
        del success_rate

    def _prepare_frame(self, img: np.ndarray) -> np.ndarray:
        frame = np.ascontiguousarray(img)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim != 3:
            raise ValueError(f"Expected HWC image, got shape {frame.shape}")
        if self.bgr_to_rgb and frame.shape[-1] == 3:
            frame = frame[..., ::-1]
        return frame

    def _frames_for_server(self, frames: np.ndarray) -> np.ndarray:
        """Subsample prefix to ``max_frames`` (LIBERO ``raw_dict_to_sample``)."""
        return subsample_trajectory_frames(frames, self.max_frames)

    def _query_server_batch(
        self,
        batch_items: Sequence[Tuple[np.ndarray, str]],
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """One HTTP round-trip for multiple env trajectories (GPU memory permitting)."""
        if not batch_items:
            return []
        samples = [
            make_progress_sample(
                frames=self._frames_for_server(frames),
                task=self.task_instruction,
                sample_id=sample_id,
                subsequence_length=int(frames.shape[0]),
            )
            for frames, sample_id in batch_items
        ]
        outputs = post_evaluate_batch_npy(
            self.server_url,
            samples,
            timeout_s=self.request_timeout_s,
            use_frame_steps=self.use_frame_steps,
        )
        return [
            (
                extract_progress_from_outputs(outputs, sample_index=i),
                extract_success_probs_from_outputs(outputs, sample_index=i),
            )
            for i in range(len(samples))
        ]

    def _query_server(
        self, frames: np.ndarray, sample_id: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        results = self._query_server_batch([(frames, sample_id)])
        return results[0]

    @staticmethod
    def _trajectory_batch_key(frames: np.ndarray) -> Tuple[int, ...]:
        """Group batchable queries: same shape => same VLM token layout (usually)."""
        return tuple(int(x) for x in frames.shape)

    def _query_server_batch_with_fallback(
        self,
        batch_items: Sequence[Tuple[np.ndarray, str]],
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        try:
            return self._query_server_batch(batch_items)
        except Exception as exc:
            if len(batch_items) <= 1:
                raise
            log.warning(
                "Robometer batch of %d failed (%s); retrying serially",
                len(batch_items),
                exc,
            )
            out: List[Tuple[np.ndarray, np.ndarray]] = []
            for frames, sample_id in batch_items:
                out.append(self._query_server_batch([(frames, sample_id)])[0])
            return out

    def _resolve_server_queries(
        self,
        query_items: Sequence[Tuple[int, np.ndarray, str, int]],
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Run server queries with same-length batching, HTTP parallelism, and fallback."""
        if not query_items:
            return {}

        grouped: Dict[Tuple[int, ...], List[Tuple[int, np.ndarray, str]]] = (
            defaultdict(list)
        )
        for env_idx, traj_frames, sample_id, _k in query_items:
            grouped[self._trajectory_batch_key(traj_frames)].append(
                (env_idx, traj_frames, sample_id)
            )

        chunks: List[Tuple[List[Tuple[np.ndarray, str]], List[int]]] = []
        for group in grouped.values():
            specs = [(traj, sid) for _, traj, sid in group]
            env_ids = [env_idx for env_idx, _, _ in group]
            max_bs = self.max_batch_size or len(specs)
            for offset in range(0, len(specs), max_bs):
                chunks.append(
                    (
                        specs[offset : offset + max_bs],
                        env_ids[offset : offset + max_bs],
                    )
                )

        def _run_chunk(
            chunk: Tuple[List[Tuple[np.ndarray, str]], List[int]],
        ) -> List[Tuple[int, Tuple[np.ndarray, np.ndarray]]]:
            specs, env_ids = chunk
            if self.batch_env_queries:
                results = self._query_server_batch_with_fallback(specs)
            else:
                results = [
                    self._query_server_batch([(frames, sid)])[0]
                    for frames, sid in specs
                ]
            return list(zip(env_ids, results))

        query_results: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        if len(chunks) == 1 or self.parallel_requests <= 1:
            for chunk in chunks:
                for env_idx, result in _run_chunk(chunk):
                    query_results[env_idx] = result
        else:
            workers = min(self.parallel_requests, len(chunks))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for pairs in pool.map(_run_chunk, chunks):
                    for env_idx, result in pairs:
                        query_results[env_idx] = result
        return query_results

    def _get_episode_frames(self, state_key: str, n_envs: int) -> List[List[np.ndarray]]:
        if state_key not in self._episode_frames:
            self._episode_frames[state_key] = [[] for _ in range(n_envs)]
        buffers = self._episode_frames[state_key]
        if len(buffers) != n_envs:
            self._episode_frames[state_key] = [[] for _ in range(n_envs)]
        return self._episode_frames[state_key]

    def _get_prev_progress(self, state_key: str, n_envs: int) -> np.ndarray:
        if state_key not in self._prev_progress:
            self._prev_progress[state_key] = np.zeros(n_envs, dtype=np.float64)
        arr = self._prev_progress[state_key]
        if arr.shape[0] != n_envs:
            self._prev_progress[state_key] = np.zeros(n_envs, dtype=np.float64)
        return self._prev_progress[state_key]

    def _get_chunk_counters(self, state_key: str, n_envs: int) -> np.ndarray:
        if state_key not in self._chunk_counters:
            self._chunk_counters[state_key] = np.zeros(n_envs, dtype=np.int64)
        arr = self._chunk_counters[state_key]
        if arr.shape[0] != n_envs:
            self._chunk_counters[state_key] = np.zeros(n_envs, dtype=np.int64)
        return self._chunk_counters[state_key]

    def _get_cached_progress(self, state_key: str, n_envs: int) -> np.ndarray:
        if state_key not in self._cached_progress:
            self._cached_progress[state_key] = np.zeros(n_envs, dtype=np.float64)
        arr = self._cached_progress[state_key]
        if arr.shape[0] != n_envs:
            self._cached_progress[state_key] = np.zeros(n_envs, dtype=np.float64)
        return self._cached_progress[state_key]

    def _get_cached_success(self, state_key: str, n_envs: int) -> np.ndarray:
        if state_key not in self._cached_success:
            self._cached_success[state_key] = np.zeros(n_envs, dtype=np.float64)
        arr = self._cached_success[state_key]
        if arr.shape[0] != n_envs:
            self._cached_success[state_key] = np.zeros(n_envs, dtype=np.float64)
        return self._cached_success[state_key]

    def _get_progress_rate(self, state_key: str, n_envs: int) -> np.ndarray:
        if state_key not in self._progress_rate:
            self._progress_rate[state_key] = np.zeros(n_envs, dtype=np.float64)
        arr = self._progress_rate[state_key]
        if arr.shape[0] != n_envs:
            self._progress_rate[state_key] = np.zeros(n_envs, dtype=np.float64)
        return self._progress_rate[state_key]

    def _get_interval_start_progress(self, state_key: str, n_envs: int) -> np.ndarray:
        if state_key not in self._interval_start_progress:
            self._interval_start_progress[state_key] = np.zeros(
                n_envs, dtype=np.float64
            )
        arr = self._interval_start_progress[state_key]
        if arr.shape[0] != n_envs:
            self._interval_start_progress[state_key] = np.zeros(
                n_envs, dtype=np.float64
            )
        return self._interval_start_progress[state_key]

    def _should_query_chunk(self, env_idx: int, state_key: str, n_envs: int) -> bool:
        counters = self._get_chunk_counters(state_key, n_envs)
        return int(counters[env_idx]) % self.query_every_n_chunks == 0

    def _bump_chunk_counter(self, env_idx: int, state_key: str, n_envs: int) -> None:
        self._get_chunk_counters(state_key, n_envs)[env_idx] += 1

    def _get_success_windows(self, state_key: str, n_envs: int) -> List[Deque[float]]:
        if state_key not in self._success_windows:
            self._success_windows[state_key] = [
                deque(maxlen=self.success_detection_duration)
                for _ in range(n_envs)
            ]
        windows = self._success_windows[state_key]
        if len(windows) != n_envs:
            self._success_windows[state_key] = [
                deque(maxlen=self.success_detection_duration)
                for _ in range(n_envs)
            ]
        return self._success_windows[state_key]

    @staticmethod
    def _libero_progress_scalar(progress_curve: np.ndarray) -> float:
        """Match ``extract_rewards_from_output``: last progress value in [0, 1]."""
        if progress_curve.size == 0:
            return 0.0
        return float(clamp_progress(progress_curve)[-1])

    def _libero_chunk_step_rewards(
        self,
        p_end: float,
        env_idx: int,
        state_key: str,
        n_envs: int,
        k: int,
    ) -> np.ndarray:
        """One LIBERO scalar per chunk/query; assign to last substep only (sum = scalar)."""
        if k <= 0:
            return np.zeros(0, dtype=np.float64)
        prev_arr = self._get_prev_progress(state_key, n_envs)
        prev_val = float(prev_arr[env_idx])
        p_end = float(p_end)
        if self.use_relative_rewards:
            scalar = self.reward_weight * (p_end - prev_val)
        else:
            scalar = self.reward_weight * p_end
        prev_arr[env_idx] = p_end
        per_sub_reward = np.zeros(k, dtype=np.float64)
        per_sub_reward[-1] = scalar
        return per_sub_reward

    def _maybe_success_terminate(
        self,
        env_idx: int,
        success_prob: float,
        state_key: str,
        n_envs: int,
        info_venv: Sequence[dict],
        terminated_venv: Optional[np.ndarray],
    ) -> bool:
        if not self.use_success_detection:
            return False
        windows = self._get_success_windows(state_key, n_envs)
        windows[env_idx].append(float(success_prob))
        if len(windows[env_idx]) < self.success_detection_duration:
            return False
        votes = sum(
            1
            for p in windows[env_idx]
            if p >= self.success_detection_threshold
        )
        if votes <= (self.success_detection_duration / 2):
            return False
        info_venv[env_idx]["success_from_reward_model"] = True
        info_venv[env_idx]["robometer_success_detected"] = True
        if terminated_venv is not None:
            terminated_venv[env_idx] = True
        return True

    def _reset_episode_state(
        self,
        env_idx: int,
        state_key: str,
        n_envs: int,
    ) -> None:
        self._get_episode_frames(state_key, n_envs)[env_idx] = []
        self._get_prev_progress(state_key, n_envs)[env_idx] = 0.0
        self._get_success_windows(state_key, n_envs)[env_idx].clear()
        self._get_chunk_counters(state_key, n_envs)[env_idx] = 0
        self._get_cached_progress(state_key, n_envs)[env_idx] = 0.0
        self._get_cached_success(state_key, n_envs)[env_idx] = 0.0
        self._get_progress_rate(state_key, n_envs)[env_idx] = 0.0
        self._get_interval_start_progress(state_key, n_envs)[env_idx] = 0.0

    def _update_progress_rate_after_query(
        self,
        env_idx: int,
        state_key: str,
        n_envs: int,
        k: int,
        p_end: float,
    ) -> None:
        interval_start = self._get_interval_start_progress(state_key, n_envs)
        rates = self._get_progress_rate(state_key, n_envs)
        cached = self._get_cached_progress(state_key, n_envs)
        p_start = float(interval_start[env_idx])
        denom = max(self.query_every_n_chunks * max(k, 1), 1)
        rates[env_idx] = (float(p_end) - p_start) / float(denom)
        cached[env_idx] = float(p_end)
        interval_start[env_idx] = float(p_end)

    def _apply_env_chunk_rewards(
        self,
        env_idx: int,
        state_key: str,
        n_envs: int,
        k: int,
        p_end: float,
        success_prob_last: float,
        reward_venv: np.ndarray,
        info_venv: Sequence[dict],
        terminated_venv: Optional[np.ndarray],
        *,
        queried_server: bool,
        write_reward: bool = True,
    ) -> None:
        if write_reward:
            per_sub_reward = self._libero_chunk_step_rewards(
                p_end, env_idx, state_key, n_envs, k
            )
        else:
            per_sub_reward = np.zeros(k, dtype=np.float64)

        if self.shaping_per_step_mean and per_sub_reward.size > 0:
            norm = (
                self.shaping_step_norm
                if self.shaping_step_norm > 0.0
                else float(per_sub_reward.shape[0])
            )
            if norm > 0.0:
                per_sub_reward = per_sub_reward / norm

        chunk_robo_reward = float(per_sub_reward.sum())
        reward_venv[env_idx] = float(reward_venv[env_idx]) + chunk_robo_reward
        info_venv[env_idx]["robometer_reward_chunk"] = chunk_robo_reward
        info_venv[env_idx]["robometer_progress_chunk"] = float(p_end)
        info_venv[env_idx]["robometer_queried_server"] = queried_server
        info_venv[env_idx]["robometer_success_prob_last"] = float(success_prob_last)
        self._get_cached_success(state_key, n_envs)[env_idx] = float(success_prob_last)
        self._maybe_success_terminate(
            env_idx,
            float(success_prob_last),
            state_key,
            n_envs,
            info_venv,
            terminated_venv,
        )

        traj = info_venv[env_idx].get("full_trajectory")
        if isinstance(traj, dict):
            traj_rewards = traj.get("rewards")
            if (
                traj_rewards is not None
                and len(traj_rewards) == per_sub_reward.shape[0]
            ):
                traj["rewards"] = [
                    float(traj_rewards[i]) + float(per_sub_reward[i])
                    for i in range(len(traj_rewards))
                ]
                info_venv[env_idx]["robometer_per_substep_reward"] = (
                    per_sub_reward.tolist()
                )

    def shape(
        self,
        reward_venv: np.ndarray,
        info_venv: Sequence[dict],
        terminated_venv: Optional[np.ndarray] = None,
        truncated_venv: Optional[np.ndarray] = None,
        state_key: str = "train",
    ) -> np.ndarray:
        if self.reward_weight <= 0.0:
            return reward_venv

        n_envs = len(info_venv)
        episode_buffers = self._get_episode_frames(state_key, n_envs)
        cached_progress = self._get_cached_progress(state_key, n_envs)
        cached_success = self._get_cached_success(state_key, n_envs)
        progress_rates = self._get_progress_rate(state_key, n_envs)
        interval_start = self._get_interval_start_progress(state_key, n_envs)

        # Pending work per env after extending frame buffers.
        pending: List[dict] = []

        for env_idx in range(n_envs):
            chunk = info_venv[env_idx].get("robometer_chunk_images")
            if chunk is None:
                chunk = info_venv[env_idx].get("dino_chunk_images")
            if not chunk:
                continue

            new_frames: List[np.ndarray] = []
            for img_dict in chunk:
                if not isinstance(img_dict, Mapping):
                    continue
                img = img_dict.get(self.camera_key)
                if img is None:
                    continue
                new_frames.append(self._prepare_frame(img))

            if not new_frames:
                continue

            episode_buffers[env_idx].extend(new_frames)
            k = len(new_frames)
            should_query = self._should_query_chunk(env_idx, state_key, n_envs)
            pending.append(
                {
                    "env_idx": env_idx,
                    "k": k,
                    "should_query": should_query,
                    "traj_frames": np.stack(episode_buffers[env_idx], axis=0),
                }
            )

        query_items: List[Tuple[int, np.ndarray, str, int]] = []
        for item in pending:
            if not item["should_query"]:
                continue
            env_idx = item["env_idx"]
            query_items.append(
                (
                    env_idx,
                    item["traj_frames"],
                    f"{state_key}_{env_idx}",
                    item["k"],
                )
            )

        query_results: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        if query_items:
            try:
                query_results = self._resolve_server_queries(query_items)
            except Exception as exc:
                log.error("Robometer server request failed: %s", exc)
                raise

        for item in pending:
            env_idx = item["env_idx"]
            k = item["k"]
            should_query = item["should_query"]

            if should_query:
                progress_curve, success_curve = query_results[env_idx]
                p_end = self._libero_progress_scalar(progress_curve)
                if success_curve.size > 0:
                    success_prob_last = float(success_curve[-1])
                else:
                    success_prob_last = float(
                        self._get_cached_success(state_key, n_envs)[env_idx]
                    )
                if int(self._get_chunk_counters(state_key, n_envs)[env_idx]) == 0:
                    interval_start[env_idx] = 0.0
                self._update_progress_rate_after_query(
                    env_idx, state_key, n_envs, k, p_end
                )
                cached_progress[env_idx] = p_end
                queried_server = True
            else:
                if self.query_fill_mode == "linear" and float(
                    progress_rates[env_idx]
                ) != 0.0:
                    p_end = float(
                        clamp_progress(
                            np.array(
                                [
                                    float(cached_progress[env_idx])
                                    + float(progress_rates[env_idx]),
                                ]
                            )
                        )[-1]
                    )
                else:
                    p_end = float(cached_progress[env_idx])
                cached_progress[env_idx] = p_end
                success_prob_last = float(
                    self._get_cached_success(state_key, n_envs)[env_idx]
                )
                queried_server = False

            # LIBERO: reward only when the model is queried. Skipped chunks (hold)
            # update cached progress but must not repeat dense reward every chunk.
            apply_reward = queried_server or self.use_relative_rewards

            self._apply_env_chunk_rewards(
                env_idx,
                state_key,
                n_envs,
                k,
                p_end if apply_reward else float(
                    self._get_prev_progress(state_key, n_envs)[env_idx]
                ),
                success_prob_last,
                reward_venv,
                info_venv,
                terminated_venv,
                queried_server=queried_server,
                write_reward=apply_reward,
            )
            self._bump_chunk_counter(env_idx, state_key, n_envs)

        if terminated_venv is not None or truncated_venv is not None:
            for env_idx in range(n_envs):
                done = False
                if terminated_venv is not None and bool(terminated_venv[env_idx]):
                    done = True
                if truncated_venv is not None and bool(truncated_venv[env_idx]):
                    done = True
                if done:
                    self._reset_episode_state(env_idx, state_key, n_envs)

        return reward_venv
