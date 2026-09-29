"""HTTP client for a running Robometer eval server (no robometer package import)."""

from __future__ import annotations

import io
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "robometer_client requires `requests` (pip install requests)"
    ) from exc


def _numpy_to_npy_file_tuple(arr: np.ndarray, filename: str) -> Tuple[str, io.BytesIO, str]:
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    return (filename, buf, "application/octet-stream")


def build_multipart_payload(
    samples: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Build multipart form data for ``/evaluate_batch_npy``."""
    files: Dict[str, Any] = {}
    data: Dict[str, str] = {}
    numpy_fields = ["frames", "lang_vector", "video_embeddings"]

    for i, sample in enumerate(samples):
        sample_copy = json.loads(json.dumps(sample, default=str))
        traj = sample.get("trajectory", {})
        traj_copy = sample_copy.get("trajectory", {})

        for field in numpy_fields:
            val = traj.get(field, None)
            if val is None:
                continue
            if hasattr(val, "detach") and hasattr(val, "cpu"):
                val = val.detach().cpu().numpy()
            if isinstance(val, np.ndarray):
                file_key = f"sample_{i}_trajectory_{field}"
                files[file_key] = _numpy_to_npy_file_tuple(val, f"{file_key}.npy")
                traj_copy[field] = {"__numpy_file__": file_key}
            else:
                traj_copy[field] = val

        if "frames_shape" in traj_copy and isinstance(
            traj_copy["frames_shape"], (tuple, list)
        ):
            traj_copy["frames_shape"] = [int(x) for x in traj_copy["frames_shape"]]

        sample_copy["trajectory"] = traj_copy
        data[f"sample_{i}"] = json.dumps(sample_copy)

    return files, data


def linspace_subsample_frames(
    frames: np.ndarray, num_frames: int = 16
) -> Tuple[np.ndarray, List[int]]:
    """Match ``robometer.data.datasets.helpers.linspace_subsample_frames`` (LIBERO path)."""
    if frames.size == 0:
        return frames, []
    total_frames = int(frames.shape[0])
    if total_frames <= num_frames:
        return frames, list(range(total_frames))
    if num_frames == 1:
        return frames[-1:], [total_frames - 1]
    indices = np.linspace(0, total_frames - 1, num_frames)
    indices = np.rint(indices).astype(int).tolist()
    indices[0] = 0
    indices[-1] = total_frames - 1
    for k in range(1, len(indices)):
        if indices[k] < indices[k - 1]:
            indices[k] = indices[k - 1]
        if indices[k] >= total_frames:
            indices[k] = total_frames - 1
    return frames[indices], indices


def subsample_trajectory_frames(
    frames: np.ndarray, max_frames: int
) -> np.ndarray:
    """Cap frames sent to eval server (aligns with ``raw_dict_to_sample(..., max_frames)``)."""
    if max_frames <= 0 or frames.shape[0] <= max_frames:
        return frames
    subsampled, _ = linspace_subsample_frames(frames, num_frames=max_frames)
    return subsampled


def make_progress_sample(
    frames: np.ndarray,
    task: str,
    sample_id: str,
    subsequence_length: int,
) -> Dict[str, Any]:
    return {
        "sample_type": "progress",
        "trajectory": {
            "frames": frames,
            "frames_shape": tuple(frames.shape),
            "task": task,
            "id": sample_id,
            "metadata": {"subsequence_length": int(subsequence_length)},
            "video_embeddings": None,
        },
    }


def post_evaluate_batch_npy(
    eval_server_url: str,
    samples: List[Dict[str, Any]],
    *,
    timeout_s: float = 120.0,
    use_frame_steps: bool = False,
) -> Dict[str, Any]:
    files, data = build_multipart_payload(samples)
    data["use_frame_steps"] = "true" if use_frame_steps else "false"
    url = eval_server_url.rstrip("/") + "/evaluate_batch_npy"
    resp = requests.post(url, files=files, data=data, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def extract_progress_from_outputs(
    outputs: Dict[str, Any], sample_index: int = 0
) -> np.ndarray:
    outputs_progress = outputs.get("outputs_progress")
    if outputs_progress is None:
        raise ValueError("No `outputs_progress` in server response")
    progress_pred = outputs_progress.get("progress_pred", [])
    if not progress_pred or len(progress_pred) <= sample_index:
        return np.array([], dtype=np.float32)
    return np.array(progress_pred[sample_index], dtype=np.float32)


def extract_success_probs_from_outputs(
    outputs: Dict[str, Any], sample_index: int = 0
) -> np.ndarray:
    outputs_success = outputs.get("outputs_success") or {}
    success_probs = outputs_success.get("success_probs", [])
    if not success_probs or len(success_probs) <= sample_index:
        return np.array([], dtype=np.float32)
    return np.array(success_probs[sample_index], dtype=np.float32)


def health_check(eval_server_url: str, timeout_s: float = 5.0) -> bool:
    try:
        url = eval_server_url.rstrip("/") + "/health"
        resp = requests.get(url, timeout=timeout_s)
        return resp.status_code == 200
    except Exception:
        return False
