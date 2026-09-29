"""Deterministic multienvironment reward trace with a download-free encoder."""
from pathlib import Path
import numpy as np
import torch


class Encoder:
    def compute_embeddings(self, images):
        x = images.float().mean(dim=(1, 2, 3)) / 255
        return torch.stack((x, 1 - x, x.square() + 0.1), dim=-1).unsqueeze(1)


def reward_trace(module, cls, directory):
    encoder = Encoder()
    original = module.build_similarity_encoder
    module.build_similarity_encoder = lambda *args, **kwargs: encoder
    keys = ["base", "wrist"]
    path = Path(directory) / "positive.pt"
    torch.save({"camera_embeddings": {
        key: encoder.compute_embeddings(torch.stack([torch.full((3, 4, 4), v) for v in (50, 150, 230)])) for key in keys},
        "camera_frame_indices": {key: torch.arange(3) for key in keys}}, path)
    cfg = {
        "dino_goal_source_mode": "contrastive_prompt", "image_keys": keys,
        "dino_reward_weight": 1.0, "dino_contrastive_lambda": 0.9,
        "dino_logsumexp_beta": 10.0, "dino_rel_diff_decay_enabled": True,
        "dino_rel_diff_pi": 0.99, "adaptive_dense_weight_max": 1.0,
        "adaptive_success_rate_window_size": 100,
        "dino_shaping_per_step_mean": True, "dino_shaping_step_norm": 3,
        "dino_positive_buffer": {"buffer_path": str(path), "camera_keys": keys,
            "sample_batch_size": 3, "sampling_mode": "uniform", "seed": 4,
            "online": {"enabled": True, "buffer_size": 2, "sample_batch_size": 3,
                       "store_images": False, "online_mix_ratio": 0.5, "seed": 5}},
        "dino_negative_buffer": {"buffer_size": 2, "sample_batch_size": 3, "seed": 6},
    }
    try:
        shaper = cls(cfg, device="cpu")
        outputs = []
        for step in range(5):
            infos = []
            for env in range(2):
                length = 2 + env
                frames = [{key: np.full((4, 4, 3), 30 + step * 30 + env * 10 + sub * 4 + cam, dtype=np.uint8)
                           for cam, key in enumerate(keys)} for sub in range(length)]
                success = step in (1, 2, 3) and env == 0
                sparse = [0.0] * (length - 1) + [float(success)]
                infos.append({"dino_chunk_images": frames, "dino_chunk_env_frame_indices": list(range(step * 3, step * 3 + length)), "env_reward_chunk_sum": float(success),
                              "full_trajectory": {"rewards": sparse}})
            done = np.array([step in (1, 2, 3)] * 2)
            reward = shaper.shape(np.array([info["env_reward_chunk_sum"] for info in infos]),
                                  infos, done, np.array([False, False]))
            outputs.append({"reward": reward.tolist(),
                "substep": [info["full_trajectory"]["rewards"] for info in infos],
                "positive_size": shaper.dino_online_positive_buffer.size("base"),
                "negative_size": shaper.dino_negative_buffer.size("base"),
                "previous": shaper._get_prev_potential("train", 2).tolist()})
            shaper.refresh_adaptive_dense_weight_before_update(0.25)
        return outputs
    finally:
        module.build_similarity_encoder = original
