from typing import List, Optional, Sequence
import torch

class EmbeddingBuffer:
    """Fixed-size per-camera queue for online embeddings (pos or neg)."""

    def __init__(
        self,
        camera_keys: List[str],
        max_size: int,
        seed: int = 0,
        store_images: bool = False,
    ):
        if int(max_size) <= 0:
            raise ValueError(f"max_size must be > 0, got {max_size}")
        self.camera_keys = list(camera_keys)
        self.max_size = int(max_size)
        self.store_images = bool(store_images)
        self.seed = int(seed)
        self.random_generator = torch.Generator(device="cpu")
        self.random_generator.manual_seed(self.seed)
        self._buffers = {camera_key: None for camera_key in self.camera_keys}
        self._image_buffers = {camera_key: None for camera_key in self.camera_keys}
        self._frame_index_buffers = {
            camera_key: None for camera_key in self.camera_keys
        }
        self._write_ptr = {camera_key: 0 for camera_key in self.camera_keys}
        self._sizes = {camera_key: 0 for camera_key in self.camera_keys}

    def size(self, camera_key: str) -> int:
        return int(self._sizes[camera_key])

    def add_embeddings(
        self,
        camera_key: str,
        embeddings: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        env_frame_indices: Optional[Sequence[int]] = None,
    ) -> None:
        if camera_key not in self._buffers:
            raise KeyError(
                f"Unknown camera key '{camera_key}'. Available: {self.camera_keys}"
            )
        if not torch.is_tensor(embeddings) or embeddings.ndim != 3:
            raise ValueError(
                f"embeddings must have shape [M,P,D], got {type(embeddings)}"
            )
        if embeddings.shape[0] <= 0:
            return
        data = embeddings.detach().to(device="cpu", dtype=torch.float32).contiguous()
        m = int(data.shape[0])
        if env_frame_indices is not None and len(env_frame_indices) != m:
            raise ValueError(
                f"env_frame_indices length ({len(env_frame_indices)}) must match "
                f"embeddings batch ({m}) for camera '{camera_key}'"
            )
        index_data = (
            torch.tensor(list(env_frame_indices), dtype=torch.long)
            if env_frame_indices is not None
            else torch.full((m,), -1, dtype=torch.long)
        )

        image_data = None
        if self.store_images:
            if images is None:
                raise ValueError("store_images=True requires images.")
            image_data = images.detach().to(device="cpu", dtype=torch.uint8).contiguous()

        if self._buffers[camera_key] is None:
            self._buffers[camera_key] = torch.empty(
                (self.max_size, data.shape[1], data.shape[2]), dtype=torch.float32
            )
        if self.store_images and self._image_buffers[camera_key] is None:
            self._image_buffers[camera_key] = torch.empty(
                (
                    self.max_size,
                    image_data.shape[1],
                    image_data.shape[2],
                    image_data.shape[3],
                ),
                dtype=torch.uint8,
            )
        if self._frame_index_buffers[camera_key] is None:
            self._frame_index_buffers[camera_key] = torch.full(
                (self.max_size,), -1, dtype=torch.long
            )

        if m >= self.max_size:
            self._buffers[camera_key].copy_(data[-self.max_size :])
            if self.store_images:
                self._image_buffers[camera_key].copy_(image_data[-self.max_size :])
            self._frame_index_buffers[camera_key].fill_(-1)
            self._frame_index_buffers[camera_key][:] = index_data[-self.max_size :]
            self._write_ptr[camera_key] = 0
            self._sizes[camera_key] = self.max_size
            return

        ptr = int(self._write_ptr[camera_key])
        end = ptr + m
        if end <= self.max_size:
            self._buffers[camera_key][ptr:end] = data
            if self.store_images:
                self._image_buffers[camera_key][ptr:end] = image_data
            self._frame_index_buffers[camera_key][ptr:end] = index_data
        else:
            first = self.max_size - ptr
            self._buffers[camera_key][ptr:] = data[:first]
            self._buffers[camera_key][: end - self.max_size] = data[first:]
            if self.store_images:
                self._image_buffers[camera_key][ptr:] = image_data[:first]
                self._image_buffers[camera_key][: end - self.max_size] = image_data[
                    first:
                ]
            self._frame_index_buffers[camera_key][ptr:] = index_data[:first]
            self._frame_index_buffers[camera_key][: end - self.max_size] = index_data[
                first:
            ]
        self._write_ptr[camera_key] = (ptr + m) % self.max_size
        self._sizes[camera_key] = min(self.max_size, int(self._sizes[camera_key]) + m)

    def sample_batch(
        self,
        camera_key: str,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        k = int(batch_size)
        if k <= 0:
            raise ValueError(f"batch_size must be > 0, got {k}")
        size = int(self._sizes[camera_key])
        if size <= 0:
            raise RuntimeError(f"Online embedding buffer for '{camera_key}' is empty")
        valid = self._buffers[camera_key][:size]
        indices = torch.randint(
            low=0, high=size, size=(k,), generator=self.random_generator
        ).long()
        out = valid.index_select(0, indices)
        if device is not None:
            out = out.to(device=device)
        return out
