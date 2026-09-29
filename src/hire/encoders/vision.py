"""Frozen visual encoders that return patch-compatible embeddings for HiRE."""

from __future__ import annotations

import logging
import os
import time
from typing import Optional, Union

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

log = logging.getLogger(__name__)

_ENCODER_DEBUG_DUMP_COUNTS: dict[str, int] = {}

# Pinned for Python 3.8 (newer dinov2 main uses PEP 604 unions at import time).
_DINOV2_PINNED_COMMIT = "b48308a394a04ccb9c4dd3a1f0a4daa1ce0579b8"


def validate_encoder_inputs(
    name: str,
    images: torch.Tensor,
    *,
    first_call: bool,
) -> None:
    """Validate encoder inputs; optionally dump debug frames on first call."""
    assert torch.is_tensor(images), (
        f"{name} expects torch.Tensor, got {type(images)}"
    )
    assert images.ndim == 4 and images.shape[1] == 3, (
        f"{name} expects [B, 3, H, W], got {tuple(images.shape)}"
    )
    assert images.shape[-1] >= 16 and images.shape[-2] >= 16, (
        f"{name} got image too small: H,W={tuple(images.shape[-2:])}"
    )
    assert images.dtype in (
        torch.uint8,
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ), f"{name} unsupported dtype {images.dtype}"

    if not first_call:
        return

    if images.dtype != torch.uint8:
        with torch.no_grad():
            probe = images.detach().float()
            mx = float(probe.amax().item())
            mn = float(probe.amin().item())
        log.info(
            "[%s] first-call value range probe: min=%.4f max=%.4f dtype=%s shape=%s device=%s",
            name,
            mn,
            mx,
            images.dtype,
            tuple(images.shape),
            images.device,
        )
        assert mx > 1.5, (
            f"{name}: input looks already in [0,1] (max={mx:.4f}); expected 0-255 raw pixels."
        )
        assert mx <= 256.0 + 1e-3, (
            f"{name}: input exceeds 0-255 (max={mx:.4f}); preprocessing likely off."
        )

    save_dir = os.environ.get("HIRE_ENCODER_DEBUG_DIR", os.environ.get("DICE_ENCODER_DEBUG_DIR", "")).strip()
    if not save_dir:
        return
    max_dumps = int(os.environ.get("HIRE_ENCODER_DEBUG_MAX", os.environ.get("DICE_ENCODER_DEBUG_MAX", "4")))
    count = _ENCODER_DEBUG_DUMP_COUNTS.get(name, 0)
    if count >= max_dumps:
        return
    _ENCODER_DEBUG_DUMP_COUNTS[name] = count + 1

    sub_dir = os.path.join(save_dir, name)
    os.makedirs(sub_dir, exist_ok=True)
    n_to_save = min(4, int(images.shape[0]))
    with torch.no_grad():
        flat = images[:n_to_save].detach().float()
        per_c_mean = flat.mean(dim=(0, 2, 3)).tolist()
        imgs_u8 = (
            torch.clamp(flat, 0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .contiguous()
            .cpu()
            .numpy()
        )
    for i in range(n_to_save):
        Image.fromarray(imgs_u8[i]).save(
            os.path.join(sub_dir, f"call{count:03d}_img{i:02d}.png")
        )
    stats_path = os.path.join(sub_dir, f"call{count:03d}_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"tag={name}\n")
        f.write(f"shape={tuple(images.shape)}\n")
        f.write(f"dtype={images.dtype}\n")
        f.write(f"device={images.device}\n")
        f.write(f"per_channel_mean={per_c_mean}\n")
    log.info(
        "[%s] dumped %d encoder-input PNGs and stats to %s",
        name,
        n_to_save,
        sub_dir,
    )


class DinoV2Encoder(nn.Module):
    """DINOv2 patch encoder for similarity-based rewards."""

    def __init__(self, device: Union[str, torch.device] = "cuda"):
        super().__init__()
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=(123.675, 116.28, 103.53),
                    std=(58.395, 57.12, 57.375),
                ),
            ]
        )
        self.device = (
            torch.device(device) if not isinstance(device, torch.device) else device
        )
        self._inputs_validated = False
        log.info("[DinoV2Encoder] Loading DINOv2 model from torch.hub...")
        self.encoder = self._load_encoder_with_retry()
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.encoder.eval()
        self.encoder.to(self.device)
        log.info("[DinoV2Encoder] DINOv2 model loaded successfully")

    def _load_encoder_with_retry(self):
        local_repo = os.environ.get("DINO_REPO_LOCAL", "").strip()
        if local_repo and os.path.isdir(local_repo):
            return torch.hub.load(local_repo, "dinov2_vits14", source="local")
        pinned = f"facebookresearch/dinov2:{_DINOV2_PINNED_COMMIT}"
        retries = int(os.environ.get("HIRE_DINO_LOAD_RETRIES", os.environ.get("DICE_DINO_LOAD_RETRIES", "3")))
        retry_wait_s = float(os.environ.get("HIRE_DINO_RETRY_WAIT_S", os.environ.get("DICE_DINO_RETRY_WAIT_S", "3")))
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                return torch.hub.load(
                    pinned, "dinov2_vits14", trust_repo=True, skip_validation=True
                )
            except Exception as exc:
                last_err = exc
                log.warning(
                    "[DinoV2Encoder] torch.hub load failed (%s/%s): %s",
                    attempt,
                    retries,
                    repr(exc),
                )
                if attempt < retries:
                    time.sleep(retry_wait_s)
        raise RuntimeError(
            "Failed to load DINOv2 from torch.hub after retries. "
            "Set DINO_REPO_LOCAL=/path/to/facebookresearch_dinov2 for offline use. "
            f"Last error: {repr(last_err)}"
        )

    @torch.no_grad()
    def compute_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """Encode [B, C, H, W] images in 0-255 range -> [B, P, D] patch tokens."""
        validate_encoder_inputs(
            "DinoV2Encoder", images, first_call=not self._inputs_validated
        )
        self._inputs_validated = True
        if images.dtype == torch.uint8:
            images = images.float()
        x = self.image_transform(images.to(self.device))
        features = self.encoder.forward_features(x)
        return features["x_norm_patchtokens"]


class LIVEncoder(nn.Module):
    """LIV global image encoder; returns [B, 1, D] for patch-mean compatibility."""

    def __init__(
        self,
        device: Union[str, torch.device] = "cuda",
        modelid: str = "resnet50",
    ):
        super().__init__()
        try:
            from liv import load_liv
        except ImportError as exc:
            raise ImportError(
                "LIVEncoder requires the `liv` package. Install from "
                "https://github.com/penn-pal-lab/LIV "
                "(e.g. `pip install -e .` plus `pip install -e liv/models/clip`)."
            ) from exc

        self.device = (
            torch.device(device) if not isinstance(device, torch.device) else device
        )
        log.info("[LIVEncoder] Loading LIV model (id=%s) on %s...", modelid, self.device)
        liv = load_liv(modelid=modelid)
        liv.eval()
        for p in liv.parameters():
            p.requires_grad = False
        try:
            liv.to(self.device)
        except Exception as exc:
            log.warning("[LIVEncoder] could not move LIV to %s: %s", self.device, exc)
        self.encoder = liv
        self._inner = liv.module if hasattr(liv, "module") else liv
        self.output_dim = int(getattr(self._inner, "output_dim", 0)) or None
        self._inputs_validated = False
        log.info("[LIVEncoder] Loaded LIV (output_dim=%s)", self.output_dim)

    @torch.no_grad()
    def compute_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        validate_encoder_inputs(
            "LIVEncoder", images, first_call=not self._inputs_validated
        )
        self._inputs_validated = True
        if images.dtype == torch.uint8:
            images = images.float()
        feats = self.encoder(input=images.to(self.device), modality="vision")
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)
        if feats.dim() != 2:
            raise RuntimeError(
                f"Unexpected LIV image embedding rank {feats.dim()}, expected 2 ([B, D])."
            )
        return feats.unsqueeze(1).contiguous().float()


class SigLIPEncoder(nn.Module):
    """SigLIP vision encoder; returns [B, P, D] patch embeddings."""

    def __init__(
        self,
        device: Union[str, torch.device] = "cuda",
        model_id: Optional[str] = None,
    ):
        super().__init__()
        try:
            from transformers import AutoImageProcessor, SiglipVisionModel
        except ImportError as exc:
            raise ImportError(
                "SigLIPEncoder requires `transformers` with SigLIP support."
            ) from exc

        self.device = (
            torch.device(device) if not isinstance(device, torch.device) else device
        )
        self.model_id = model_id or os.environ.get(
            "HIRE_SIGLIP_MODEL", os.environ.get("DICE_SIGLIP_MODEL", "google/siglip-base-patch16-224")
        )
        log.info(
            "[SigLIPEncoder] Loading SigLIP vision model (%s) on %s...",
            self.model_id,
            self.device,
        )

        processor = AutoImageProcessor.from_pretrained(self.model_id)
        image_size = self._resolve_image_size(processor)
        image_mean = tuple(
            float(v) * 255.0 for v in getattr(processor, "image_mean", (0.5, 0.5, 0.5))
        )
        image_std = tuple(
            float(v) * 255.0 for v in getattr(processor, "image_std", (0.5, 0.5, 0.5))
        )
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.Normalize(mean=image_mean, std=image_std),
            ]
        )

        self.encoder = SiglipVisionModel.from_pretrained(self.model_id)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        self.encoder.to(self.device)
        self.output_dim = int(self.encoder.config.hidden_size)
        self._inputs_validated = False
        log.info("[SigLIPEncoder] Loaded SigLIP (output_dim=%s)", self.output_dim)

    @staticmethod
    def _resolve_image_size(processor) -> int:
        size = getattr(processor, "size", None) or {}
        if isinstance(size, dict):
            if "height" in size:
                return int(size["height"])
            if "shortest_edge" in size:
                return int(size["shortest_edge"])
        return 224

    @torch.no_grad()
    def compute_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        validate_encoder_inputs(
            "SigLIPEncoder", images, first_call=not self._inputs_validated
        )
        self._inputs_validated = True
        if images.dtype == torch.uint8:
            images = images.float()
        x = self.image_transform(images.to(self.device))
        outputs = self.encoder(pixel_values=x)
        return outputs.last_hidden_state.contiguous().float()


def build_similarity_encoder(encoder_kind: str, device: str = "cuda") -> nn.Module:
    """Construct a similarity encoder by name (dino | liv | siglip)."""
    kind = (encoder_kind or "dino").strip().lower()
    if kind == "dino":
        return DinoV2Encoder(device=device)
    if kind == "liv":
        return LIVEncoder(device=device)
    if kind == "siglip":
        return SigLIPEncoder(device=device)
    raise ValueError(
        f"Unsupported similarity encoder_kind={encoder_kind}. "
        "Expected one of: dino, liv, siglip."
    )
