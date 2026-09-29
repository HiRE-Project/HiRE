"""Contrastive potentials over corresponding normalized visual patches."""
from typing import Optional
import torch
import torch.nn.functional as F


def smooth_max(values: torch.Tensor, temperature: float, dim: int = -1) -> torch.Tensor:
    """LogSumExp / temperature, retaining the experiment's unnormalized convention."""
    return torch.logsumexp(float(temperature) * values, dim=dim) / float(temperature)


def reference_similarity(current: torch.Tensor, references: Optional[torch.Tensor], temperature: float) -> torch.Tensor:
    if references is None or references.numel() == 0:
        return current.new_zeros(current.shape[0])
    references = references.to(device=current.device, dtype=current.dtype)
    pairwise = torch.einsum("bpd,kpd->bkp", F.normalize(current, dim=-1), F.normalize(references, dim=-1))
    return smooth_max(pairwise.mean(dim=-1), temperature)


def contrastive_score(positive, negative, weight=0.9):
    return positive - float(weight) * negative
