from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ProbabilityStats:
    weights: torch.Tensor
    weight_mean: float
    weight_min: float
    weight_max: float
    weight_clipped_frac: float


def cosine_similarity(vector: np.ndarray, query: np.ndarray, eps: float = 1e-12) -> float:
    v = np.asarray(vector, dtype="float32")
    q = np.asarray(query, dtype="float32")
    denom = (float(np.linalg.norm(v)) * float(np.linalg.norm(q))) + eps
    return float(np.dot(v, q) / denom)


def simhash_collision_probability(cosine: float, eps: float = 1e-7) -> float:
    clipped = float(np.clip(cosine, -1.0 + eps, 1.0 - eps))
    return float(1.0 - np.arccos(clipped) / np.pi)


def lsh_sampling_probability(
    vector: np.ndarray,
    query: np.ndarray,
    k: int,
    bucket_size: int,
    attempts: int,
    eps: float = 1e-12,
) -> tuple[float, float, float, float, bool]:
    if bucket_size <= 0 or attempts <= 0:
        raise ValueError("bucket_size and attempts must be positive")
    cos = cosine_similarity(vector, query)
    cp = simhash_collision_probability(cos)
    cp_k = float(cp**k)
    p = float(cp_k * ((1.0 - cp_k) ** (attempts - 1)) * (1.0 / bucket_size))
    clamped = p < eps
    return float(max(p, eps)), cp, cp_k, cos, clamped


def corrected_loss_weights(
    probabilities: torch.Tensor,
    train_size: int,
    mode: str,
    max_weight: float = 10.0,
    eps: float = 1e-12,
) -> ProbabilityStats:
    if mode == "none":
        weights = torch.ones_like(probabilities)
        clipped_frac = 0.0
    else:
        raw = 1.0 / (float(train_size) * torch.clamp(probabilities, min=eps))
        if mode == "full":
            weights = raw
            clipped_frac = 0.0
        elif mode == "clipped":
            weights = torch.clamp(raw, max=max_weight)
            clipped_frac = float((raw > max_weight).float().mean().detach().cpu())
        elif mode == "sqrt":
            weights = torch.sqrt(raw)
            clipped_frac = 0.0
        elif mode == "normalized":
            weights = raw / torch.clamp(raw.mean(), min=eps)
            clipped_frac = 0.0
        else:
            raise ValueError(f"Unsupported correction mode: {mode}")
    detached = weights.detach().float().cpu()
    return ProbabilityStats(
        weights=weights,
        weight_mean=float(detached.mean()),
        weight_min=float(detached.min()),
        weight_max=float(detached.max()),
        weight_clipped_frac=clipped_frac,
    )

