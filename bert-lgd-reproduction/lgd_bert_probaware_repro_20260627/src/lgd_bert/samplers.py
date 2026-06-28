from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lsh import SimHashLSH
from .probability import lsh_sampling_probability


@dataclass
class SampleBatch:
    indices: list[int]
    probabilities: list[float]
    collision_probs: list[float]
    collision_probs_k: list[float]
    cosines: list[float]
    bucket_sizes: list[int]
    attempts: list[int]
    fallback_flags: list[bool]
    probability_clamped_flags: list[bool]

    def stats(self) -> dict[str, float]:
        p = np.asarray(self.probabilities, dtype="float64")
        cp = np.asarray(self.collision_probs, dtype="float64") if self.collision_probs else np.asarray([0.0])
        cp_k = np.asarray(self.collision_probs_k, dtype="float64") if self.collision_probs_k else np.asarray([0.0])
        buckets = np.asarray([size for size in self.bucket_sizes if size > 0], dtype="float64")
        attempts = np.asarray(self.attempts, dtype="float64")
        fallbacks = np.asarray(self.fallback_flags, dtype="float64")
        return {
            "sampler/p_mean": float(p.mean()),
            "sampler/p_min": float(p.min()),
            "sampler/p_max": float(p.max()),
            "sampler/cp_mean": float(cp.mean()),
            "sampler/cpK_mean": float(cp_k.mean()),
            "sampler/bucket_size_mean": float(buckets.mean()) if buckets.size else 0.0,
            "sampler/bucket_size_max": float(buckets.max()) if buckets.size else 0.0,
            "sampler/attempts_mean": float(attempts.mean()) if attempts.size else 0.0,
            "sampler/fallback_rate": float(fallbacks.mean()) if fallbacks.size else 0.0,
            "sampler/probability_clamped_frac": float(np.mean(self.probability_clamped_flags)) if self.probability_clamped_flags else 0.0,
        }


class RandomBatchSampler:
    def __init__(self, train_size: int, seed: int = 0) -> None:
        self.train_size = int(train_size)
        self.rng = np.random.default_rng(seed)

    def sample_batch(self, batch_size: int) -> SampleBatch:
        indices = self.rng.integers(0, self.train_size, size=int(batch_size)).astype(int).tolist()
        return SampleBatch(
            indices=indices,
            probabilities=[1.0 / self.train_size] * len(indices),
            collision_probs=[0.0] * len(indices),
            collision_probs_k=[0.0] * len(indices),
            cosines=[0.0] * len(indices),
            bucket_sizes=[self.train_size] * len(indices),
            attempts=[1] * len(indices),
            fallback_flags=[False] * len(indices),
            probability_clamped_flags=[False] * len(indices),
        )


class LGDBatchSampler:
    def __init__(
        self,
        lsh: SimHashLSH,
        stored_vectors: np.ndarray,
        train_size: int,
        seed: int = 0,
        probability_eps: float = 1e-12,
    ) -> None:
        self.lsh = lsh
        self.stored_vectors = np.asarray(stored_vectors, dtype="float32")
        self.train_size = int(train_size)
        self.rng = np.random.default_rng(seed)
        self.probability_eps = float(probability_eps)

    def update_index(self, lsh: SimHashLSH, stored_vectors: np.ndarray) -> None:
        self.lsh = lsh
        self.stored_vectors = np.asarray(stored_vectors, dtype="float32")

    def sample_one(self, query: np.ndarray) -> tuple[int, float, float, float, float, int, int, bool, bool]:
        table_order = self.rng.permutation(self.lsh.l)
        for attempts, table_id in enumerate(table_order, start=1):
            _key, bucket = self.lsh.query_bucket(query, int(table_id))
            if not bucket:
                continue
            index = int(self.rng.choice(bucket))
            bucket_size = int(len(bucket))
            p, cp, cp_k, cos, clamped = lsh_sampling_probability(
                self.stored_vectors[index],
                query,
                self.lsh.k,
                bucket_size,
                attempts,
                eps=self.probability_eps,
            )
            return index, p, cp, cp_k, cos, bucket_size, attempts, False, clamped
        index = int(self.rng.integers(0, self.train_size))
        return index, 1.0 / self.train_size, 0.0, 0.0, 0.0, self.train_size, self.lsh.l, True, False

    def sample_batch(self, batch_size: int, query: np.ndarray) -> SampleBatch:
        rows = [self.sample_one(query) for _ in range(int(batch_size))]
        return SampleBatch(
            indices=[row[0] for row in rows],
            probabilities=[row[1] for row in rows],
            collision_probs=[row[2] for row in rows],
            collision_probs_k=[row[3] for row in rows],
            cosines=[row[4] for row in rows],
            bucket_sizes=[row[5] for row in rows],
            attempts=[row[6] for row in rows],
            fallback_flags=[row[7] for row in rows],
            probability_clamped_flags=[row[8] for row in rows],
        )

