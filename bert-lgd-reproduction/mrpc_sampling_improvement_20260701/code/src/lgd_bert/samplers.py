from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lsh import SimHashLSH
from .probability import lsh_sampling_probability


@dataclass
class SampleRow:
    index: int
    probability: float
    collision_prob: float
    collision_prob_k: float
    cosine: float
    bucket_size: int
    attempts: int
    fallback: bool
    probability_clamped: bool
    source: str
    table_id: int | None = None
    bucket_key: int | None = None
    resample_count: int = 0


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
    sources: list[str]
    table_ids: list[int | None]
    bucket_keys: list[int | None]
    resample_counts: list[int]

    def stats(self) -> dict[str, float]:
        p = np.asarray(self.probabilities, dtype="float64")
        cp = np.asarray(self.collision_probs, dtype="float64") if self.collision_probs else np.asarray([0.0])
        cp_k = np.asarray(self.collision_probs_k, dtype="float64") if self.collision_probs_k else np.asarray([0.0])
        buckets = np.asarray([size for size in self.bucket_sizes if size > 0], dtype="float64")
        attempts = np.asarray(self.attempts, dtype="float64")
        fallbacks = np.asarray(self.fallback_flags, dtype="float64")
        unique_count = len(set(self.indices))
        duplicate_count = len(self.indices) - unique_count
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
            "sampler/unique_count": float(unique_count),
            "sampler/duplicate_count": float(duplicate_count),
            "sampler/duplicate_frac": float(duplicate_count / max(len(self.indices), 1)),
            "sampler/resample_count_mean": float(np.mean(self.resample_counts)) if self.resample_counts else 0.0,
        }

    @classmethod
    def from_rows(cls, rows: list[SampleRow]) -> "SampleBatch":
        return cls(
            indices=[row.index for row in rows],
            probabilities=[row.probability for row in rows],
            collision_probs=[row.collision_prob for row in rows],
            collision_probs_k=[row.collision_prob_k for row in rows],
            cosines=[row.cosine for row in rows],
            bucket_sizes=[row.bucket_size for row in rows],
            attempts=[row.attempts for row in rows],
            fallback_flags=[row.fallback for row in rows],
            probability_clamped_flags=[row.probability_clamped for row in rows],
            sources=[row.source for row in rows],
            table_ids=[row.table_id for row in rows],
            bucket_keys=[row.bucket_key for row in rows],
            resample_counts=[row.resample_count for row in rows],
        )


class RandomBatchSampler:
    def __init__(self, train_size: int, seed: int = 0, mode: str = "with_replacement") -> None:
        self.train_size = int(train_size)
        self.rng = np.random.default_rng(seed)
        self.mode = mode
        self._permutation: list[int] = []
        self._cursor = 0
        if self.mode not in {"with_replacement", "epoch_shuffle"}:
            raise ValueError(f"Unsupported random sampler mode: {self.mode}")

    def reset_epoch(self) -> None:
        self._permutation = self.rng.permutation(self.train_size).astype(int).tolist()
        self._cursor = 0

    def sample_batch(self, batch_size: int) -> SampleBatch:
        if self.mode == "with_replacement":
            indices = self.rng.integers(0, self.train_size, size=int(batch_size)).astype(int).tolist()
        else:
            if not self._permutation or self._cursor >= self.train_size:
                self.reset_epoch()
            end = min(self._cursor + int(batch_size), self.train_size)
            indices = self._permutation[self._cursor : end]
            self._cursor = end
        rows = [
            SampleRow(
                index=int(index),
                probability=1.0 / self.train_size,
                collision_prob=0.0,
                collision_prob_k=0.0,
                cosine=0.0,
                bucket_size=self.train_size,
                attempts=1,
                fallback=False,
                probability_clamped=False,
                source="random",
            )
            for index in indices
        ]
        return SampleBatch.from_rows(rows)


class LGDBatchSampler:
    def __init__(
        self,
        lsh: SimHashLSH,
        stored_vectors: np.ndarray,
        train_size: int,
        seed: int = 0,
        probability_eps: float = 1e-12,
        replacement_mode: str = "with_replacement",
        max_resample_attempts: int = 128,
    ) -> None:
        self.lsh = lsh
        self.stored_vectors = np.asarray(stored_vectors, dtype="float32")
        self.train_size = int(train_size)
        self.rng = np.random.default_rng(seed)
        self.probability_eps = float(probability_eps)
        self.replacement_mode = replacement_mode
        self.max_resample_attempts = int(max_resample_attempts)
        self.used_this_epoch: set[int] = set()
        if self.replacement_mode not in {"with_replacement", "batch_without_replacement", "epoch_no_reuse"}:
            raise ValueError(f"Unsupported LGD replacement mode: {self.replacement_mode}")

    def update_index(self, lsh: SimHashLSH, stored_vectors: np.ndarray) -> None:
        self.lsh = lsh
        self.stored_vectors = np.asarray(stored_vectors, dtype="float32")

    def reset_epoch(self) -> None:
        self.used_this_epoch.clear()

    def sample_one(self, query: np.ndarray, avoid_indices: set[int] | None = None) -> SampleRow:
        avoid_indices = avoid_indices or set()
        table_order = self.rng.permutation(self.lsh.l)
        for attempts, table_id in enumerate(table_order, start=1):
            key, bucket = self.lsh.query_bucket(query, int(table_id))
            if avoid_indices:
                bucket = [index for index in bucket if int(index) not in avoid_indices]
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
            return SampleRow(
                index=index,
                probability=p,
                collision_prob=cp,
                collision_prob_k=cp_k,
                cosine=cos,
                bucket_size=bucket_size,
                attempts=attempts,
                fallback=False,
                probability_clamped=clamped,
                source="lsh",
                table_id=int(table_id),
                bucket_key=int(key),
            )
        available = np.setdiff1d(np.arange(self.train_size, dtype=np.int64), np.fromiter(avoid_indices, dtype=np.int64), assume_unique=False)
        if available.size:
            index = int(self.rng.choice(available))
        else:
            index = int(self.rng.integers(0, self.train_size))
        return SampleRow(
            index=index,
            probability=1.0 / self.train_size,
            collision_prob=0.0,
            collision_prob_k=0.0,
            cosine=0.0,
            bucket_size=self.train_size,
            attempts=self.lsh.l,
            fallback=True,
            probability_clamped=False,
            source="fallback",
        )

    def sample_batch(self, batch_size: int, query: np.ndarray) -> SampleBatch:
        rows: list[SampleRow] = []
        batch_seen: set[int] = set()
        for _ in range(int(batch_size)):
            avoid_indices: set[int] = set()
            if self.replacement_mode == "batch_without_replacement":
                avoid_indices = set(batch_seen)
            elif self.replacement_mode == "epoch_no_reuse":
                if len(self.used_this_epoch) >= self.train_size:
                    self.reset_epoch()
                avoid_indices = set(self.used_this_epoch) | set(batch_seen)

            row = self.sample_one(query, avoid_indices=avoid_indices)
            resample_count = 0
            while (
                self.replacement_mode == "batch_without_replacement"
                and row.index in batch_seen
                and resample_count < self.max_resample_attempts
            ):
                resample_count += 1
                row = self.sample_one(query, avoid_indices=set(batch_seen))
            row.resample_count = resample_count
            rows.append(row)
            batch_seen.add(row.index)
            if self.replacement_mode == "epoch_no_reuse":
                self.used_this_epoch.add(row.index)
        return SampleBatch.from_rows(rows)
