from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class LSHSummary:
    table_count: int
    k: int
    dim: int
    nonempty_bucket_mean: float
    nonempty_bucket_max: int
    nonempty_bucket_min: int


class SimHashLSH:
    def __init__(self, dim: int, k: int = 7, l: int = 10, seed: int = 0) -> None:
        self.dim = int(dim)
        self.k = int(k)
        self.l = int(l)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        projections = self.rng.normal(size=(self.l, self.k, self.dim)).astype("float32")
        projections /= np.linalg.norm(projections, axis=2, keepdims=True) + 1e-12
        self.projections = projections
        self.tables: list[dict[int, list[int]]] = [defaultdict(list) for _ in range(self.l)]

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype="float32")
        return vectors / (np.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-12)

    def hash_vector(self, vector: np.ndarray, table_id: int) -> int:
        row = self._normalize(np.asarray(vector, dtype="float32").reshape(1, -1))[0]
        bits = (self.projections[int(table_id)] @ row >= 0).astype(np.uint8)
        key = 0
        for bit in bits:
            key = (key << 1) | int(bit)
        return int(key)

    def fit(self, vectors: np.ndarray, indices: np.ndarray | None = None) -> None:
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected vectors with shape [n,{self.dim}], got {vectors.shape}")
        if indices is None:
            indices = np.arange(vectors.shape[0])
        indices = np.asarray(indices, dtype=np.int64)
        if indices.shape[0] != vectors.shape[0]:
            raise ValueError("indices length must match vectors length")
        self.tables = [defaultdict(list) for _ in range(self.l)]
        normalized = self._normalize(vectors)
        for row, index in zip(normalized, indices):
            for table_id in range(self.l):
                key = self.hash_vector(row, table_id)
                self.tables[table_id][key].append(int(index))

    def query_bucket(self, query: np.ndarray, table_id: int) -> tuple[int, list[int]]:
        key = self.hash_vector(query, table_id)
        return key, list(self.tables[int(table_id)].get(key, []))

    def all_bucket_sizes(self) -> list[int]:
        return [len(bucket) for table in self.tables for bucket in table.values()]

    def summary(self) -> LSHSummary:
        sizes = self.all_bucket_sizes()
        return LSHSummary(
            table_count=self.l,
            k=self.k,
            dim=self.dim,
            nonempty_bucket_mean=float(np.mean(sizes)) if sizes else 0.0,
            nonempty_bucket_max=int(max(sizes)) if sizes else 0,
            nonempty_bucket_min=int(min(sizes)) if sizes else 0,
        )

