from __future__ import annotations

import numpy as np

from lgd_bert.lsh import SimHashLSH
from lgd_bert.samplers import LGDBatchSampler


def test_sampler_returns_shapes_and_positive_probabilities():
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(40, 6)).astype("float32")
    lsh = SimHashLSH(dim=6, k=3, l=4, seed=2)
    lsh.fit(vectors)
    sampler = LGDBatchSampler(lsh, vectors, train_size=40, seed=3)
    batch = sampler.sample_batch(8, rng.normal(size=(6,)).astype("float32"))
    assert len(batch.indices) == 8
    assert len(batch.probabilities) == 8
    assert all(p > 0 for p in batch.probabilities)
    assert len(batch.bucket_sizes) == 8
    assert len(batch.attempts) == 8


def test_sampler_fallback_path():
    rng = np.random.default_rng(4)
    vectors = rng.normal(size=(10, 4)).astype("float32")
    lsh = SimHashLSH(dim=4, k=4, l=3, seed=5)
    lsh.fit(vectors)
    lsh.tables = [{} for _ in range(lsh.l)]
    sampler = LGDBatchSampler(lsh, vectors, train_size=10, seed=6)
    batch = sampler.sample_batch(5, rng.normal(size=(4,)).astype("float32"))
    assert all(batch.fallback_flags)
    assert all(np.isclose(p, 1.0 / 10) for p in batch.probabilities)
