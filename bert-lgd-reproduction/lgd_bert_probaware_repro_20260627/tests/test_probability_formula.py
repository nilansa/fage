from __future__ import annotations

import numpy as np

from lgd_bert.probability import lsh_sampling_probability, simhash_collision_probability


def test_collision_probability_monotonic_and_cp_k():
    low = simhash_collision_probability(-0.5)
    mid = simhash_collision_probability(0.0)
    high = simhash_collision_probability(0.8)
    assert low < mid < high
    vector = np.array([1.0, 0.0], dtype="float32")
    query = np.array([1.0, 0.0], dtype="float32")
    p, cp, cp_k, _cos, _clamped = lsh_sampling_probability(vector, query, k=7, bucket_size=4, attempts=2)
    assert np.isclose(cp_k, cp**7)
    assert np.isclose(p, cp_k * (1.0 - cp_k) / 4.0)

