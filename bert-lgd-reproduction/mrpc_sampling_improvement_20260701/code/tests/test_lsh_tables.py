from __future__ import annotations

import numpy as np

from lgd_bert.lsh import SimHashLSH


def test_every_index_appears_once_per_table():
    vectors = np.random.default_rng(0).normal(size=(32, 8)).astype("float32")
    lsh = SimHashLSH(dim=8, k=4, l=5, seed=0)
    lsh.fit(vectors)
    assert len(lsh.tables) == 5
    for table in lsh.tables:
        seen = sorted(index for bucket in table.values() for index in bucket)
        assert seen == list(range(32))
        assert all(0 <= key < 2**4 for key in table)
