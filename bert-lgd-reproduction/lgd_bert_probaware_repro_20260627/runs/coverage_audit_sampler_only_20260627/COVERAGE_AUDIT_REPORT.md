# Sample Coverage Audit

Audit mode: full 3-epoch sampler draw schedule, no BERT/classifier weight updates, no W&B.

| Task | Variant | Epoch | Train N | Draws | Unique seen | Repeated samples | Duplicate draws | Never seen | Max times seen | Fallback frac |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mrpc | random | 0 | 3668 | 3680 | 2338 | 963 | 1342 | 1330 | 7 | 0.0000 |
| mrpc | random | 1 | 3668 | 3680 | 2319 | 976 | 1361 | 1349 | 6 | 0.0000 |
| mrpc | random | 2 | 3668 | 3680 | 2278 | 978 | 1402 | 1390 | 6 | 0.0000 |
| mrpc | paper_lgd | 0 | 3668 | 3680 | 3 | 3 | 3677 | 3665 | 1256 | 0.0000 |
| mrpc | paper_lgd | 1 | 3668 | 3680 | 3 | 3 | 3677 | 3665 | 1239 | 0.0000 |
| mrpc | paper_lgd | 2 | 3668 | 3680 | 3 | 3 | 3677 | 3665 | 1234 | 0.0000 |
| mrpc | label_aware_lgd | 0 | 3668 | 3680 | 3 | 3 | 3677 | 3665 | 1256 | 0.0000 |
| mrpc | label_aware_lgd | 1 | 3668 | 3680 | 3 | 3 | 3677 | 3665 | 1239 | 0.0000 |
| mrpc | label_aware_lgd | 2 | 3668 | 3680 | 3 | 3 | 3677 | 3665 | 1234 | 0.0000 |
| rte | random | 0 | 2490 | 2496 | 1581 | 650 | 915 | 909 | 6 | 0.0000 |
| rte | random | 1 | 2490 | 2496 | 1579 | 660 | 917 | 911 | 6 | 0.0000 |
| rte | random | 2 | 2490 | 2496 | 1573 | 649 | 923 | 917 | 7 | 0.0000 |
| rte | paper_lgd | 0 | 2490 | 2496 | 14 | 14 | 2482 | 2476 | 497 | 0.0000 |
| rte | paper_lgd | 1 | 2490 | 2496 | 14 | 14 | 2482 | 2476 | 515 | 0.0000 |
| rte | paper_lgd | 2 | 2490 | 2496 | 14 | 14 | 2482 | 2476 | 511 | 0.0000 |
| rte | label_aware_lgd | 0 | 2490 | 2496 | 6 | 6 | 2490 | 2484 | 508 | 0.0000 |
| rte | label_aware_lgd | 1 | 2490 | 2496 | 6 | 6 | 2490 | 2484 | 526 | 0.0000 |
| rte | label_aware_lgd | 2 | 2490 | 2496 | 6 | 6 | 2490 | 2484 | 516 | 0.0000 |

Definitions:

- `Repeated samples`: number of distinct training examples sampled at least twice within that epoch.
- `Duplicate draws`: total extra uses beyond the first use, i.e. `draw_count - unique_seen_count`.
- `Never seen`: training examples with zero draws within that epoch.
