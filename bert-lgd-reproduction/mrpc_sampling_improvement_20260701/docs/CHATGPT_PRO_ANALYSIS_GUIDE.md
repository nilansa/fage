# ChatGPT Pro Analysis Guide

## Suggested Reading Order

1. `README.md`
2. `reports/MRPC_SAMPLING_IMPROVEMENT_REPORT.md`
3. `docs/ASSUMPTIONS_AND_SCOPE.md`
4. `docs/PIPELINE.md`
5. `docs/CODE_CHANGES.md`
6. `reports/final_metrics.csv`
7. `reports/sampling_health.csv`
8. `runs/*/sample_step_log.csv`
9. `runs/*/sample_log.csv.gz` if per-sample behavior is needed.

## Main Question

Sampling collapse appears repaired, but LGD still does not beat random. The next analysis should ask:

Does the label-aware LGD query score correlate with high-loss or high-gradient-norm MRPC examples
after the no-duplicate coverage repair?

## Evidence To Compare

Final metrics:

- `reports/final_metrics.csv`

Sampling health:

- `reports/sampling_health.csv`

Per-step sampler behavior:

- `runs/MRPC_LA_K3_L50_BWoR_refreshEpoch_noCorr_seed0/train_log.csv`
- `runs/MRPC_LA_K4_L50_BWoR_refreshEpoch_noCorr_seed0/train_log.csv`
- `runs/MRPC_LA_K5_L50_BWoR_refreshEpoch_noCorr_seed0/train_log.csv`
- `runs/MRPC_LA_K6_L50_BWoR_refreshEpoch_noCorr_seed0/train_log.csv`

Per-sample behavior:

- `runs/*/sample_log.csv.gz`

## Things To Check

- Whether lower K improves coverage at the cost of making buckets too broad.
- Whether K=3's strong coverage is mostly due to very large buckets and weak selectivity.
- Whether K=6's lower coverage but strong F1 indicates better selectivity than K=4/K=5.
- Whether sampled label ratio drift explains the accuracy differences.
- Whether `sampler/query_norm` or LSH refresh timing aligns with evaluation changes.
- Whether the label-aware score should use a different query sign, classifier margin, or loss-aware
  proxy for MRPC.

## Decision Boundary

If query-loss correlation is weak, do not move to probability correction yet. The next implementation
should be a query/loss-alignment audit or a supplement-style mini-batch bucket collector.

If query-loss correlation is strong for K=3 or K=6, use that K as the next candidate for a small
refresh-rate sweep.
