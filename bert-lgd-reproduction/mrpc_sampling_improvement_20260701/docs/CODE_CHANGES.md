# Code Changes In This Variant

The changes are scoped to the copied Sherlock audit harness in `code/`.

## Trainer Changes

Files:

- `code/src/lgd_bert/train.py`
- `code/src/lgd_bert/config.py`

Changes:

- Allow `--sampler_warmup none` and set the config default to `none`.
- Permit pre-created `--run_dir` directories from the launcher.
- Add required sampler metric names:
  - `sampler/coverage_so_far`
  - `sampler/unique_count_batch`
  - `sampler/duplicate_count_batch`
  - `sampler/duplicate_frac_batch`
  - `sampler/label0_count_batch`
  - `sampler/label1_count_batch`
- Track cumulative train coverage across all epochs in `run_summary.json`.
- Extend `sample_log.csv.gz` with:
  - `fallback`
  - `duplicate_within_batch`
  - `seen_count_so_far`
- Preserve existing per-step `sample/*` metrics for compatibility.

## Launcher

File:

- `code/scripts/launch_mrpc_sampling_improvement.py`

The launcher defines only the requested five MRPC jobs:

- random epoch shuffle.
- label-aware LGD K=3, L=50.
- label-aware LGD K=4, L=50.
- label-aware LGD K=5, L=50.
- label-aware LGD K=6, L=50.

It queues across four GPUs, writes per-run logs and command files, and creates:

- `reports/MRPC_SAMPLING_IMPROVEMENT_REPORT.md`
- `reports/final_metrics.csv`
- `reports/run_manifest.csv`
- `reports/sampling_health.csv`

## What Was Not Changed

- No optimizer changes.
- No model architecture changes.
- No probability-correction math changes.
- No RTE code path changes.
- No implementation of the full supplement-style mini-batch bucket collector.

The implemented rescue is narrower: label-aware LGD with smaller K, larger L, no correction, no
warmup, and batch-without-replacement sampling.
